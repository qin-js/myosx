import os.path as osp
import math
import abc
from torch.utils.data import DataLoader
import torch.optim
import torch.nn as nn
import torchvision.transforms as transforms
from common.timer import Timer
from common.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from collections import OrderedDict
if cfg.decoder_setting == 'normal':
    print("normal mode")
    from OSX import get_model
elif cfg.decoder_setting == 'wo_face_decoder':
    print("no face decoder")
    from OSX_WoFaceDecoder import get_model
elif cfg.decoder_setting == 'wo_decoder':
    print("no decoder")
    from OSX_WoDecoder import get_model
elif cfg.decoder_setting == "pytorch":
    print("pytorch implement")
    from model_core import get_model
from dataset import MultipleDatasets
# dynamic dataset import
for i in range(len(cfg.trainset_3d)):
    exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
for i in range(len(cfg.trainset_2d)):
    exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name='train_logs.txt')

    def get_optimizer(self, model):
        """
        构建 optimizer，只包含可训练参数。

        参数分组:
        ┌─────────────────────────────┬──────────────────┐
        │ trainable_modules           │ lr = cfg.lr      │
        │ (hand/face position_net,    │                  │
        │  hand/face decoder)         │                  │
        ├─────────────────────────────┼──────────────────┤
        │ special_trainable_modules   │ lr = cfg.lr *    │
        │ (hand_regressor,            │   cfg.lr_mult    │
        │  face_regressor)            │                  │
        └─────────────────────────────┴──────────────────┘
        """
        normal_param = []
        special_param = []

        # model.module 因为 DataParallel 包裹
        for module in model.module.trainable_modules:
            normal_param += list(module.parameters())

        for module in model.module.special_trainable_modules:
            special_param += list(module.parameters())

        optim_params = [
            {
                'params': normal_param,
                'lr': cfg.lr,
                'name': 'new_modules',
            },
            {
                'params': special_param,
                'lr': cfg.lr * cfg.lr_mult,
                'name': 'regressors',
            },
        ]

        optimizer = torch.optim.Adam(optim_params, lr=cfg.lr)

        # 打印参数统计
        self.logger.info(f"Optimizer 参数分组:")
        self.logger.info(f"  new_modules: {sum(p.numel() for p in normal_param):,} params, "
                         f"lr={cfg.lr}")
        self.logger.info(f"  regressors:  {sum(p.numel() for p in special_param):,} params, "
                         f"lr={cfg.lr * cfg.lr_mult}")

        return optimizer

    #  模型保存：区分冻结/训练模块
    # ================================================================
    def save_model(self, state, epoch):
        """
        保存 checkpoint:
        真正的极致节省空间：只保存发生了梯度更新的模块参数。
        """
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))

        # ---- 提取可训练模块(包含新加入的模块 + 微调的Regressor) ----
        trainable_state = {}
        for k, v in state['network'].items():
            # 去掉 "module." 前缀后检查模块名
            clean_key = k.replace('module.', '', 1)
            module_name = clean_key.split('.')[0]

            # ⚠️ 注意：这里必须确保你的 trainable_module_names 中
            # 包含了 'hand_regressor' 和 'face_regressor'，因为它们也在参与训练！
            # 如果没包含，可以使用 param.requires_grad 判断，或者显式写出：
            # if module_name in self.model.module.trainable_module_names or \
            #    module_name in['hand_regressor', 'face_regressor']:
            if module_name in self.model.module.trainable_module_names:
                # 过滤掉不需要的 smplx_layer
                if 'smplx_layer' not in k:
                    trainable_state[k] = v

        # ---- 核心改动：用精简版替换掉庞大的完整模型 ----
        state['network'] = trainable_state 

        # 记录冻结/训练信息（便于恢复时验证）
        state['frozen_modules'] = self.model.module.frozen_modules
        state['trainable_module_names'] = self.model.module.trainable_module_names

        torch.save(state, file_path)
        self.logger.info(f"Write lightweight snapshot into {file_path}")
        self.logger.info(f"  Saved trainable parameters only: {len(state['network'])} tensors.")

    # ================================================================
    #  模型加载：从预训练 checkpoint 加载冻结模块权重
    # ================================================================
    def load_model(self, model, optimizer):
        """
        加载策略:

        情况 1: cfg.continue_train = True
          → 从我们自己保存的 checkpoint 恢复（包含新模块权重）

        情况 2: cfg.pretrained_model_path 存在
          → 从原始 OSX 预训练模型加载冻结模块权重
          → 新模块保持随机初始化
        """
        if cfg.continue_train:
            # ---- 恢复训练：加载我们自己的 checkpoint ----
            start_epoch, model, optimizer = self._load_resume_checkpoint(
                model, optimizer
            )
        elif cfg.pretrained_model_path is not None:
            # ---- 新训练：从原始 OSX 加载冻结模块 ----
            start_epoch = 0
            self._load_pretrained_frozen(model, cfg.pretrained_model_path)
        else:
            start_epoch = 0
            self.logger.warning("⚠️ 未提供预训练模型！冻结模块使用随机初始化（不推荐）")

        return start_epoch, model, optimizer

    def _load_pretrained_frozen(self, model, ckpt_path):
        """
        从原始 OSX 预训练 checkpoint 加载冻结模块和 Regressor 的权重。
        显式跳过新重写的 PositionNet 和 Decoder。
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # 获取原始字典
        if 'network' in ckpt: pretrained_dict = ckpt['network']
        elif 'state_dict' in ckpt: pretrained_dict = ckpt['state_dict']
        elif 'model' in ckpt: pretrained_dict = ckpt['model']
        else: pretrained_dict = ckpt

        # 1. 预处理：统一剥离 'module.' 前缀，方便后续做纯净的字符串匹配
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        
        model_dict = model.state_dict() # 这里的 key 都有 'module.' 前缀 (因为用了 DataParallel)
        new_state_dict = {}

        loaded_keys = []
        skipped_keys = []
        shape_mismatch =[]

        for k, v in pretrained_dict.items():
            
            # [映射逻辑 1]: 显式跳过全新重写的模块 (让它们保持随机初始化)
            if k.startswith('hand_decoder.') or k.startswith('face_decoder.') or \
               k.startswith('hand_position_net.') or k.startswith('face_position_net.'):
                skipped_keys.append(k)
                continue
                
            #[映射逻辑 2]: Encoder 的 last_norm 名字替换
            if 'encoder.last_norm.' in k:
                k = k.replace('encoder.last_norm.', 'encoder.norm.')
                
            # 找到对应在你当前 DataParallel 模型中的 key
            target_key = 'module.' + k
            
            #[映射逻辑 3]: 处理位置编码的自动扩充 (193 -> 223)
            if k == 'encoder.pos_embed' and target_key in model_dict:
                target_shape = model_dict[target_key].shape
                if v.shape[1] == 193 and target_shape[1] == 223:
                    cls_pos = v[:, :1, :]
                    patch_pos = v[:, 1:, :]
                    task_pos = cls_pos.repeat(1, 31, 1)
                    v = torch.cat([task_pos, patch_pos], dim=1) # 请确认你的拼接顺序
                    self.logger.info(f"✨ 自动扩充 pos_embed 形状: (1,193,1024) -> {v.shape}")

            # 验证形状并存入新字典
            if target_key in model_dict:
                if model_dict[target_key].shape == v.shape:
                    new_state_dict[target_key] = v
                    loaded_keys.append(target_key)
                else:
                    shape_mismatch.append(f"{target_key}: ckpt {v.shape} vs model {model_dict[target_key].shape}")
            else:
                skipped_keys.append(k)

        # 一键加载所有对齐的权重 (包含冻结的 backbone 和需要微调的 regressors)
        model.load_state_dict(new_state_dict, strict=False)

        # ---- 日志汇报 ----
        self.logger.info(f"从 {ckpt_path} 加载预训练权重:")
        self.logger.info(f"  ✅ 成功加载: {len(loaded_keys)} 个参数")
        self.logger.info(f"  ⏭️  故意跳过/未找到: {len(skipped_keys)} 个参数 (通常是新的 Decoder/PositionNet)")

        if shape_mismatch:
            self.logger.warning(f"  ⚠️  形状不匹配: {len(shape_mismatch)} 个")
            for s in shape_mismatch[:5]:
                self.logger.warning(f"     {s}")
                
        # 验证你的冻结模块是否全都被覆盖了
        frozen_missing =[name for name, param in model.named_parameters() 
                          if name.replace('module.', '', 1).split('.')[0] in model.module.frozen_modules 
                          and name not in loaded_keys]
        
        if frozen_missing:
            self.logger.warning(f"  ⚠️  严重警告: 冻结模块中有 {len(frozen_missing)} 个参数未被预训练权重覆盖!")
            for k in frozen_missing[:10]: self.logger.warning(f"     {k}")
        else:
            self.logger.info(f"  🎯 冻结模块安全检查: 完美! 所有冻结参数均已成功加载预训练权重。")
        # (可选) 同时加载回归网络的预训练权重用于微调
        # self._load_regressor_weights(model, pretrained_dict)

    def _load_regressor_weights(self, model, pretrained_dict):
        """
        从预训练模型加载回归网络权重用于微调。
        原始 OSX 的 hand_regressor 和 face_regressor 结构没变，可以直接加载。
        """
        model_dict = model.state_dict()
        regressor_loaded = 0

        for k, v in pretrained_dict.items():
            candidates = [k]
            if not k.startswith('module.'):
                candidates.append('module.' + k)

            for candidate_key in candidates:
                if candidate_key in model_dict:
                    clean = candidate_key.replace('module.', '', 1)
                    module_name = clean.split('.')[0]

                    # 只加载回归网络
                    if module_name in ('hand_regressor', 'face_regressor'):
                        if model_dict[candidate_key].shape == v.shape:
                            model_dict[candidate_key] = v
                            regressor_loaded += 1
                    break

        model.load_state_dict(model_dict, strict=False)
        self.logger.info(f"  回归网络预训练加载: {regressor_loaded} 个参数")

    def _load_resume_checkpoint(self, model, optimizer):
        """
        恢复训练（两步走加载策略）：
        1. 先加载冻结基座（从原始 OSX）
        2. 再加载训练进度（从自己的轻量级 Checkpoint）
        """
        # --- 步骤 1: 加载冻结的 Backbone ---
        if cfg.pretrained_model_path is not None:
            self.logger.info("Resume Step 1: Loading frozen backbone from original pretrained model...")
            self._load_pretrained_frozen(model, cfg.pretrained_model_path)
        else:
            self.logger.warning("Resume Step 1: No pretrained model found! Backbone will be random!")

        # --- 步骤 2: 加载自己保存的轻量级 Checkpoint ---
        ckpt_path = cfg.continue_train_path # 假设你配置里指定了要恢复的 snapshot 路径
        self.logger.info(f"Resume Step 2: Loading trained modules from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        start_epoch = ckpt.get('epoch', 0) + 1

        if 'network' in ckpt:
            model_dict = model.state_dict()
            trained_dict = ckpt['network']  # 这里面现在只有 trainable 部分

            loaded = 0
            for k, v in trained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded += 1

            # strict=False 允许只覆盖可训练部分，不动刚刚加载好的 Backbone
            model.load_state_dict(model_dict, strict=False) 
            
            self.logger.info(f"  成功覆盖了 {loaded} 个已训练的参数张量")

        # ---- 加载 optimizer 状态 ----
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                self.logger.info(f"  成功恢复 optimizer 状态")
            except Exception as e:
                self.logger.warning(f"⚠️ 无法恢复 optimizer 状态: {e}")

        self.logger.info(f"从 epoch {start_epoch} 准备继续训练!")
        return start_epoch, model, optimizer
        
    # ================================================================
    #  学习率获取（不变）
    # ================================================================
    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    # ================================================================
    #  数据加载（不变）
    # ================================================================
    def _make_batch_generator(self):
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))
        valid_loader_num = 0
        if len(trainset3d_loader) > 0:
            trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []
        if len(trainset2d_loader) > 0:
            trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []

        if valid_loader_num > 1:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(
            dataset=trainset_loader,
            batch_size=cfg.num_gpus * cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_thread,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,    # 保持 worker 存活
            prefetch_factor=3,          # 内存充足，多预取
        )

    # ================================================================
    #  模型构建：集成冻结逻辑
    # ================================================================
    def _make_model(self):
        self.logger.info("Creating graph and optimizer...")

        # 1. 构建模型
        model = get_model('train')
        model = DataParallel(model).cuda()

        # 2. 加载预训练权重（冻结模块从原始 OSX 加载）
        optimizer = self.get_optimizer(model)  # 先创建 optimizer（只包含可训练参数）

        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
            if cfg.pretrained_model_path is not None:
                self._load_pretrained_frozen(model, cfg.pretrained_model_path)

        # 3. 冻结模块
        model.module.freeze_modules()

        # 4. 设置训练模式（重写的 train() 会保持冻结模块 eval）
        model.train()

        # 5. 验证冻结状态
        self._verify_freeze_status(model)

        # 6. Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            cfg.end_epoch * self.itr_per_epoch,
            eta_min=1e-6,
        )

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    # ================================================================
    #  验证工具
    # ================================================================
    def _verify_freeze_status(self, model):
        """验证冻结/训练状态是否正确"""
        frozen_count = 0
        trainable_count = 0
        errors = []

        for name, param in model.named_parameters():
            clean_name = name.replace('module.', '', 1)
            module_name = clean_name.split('.')[0]

            is_frozen_module = module_name in model.module.frozen_modules
            is_trainable_module = module_name in model.module.trainable_module_names

            if is_frozen_module:
                if param.requires_grad:
                    errors.append(f"❌ {name}: 冻结模块但 requires_grad=True")
                frozen_count += 1
            elif is_trainable_module:
                if not param.requires_grad:
                    errors.append(f"❌ {name}: 训练模块但 requires_grad=False")
                trainable_count += 1

        # 检查 BN 层模式
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                clean_name = name.replace('module.', '', 1)
                top_module = clean_name.split('.')[0]
                if top_module in model.module.frozen_modules and module.training:
                    errors.append(f"❌ {name}: 冻结模块的 BN 应该在 eval 模式")

        if errors:
            self.logger.error("冻结状态验证失败:")
            for e in errors:
                self.logger.error(f"  {e}")
            raise RuntimeError("冻结状态异常，请检查 freeze_modules()")
        else:
            self.logger.info(f"✅ 冻结状态验证通过: "
                             f"冻结 {frozen_count} 个参数, "
                             f"训练 {trainable_count} 个参数")



class Tester(Base):
    def __init__(self, test_epoch=None):
        if test_epoch is not None:
            self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self):
        self.logger.info('Load checkpoint from {}'.format(cfg.pretrained_model_path))

        # prepare network
        # —————————————— 旧代码 (MMCV 实现) ——————————————————
        if cfg.decoder_setting != "pytorch":
            self.logger.info("Creating graph...")
            model = get_model('test')
            model = DataParallel(model).cuda()
            ckpt = torch.load(cfg.pretrained_model_path)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['network'].items():
                k = k.replace('module.backbone', 'module.encoder').replace('body_rotation_net', 'body_regressor').replace(
                    'hand_rotation_net', 'hand_regressor')
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()

            self.model = model
            return model
        # ————————————————————————————————————————

        # —————————————— 新代码 (纯 PyTorch 实现) ——————————————————
        # 假设 model 已经是你用 get_model() 获取到的纯 PyTorch 模型
        model = get_model('test1')
        model = model.cuda()
        
        from collections import OrderedDict

        # 最终修正版：基于真实 Key 结构的精确映射
        # ==========================================================================
        print(f"Loading checkpoint from {cfg.pretrained_model_path} ...")
        ckpt = torch.load(cfg.pretrained_model_path, map_location='cpu')
        src_state_dict = ckpt['network'] if 'network' in ckpt else ckpt['state_dict']
        
        # 1. 预处理：去掉 module. 前缀
        src_state_dict = {k.replace('module.', ''): v for k, v in src_state_dict.items()}
        new_state_dict = OrderedDict()
        
        print("开始执行精确映射...")
        print("源模型 Key 数量：", len(src_state_dict))
        print("目标模型 Key 数量：", len(model.state_dict()))


        # 获取当前 PyTorch 模型的 state_dict
        model_state_dict = model.state_dict()

        # 统计用
        exact_match_count = 0
        mapped_count = 0

        print("开始精准匹配与智能映射权重...")

        for model_key in model_state_dict.keys():
            
            # -----------------------------------------------------------
            # [特殊处理 1] pos_embed 形状适配 (Task Token + Patch Token)
            # -----------------------------------------------------------
            if 'encoder.pos_embed' == model_key:
                target_shape = model_state_dict[model_key].shape
                v = src_state_dict.get(model_key)
                if v is not None and v.shape != target_shape:
                    # 严格按照你旧代码的顺序：Task Token 在前，Patch Token 在后
                    cls_pos = v[:, 0:1, :]
                    patch_pos = v[:, 1:, :]
                    task_pos = cls_pos.repeat(1, 31, 1)
                    v = torch.cat([task_pos, patch_pos], dim=1)
                    print(f"  [Encoder] Resized pos_embed to {v.shape}")
                new_state_dict[model_key] = v
                continue

            # -----------------------------------------------------------
            # [策略 A] 精确匹配 (涵盖了所有的 Backbone, ROI, BN层等)
            # -----------------------------------------------------------
            if model_key in src_state_dict:
                v = src_state_dict[model_key]
                if v.shape == model_state_dict[model_key].shape:
                    new_state_dict[model_key] = v
                    exact_match_count += 1
                else:
                    print(f"  [Shape Mismatch] {model_key}: ckpt {v.shape} != model {model_state_dict[model_key].shape}")
                continue

            # -----------------------------------------------------------
            # [策略 B] Encoder 的 last_norm 映射
            # -----------------------------------------------------------
            if 'encoder.norm.' in model_key:
                src_key = model_key.replace('encoder.norm.', 'encoder.last_norm.')
                if src_key in src_state_dict and src_state_dict[src_key].shape == model_state_dict[model_key].shape:
                    new_state_dict[model_key] = src_state_dict[src_key]
                    mapped_count += 1
                    continue

            # -----------------------------------------------------------
            # [策略 C] Decoder Transformer 的智能映射 (极其重要，防止手脸扭曲)
            # -----------------------------------------------------------
            if 'decoder.layers.' in model_key:
                src_key = model_key
                # 1. 补全 MMCV 的 Transformer 路径
                if 'hand_decoder.layers.' in model_key:
                    src_key = src_key.replace('hand_decoder.layers.', 'hand_decoder.keypoint_head.transformer.decoder.layers.')
                elif 'face_decoder.layers.' in model_key:
                    src_key = src_key.replace('face_decoder.layers.', 'face_decoder.keypoint_head.transformer.decoder.layers.')
                    
                # 2. 映射 Self-Attention
                src_key = src_key.replace('.self_attn.in_proj_', '.attentions.0.attn.in_proj_')
                src_key = src_key.replace('.self_attn.out_proj.', '.attentions.0.attn.out_proj.')
                
                # 3. 映射 Cross-Attention
                src_key = src_key.replace('.cross_attn.sampling_offsets.', '.attentions.1.sampling_offsets.')
                src_key = src_key.replace('.cross_attn.attention_weights.', '.attentions.1.attention_weights.')
                # MMCV 的 value_proj 命名非常特别，带有 weight 和 bias 后缀
                src_key = src_key.replace('.cross_attn.value_proj.weight', '.attentions.1.value_proj_weight.weight')
                src_key = src_key.replace('.cross_attn.value_proj.bias', '.attentions.1.value_proj_bias.weight') # MMCV bias也叫weight
                src_key = src_key.replace('.cross_attn.output_proj.', '.attentions.1.output_proj.')
                
                # 4. 映射 FFN 和 Norm
                src_key = src_key.replace('.linear1.', '.ffns.0.layers.0.0.')
                src_key = src_key.replace('.linear2.', '.ffns.0.layers.1.')
                src_key = src_key.replace('.norm1.', '.norms.0.')
                src_key = src_key.replace('.norm2.', '.norms.1.')
                src_key = src_key.replace('.norm3.', '.norms.2.')

                # 尝试从源字典中获取并校验形状
                if src_key in src_state_dict:
                    v = src_state_dict[src_key]
                    if v.shape == model_state_dict[model_key].shape:
                        new_state_dict[model_key] = v
                        mapped_count += 1
                continue

        # 执行最终加载
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        print("\n" + "="*40)
        print("✅ 权重加载完成报告:")
        print("="*40)
        print(f"🎯 精确匹配加载: {exact_match_count} 个张量 (含 BN 层的统计参数)")
        print(f"🔗 智能映射加载: {mapped_count} 个张量 (主要为 Decoder 的 Transformer 层)")

        # 帮你排查除了你还没重写完的部分，是否还有其他遗漏
        real_missing = [m for m in missing if not ('decoder' in m)]
        if real_missing or unexpected:
            print(f"\n⚠️ 警告: 以下基础网络部分存在未加载的参数 (请检查拼写):")
            for m in real_missing:
                print(f"  - {m}")
            for m in unexpected:
                print(f"  - {m}")
        else:
            print("\n🎉 完美！骨干网络已全部精准对齐，Decoder 的核心 Transformer 层也已成功映射，不再是随机噪声！")


        # --- 步骤 2: 加载自己保存的轻量级 Checkpoint ---
        ckpt_path = cfg.continue_train_path # 假设你配置里指定了要恢复的 snapshot 路径
        self.logger.info(f"Resume Step 2: Loading trained modules from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu')

        if 'network' in ckpt:
            model_dict = model.state_dict()
            trained_dict = ckpt['network']  # 这里面现在只有 trainable 部分

            loaded = 0
            for k, v in trained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded += 1

            # strict=False 允许只覆盖可训练部分，不动刚刚加载好的 Backbone
            missing, unexpected = model.load_state_dict(model_dict, strict=False) 
            self.logger.info(f"  成功覆盖了 {loaded} 个已训练的参数张量")

            if missing:
                print(f"\n⚠️ 警告: 以下可训练网络部分存在未加载的参数 (请检查拼写):")
                for m in missing:
                    print(f"  - {m}")
            if unexpected:
                print(f"仍有未预期参数！！！！！！")
                for m in unexpected:
                    print(f"  - {m}")
            if not missing and not unexpected:
                print("\n🎉 完美！可训练网络已全部加载")
        else:
            raise Exception("Checkpoint 中不存在 'network' 字段，请检查是否正确加载")
        
        model.eval()
        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)

class Demoer(Base):
    def __init__(self, test_epoch=None):
        if test_epoch is not None:
            self.test_epoch = int(test_epoch)
        super(Demoer, self).__init__(log_name='test_logs.txt')

    def _make_model(self):
        self.logger.info('Load checkpoint from {}'.format(cfg.pretrained_model_path))

        # prepare network
        # —————————————— 旧代码 ——————————————————
        if cfg.decoder_setting != "pytorch":
            self.logger.info("Creating graph...")
            model = get_model('test')
            model = DataParallel(model).cuda()
            ckpt = torch.load(cfg.pretrained_model_path)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['network'].items():
                k = k.replace('module.backbone', 'module.encoder').replace('body_rotation_net', 'body_regressor').replace(
                    'hand_rotation_net', 'hand_regressor')
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()

            self.model = model
            return model
        # ————————————————————————————————————————

        # 假设 model 已经是你用 get_model() 获取到的纯 PyTorch 模型
        model = get_model('test1')
        with open("my_model_params.txt", "w", encoding="utf-8") as f:
            total = 0
            for name, param in model.named_parameters():
                shape = tuple(param.shape)
                num = param.numel()
                total += num
                f.write(f"{name}\t{shape}\t{num}\n")
            f.write(f"\nTotal params: {total}\n")
        model = model.cuda()
        print(f"box_net conv1:{model.box_net.deconv[1]}")
        from collections import OrderedDict

        # 最终修正版：基于真实 Key 结构的精确映射
        # ==========================================================================
        print(f"Loading checkpoint from {cfg.pretrained_model_path} ...")
        ckpt = torch.load(cfg.pretrained_model_path, map_location='cpu')
        src_state_dict = ckpt['network'] if 'network' in ckpt else ckpt['state_dict']
        
        # 1. 预处理：去掉 module. 前缀
        src_state_dict = {k.replace('module.', ''): v for k, v in src_state_dict.items()}
        new_state_dict = OrderedDict()
        
        print("开始执行精确映射...")
        print("源模型 Key 数量：", len(src_state_dict))
        print("目标模型 Key 数量：", len(model.state_dict()))


        # 获取当前 PyTorch 模型的 state_dict
        model_state_dict = model.state_dict() # 注意：如果这段代码在模型类内部，请将 model 替换为 self

        # 统计用
        exact_match_count = 0
        mapped_count = 0

        print("开始精准匹配与智能映射权重...")

        for model_key in model_state_dict.keys():
            
            # -----------------------------------------------------------
            # [特殊处理 1] pos_embed 形状适配 (Task Token + Patch Token)
            # -----------------------------------------------------------
            if 'encoder.pos_embed' == model_key:
                target_shape = model_state_dict[model_key].shape
                v = src_state_dict.get(model_key)
                if v is not None and v.shape != target_shape:
                    # 严格按照你旧代码的顺序：Task Token 在前，Patch Token 在后
                    cls_pos = v[:, 0:1, :]
                    patch_pos = v[:, 1:, :]
                    task_pos = cls_pos.repeat(1, 31, 1)
                    v = torch.cat([task_pos, patch_pos], dim=1)
                    print(f"  [Encoder] Resized pos_embed to {v.shape}")
                new_state_dict[model_key] = v
                continue

            # -----------------------------------------------------------
            # [策略 A] 精确匹配 (涵盖了所有的 Backbone, ROI, BN层等)
            # -----------------------------------------------------------
            if model_key in src_state_dict:
                v = src_state_dict[model_key]
                if v.shape == model_state_dict[model_key].shape:
                    new_state_dict[model_key] = v
                    exact_match_count += 1
                else:
                    print(f"  [Shape Mismatch] {model_key}: ckpt {v.shape} != model {model_state_dict[model_key].shape}")
                continue

            # -----------------------------------------------------------
            # [策略 B] Encoder 的 last_norm 映射
            # -----------------------------------------------------------
            if 'encoder.norm.' in model_key:
                src_key = model_key.replace('encoder.norm.', 'encoder.last_norm.')
                if src_key in src_state_dict and src_state_dict[src_key].shape == model_state_dict[model_key].shape:
                    new_state_dict[model_key] = src_state_dict[src_key]
                    mapped_count += 1
                    continue

            # -----------------------------------------------------------
            # [策略 C] Decoder Transformer 的智能映射 (极其重要，防止手脸扭曲)
            # -----------------------------------------------------------
            if 'decoder.layers.' in model_key:
                src_key = model_key
                # 1. 补全 MMCV 的 Transformer 路径
                if 'hand_decoder.layers.' in model_key:
                    src_key = src_key.replace('hand_decoder.layers.', 'hand_decoder.keypoint_head.transformer.decoder.layers.')
                elif 'face_decoder.layers.' in model_key:
                    src_key = src_key.replace('face_decoder.layers.', 'face_decoder.keypoint_head.transformer.decoder.layers.')
                    
                # 2. 映射 Self-Attention
                src_key = src_key.replace('.self_attn.in_proj_', '.attentions.0.attn.in_proj_')
                src_key = src_key.replace('.self_attn.out_proj.', '.attentions.0.attn.out_proj.')
                
                # 3. 映射 Cross-Attention
                src_key = src_key.replace('.cross_attn.sampling_offsets.', '.attentions.1.sampling_offsets.')
                src_key = src_key.replace('.cross_attn.attention_weights.', '.attentions.1.attention_weights.')
                # MMCV 的 value_proj 命名非常特别，带有 weight 和 bias 后缀
                src_key = src_key.replace('.cross_attn.value_proj.weight', '.attentions.1.value_proj_weight.weight')
                src_key = src_key.replace('.cross_attn.value_proj.bias', '.attentions.1.value_proj_bias.weight') # MMCV bias也叫weight
                src_key = src_key.replace('.cross_attn.output_proj.', '.attentions.1.output_proj.')
                
                # 4. 映射 FFN 和 Norm
                src_key = src_key.replace('.linear1.', '.ffns.0.layers.0.0.')
                src_key = src_key.replace('.linear2.', '.ffns.0.layers.1.')
                src_key = src_key.replace('.norm1.', '.norms.0.')
                src_key = src_key.replace('.norm2.', '.norms.1.')
                src_key = src_key.replace('.norm3.', '.norms.2.')

                # 尝试从源字典中获取并校验形状
                if src_key in src_state_dict:
                    v = src_state_dict[src_key]
                    if v.shape == model_state_dict[model_key].shape:
                        new_state_dict[model_key] = v
                        mapped_count += 1
                    # 注意：如果维度不匹配(如query_embed)，这里会安全地跳过，等待你后续的重构
                continue

        # 执行最终加载
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        print("\n" + "="*40)
        print("✅ 权重加载完成报告:")
        print("="*40)
        print(f"🎯 精确匹配加载: {exact_match_count} 个张量 (含 BN 层的统计参数)")
        print(f"🔗 智能映射加载: {mapped_count} 个张量 (主要为 Decoder 的 Transformer 层)")

        # 帮你排查除了你还没重写完的部分，是否还有其他遗漏
        real_missing = [m for m in missing if not ('decoder' in m)]
        if real_missing:
            print(f"\n⚠️ 警告: 以下基础网络部分存在未加载的参数 (请检查拼写):")
            for m in real_missing:
                print(f"  - {m}")
        else:
            print("\n🎉 完美！骨干网络已全部精准对齐，Decoder 的核心 Transformer 层也已成功映射，不再是随机噪声！")

        # --- 步骤 2: 加载自己保存的轻量级 Checkpoint ---
        ckpt_path = cfg.continue_train_path # 假设你配置里指定了要恢复的 snapshot 路径
        self.logger.info(f"Resume Step 2: Loading trained modules from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu')

        if 'network' in ckpt:
            model_dict = model.state_dict()
            trained_dict = ckpt['network']  # 这里面现在只有 trainable 部分

            loaded = 0
            for k, v in trained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded += 1

            # strict=False 允许只覆盖可训练部分，不动刚刚加载好的 Backbone
            missing, unexpected = model.load_state_dict(model_dict, strict=False) 
            self.logger.info(f"  成功覆盖了 {loaded} 个已训练的参数张量")

            if missing:
                print(f"\n⚠️ 警告: 以下可训练网络部分存在未加载的参数 (请检查拼写):")
                for m in missing:
                    print(f"  - {m}")
            elif unexpected:
                print(f"\n⚠️ 警告: 以下可训练网络部分存在未预期的参数 (请检查拼写):")
                for m in unexpected:
                    print(f"  - {m}")
            else:
                print("\n🎉 完美！可训练网络已全部加载")
        else:
            raise Exception("Checkpoint 中不存在 'network' 字段，请检查是否正确加载")
        
        self.model = model
# model.eval()