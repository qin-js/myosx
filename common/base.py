import os.path as osp
import math
import abc
from torch.utils.data import DataLoader
import torch.optim
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
        normal_param = []
        special_param = []
        for module in model.module.special_trainable_modules:
            special_param += list(module.parameters())
            # print(module)
        for module in model.module.trainable_modules:
            normal_param += list(module.parameters())
        optim_params = [
            {  # add normal params first
                'params': normal_param,
                'lr': cfg.lr
            },
            {
                'params': special_param,
                'lr': cfg.lr * cfg.lr_mult
            },
        ]
        optimizer = torch.optim.Adam(optim_params, lr=cfg.lr)
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))

        # do not save smplx layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'smplx_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        if cfg.pretrained_model_path is not None:
            ckpt_path = cfg.pretrained_model_path
            ckpt = torch.load(ckpt_path)
            start_epoch = 0
            model.load_state_dict(ckpt['network'], strict=False)
            self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        else:
            start_epoch = 0

        return start_epoch, model, optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
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
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus * cfg.train_batch_size,
                                          shuffle=True, num_workers=cfg.num_thread, pin_memory=True, drop_last=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model('train')
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.end_epoch * self.itr_per_epoch,
                                                               eta_min=1e-6)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer


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
        model = get_model('test')
        model = model.cuda()

         # 智能权重加载器 (Smart Adapter)
        # ==============================================================================
        print(f"Loading checkpoint from {cfg.pretrained_model_path} ...")
        ckpt = torch.load(cfg.pretrained_model_path, map_location='cpu')
        src_state_dict = ckpt['network'] if 'network' in ckpt else ckpt['state_dict']
        
        # 清洗源权重 key，去掉 module. 前缀
        src_state_dict = {k.replace('module.', ''): v for k, v in src_state_dict.items()}
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        model_state_dict = model.state_dict()
        
        print("开始模糊匹配权重...")
        
        for model_key in model_state_dict.keys():
            # -----------------------------------------------------------
            # 特殊处理: pos_embed 形状适配
            if 'encoder.pos_embed' == model_key:
                target_shape = model_state_dict[model_key].shape
                v = src_state_dict[model_key]
                if v.shape != target_shape:
                    print(f"  [Encoder] Resizing pos_embed: {v.shape} -> {target_shape}")
                    cls_pos = v[:, 0:1, :]
                    patch_pos = v[:, 1:, :]
                    task_pos = cls_pos.repeat(1, 31, 1)
                    v = torch.cat([task_pos, patch_pos], dim=1)
            
                new_state_dict[model_key] = v
                continue
            # 1. 策略 A: 精确匹配 (最优先)
            # -----------------------------------------------------------
            if model_key in src_state_dict:
                new_state_dict[model_key] = src_state_dict[model_key]
                continue
                
            # -----------------------------------------------------------
            # 2. 策略 B: 针对 StandardViT (Encoder) 的后缀匹配
            # -----------------------------------------------------------
            if 'encoder.' in model_key:
                # 去掉 encoder. 前缀，找后缀
                # 例如: encoder.norm.weight -> 找 ...norm.weight
                suffix = model_key.replace('encoder.', '') 

                # [新增] 如果是 norm，尝试匹配 last_norm
                if suffix == 'norm.weight':
                    suffix = 'last_norm.weight'
                elif suffix == 'norm.bias':
                    suffix = 'last_norm.bias'
                
                # 在源权重中寻找以 suffix 结尾，且包含 backbone 或 encoder 的 key
                candidates = [k for k in src_state_dict.keys() if k.endswith(suffix) and ('backbone' in k or 'encoder' in k)]
                
                if len(candidates) == 1:
                    src_key = candidates[0]
                    v = src_state_dict[src_key]
                    
                    # 特殊处理: pos_embed 形状适配
                    if 'pos_embed' in model_key:
                        target_shape = model_state_dict[model_key].shape
                        if v.shape != target_shape:
                            # print(f"  [Encoder] Resizing pos_embed: {v.shape} -> {target_shape}")
                            cls_pos = v[:, 0:1, :]
                            patch_pos = v[:, 1:, :]
                            task_pos = cls_pos.repeat(1, 31, 1)
                            v = torch.cat([task_pos, patch_pos], dim=1)
                    
                    new_state_dict[model_key] = v
                    continue

            # -----------------------------------------------------------
            # 3. 策略 C: 针对 Decoder 的智能映射
            # -----------------------------------------------------------
            if 'decoder' in model_key:
                # 构造 MMCV 风格的 key 进行尝试
                # PyTorch Key: hand_decoder.layers.0.linear1.weight
                # 目标 MMCV Key 可能长这样: ...transformer.decoder.layers.0.ffns.0.layers.0.weight
                
                # 提取核心部分: layers.0.linear1.weight
                core_key = model_key.split('decoder.')[-1] 
                
                # 映射表: PyTorch命名 -> MMCV命名片段
                mapping = {
                    'linear1': 'ffns.0.layers.0', # 或 ffns.0.layers.0.linear
                    'linear2': 'ffns.0.layers.1', # 注意 MMCV 可能是 layers.1
                    'norm1': 'norms.0',
                    'norm2': 'norms.1',
                    'norm3': 'norms.2',
                    'self_attn': 'attentions.0',
                    'cross_attn': 'attentions.1',
                    'value_proj': 'value_project', # 有时是 value_project
                    'output_proj': 'output_project'
                }
                
                search_key = core_key
                for py_name, mmcv_name in mapping.items():
                    search_key = search_key.replace(py_name, mmcv_name)
                
                # 尝试在源权重中找包含 search_key 的项
                # 过滤条件: 必须包含 transformer.decoder
                candidates = [k for k in src_state_dict.keys() 
                            if search_key in k 
                            and ('transformer.decoder' in k)
                            and model_key.split('.')[0] in k] # 确保 hand 对应 hand, face 对应 face
                
                # 如果没找到，尝试 linear2 的另一种可能 (layers.3)
                if not candidates and 'linear2' in model_key:
                    alt_key = search_key.replace('ffns.0.layers.1', 'ffns.0.layers.3')
                    candidates = [k for k in src_state_dict.keys() if alt_key in k and 'transformer.decoder' in k]

                if len(candidates) >= 1:
                    # 取最短的那个通常是正确的 (避免匹配到 gradients 等奇怪东西)
                    best_candidate = min(candidates, key=len)
                    # print(f"  [Decoder] Mapping {model_key} <- {best_candidate}")
                    new_state_dict[model_key] = src_state_dict[best_candidate]
                    continue

            # 5. 策略 E: Self-Attention 权重合并 (解决 in_proj_weight 缺失)
            # -----------------------------------------------------------
            if 'self_attn.in_proj_weight' in model_key:
                # model_key 例子: hand_decoder.layers.0.self_attn.in_proj_weight
                # 我们需要找到源权重中的 q_proj, k_proj, v_proj
                
                # 1. 构建源权重的基准前缀
                # 尝试将 .self_attn.in_proj_weight 替换为 MMCV 风格的前缀
                # 假设 MMCV 路径: ...transformer.decoder.layers.0.attentions.0.attn.
                
                # 提取 layers.x
                layer_idx = model_key.split('layers.')[1].split('.')[0] # 获取 '0', '1' 等
                decoder_type = 'hand' if 'hand' in model_key else 'face'
                
                # 构造搜索前缀
                prefix_candidates = [
                    f"{decoder_type}_decoder.keypoint_head.transformer.decoder.layers.{layer_idx}.attentions.0.attn.", # MMCV 标准
                    f"transformer.decoder.layers.{layer_idx}.attentions.0.attn.", # 简化版
                ]
                
                q, k, v = None, None, None
                
                # 在源权重中搜索 q, k, v
                for prefix in prefix_candidates:
                    # 注意：有些权重叫 q_proj, 有些叫 in_proj_q 等，这里假设最常见的 q_proj
                    q_name = prefix + "in_proj_weight" # 有时 MMCV 也是合并的，只是路径深
                    if q_name in src_state_dict:
                        new_state_dict[model_key] = src_state_dict[q_name]
                        break
                    
                    # 如果是分离的
                    q_name = prefix + "q_proj.weight"
                    k_name = prefix + "k_proj.weight"
                    v_name = prefix + "v_proj.weight"
                    
                    # 尝试查找 (模糊匹配)
                    q_cand = [key for key in src_state_dict.keys() if key.endswith(q_name)]
                    k_cand = [key for key in src_state_dict.keys() if key.endswith(k_name)]
                    v_cand = [key for key in src_state_dict.keys() if key.endswith(v_name)]
                    
                    if q_cand and k_cand and v_cand:
                        q = src_state_dict[q_cand[0]]
                        k = src_state_dict[k_cand[0]]
                        v = src_state_dict[v_cand[0]]
                        
                        # [核心操作] 合并 Q, K, V -> in_proj_weight
                        # print(f"  Merging QKV for {model_key}")
                        new_state_dict[model_key] = torch.cat([q, k, v], dim=0)
                        break
                
                if model_key in new_state_dict:
                    continue

            # 处理 bias (同理)
            if 'self_attn.in_proj_bias' in model_key:
                layer_idx = model_key.split('layers.')[1].split('.')[0]
                # 简化逻辑，只找后缀
                suffix_q = f"layers.{layer_idx}.attentions.0.attn.q_proj.bias"
                suffix_k = f"layers.{layer_idx}.attentions.0.attn.k_proj.bias"
                suffix_v = f"layers.{layer_idx}.attentions.0.attn.v_proj.bias"
                
                q_cand = [key for key in src_state_dict.keys() if key.endswith(suffix_q)]
                k_cand = [key for key in src_state_dict.keys() if key.endswith(suffix_k)]
                v_cand = [key for key in src_state_dict.keys() if key.endswith(suffix_v)]
                
                if q_cand and k_cand and v_cand:
                    q = src_state_dict[q_cand[0]]
                    k = src_state_dict[k_cand[0]]
                    v = src_state_dict[v_cand[0]]
                    new_state_dict[model_key] = torch.cat([q, k, v], dim=0)
                    continue
                    
            # 处理 out_proj (MMCV 叫 out_proj 或 output_proj)
            if 'self_attn.out_proj.weight' in model_key:
                layer_idx = model_key.split('layers.')[1].split('.')[0]
                suffix = f"layers.{layer_idx}.attentions.0.attn.out_proj.weight"
                cand = [key for key in src_state_dict.keys() if key.endswith(suffix)]
                if cand:
                    new_state_dict[model_key] = src_state_dict[cand[0]]
                    continue

            # -----------------------------------------------------------
            # 4. 策略 D: Regressor 简单替换
            # -----------------------------------------------------------
            if 'regressor' in model_key:
                old_name = model_key.replace('regressor', 'rotation_net')
                candidates = [k for k in src_state_dict.keys() if k.endswith(old_name)]
                if candidates:
                    new_state_dict[model_key] = src_state_dict[candidates[0]]

        # 执行加载
        msg = model.load_state_dict(new_state_dict, strict=False)
        
        # 最终检查
        print(f"\nMissing Keys Count: {len(msg.missing_keys)}")
        if len(msg.missing_keys) > 0:
            print("Still missing (example):", msg.missing_keys[:3])
            with open('missing_keys.txt', 'w') as f:
                f.write('\n'.join(msg.missing_keys))
            with open('unexpected_keys.txt', 'w') as f:
                f.write('\n'.join(msg.unexpected_keys))
        self.model = model
# model.eval()