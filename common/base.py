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
        return 
        # ————————————————————————————————————————

        # 假设 model 已经是你用 get_model() 获取到的纯 PyTorch 模型
        model = get_model('test')
        model = model.cuda()

         # 智能权重加载器 (Smart Adapter)
        # ==============================================================================
        print(f"Loading checkpoint from {cfg.pretrained_model_path} ...")
        ckpt = torch.load(cfg.pretrained_model_path, map_location='cpu')
        
        if 'network' in ckpt: src_state_dict = ckpt['network']
        elif 'state_dict' in ckpt: src_state_dict = ckpt['state_dict']
        else: src_state_dict = ckpt
        
        new_state_dict = OrderedDict()
        
        for k, v in src_state_dict.items():
            if k.startswith('module.'): k = k[7:]
            
            # ----------------------------------------------------------------------
            # 1. Encoder (ViT) 
            # ----------------------------------------------------------------------
            if k.startswith('backbone.') or k.startswith('encoder.'):
                new_k = k.replace('backbone.', 'encoder.')
                
                # 处理 pos_embed 形状扩展 (193 -> 223)
                if 'pos_embed' in new_k:
                    if v.shape[1] == 193 and model.encoder.pos_embed.shape[1] == 223:
                        # print(f"Adapting pos_embed {v.shape} -> {model.encoder.pos_embed.shape}")
                        cls_pos = v[:, 0:1, :]
                        patch_pos = v[:, 1:, :]
                        task_pos = cls_pos.repeat(1, 31, 1) # 复制 31 份
                        v = torch.cat([task_pos, patch_pos], dim=1)
                
                # 务必保存修改后的 Key
                new_state_dict[new_k] = v
                continue # 结束当前循环，处理下一个 Key

            # ----------------------------------------------------------------------
            # 2. Regressor 头
            # ----------------------------------------------------------------------
            k = k.replace('body_rotation_net', 'body_regressor')
            k = k.replace('hand_rotation_net', 'hand_regressor')
            
            # ----------------------------------------------------------------------
            # 3. Decoder (Hand/Face) - MMCV 到 PyTorch 的复杂映射
            # ----------------------------------------------------------------------
            if 'transformer.decoder.' in k:
                # 3.1 去掉前缀
                k = k.replace('keypoint_head.transformer.decoder.', '')
                
                # 3.2 映射 Norm 层 (MMCV: norms.0 -> PyTorch: norm1)
                k = k.replace('norms.0', 'norm1')
                k = k.replace('norms.1', 'norm2')
                k = k.replace('norms.2', 'norm3')
                
                # 3.3 映射 Feed Forward Network (FFN)
                # MMCV: ffns.0.layers.0 -> PyTorch: linear1
                # MMCV: ffns.0.layers.1 -> PyTorch: linear2
                if 'ffns.0.layers.0' in k:
                    k = k.replace('ffns.0.layers.0', 'linear1')
                elif 'ffns.0.layers.1' in k:
                    k = k.replace('ffns.0.layers.1', 'linear2')
                
                # 3.4 映射 Self-Attention (attentions.0)
                if 'attentions.0' in k:
                    k = k.replace('attentions.0', 'self_attn')
                    k = k.replace('.attn.', '.') # 去掉 MMCV wrapper 层
                
                # 3.5 映射 Cross-Attention (attentions.1 / Deformable)
                elif 'attentions.1' in k:
                    k = k.replace('attentions.1', 'cross_attn')
                    # 你的 PyTorchMSDeformAttn 定义的变量名
                    k = k.replace('output_project', 'output_proj') 
                    k = k.replace('sampling_offsets', 'sampling_offsets') # 保持不变
                    k = k.replace('attention_weights', 'attention_weights') # 保持不变
                    k = k.replace('value_proj', 'value_proj') # 保持不变
            
            # 保存 Decoder 及其他层的权重
            new_state_dict[k] = v

        # 执行加载
        msg = model.load_state_dict(new_state_dict, strict=False)
        
        # 打印最终结果
        print("\n========== 加载报告 ==========")
        print(f"Missing Keys: {len(msg.missing_keys)}")
        # 过滤掉一些无关紧要的 missing keys (比如 clip loss 的 logit_scale)
        real_missing = [k for k in msg.missing_keys if 'encoder' in k or 'decoder' in k]
        if len(real_missing) > 0:
            print("警告：关键层依然缺失 (前5个):")
            for k in real_missing[:5]: print(f" - {k}")
        else:
            print("完美！Encoder 和 Decoder 权重已全部加载。")
        self.model = model
# model.eval()