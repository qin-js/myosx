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
        # self.logger.info("Creating graph...")
        # model = get_model('test')
        # model = DataParallel(model).cuda()
        # ckpt = torch.load(cfg.pretrained_model_path)

        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in ckpt['network'].items():
        #     k = k.replace('module.backbone', 'module.encoder').replace('body_rotation_net', 'body_regressor').replace(
        #         'hand_rotation_net', 'hand_regressor')
        #     new_state_dict[k] = v
        # model.load_state_dict(new_state_dict, strict=False)
        # model.eval()

        # self.model = model
        # ————————————————————————————————————————

        # 假设 model 已经是你用 get_model() 获取到的纯 PyTorch 模型
        model = get_model('test')
        # model = model.cuda() # 建议先不上 DataParallel，调试通了再加

        print(f"Loading checkpoint from {cfg.pretrained_model_path} ...")
        ckpt = torch.load(cfg.pretrained_model_path, map_location='cpu')

        # 1. 获取原始 state_dict
        if 'network' in ckpt:
            src_state_dict = ckpt['network']
        elif 'state_dict' in ckpt:
            src_state_dict = ckpt['state_dict']
        else:
            src_state_dict = ckpt

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        model_keys = list(model.state_dict().keys())

        for k, v in src_state_dict.items():
            # 1. 基础清理
            if k.startswith('module.'):
                k = k[7:]
            
            # 2. Encoder 映射 (保持之前的逻辑)
            if k.startswith('encoder.') or k.startswith('backbone.'):
                # 处理 task_tokens -> cls_token 的映射 (如果此时你想强行加载)
                if 'task_tokens' in k:
                    k = k.replace('task_tokens', 'cls_token') # 形状可能不匹配，如果报错就忽略这个key
                
                # 确保前缀是 encoder. (匹配 model.encoder)
                k = k.replace('backbone.', 'encoder.')

            # 3. Regressor 映射 (保持之前的逻辑)
            k = k.replace('body_rotation_net', 'body_regressor')
            k = k.replace('hand_rotation_net', 'hand_regressor')

            # 4. === 核心：Decoder Attention 映射 ===
            # 原 Key: hand_decoder.keypoint_head.transformer.decoder.layers.0.attentions.1.sampling_offsets.weight
            # 新 Key: hand_decoder.layers.0.cross_attn.sampling_offsets.weight
            
            if 'decoder' in k and 'attentions' in k:
                # 去掉中间冗余路径
                k = k.replace('keypoint_head.transformer.decoder.', '')
                
                # 映射 attentions.1 -> cross_attn
                if 'attentions.1' in k:
                    k = k.replace('attentions.1', 'cross_attn')
                # 映射 attentions.0 -> self_attn
                elif 'attentions.0' in k:
                    k = k.replace('attentions.0', 'self_attn')
            
            new_state_dict[k] = v

        # 执行加载
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        # 打印诊断信息
        print(f"成功加载 Key 数量: {len(new_state_dict) - len(missing_keys)}")
        print(f"缺失 Key 数量: {len(missing_keys)}")

        if len(missing_keys) > 0:
            print("前 100 个缺失的 Key (请检查是否是关键层):")
            for k in missing_keys[:100]:
                print(f" - {k}")

        # 诊断脚本
        print("\n========== 权重 Key 诊断 ==========")
        keys = list(ckpt['network'].keys())

        # 1. 打印前 30 个 Key (看看 Encoder 叫什么)
        # print("--- Top 30 Keys ---")
        # for k in keys[:30]:
        #     print(k)

        # 2. 搜索包含 'hand' 的 Key (看看 Hand Decoder 叫什么)
        print("\n--- Hand Keys Sample ---")
        hand_keys = [k for k in keys if 'hand' in k]
        for k in hand_keys[:20]: # 只打前20个
            print(k)

        # 3. 搜索包含 'sampling_offsets' 的 Key (看看 Attention 在哪)
        print("\n--- Attention Keys Sample ---")
        attn_keys = [k for k in keys if 'sampling_offsets' in k]
        for k in attn_keys[:5]:
            print(k)

        # 重点检查 Attention
        attn_loaded = any("sampling_offsets" in k and k not in missing_keys for k in model.state_dict())
        if attn_loaded:
            print(">>> 恭喜：Deformable Attention 权重加载成功！")
        else:
            print(">>> 警告：Deformable Attention 权重依然缺失，可能需要进一步检查 Key 前缀。")

        self.model = model

# model.eval()