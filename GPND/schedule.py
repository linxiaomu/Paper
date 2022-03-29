from save_to_csv import save_results
import logging
import sys
import utils.multiprocessing
from defaults import get_cfg_defaults
import os

full_run = True

# 记录器
logger = logging.getLogger("logger")
# 设置记录器阈值
logger.setLevel(logging.DEBUG)
# 它可将日志记录输出发送到数据流例如 sys.stdout, sys.stderr 或任何文件类对象
# 返回一个新的 StreamHandler类。
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
# 将指定的处理器ch添加到此记录器
logger.addHandler(ch)

if len(sys.argv) > 1:
    cfg_file = 'configs/' + sys.argv[1]
else:
    cfg_file = 'configs/' + input("Config file:")

mul = 0.2

settings = []

classes_count = 10

for fold in range(5 if full_run else 1):
    for i in range(classes_count):
        settings.append(dict(fold=fold, digit=i))

# 获取默认配置
cfg = get_cfg_defaults()
# 合并特定配置
cfg.merge_from_file(cfg_file)
# 在初始设置之后，最好通过调用freeze（）方法将配置冻结以防止进一步修改。
cfg.freeze()


def f(setting):
    import train_AAE
    import novelty_detector

    fold_id = setting['fold']
    inliner_classes = setting['digit']

    train_AAE.train(fold_id, [inliner_classes], inliner_classes, cfg=cfg)

    res = novelty_detector.main(fold_id, [inliner_classes], inliner_classes, classes_count, mul, cfg=cfg)
    return res


if __name__ == '__main__':
    gpu_count = utils.multiprocessing.get_gpu_count()

    results = utils.multiprocessing.map(f, gpu_count, settings)

    save_results(results, os.path.join(cfg.OUTPUT_FOLDER, cfg.RESULTS_NAME))
