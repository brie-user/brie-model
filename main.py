import argparse
import datetime
import os
import random
import sys
from scipy.sparse import sputils
import yaml
import torch
import time
import numpy as np
from collections import defaultdict, OrderedDict
import logging
from src.model_handler import ModelHandler
import glob
import shutil
################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(config):
    prepare_logging(config)
    if config["save_scripts"] == True:
        save_main_script(config)
    print_config(config)
    set_random_seed(config['seed'])
    model = ModelHandler(config)
    f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test, recall, fbeta2_1_gnn, fbeta4_1_gnn, recall_gnn1 = model.train()

    logging.info("F1-Macro: {}".format(f1_mac_test))
    logging.info("AUC: {}".format(auc_test))
    logging.info("G-Mean: {}".format(gmean_test))
    logging.info("Recall: {}".format(recall))
    logging.info("Fbeta_2_1 {}".format(fbeta2_1_gnn))
    logging.info("Fbeta_4_1 {}".format(fbeta4_1_gnn))
    logging.info("Recall_gnn1 {}".format(recall_gnn1))


def multi_run_main(config):
    prepare_logging(config)
    print_config(config)
    if config["save_scripts"] == True:
        save_main_script(config)
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    f1_list, f1_1_list, f1_0_list, auc_list, gmean_list, recall_list, fbeta2_1_list, fbeta4_1_list, recall_gnn1_list = [
    ], [], [], [], [], [], [], [], []
    configs = grid(config)
    for i, cnf in enumerate(configs):
        logging.info('Running {}:\n'.format(i))
        print_config(cnf)
        # for k in hyperparams:
        #     cnf['save_dir'] += '{}_{}_'.format(k, cnf[k])
        set_random_seed(cnf['seed'])
        st = time.time()
        model = ModelHandler(cnf)
        f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test, recall, fbeta2_1_gnn, fbeta4_1_gnn, recall_gnn1 = model.train()
        f1_list.append(f1_mac_test)
        f1_1_list.append(f1_1_test)
        f1_0_list.append(f1_0_test)
        auc_list.append(auc_test)
        gmean_list.append(gmean_test)
        recall_list.append(recall)
        fbeta2_1_list.append(fbeta2_1_gnn)
        fbeta4_1_list.append(fbeta4_1_gnn)
        recall_gnn1_list.append(recall_gnn1)
        logging.info(
            "Running {} done, elapsed time {}s".format(i, time.time()-st))

    logging.info("F1-Macro: {}".format(f1_list))
    logging.info("AUC: {}".format(auc_list))
    logging.info("G-Mean: {}".format(gmean_list))
    logging.info("Recall: {}".format(recall_list))
    logging.info("Fbeta_2_1 {}".format(fbeta2_1_list))
    logging.info("Fbeta_4_1 {}".format(fbeta4_1_list))
    logging.info("Recall_gnn1 {}".format(recall_gnn1_list))

    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list, ddof=1)
    f1_1_mean, f1_1_std = np.mean(f1_1_list), np.std(f1_1_list, ddof=1)
    f1_0_mean, f1_0_std = np.mean(f1_0_list), np.std(f1_0_list, ddof=1)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list, ddof=1)
    gmean_mean, gmean_std = np.mean(gmean_list), np.std(gmean_list, ddof=1)
    recall_mean, recall_std = np.mean(recall_list), np.std(recall_list, ddof=1)
    fbeta2_1_mean, fbeta2_1_std = np.mean(
        fbeta2_1_list), np.std(fbeta2_1_list, ddof=1)
    fbeta4_1_mean, fbeta4_1_std = np.mean(
        fbeta4_1_list), np.std(fbeta4_1_list, ddof=1)
    recall_gnn1_mean, recall_gnn1_std = np.mean(
        recall_gnn1_list), np.std(recall_gnn1_list, ddof=1)

    logging.info("F1-Macro: {}+{}".format(f1_mean, f1_std))
    logging.info("F1-binary-1: {}+{}".format(f1_1_mean, f1_1_std))
    logging.info("F1-binary-0: {}+{}".format(f1_0_mean, f1_0_std))
    logging.info("AUC: {}+{}".format(auc_mean, auc_std))
    logging.info("G-Mean: {}+{}".format(gmean_mean, gmean_std))
    logging.info("Recall: {}+{}".format(recall_mean, recall_std))
    logging.info("Fbeta_2_1: {}+{}".format(fbeta2_1_mean, fbeta2_1_std))
    logging.info("Fbeta_4_1: {}+{}".format(fbeta4_1_mean, fbeta4_1_std))
    logging.info(
        "Recall_gnn1: {}+{}".format(recall_gnn1_mean, recall_gnn1_std))


################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True,
                        type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true',
                        help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def prepare_logging(config):
    log_format = "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")
    timestamp = time.time()
    timestamp = datetime.datetime.fromtimestamp(
        int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
    config['exp_name'] = timestamp+f" {config['data_name']} {config['model']}"
    if not os.path.exists(os.path.join(config['save_dir'], config['exp_name'])):
        os.makedirs(os.path.join(config['save_dir'], config['exp_name']))
    fh = logging.FileHandler(os.path.join(
        config['save_dir'], config['exp_name'], 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def save_main_script(config):
    def create_exp_dir(path, scripts_to_save=None):
        if not os.path.exists(path):
            os.mkdir(path)
        logging.info('Experiment dir : {}'.format(path))

        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(
                    path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    create_exp_dir(os.path.join(config['save_dir'], config['exp_name']),
                   scripts_to_save=glob.glob('./src/*.py'))


def print_config(config):
    logging.info("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        logging.info("{} -->   {}".format(keystr, val))
    logging.info("**************** MODEL CONFIGURATION ****************")


def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce

        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()),
                   dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i]
         for i, k in enumerate(sin)}
    ) for vv in grd]


################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])

    if cfg['multi_run']:
        multi_run_main(config)
    else:
        main(config)
