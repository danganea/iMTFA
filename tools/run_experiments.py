"""

    #TODO(): Write usage up:
    - extra options (multiple score thresh etc)
    - multiple seeds
    - multiple shots
    Usage: 
"""
# CV2 needed before anything else
import cv2

import argparse
import os
import re
import shutil
import itertools
from ast import literal_eval as make_tuple
from enum import Enum
from subprocess import PIPE, STDOUT, Popen
from typing import List, Optional, Tuple
import yaml

from detectron2.data import MetadataCatalog

from OurPaper.training import combine_base_novel_models


class ExtraOption(Enum):
    SCORE_THRESH = 1
    ALPHA_WEIGHTING = 2
    USE_ORACLE = 3,
    LIKE_ONESHOT_PAPER = 4,
    LIKE_FGN_PAPER = 5,
    COSIM_LOGIT = 6,
    COSINE_SCALE = 7,
    SET_VOC_TEST = 8


class DatasetConfigs():
    SUPPORTED_DATASETS = ['coco']
    FSDET_DEFAULT_BATCHSIZE = 16
    FSDET_IMGS_PER_GPU = 4

    def __init__(self, dataset_name, dataset_mode, method, num_gpus=2, imgs_per_gpu=None, is_correct_train_iters = False):
        if dataset_name not in DatasetConfigs.SUPPORTED_DATASETS:
            raise ValueError('dataset name not in supported datasets')
        self.dataset = 'coco' if 'coco' in dataset_name else None
        self.dataset_mode = dataset_mode

        #TODO: Initially we trained using a wrong number of iterations and settings. As a hack we append 'correct' to newly used stuff.
        # FOR THE READER: "Only consider the "CORRECT" path being called"
        if 'all' in self.dataset_mode and is_correct_train_iters:
            self.dataset_mode += '-correct'
        self.method = method  # 'Default', 'metric-...' etc

        if self.dataset == 'coco':
            self.default_batchsize = DatasetConfigs.FSDET_DEFAULT_BATCHSIZE
            self.imgs_per_gpu = DatasetConfigs.FSDET_IMGS_PER_GPU
            self.config_dir = 'configs/coco-experiments/'
            self.checkpoint_dir = 'checkpoints/nfsmirror/faster_rcnn/'

        if imgs_per_gpu:
            self.imgs_per_gpu = imgs_per_gpu

        self.current_batch_size = self.imgs_per_gpu * num_gpus
        if self.default_batchsize % self.current_batch_size != 0:
            raise ValueError('default batch size must divide current batch size')

    @property
    def iteration_multiplier(self):
        """
        For some datasets (FsDet COCO) our iteration number defined in 'get_learning_iterations' is w.r.t a
        previous batch size. In order to train on a different batch size, this function adjust the iterations with LR rule.

        :param current_batch_size: Number of images * number of GPUs we currently have
        :param previous_batch_size: The batch size for which the default learning rate was set on a particular dataset.
        :return: Amount to adjust the iterations  by (using learning rate rule) given our current batch size.
        """
        return self.default_batchsize // self.current_batch_size

    def get_learning_iterations(self):
        # Iterations w.r.t to FSDET_DEFAULT_BATCHSIZE from FsDet. Added iteration_multiplier.
        iteration_multiplier = self.iteration_multiplier
        COCO_ITERS_MAP = {
            'novel': {
                # STEPS, MAX_ITERATIONS
                1: (10000 * iteration_multiplier, 500 * iteration_multiplier),  # 200 epochs
                2: (10000 * iteration_multiplier, 1500 * iteration_multiplier),
                3: (10000 * iteration_multiplier, 1500 * iteration_multiplier),
                5: (10000 * iteration_multiplier, 1500 * iteration_multiplier),  # 120 epochs
                10: (10000 * iteration_multiplier, 2000 * iteration_multiplier),  # 80 epochs
                30: (10000 * iteration_multiplier, 6000 * iteration_multiplier),  # 80 epochs
            },
            # 'all':
            #     {
            #         1: (14400 * iteration_multiplier, 16000 * iteration_multiplier),
            #         2: (28800 * iteration_multiplier, 32000 * iteration_multiplier),
            #         3: (43200 * iteration_multiplier, 48000 * iteration_multiplier),
            #         5: (72000 * iteration_multiplier, 80000 * iteration_multiplier),
            #         10: (144000 * iteration_multiplier, 160000 * iteration_multiplier),
            #         30: (216000 * iteration_multiplier, 240000 * iteration_multiplier),
            # },
            # We know iterations for fine-tuning on 'novel'. For 'all', we have 4 times more classes.
            # Hence we can just choose to x4 the iteration from novel and add maybe 1000.
            # Steps are added by hand, whatever I think works best
            # NOTE(): This was the initial thought, but we later found out that actually this
            # number results in overfitting, so we JUST MULTIPLY BY 2
            'all':
                {
                    1: (2350 * iteration_multiplier, 3000 * iteration_multiplier),
                    2: (6500 * iteration_multiplier, 3000 * iteration_multiplier),
                    3: (6500 * iteration_multiplier, 3000 * iteration_multiplier),
                    5: (6500 * iteration_multiplier, 3000 * iteration_multiplier),
                    10: (8800 * iteration_multiplier, 5500 * iteration_multiplier),
                    30: (24500 * iteration_multiplier, 12000 * iteration_multiplier), #TODO ITERS for 30
            },
            'all-correct':
                {
                    1: (10000 * iteration_multiplier, 10000 * iteration_multiplier),
                    # 2: (6500 * iteration_multiplier, 3000 * iteration_multiplier),
                    # 3: (6500 * iteration_multiplier, 3000 * iteration_multiplier),
                    5: (36000 * iteration_multiplier, 36000 * iteration_multiplier),
                    10: (100000 * iteration_multiplier, 48000 * iteration_multiplier),
                    # 30: (24500 * iteration_multiplier, 12000 * iteration_multiplier), #TODO ITERS for 30
            }

        }
        ITER_MAP = {'coco': COCO_ITERS_MAP}

        return ITER_MAP[self.dataset][self.dataset_mode]

.


class ExtraOptionHandler():
    def __init__(self, extra_options: List[Tuple] = []):
        self.extra_options = extra_options
        self.extra_values = [val if isinstance(val, list) else [val]
                             for _, val in self.extra_options]
        self.extra_names = [name for name, _ in self.extra_options]

    def value_iter(self):
        return itertools.product(*self.extra_values)

    @staticmethod
    def parse_extra_options(args) -> List:
        extra_options = []
        if args.score_thresh:
            extra_options.append((ExtraOption.SCORE_THRESH, args.score_thresh))
        if args.alpha_weighting:
            extra_options.append((ExtraOption.ALPHA_WEIGHTING, args.alpha_weighting))
        if args.use_oracle:
            extra_options.append((ExtraOption.USE_ORACLE, True))
        if args.like_oneshot_paper:
            extra_options.append((ExtraOption.LIKE_ONESHOT_PAPER, True))
        if args.like_fgn_paper:
            extra_options.append((ExtraOption.LIKE_FGN_PAPER, True))
        if args.cosim_logit:
            extra_options.append((ExtraOption.COSIM_LOGIT, 'pure-metric-averaged'))
        if args.cosine_scale:
            extra_options.append((ExtraOption.COSINE_SCALE, args.cosine_scale))
        if args.set_voc_test:
            extra_options.append((ExtraOption.SET_VOC_TEST, args.set_voc_test))



        return extra_options

    @staticmethod
    def perform_extra_option_work(output_dir: str, extra_option_names: List[ExtraOption],
                                  extra_option_values: List,
                                  dry_run: bool = True) -> Tuple[str, str]:
        """Performs the extra work needed if we have extra options and outputs the extra string needed
        to be passed to the shell command string for training.
        This generally means creating extra directories or copying the models from our existing output
        directory to another directory.

        :return: Tuple of strings to be appended to ran shell command and the new output dir, 
        if it was modified
        """
        assert extra_option_names is not None

        suffixes = ExtraOptionHandler._get_extra_option_suffixes(extra_option_names, extra_option_values)
        new_output_dir = output_dir + suffixes

        print(f"Copying from {output_dir + '/model_final.pth'} to {new_output_dir + '/model_final.pth'}")
        if not dry_run:
            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir, exist_ok=False)
                # print(f"Copying from {output_dir + '/model_final.pth'} to {new_output_dir} + '/model_final.pth")
                # Copy files from base dir so that we can emulate training from this new extra-opt folder
                shutil.copy(output_dir + '/model_final.pth', new_output_dir + '/model_final.pth')
                shutil.copy(output_dir + '/last_checkpoint', new_output_dir + '/last_checkpoint')
            else:
                print(f"Output dir {new_output_dir} already exists! Skipping model copy!")

        extra_params = f' OUTPUT_DIR {new_output_dir} '
        extra_params += ExtraOptionHandler._get_params_to_pass(
            extra_option_names, extra_option_values)

        return extra_params, new_output_dir

    @staticmethod
    def _get_param(extra_option_name, extra_option_value):
        """Given param and a value returns what options we need to pass to main script"""
        extra_opt_str = ''
        if extra_option_name == ExtraOption.SCORE_THRESH:
            extra_opt_str += f' MODEL.ROI_HEADS.SCORE_THRESH_TEST {extra_option_value}'
        elif extra_option_name == ExtraOption.ALPHA_WEIGHTING:
            raise NotImplementedError
        elif extra_option_name == ExtraOption.USE_ORACLE:
            raise NotImplementedError
        elif extra_option_name == ExtraOption.LIKE_ONESHOT_PAPER:
            extra_opt_str += f' TEST.EVAL_METHOD like_oneshot'
        elif extra_option_name == ExtraOption.COSIM_LOGIT:
            extra_opt_str += f' TEST.MODEL_METHOD {extra_option_value}'
        elif extra_option_name == ExtraOption.LIKE_FGN_PAPER:
            extra_opt_str += f' TEST.EVAL_METHOD like_fgn'
        elif extra_option_name == ExtraOption.COSINE_SCALE:
            extra_opt_str += f' MODEL.ROI_HEADS.COSINE_SCALE {extra_option_value}'
        elif extra_option_name == ExtraOption.SET_VOC_TEST:
            extra_opt_str += " DATASETS.TEST \"('VOCSegm_val_novel',)\""
        else:
            raise NotImplementedError
        
        # Make sure we have an extra space at the end of every option in the command line
        extra_opt_str += ' '
        return extra_opt_str

    @staticmethod
    def _get_params_to_pass(extra_opt_names, extra_opt_values):
        """Given list of params and values returns what options we need to pass to main script"""
        extra_params = ''
        for name, val in zip(extra_opt_names, extra_opt_values):
            extra_params += ExtraOptionHandler._get_param(name, val)
        return extra_params

    @staticmethod
    def _get_extra_option_suffixes(extra_option_names, extra_option_values):
        """
        Given a list of extra options and extra values for an experiment, returns suffixes to
        append to the output path
        """
        assert len(extra_option_names) == len(extra_option_values)

        suffixes = ''
        for name, val in zip(extra_option_names, extra_option_values):
            suffixes += ExtraOptionHandler._get_extra_option_suffix(name, val)
        return suffixes

    @staticmethod
    def _get_extra_option_suffix(extra_option_name, extra_option_value) -> str:
        """
            Compute suffix to be appended to path.  Based on extra options for training (Oracle etc.)
            If no suffix is required then returns an empty string.
        """
        if extra_option_name is None:
            return ''

        def float_to_sanitizedstr(number: int) -> str:
            """
                Examples:
                1.00 -> 100
                0.5  -> 050
                0.05 -> 005
            """
            number = f'{number:.2f}'
            leading, tail = number.split('.')
            return leading + tail  # Get rid of the '.'

        extra_suffix = ''
        if extra_option_name == ExtraOption.SCORE_THRESH:
            extra_suffix = '_SCORE_THRESH_' + float_to_sanitizedstr(extra_option_value)
        elif extra_option_name == ExtraOption.ALPHA_WEIGHTING:
             extra_suffix = '_ALPHA_WEIGHT_' + float_to_sanitizedstr(extra_option_value)
             raise NotImplementedError
        elif extra_option_name == ExtraOption.USE_ORACLE:
            extra_suffix = '_ORACLE'
            raise NotImplementedError
        elif extra_option_name == ExtraOption.LIKE_ONESHOT_PAPER:
            extra_suffix = '_LIKE_ONESHOT'
        elif extra_option_name == ExtraOption.LIKE_FGN_PAPER:
            extra_suffix = '_LIKE_FGN'
        elif extra_option_name == ExtraOption.COSIM_LOGIT:
            extra_suffix = '_CSLOGIT'
        elif extra_option_name == ExtraOption.COSINE_SCALE:
            extra_suffix = '_CSSCALE_' +float_to_sanitizedstr(extra_option_value)
        elif extra_option_name == ExtraOption.SET_VOC_TEST:
            extra_suffix = '_TEST_VOC'
        elif not extra_option_name:
            raise ValueError("Can't get suffix from current extra option name!")

        return extra_suffix


def parse_args(custom_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=2)
    parser.add_argument('--imgs-per-gpu', type=int)
    parser.add_argument('--shots', type=int, nargs='+', default=[1],
                        help='Shots to run experiments over')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 10],
                        help='Range of seeds to run')
    parser.add_argument('--root', type=str, default='./', help='Root of data')

    parser.add_argument(
        "--method", choices=['default', 'pure-metric-averaged', 'pure-metric-finetuned'], type=str,
        default='default')
    parser.add_argument("--fewshot-finetune-source", type=str,
                        help="Used in case of pure-metric-finetuned to retrieve novel shots path to add to base shots. Example: mask_rcnn_R_50_FPN_ft_fullclsag_cos_bh_novel_1shot")
    parser.add_argument('--ckpt-freq', type=int, required=True,
                        help='Frequency of saving checkpoints')
    # Model
    parser.add_argument('--base-config', type=str, help='Base config for 1shot. '
                        'All generated configs are based on this one', required=True)
    parser.add_argument('--no-dry-run', action='store_true',
                        help='Enabled by default such that script has no affect')
    parser.add_argument('--no-eval', action='store_true', help="Disables evaluation if given "
                        "Especially useful if we want to only obtrain some model via a 'method' but not evaluate")
    parser.add_argument('--skip-config-write', action='store_true')

    parser.add_argument('--cosim-logit', action='store_true')
    parser.add_argument('--score-thresh', type=float, nargs='+')
    parser.add_argument('--alpha-weighting', type=float, nargs='+')
    parser.add_argument('--cosine-scale', type=float, nargs='+')
    parser.add_argument('--use-oracle', action='store_true')
    parser.add_argument('--like-oneshot-paper', action='store_true')
    parser.add_argument('--like-fgn-paper', action='store_true')
    parser.add_argument('--set-voc-test', action='store_true')
    # Datasets
    datasets_group = parser.add_mutually_exclusive_group(required=True)
    # COCO arguments
    datasets_group.add_argument('--coco', action='store_true', help='Use COCO dataset')

    args = parser.parse_args(custom_args)

    extra_options = ExtraOptionHandler.parse_extra_options(args)

    if not args.no_dry_run:
        print('Doing a dry-run of our code!')

    # These configs leverage the fact that our other training methods start with 'metric'
    # Ex1 : configs/coco-experiments/mask_rcnn_R_50_FPN_fullclsag_metric_avg_cos_bh_all_1shot.yaml
    # Ex2: configs/coco-experiments/mask_rcnn_R_50_FPN_fullclsag_metric_finetunedw_cos_bh_all_1shot.yaml
    if 'metric' in args.base_config and 'metric' not in args.method:
        raise ValueError("Likely wanted to pass in method being finetuned")
    print(args.base_config, args.method)

    return args, extra_options


def load_yaml_file(fname):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_shell_cmd(cmd: str):
    """Runs a command in a shell and pipes the output to print it

    :param cmd: Command
    """
    print(f'Executing : {cmd}')
    if args.no_dry_run:
        p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
        while True:
            line = p.stdout.readline().decode('utf-8')
            if not line:
                break
            print(line)


def get_base_model_path(current_model_name: str) -> str:
    """
        Given the current model, return the 'base_model' type for this model.
        The 'base model' is the model trained on the base_classes starting from scratch.
        For example:

        I. If using a fully class agnnostic model one could have:
            - Class agnostic with cosine similarity
            - Class agnostic without cosine similarity (using 'fc')
        II. If using non-class agnostic model ('fsdet'), then one could have:
            -Normal FSDet configuration here.
    """
    # No 'base' model defined unless our current model is 'all'.
    # This is because the only valid context to call this function is if we want to create an
    # 'all' model.

    assert 'all' in current_model_name
    if args.coco:
        checkpoint_dir = 'checkpoints/coco/'

    if 'fullclsag' in current_model_name and 'cos_bh' in current_model_name:
        # Cosine Similarity head with fully class agnostic base model
        base_filename = "faster_rcnn_R_50_ft_fullclsag_cos_bh_base/model_final.pth"
    elif 'fullclsag' in current_model_name:
        # Fully class agnostic model, with fully connected head
        base_filename = 'checkpoints/coco/faster_rcnn_R_50_FPN_fc_fullclsag_base/model_final.pth'
    else:
        raise NotImplementedError

    return os.path.join(checkpoint_dir, base_filename)


def retrieve_metric_finetuned_fewshot_source(output_dir: str, shot: int, ft_source : str) -> str:
    """
    This function is only called in the metric-finetunedw case.

    Given the output dir of the model and the shot number for which we are evaluating, retrieves
    the path towards a fine-tuned model on the novel dataset. 
    Uses 'fullclsag_cos_bh_novel' as a base dataset
    """
    output_root, _ = os.path.split(output_dir)
    # Hardcode the name for the fewshot novel and replace shot number
    assert '1shot' in ft_source

    # Previous code:
    # fewshot_novel_name = 'mask_rcnn_R_50_FPN_ft_fullclsag_cos_bh_novel_1shot'.replace(
    #     '1shot', f'{shot}shot')

    fewshot_novel_name = ft_source.replace(
        '1shot', f'{shot}shot')

    fewshot_model_source = os.path.join(output_root, fewshot_novel_name, 'model_final.pth')

    if not os.path.exists(fewshot_model_source):
        raise ValueError('Fewshot model source does not exist!')

    return fewshot_model_source


def get_output_name_from_base(base_name: str, shot: int) -> str:
    """Returns a filename from the base config path by replacing the shot number

    :param base_name: The base config path.
    :param shot: The shot number for the current experiment
    :return: A filename containing shot. No directories included, just a file name.
    """
    assert '1shot' in base_name or f'{shot}shot' in base_name

    output_cfg_name = base_name.replace('1shot', f'{shot}shot')
    _, output_cfg_name = os.path.split(output_cfg_name)

    return output_cfg_name


def get_train_dataset_name(config_train_name: str, shot: int, seed: int) -> str:
    """Returns a modified version of a given train dataset with proper shot and seed numbers"""
    # We replace the given config name's shot number with ours, unless they are already equal.
    if '1shot' in config_train_name:  # Replace shot number
        config_train_name = config_train_name.replace('1shot', str(shot) + 'shot')
    elif f'{shot}shot' in config_train_name:  # Leave shot as-is
        assert False
        # pass
    else:
        assert False

    # Modify or add seed number in name.
    if 'seed' in config_train_name:
        # Replace current seed number
        # Example: name_seedX with name_seed{SEED}, where SEED is our current seed
        assert len(re.findall('seed.', config_train_name)) == 1
        config_train_name = re.sub('seed.', 'seed{}'.format(seed), config_train_name)
    else:  # Append seed string
        config_train_name = config_train_name + f'_seed{seed}'

    return config_train_name


def get_config(seed, shot):
    """
        Uses a given base 1-shot config to replicate it for  'shot' and 'seed'.
        Changes dataset training split, cfg.OUTPUT_DIR and iteration number and steps accordingly.
    """
    base_config_path: str = args.base_config
    assert '1shot' in base_config_path

    dataset_mode = 'novel' if '_novel' in base_config_path else 'all'
    dataset_config = DatasetConfigs('coco' if args.coco else None, dataset_mode, args.method,
                                    args.num_gpus, args.imgs_per_gpu, is_correct_train_iters='correct' in base_config_path)

    seed_str = f'seed{seed}'
    dataset_split = re.findall('split.', base_config_path)
    assert len(dataset_split) <= 1
    dataset_split = dataset_split[0] if dataset_split else ''
    output_cfg_name = get_output_name_from_base(base_config_path, shot)
    model_output_root = os.path.join(args.root, dataset_config.checkpoint_dir, dataset_split, seed_str)

    os.makedirs(model_output_root, exist_ok=True)

    output_dir = os.path.join(model_output_root,
                              os.path.splitext(output_cfg_name)[0])

    result_config = load_yaml_file(base_config_path)
    result_config = _fill_config(result_config, shot, dataset_split, seed, dataset_config, output_dir)

    print(yaml.dump(result_config))

    dry_run_config = not args.no_dry_run or args.skip_config_write

    output_cfg_fullpath = _save_config(dataset_config.config_dir, dataset_split,
                                       seed_str, output_cfg_name, result_config, dry_run_config)

    return output_cfg_fullpath, result_config


def _fill_config(result_config, shot, dataset_split, seed, dataset_config, output_dir):
    """Fill the resulting config with all the relevant experiment info in it's YAML file"""
    if not dataset_split:
        base_rcnn_config = '../../Base-RCNN-FPN.yaml'
    else:
        base_rcnn_config = '../../../Base-RCNN-FPN.yaml'

    result_config['_BASE_'] = base_rcnn_config
    result_config['DATASETS']['TRAIN'] = make_tuple(result_config['DATASETS']['TRAIN'])
    result_config['DATASETS']['TEST'] = make_tuple(result_config['DATASETS']['TEST'])

    previous_train_name = result_config['DATASETS']['TRAIN'][0]
    new_train_name = get_train_dataset_name(previous_train_name, shot, seed)
    result_config['DATASETS']['TRAIN'] = (new_train_name,)

    previous_batch_size = result_config['SOLVER']['IMS_PER_BATCH']
    result_config['SOLVER']['IMS_PER_BATCH'] = dataset_config.current_batch_size

    ITERS = dataset_config.get_learning_iterations()
    result_config['SOLVER']['MAX_ITER'] = ITERS[shot][1]
    result_config['SOLVER']['STEPS'] = (ITERS[shot][0],)
    if args.ckpt_freq:
        result_config['SOLVER']['CHECKPOINT_PERIOD'] = ITERS[shot][1] // args.ckpt_freq
    else:
        result_config['SOLVER']['CHECKPOINT_PERIOD'] =  1000000 # <- So we don't checkpoint at all

    lr_modifier = previous_batch_size // dataset_config.current_batch_size
    if previous_batch_size % dataset_config.current_batch_size != 0:
        raise ValueError(
            f'Previous batch size in config - {previous_batch_size} is not divided by the '
            f'current desired batch size {dataset_config.current_batch_size}')

    base_config_lr = result_config['SOLVER']['BASE_LR']
    result_config['SOLVER']['BASE_LR'] = base_config_lr * (1.0 / lr_modifier)
    result_config['OUTPUT_DIR'] = output_dir

    if 'all' in dataset_config.dataset_mode  and 'default' in dataset_config.method:
        weights_path = result_config['MODEL']['WEIGHTS']
        result_config['MODEL']['WEIGHTS'] = _get_weights_all_dataset(weights_path, shot, seed)

    return result_config


def _save_config(config_dir, split_str, seed_str, output_cfg_name, result_config, dry_run=False):
    output_cfg_fullpath = os.path.join(args.root, config_dir, split_str, seed_str, output_cfg_name)

    print(f'Writing to {output_cfg_fullpath}')
    if not dry_run:
        print("Writing config!")
        os.makedirs(os.path.split(output_cfg_fullpath)[0], exist_ok=True)
        with open(output_cfg_fullpath, 'w') as fp:
            fp.write('# Seed config autogenerated \n'
                     '# NUM_GPUS : {}\n'.format(args.num_gpus))
            yaml.dump(result_config, fp)
    return output_cfg_fullpath


def execute_train(model_path, method, num_gpus, config_path, fewshot_model_source='') -> bool:
    """Executes the training script by sending it to an externall shell. Returns True if executed"""

    train_cmd = f'python tools/run_train.py ' \
                f'--method {method} --dist-url auto --num-gpus {num_gpus} ' \
                f'--config-file {config_path} --resume '
    if fewshot_model_source:
        train_cmd = train_cmd + f'--src1 {fewshot_model_source}'

    if not os.path.exists(model_path):
        executed_train = True
        run_shell_cmd(train_cmd)
    else:
        print(f'Train command would have executed : {train_cmd}')
        executed_train = False
        print(f'Skipping train step for {model_path}')

    return executed_train


def execute_test(output_dir: str, method: str, num_gpus: int, cfg_path: str, extra_opt_string='') -> bool:
    """Executes the testing script by sending it to an externall shell. Returns True if executed"""
    result_dir = os.path.join(args.root, output_dir)
    results_path = os.path.join(result_dir, 'inference', 'res_final.json')
    test_cmd = f'python tools/run_train.py --dist-url auto --num-gpus {num_gpus} ' \
                f'--method {method} --config-file {cfg_path} --resume --eval-only {extra_opt_string} '

    if not os.path.exists(results_path):
        executed_test = True
        run_shell_cmd(test_cmd)
    else:
        print(f'Test command would have executed {test_cmd}')
        executed_test = False
        print(f'Skipping test step for {results_path}')
    return executed_test


def _get_weights_all_dataset(weights_path, shot, seed) -> str:
    """Returns the path to a model with weights having base + novel with novel finetuned using
    a `pure-metric-finetuned` method """
    assert 'finetunedw' in weights_path or 'combined' in weights_path  # We expect to get this kind of config file
    if '1shot' not in weights_path:
        raise ValueError(f'{weights_path} does not have 1shot in it')
    if 'seed0' not in weights_path:
        raise ValueError(f'{weights_path} does not have seed0 in it')

    weights_path = weights_path.replace('1shot', f'{shot}shot').replace('seed0', f'seed{seed}')

    if not os.path.exists(weights_path):
        raise ValueError(f'{weights_path} does not exist')

    return weights_path


def run_experiment(cfg_path, cfg_dict, shot, extra_option_name=[], extra_option_value=[]):
    """
    Run training and evaluation scripts based on given config files.
    """
    # Train setup and execute
    output_dir: str = cfg_dict['OUTPUT_DIR']
    model_path = os.path.join(args.root, output_dir, 'model_final.pth')

    dataset_mode = 'novel' if '_novel' in args.base_config else 'all'
    fewshot_model_source = ''
    if dataset_mode == 'all':
        if 'metric-finetuned' in args.method:
            fewshot_model_source = retrieve_metric_finetuned_fewshot_source(output_dir, shot, args.fewshot_finetune_source)
        elif 'default' in args.method:
            pass

    executed_train = execute_train(model_path, args.method, args.num_gpus, cfg_path,
                                   fewshot_model_source)
    if args.no_eval:
        print("Skipping eval due to no-eval flag!")
        return
    extra_option_str = ''
    if extra_option_name and executed_train:
        raise ValueError('Executed training but passed extra test options or dataset != all')
    if extra_option_name:  # We have extra options we want to add
        extra_option_str, output_dir = ExtraOptionHandler.perform_extra_option_work(output_dir,
                                                                 extra_option_name,
                                                                 extra_option_value,
                                                                 dry_run=not args.no_dry_run)
    executed_test = execute_test(output_dir, args.method, args.num_gpus, cfg_path,
                                 extra_opt_string=extra_option_str)


def main(args, extra_options):

    extra_options = ExtraOptionHandler(extra_options)
    for shot in args.shots:
        for seed in range(args.seeds[0], args.seeds[1]):

            print('Seed: {}, Shot: {}'.format(seed, shot))
            config_path, config_dict = get_config(seed, shot)

            for extra_opt_values in extra_options.value_iter():
                    run_experiment(config_path, config_dict, shot,
                                   extra_option_name=extra_options.extra_names,
                                   extra_option_value=extra_opt_values)


if __name__ == '__main__':
    args, extra_options =parse_args()

    main(args, extra_options)
