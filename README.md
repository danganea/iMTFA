## iMTFA and MTFA
This README is still a work in progress, expect a refactor very soon.



This code is based on Detectron2 and parts of TFA's source code. We have edited the detectron2 source code to better match the few-shot instance segmentation task.

We advise the users to create a new conda environment and install our source code in the same way as the detectron2 source code. See [INSTALL.md](INSTALL.md).

After setting up the dependencies, installation should simply be:

`pip install -e .` in this folder.


## Configurations

For simplicity in running/automating the scripts, the naming scheme of these experiments is different than that in the paper.

All configs can be found in the `configs/coco-experiments` directory.


MTFA's first training stage is: `configs/coco-experiments/mask_rcnn_R_50_FPN_fc_fsdet_base.yaml`

iMTFA's first training stage is: `configs/coco-experiments/mask_rcnn_R_50_FPN_fc_fullclsag_base.yaml`
iMTFA's second training stage is: `configs/coco-experiments/mask_rcnn_R_50_FPN_ft_fullclsag_cos_bh_base.yaml`


1shot,5shot and 10_shot MTFA configs for the NOVEL classes are named as such:

`configs/coco-experiments/mask_rcnn_R_50_FPN_ft_fsdet_cos_novel_{shot_number}shot.yaml`

1shot,5shot and 10_shot MTFA configs for the ALL classes are named as such:

`configs/coco-experiments/mask_rcnn_R_50_FPN_ft_fsdet_cos_correct_all_{shot_number}shot.yaml`

1shot,5shot and 10_shot iMTFA configs for the NOVEL + ALL classes are named as such:

`configs/coco-experiments/mask_rcnn_R_50_FPN_fullclsag_metric_avg_cos_bh_normalized_{all/novel}_{shot_number}shot.yaml`


For COCO-Split-2 in the paper, all experiments have an appended `_split1` or are in the related `split1` directory.



Experiments on COCO2VOC

iMTFA : 

`configs/coco-experiments/mask_rcnn_R_50_FPN_fullclsag_metric_avg_cos_bh_normalized_novel_1shot_LIKE_FGN_CSSCALE_100_TEST_VOC.yaml`

MTFA:

`configs/coco-experiments/mask_rcnn_R_50_FPN_ft_fsdet_cos_novel_1shot_LIKE_FGN_TEST_VOC.yaml`

Experiments for every shot are generated in the `tools/run_experiments.py` script. This is why there are no configs for the
alpha value. These are generated automatically.


The `*_metric_avg_directcos_normalized_moreiters_novel_{}shot` experiments are 
for the One-Shot-Cosine and the `_metric_avg_FC` experiments are for the One-Shot-FC, detailed in the paper


## Models

Models temporarily available here:

https://1drv.ms/u/s!Ako0GB-Fly5dgaI6a9w7V7qGexkiiA?e=UUzeYV

Note: Not 100% of the models here are in the paper. The naming scheme follows the naming scheme above. You'll notice that in order to have a fair comparison between a first and second stage of training, we use a 'moreiters' setup for the first stage. This is to account for a larger number of iteration steps when training. In practice, we notice that training the cosine head for more iterations does help a bit, but not enough vs the two-stage approach.

### Running the scripts

Currently the scripts are somewhat convoluted to run. We entend to make this documentation nicer in the near future.

For now:

To run the training, the `tools/run_train.py` script is used. Run it with `-h` to get all available options

### Seting up the data

We use the same `datasets` folder used in Detectron2 and TFA. Download and unzip the cocosplit folder [here](https://drive.google.com/file/d/12jGNdhdL8jz5YO8Gz5P-liNtY7eAz6Av/view?usp=sharing).

Also, setup a `coco` directory in `datasets`, exactly the same way as TFA. For this, just download COCO2014 train + val and place them in trainval, similarly download COCO2014 test.


### Generating the few-shots

See `prepare_coco_few_shot.py` for generating them manually, but the `cocosplit` folder provided above already includes the splits


### Results

The main results can be found in the `paper_results_and_supp` folder

### Aggregating the results

`tools/aggregate_seeds.py` is the script which produces averages of all shots for an experiment.
`tools/aggregate_to_csv.py` produces CSV files for all aggregate seeds for an experiment.


### Additional comments

Additional explanations will be available soon.

Note: **Not all experiments in configs are used. Not all experiments in configs/coco-experiments/ are used in the paper.**
