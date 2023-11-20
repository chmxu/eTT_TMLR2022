# Exploring Efficient Few-shot Adaptation for Vision Transformers
This is an official implementation in Pytorch of eTT. Our paper is available at [link](https://arxiv.org/pdf/2301.02419.pdf).




### Data Preparation
This repo adopts the same data structure as [TSA](https://github.com/VICO-UoE/URL). We simply quote the original data preparation here, thank the authors of TSA for the contribution.

* Follow the "User instructions" in the [Meta-Dataset repository](https://github.com/google-research/meta-dataset) for "Installation" and "Downloading and converting datasets".
    * Edit ```./meta-dataset/data/reader.py``` in the meta-dataset repository to change ```dataset = dataset.batch(batch_size, drop_remainder=False)``` to ```dataset = dataset.batch(batch_size, drop_remainder=True)```. (The code can run with ```drop_remainder=False```, but in our work, we drop the remainder such that we will not use very small batch for some domains and we recommend to drop the remainder for reproducing our methods.)
    * To test unseen domain (out-of-domain) performance on additional datasets, i.e. MNIST, CIFAR-10 and CIFAR-100, follow the installation instruction in the [CNAPs repository](https://github.com/cambridge-mlg/cnaps) to get these datasets.
    * Run the following commands.
    ```shell script
    ulimit -n 50000
    export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>
    export RECORDS=<the directory where tf-records of MetaDataset are stored>
    ```

### Training & Inference
To run this code you need to get a DINO-pretrained network weight. We recomment to re-run the original [DINO](https://github.com/facebookresearch/dino) using the meta-train set of ImageNet as training data. To do this, you need to clone the DINO repo and copy all files in ```pretrain_code_snippet``` in the DINO folder and run the training script. In detail,
```shell script
git clone https://github.com/facebookresearch/dino.git

cp -rf pretrain_code_snippet/* dino/

python -m torch.distributed.launch --nproc_per_node=8 main_dino_metadataset.py --arch {ARCH} --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```
This is the main setting in our paper. Technically speaking you can also use the 1000-class DINO weight provided in the original repo for the experiments.

After getting the pretrained weight you can run the meta-testing as follow:

```shell script
python test_extractor_pa_vit_prefix.py --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco --model.ckpt {WEIGHT PATH}
```
The code adopts ViT-small as default backbone structure can be modified according to your requirement.  

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@article{xu2023exploring,
  title={Exploring efficient few-shot adaptation for vision transformers},
  author={Xu, Chengming and Yang, Siqian and Wang, Yabiao and Wang, Zhanxiong and Fu, Yanwei and Xue, Xiangyang},
  journal={arXiv preprint arXiv:2301.02419},
  year={2023}
}
```

## Acknowledgement
We modify our code from [TSA](https://github.com/VICO-UoE/URL).