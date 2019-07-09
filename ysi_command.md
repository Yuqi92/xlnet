## Create tf record (important to make sure otherwise the file name is wrong)

correct file name should be ``record_info-train-0-0.bsz-16.seqlen-128.reuse-64.bi.alpha-6.beta-1.fnp-85.json``

```
sudo docker run --name=create_tfrecord \
  --runtime=nvidia -d -v /collab/ysi/work/pretrain_xlnet/xlnet:/tmp/file \
  -e NVIDIA_VISIBLE_DEVICES=1 \
  --user $(id -u):$(id -g) -w /tmp/file ysi/customized-xlnet:0.1 python data_utils.py \
  --use_tpu=False \
  --bsz_per_host=16 \
  --num_core_per_host=1 \
  --uncased=False \
  --seq_len=128 \
  --reuse_len=64 \
  --input_glob=./mimic_data/mimic_xlnet.txt \
  --save_dir=./tf_record_mimic_seqlen128 \
  --num_passes=20 \
  --bi_data=True \
  --sp_path=./xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 

```

## Pretrain 

```
sudo docker run --name=pretrain_xlnet_mimic \
  --runtime=nvidia -d -v /collab/ysi/work/pretrain_xlnet/xlnet:/tmp/file \
  -e NVIDIA_VISIBLE_DEVICES=2 \
  --user $(id -u):$(id -g) -w /tmp/file ysi/customized-xlnet:0.1 python train_gpu.py \
  --record_info_dir=./tf_record_mimic_seqlen128/tfrecords \
  --model_dir= ./mimic_xlnet_cased_large \
  --init_checkpoint=./xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --train_batch_size=16 \
  --train_steps=500000 \
  --save_steps=10000 \
  --num_core_per_host=1 \
  --seq_len=128 \
  --reuse_len=64 \
  --perm_size=64 \
  --n_layer=24 \
  --d_model=1024 \
  --d_embed=1024 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=4096 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --uncased=False
```

## Run classification on imdb data

1. Train

```
sudo docker run --name=imdb \
--runtime=nvidia -d -v /collab/ysi/work/finetune_xlnet:/tmp/file \
-e NVIDIA_VISIBLE_DEVICES=2 \
--user $(id -u):$(id -g) -w /tmp/file ysi/customized-xlnet:0.1 \
python run_classifier.py \
  --do_train=True \
  --do_eval=False \
  --task_name=imdb \
  --data_dir=./aclImdb \
  --output_dir=./results/proc_data/imdb \
  --model_dir=./results/models/imdb \
  --uncased=False \
  --spiece_model_file=./xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=./xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --init_checkpoint=./xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=5e-5 \
  --train_steps=5000 \
  --warmup_steps=120 \
  --save_steps=100 \
  --predict_ckpt=True 
```

2. Eval

```
sudo docker run --name=imdb_eval_allckpt \
--runtime=nvidia -d -v /collab/ysi/work/finetune_xlnet:/tmp/file \
-e NVIDIA_VISIBLE_DEVICES=2 \
--user $(id -u):$(id -g) -w /tmp/file ysi/customized-xlnet:0.1 \
python run_classifier.py \
  --do_train=False \
  --do_eval=True \
  --task_name=imdb \
  --data_dir=./aclImdb \
  --output_dir=./results/proc_data/imdb \
  --model_dir=./results/models/imdb \
  --uncased=False \
  --spiece_model_file=./xlnet_cased_L-24_H-1024_A-16/spiece.model \
  --model_config_path=./xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=5e-5
  --eval_all_ckpt=True
```


## Run NER on conll2003 data

```
python prepro/prepro_conll.py \
--data_format json \
--input_file ./data/ner/conll2003/raw/dev.txt \
--output_file ./data/ner/conll2003/dev-conll2003/dev-conll2003.json


sudo docker run --name=xlnet_ner \
  --runtime=nvidia -d -v /collab/ysi/work/xlnet_extension_tf:/tmp/file \
  -e NVIDIA_VISIBLE_DEVICES=1 \
  --user $(id -u):$(id -g) -w /tmp/file ysi/xlnet_ner_py3:0.1  python run_ner.py \
    --spiece_model_file=./xlnet_ckpt_260k/spiece.model \
    --model_config_path=./xlnet_ckpt_260k/xlnet_config.json \
    --init_checkpoint=./xlnet_ckpt_260k/model.ckpt \
    --task_name=conll2003 \
    --data_dir=data/ner/conll2003 \
    --output_dir=output/ner/conll2003/data \
    --model_dir=output/ner/conll2003/checkpoint \
    --export_dir=output/ner/conll2003/export \
    --max_seq_length=128 \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --predict_batch_size=16 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --learning_rate=2e-5 \
    --train_steps=4000 \
    --warmup_steps=100 \
    --save_steps=500 \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --do_export=true
```
