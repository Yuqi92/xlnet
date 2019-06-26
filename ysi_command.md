## Create tf record

```
sudo docker run --name=create_tfrecord \
	--runtime=nvidia -d -v /collab/ysi/work/pretrain_xlnet/xlnet:/tmp/file \
	-e NVIDIA_VISIBLE_DEVICES=1 \
	--user $(id -u):$(id -g) -w /tmp/file ysi/customized-xlnet:0.1 python data_utils.py \
	--use_tpu=False \
	--bsz_per_host=32 \
	--num_core_per_host=1 \
	--uncased = False \
	--seq_len=512 \
	--reuse_len=256 \
	--input_glob=./mimic_data/mimic_xlnet.txt \
	--save_dir=./tf_record_mimic \
	--num_passes=20 \
	--bi_data=True \
	--sp_path=./xlnet_cased_L-24_H-1024_A-16/spiece.model \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=85 

```

## Run classification on imdb data
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
