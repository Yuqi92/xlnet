create tf record


sudo docker run --name=create_tfrecord --runtime=nvidia -d -v /collab/ysi/work/pretrain_xlnet/xlnet:/tmp/file \
	-e NVIDIA_VISIBLE_DEVICES=1 \
        --user $(id -u):$(id -g) -w /tmp/file ysi/customerized-xlnet:0.2 python data_utils.py \
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
