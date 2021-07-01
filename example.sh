CUDA_VISIBLE_DEVICES=2 python main_pretrain.py --gat_split 10 --model_name UPGAN --data_folder /gaolehe/data/kb_final/ --dataset book --batch_size 4096 --embedding_size 100 --n_epochs 1000 --lr 1e-3 --decay_rate 0.99 --checkpoint_dir checkpoint/good_pretrain/book --n_sample 200 --l2_lambda 1e-5 --rs_sample 20 --rs_sample_flag --kg_sample 10 --kg_sample_flag --eval_every 5 --experiment_name book-UPGAN-mlp-200sample-norm-emb --query_weight --load_ckpt_file init/book-DistMult-bce-decay.ckpt --norm_emb
CUDA_VISIBLE_DEVICES=1 python main_ifgan.py --gat_split 10 --model_name UPGAN --G_name generator_concat --data_folder /gaolehe/data/kb_final/ --dataset book --batch_size 4096 --embedding_size 100 --n_epochs 50 --lr 1e-4 --lr_g 1e-4 --decay_rate 0.0 --checkpoint_dir checkpoint/upgan --n_sample 1024 --n_sample_gen 200 --l2_lambda 1e-5 --l2_lambda_g 1e-5 --rs_sample 20 --rs_sample_flag --kg_sample 10 --kg_sample_flag --eval_every 3 --experiment_name book-norm_ent-concat-noise --query_weight --load_ckpt_file good_pretrain/book/book-UPGAN-mlp-200sample-norm-emb.ckpt --load_ckpt_G init/book-DistMult-bce-decay.ckpt --norm_emb --reward_type baseline-softmax --lambda_smooth 0.01 --sigma 1.0


#AmazonBook
python main_pretrain.py --gat_split 10 --model_name UPGAN --data_folder data/ --dataset AmazonBook --batch_size 4096 --embedding_size 100 --n_epochs 100 --lr 1e-3 --decay_rate 0.99 --checkpoint_dir checkpoint/good_pretrain --n_sample 200 --l2_lambda 1e-5 --rs_sample 20 --rs_sample_flag --kg_sample 10 --kg_sample_flag --eval_every 5 --experiment_name book-UPGAN-mlp-200sample-norm-emb --query_weight --load_ckpt_file init/book-DistMult-100dim.ckpt --norm_emb
python main_ifgan.py --gat_split 10 --model_name UPGAN --G_name generator_concat-ConvE --data_folder data/ --dataset AmazonBook --batch_size 2048  --n_epochs 100 --lr 1e-4 --lr_g 1e-4 --decay_rate 0.0 --checkpoint_dir checkpoint --n_sample 1024 --n_sample_gen 200 --l2_lambda 1e-5 --l2_lambda_g 1e-5 --rs_sample 20 --rs_sample_flag --kg_sample 10 --kg_sample_flag --eval_every 3 --experiment_name book-norm_ent-concat-noise --query_weight --load_ckpt_file good_pretrain/book-UPGAN-mlp-200sample-norm-emb.ckpt  --norm_emb --reward_type baseline-softmax --lambda_smooth 0.01 --sigma 1.0

#LastFM
python main_pretrain.py --gat_split 10 --model_name UPGAN --data_folder data/ --dataset LastFM --batch_size 4096 --embedding_size 100 --n_epochs 100 --lr 1e-3 --decay_rate 0.99 --checkpoint_dir checkpoint/good_pretrain --n_sample 200 --l2_lambda 1e-5 --rs_sample 20 --rs_sample_flag --kg_sample 10 --kg_sample_flag --eval_every 5 --experiment_name music-UPGAN-mlp-200sample-norm-emb --query_weight --load_ckpt_file init/music-DistMult-100dim.ckpt --norm_emb
python main_ifgan.py --gat_split 10 --model_name UPGAN --G_name generator_concat-ConvE --data_folder data/ --dataset LastFM --batch_size 2048  --n_epochs 100 --lr 1e-4 --lr_g 1e-4 --decay_rate 0.0 --checkpoint_dir checkpoint --n_sample 1024 --n_sample_gen 200 --l2_lambda 1e-5 --l2_lambda_g 1e-5 --rs_sample 20 --rs_sample_flag --kg_sample 10 --kg_sample_flag --eval_every 3 --experiment_name music-norm_ent-concat-noise --query_weight --load_ckpt_file good_pretrain/music-UPGAN-mlp-200sample-norm-emb.ckpt  --norm_emb --reward_type baseline-softmax --lambda_smooth 0.01 --sigma 1.0

#Movielens
python main_pretrain.py --gat_split 10 --model_name UPGAN --data_folder data/ --dataset Movielens --batch_size 4096 --embedding_size 100 --n_epochs 100 --lr 1e-3 --decay_rate 0.99 --checkpoint_dir checkpoint/good_pretrain --n_sample 200 --l2_lambda 1e-5 --rs_sample 20 --rs_sample_flag --kg_sample 10 --kg_sample_flag --eval_every 5 --experiment_name movie-UPGAN-mlp-200sample-norm-emb --query_weight --load_ckpt_file init/movie-DistMult-100dim.ckpt --norm_emb
python main_ifgan.py --gat_split 10 --model_name UPGAN --G_name generator_concat-ConvE --data_folder data/ --dataset Movielens --batch_size 2048  --n_epochs 100 --lr 1e-4 --lr_g 1e-4 --decay_rate 0.0 --checkpoint_dir checkpoint --n_sample 1024 --n_sample_gen 200 --l2_lambda 1e-5 --l2_lambda_g 1e-5 --rs_sample 20 --rs_sample_flag --kg_sample 10 --kg_sample_flag --eval_every 3 --experiment_name movie-norm_ent-concat-noise --query_weight --load_ckpt_file good_pretrain/movie-UPGAN-mlp-200sample-norm-emb.ckpt  --norm_emb --reward_type baseline-softmax --lambda_smooth 0.01 --sigma 1.0