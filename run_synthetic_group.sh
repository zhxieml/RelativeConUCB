IN_FOLDER="data/synthetic/synthetic_group"
OUT_FOLDER="result/synthetic_group"
mkdir -p ${OUT_FOLDER}

python run_exp_seq_bernoulli.py --num_repeat 10 --in_folder ${IN_FOLDER}/synthetic_sigma_0.0 --out_folder ${OUT_FOLDER} --arm_pool_size 50
python run_exp_seq_bernoulli.py --num_repeat 10 --in_folder ${IN_FOLDER}/synthetic_sigma_0.5 --out_folder ${OUT_FOLDER} --arm_pool_size 50
python run_exp_seq_bernoulli.py --num_repeat 10 --in_folder ${IN_FOLDER}/synthetic_sigma_1.0 --out_folder ${OUT_FOLDER} --arm_pool_size 50
python run_exp_seq_bernoulli.py --num_repeat 10 --in_folder ${IN_FOLDER}/synthetic_sigma_2.0 --out_folder ${OUT_FOLDER} --arm_pool_size 50
python run_exp_seq_bernoulli.py --num_repeat 10 --in_folder ${IN_FOLDER}/synthetic_sigma_5.0 --out_folder ${OUT_FOLDER} --arm_pool_size 50
