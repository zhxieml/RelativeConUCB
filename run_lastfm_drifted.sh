IN_FOLDER="data/lastfm/lastfm_drifted"
OUT_FOLDER="result"
mkdir -p ${OUT_FOLDER}

python run_exp_seq_bernoulli.py --num_repeat 10 --in_folder ${IN_FOLDER} --out_folder ${OUT_FOLDER} --arm_pool_size 50
