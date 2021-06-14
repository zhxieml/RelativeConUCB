OUT_FOLDER="data/synthetic"
mkdir -p ${OUT_FOLDER}

python scripts/generate_synthetic_group_data.py -o ${OUT_FOLDER}
