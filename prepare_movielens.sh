IN_FOLDER="/mnt/zhxie_hdd/dataset/movielens-2k"
OUT_FOLDER="data/movielens"
LIBMF_EXE="/home/zhxie/GITHUB/libmf/mf-train"
mkdir -p ${OUT_FOLDER}
echo "=========Init: input from ${IN_FOLDER} and output to ${OUT_FOLDER}, using libmf in ${LIBMF_EXE}========="

echo "=========Task: extracting rating matrix & relations========="
python scripts/generate_movielens_data.py -i ${IN_FOLDER} -o ${OUT_FOLDER}/movielens

echo "=========Task: generating features by libmf========="
chmod +x ${LIBMF_EXE}
${LIBMF_EXE} --quiet -k 49 -f 10 -s 2 -t 50 ${OUT_FOLDER}/movielens/rating_oneclass_train.txt ${OUT_FOLDER}/movielens/raw_feats_train.txt
${LIBMF_EXE} --quiet -k 49 -f 10 -s 2 -t 50 ${OUT_FOLDER}/movielens/rating_oneclass_test.txt ${OUT_FOLDER}/movielens/raw_feats_test.txt

echo "=========Task: preparing features for real-data experiment========="
python scripts/extract_raw_feats.py -i ${OUT_FOLDER}/movielens
python scripts/normalize_raw_feats.py -i ${OUT_FOLDER}/movielens

echo "=========Task: preparing ground-truth feedback for real-data experiment========="
python scripts/generate_affinity_matrix.py -i ${OUT_FOLDER}/movielens
cp -rf ${OUT_FOLDER}/movielens ${OUT_FOLDER}/movielens_drifted

echo "=========Task: preparing ground-truth feedback for drifted real-data experiment========="
mv ${OUT_FOLDER}/movielens_drifted/affinity.npy ${OUT_FOLDER}/movielens_drifted/affinity_raw.npy
python scripts/generate_drifted_affinity_matrix.py -i ${OUT_FOLDER}/movielens_drifted

echo "=========Task: removing redundant files (optional)========="
rm ${OUT_FOLDER}/movielens/rating_oneclass*
rm ${OUT_FOLDER}/movielens/*raw*
rm ${OUT_FOLDER}/movielens_drifted/rating_oneclass*
rm ${OUT_FOLDER}/movielens_drifted/*raw*
