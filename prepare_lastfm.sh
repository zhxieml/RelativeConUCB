IN_FOLDER="/mnt/zhxie_hdd/dataset/lastfm"
OUT_FOLDER="data/lastfm"
LIBMF_EXE="/home/zhxie/GITHUB/libmf/mf-train"
mkdir -p ${OUT_FOLDER}
echo "=========Init: input from ${IN_FOLDER} and output to ${OUT_FOLDER}, using libmf in ${LIBMF_EXE}========="

echo "=========Task: extracting rating matrix & relations========="
python scripts/generate_lastfm_data.py -i ${IN_FOLDER} -o ${OUT_FOLDER}/lastfm

echo "=========Task: generating features by libmf========="
chmod +x ${LIBMF_EXE}
${LIBMF_EXE} --quiet -k 49 -f 10 -s 2 -t 50 ${OUT_FOLDER}/lastfm/rating_oneclass_train.txt ${OUT_FOLDER}/lastfm/raw_feats_train.txt
${LIBMF_EXE} --quiet -k 49 -f 10 -s 2 -t 50 ${OUT_FOLDER}/lastfm/rating_oneclass_test.txt ${OUT_FOLDER}/lastfm/raw_feats_test.txt

echo "=========Task: preparing features for real-data experiment========="
python scripts/extract_raw_feats.py -i ${OUT_FOLDER}/lastfm
python scripts/normalize_raw_feats.py -i ${OUT_FOLDER}/lastfm

echo "=========Task: preparing ground-truth feedback for real-data experiment========="
python scripts/generate_affinity_matrix.py -i ${OUT_FOLDER}/lastfm
cp -rf ${OUT_FOLDER}/lastfm ${OUT_FOLDER}/lastfm_drifted

echo "=========Task: preparing ground-truth feedback for drifted real-data experiment========="
mv ${OUT_FOLDER}/lastfm_drifted/affinity.npy ${OUT_FOLDER}/lastfm_drifted/affinity_raw.npy
python scripts/generate_drifted_affinity_matrix.py -i ${OUT_FOLDER}/lastfm_drifted

echo "=========Task: removing redundant files (optional)========="
rm ${OUT_FOLDER}/lastfm/rating_oneclass*
rm ${OUT_FOLDER}/lastfm/*raw*
rm ${OUT_FOLDER}/lastfm_drifted/rating_oneclass*
rm ${OUT_FOLDER}/lastfm_drifted/*raw*
