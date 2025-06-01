# tokenize
echo "tokenize"
SRC_FILE=/data3/hcjiang/data_augment/data/news/news.seq.400w.txt

python ../utils/segment_thulac.py --data_file $SRC_FILE --thulac_file $SRC_FILE".thulac"  # 分字

echo "Parse"
# Dep Parse
CUDA_VISIBLE_DEVICES=5,6,7 python ../src_gopar/parse.py $SRC_FILE".thulac" $SRC_FILE.dep.conll_predict
# Con Parse
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python ../src_gopar/con_parse.py $SRC_FILE".thulac" $SRC_FILE.con.conll_predict

echo "generate"
python ../syntax_gec_error_generate.py

echo "analyse"
python ../edit_analyse.py