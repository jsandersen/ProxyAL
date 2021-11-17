#cd ./encodings/
#python encode_elmo.py --data_set hate_speech
#cd ..

cd ./eval_proxy/
bash run_multiple.sh
cd ..

cd ./eval_bert/
bash run_multiple.sh