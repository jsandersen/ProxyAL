#!/bin/bash

python run_bert.py --dataset hate_speech --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
python run_bert.py --dataset hate_speech --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random 

#python run_bert.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
#python run_bert.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
#python run_bert.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
#python run_bert.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
#python run_bert.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density
#python run_bert.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random 

#python run_bert.py --dataset imdb --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
#python run_bert.py --dataset imdb --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
#python run_bert.py --dataset imdb --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
#python run_bert.py --dataset imdb --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
#python run_bert.py --dataset imdb --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density
#python run_bert.py --dataset imdb --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random 

#python run_bert.py --dataset imdb --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc
#python run_bert.py --dataset imdb --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli
#python run_bert.py --dataset imdb --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density
#python run_bert.py --dataset imdb --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls
#python run_bert.py --dataset imdb --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density
#python run_bert.py --dataset imdb --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random 

#python run_bert.py --dataset imdb --steps 100 --encoding sif --save_repeat 5 --query_strategy unc
#python run_bert.py --dataset imdb --steps 100 --encoding sif --save_repeat 5 --query_strategy isli
#python run_bert.py --dataset imdb --steps 100 --encoding sif --save_repeat 5 --query_strategy isli_density
#python run_bert.py --dataset imdb --steps 100 --encoding sif --save_repeat 5 --query_strategy isls
#python run_bert.py --dataset imdb --steps 100 --encoding sif --save_repeat 5 --query_strategy isls_density
#python run_bert.py --dataset imdb --steps 100 --encoding sif --save_repeat 5 --query_strategy unc --random 