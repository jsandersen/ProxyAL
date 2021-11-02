#!/bin/bash

python run_al_experiment.py --name j --dataset app_store --steps 100 --encoding sbert --save_repeat 1 --query_strategy unc
python run_al_experiment.py --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
python run_al_experiment.py --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
python run_al_experiment.py --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
python run_al_experiment.py --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
python run_al_experiment.py --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density
python run_al_experiment.py --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random 


python run_al_experiment.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
python run_al_experiment.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
python run_al_experiment.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
python run_al_experiment.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
python run_al_experiment.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
python run_al_experiment.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density
python run_al_experiment.py --dataset imdb --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random 
#python run_al_experiment.py --dataset app_store  --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random 


#python run_al_experiment.py --dataset app_store  --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc