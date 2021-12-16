#!/bin/bash

python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density

python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
python run_al_experiment.py --name final --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density

python run_al_experiment.py  --name final --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc
python run_al_experiment.py  --name final --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random
python run_al_experiment.py  --name final --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density
python run_al_experiment.py  --name final --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli
python run_al_experiment.py  --name final --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density
python run_al_experiment.py  --name final --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls
python run_al_experiment.py  --name final --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density

python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density

python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
python run_al_experiment.py --name final --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density

python run_al_experiment.py  --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc
python run_al_experiment.py  --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random
python run_al_experiment.py  --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density
python run_al_experiment.py  --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli
python run_al_experiment.py  --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density
python run_al_experiment.py  --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls
python run_al_experiment.py  --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --warmstart 3  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density




#python run_al_experiment.py  --name gpu3 --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc 
#python run_al_experiment.py  --name gpu3 --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --name gpu3 --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py  --name gpu3 --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py  --name gpu3 --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py  --name gpu3 --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py   --name gpu3 --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density


#python run_al_experiment.py --name gpu2 --dataset reuters --steps 100 --encoding ebert --save_repeat 5 --query_strategy unc



#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --name cpu --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --name cpu --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density


#python run_al_experiment.py --warmstart 3 --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --warmstart 3 --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --warmstart 3 --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --warmstart 3 --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --warmstart 3 --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --warmstart 3 --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --warmstart 3 --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random

#python run_al_experiment.py  --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py  --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py  --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls
#python run_al_experiment.py  --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py  --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy isls
#python run_al_experiment.py  --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --dataset hate_speech --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc 

#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --dataset hate_speech --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random 



#python run_al_experiment.py --dataset hate_speech --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density 
#python run_al_experiment.py --dataset hate_speech --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli 
#python run_al_experiment.py --dataset hate_speech --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density 
#python run_al_experiment.py --dataset hate_speech --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls 
#python run_al_experiment.py --dataset hate_speech --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density 

#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random 

#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random
#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc

#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --warmstart 6 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density

#python run_al_experiment.py --dataset news_groups --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
#python run_al_experiment.py --dataset news_groups --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
#python run_al_experiment.py --dataset news_groups --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
#python run_al_experiment.py --dataset news_groups --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
#python run_al_experiment.py --dataset news_groups --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
#python run_al_experiment.py --dataset news_groups --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density
#python run_al_experiment.py --dataset news_groups --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random 
#python run_al_experiment.py --dataset app_store  --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random 


#python run_al_experiment.py --dataset app_store  --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc