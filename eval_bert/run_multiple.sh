#!/bin/bash


python run_bert.py --name gpu2 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
python run_bert.py --name gpu2 --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density


python run_bert.py   --name gpu2 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random
python run_bert.py   --name gpu2 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
python run_bert.py   --name gpu2 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
python run_bert.py   --name gpu2 --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density

python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls
python run_bert.py  --name gpu2  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density

#python run_bert.py --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --training_index -3



#python run_bert.py --name final --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc  --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density --training_index -3

#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc  --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density --training_index -5

#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc  --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density --training_index -3
 
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc  --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density --training_index -5

#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc  --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls --training_index -3
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density --training_index -3

#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc  --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density --training_index -5
#python run_bert.py --name final  --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli --training_index -5
#python run_bert.py  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density --training_index -5
#python run_bert.py  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls --training_index -5
#python run_bert.py  --name final --dataset reuters --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density --training_index -5
#python run_bert.py --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --training_index -5



#python run_bert.py --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc
#python run_bert.py --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc --random
#python run_bert.py --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density
#python run_bert.py --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli
#python run_bert.py --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isli_density
#python run_bert.py --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls
#python run_bert.py --name final --dataset toxic --steps 100 --encoding elmo --save_repeat 5 --query_strategy isls_density


#python run_bert.py --dataset app_store --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
#python run_bert.py --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density
#python run_bert.py --dataset app_store --steps 100 --encoding sif --save_repeat 5 --query_strategy unc_density
#python run_bert.py --dataset app_store --steps 100 --encoding elmo --save_repeat 5 --query_strategy unc_density

#python run_bert.py --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc_density
#python run_bert.py --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc_density



#python run_bert.py --dataset reuters --steps 100 --encoding sif --save_repeat 5 --query_strategy unc_density

#python run_bert.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
#python run_bert.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random 

#python run_bert.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
#python run_bert.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
#python run_bert.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
#python run_bert.py --dataset toxic --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density

#python run_bert.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy unc
#python run_bert.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy unc --random 
#python run_bert.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy isli
#python run_bert.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy isli_density
#python run_bert.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy isls
#python run_bert.py --dataset toxic --steps 100 --encoding sif --save_repeat 5 --query_strategy isls_density

#python run_bert.py --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc
#python run_bert.py --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli
#python run_bert.py --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isli_density
#python run_bert.py --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls
#python run_bert.py --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy isls_density
#python run_bert.py --dataset reuters --steps 100 --encoding sbert --save_repeat 5 --query_strategy unc --random 

#python run_bert.py --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc
#python run_bert.py --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli
#python run_bert.py --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isli_density
#python run_bert.py --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls
#python run_bert.py --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy isls_density
#python run_bert.py --dataset reuters --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random 

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