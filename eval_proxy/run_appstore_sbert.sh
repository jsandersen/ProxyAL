#!/bin/bash
python al_app_store_sbert.py --dataset app_store  --steps 100 --encoding sbert  --query_strategy unc
python al_app_store_sbert.py --dataset app_store  --steps 100 --encoding sbert  --query_strategy unc_density
python al_app_store_sbert.py --dataset app_store  --steps 100 --encoding sbert  --query_strategy unc_isls
python al_app_store_sbert.py --dataset app_store  --steps 100 --encoding sbert  --query_strategy isli
python al_app_store_sbert.py --dataset app_store  --steps 100 --encoding sbert  --query_strategy isli_density
python al_app_store_sbert.py --dataset app_store  --steps 100 --encoding sbert  --query_strategy isls
python al_app_store_sbert.py --dataset app_store  --steps 100 --encoding sbert  --query_strategy isls_density

