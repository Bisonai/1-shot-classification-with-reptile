python -u run_omniglot_bisonai.py \
       --shots 1 \
       --inner-batch 10 \
       --inner-iters 5 \
       --meta-step 1 \
       --meta-batch 5 \
       --meta-iters 100000 \
       --eval-batch 5 \
       --eval-iters 50 \
       --learning-rate 0.001 \
       --meta-step-final 0 \
       --train-shots 10 \
       --checkpoint 1shot_3way_bisonai_ckpt_o15t \
       --transductive \
       --classes 3