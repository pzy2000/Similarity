#!/usr/bin/env bash

BERT_BASE_DIR=../albert_tiny_489k

python3 ./bert_sim/create_pretraining_data.py --do_whole_word_mask=True --input_file=../data/pretrain.txt \
--output_file=../data/pretrain.tfrecord --vocab_file=$BERT_BASE_DIR/vocab.txt --do_lower_case=True \
--max_seq_length=512 --max_predictions_per_seq=51 --masked_lm_prob=0.10

nohup python3 ./bert_sim/run_pretraining.py --input_file=../data/pretrain.tfrecord  \
--output_dir=../albert_tiny_489k --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
--train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=51 \
--num_train_steps=125000 --num_warmup_steps=12500 --learning_rate=0.00176    \
--save_checkpoints_steps=2000  --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt &