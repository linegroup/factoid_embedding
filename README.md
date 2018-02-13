# Factoid Embedding


This is a simple implementation of Factoid Embedding.

## Example to run the codes
```
python main.py --path ../data_for_experiment/facebook_twitter/ \
                --source_prefix fb --target_prefix tw --source_col 0 --target_col 1  \
                --name_method tfidf --image_exist True --user_dim 1024+256 \
                --name_concatenate True \
                --image_dim 256 --n_iter 52 --supervised False\
                 --image_method vgg16 --skip_network False;
```

Or see example.sh

## Usage
```
usage: main.py [-h] [--path PATH] [--skip_network SKIP_NETWORK]
               [--source_prefix SOURCE_PREFIX] [--target_prefix TARGET_PREFIX]
               [--source_col SOURCE_COL] [--target_col TARGET_COL]
               [--name_dim NAME_DIM] [--name_concatenate NAME_CONCATENATE]
               [--name_preprocess NAME_PREPROCESS] [--name_method NAME_METHOD]
               [--screen_name_exist SCREEN_NAME_EXIST]
               [--image_exist IMAGE_EXIST] [--image_method IMAGE_METHOD]
               [--image_identical_threshold IMAGE_IDENTICAL_THRESHOLD]
               [--image_dim IMAGE_DIM]
               [--cosine_embedding_batch_size COSINE_EMBEDDING_BATCH_SIZE]
               [--cosine_embedding_learning_rate COSINE_EMBEDDING_LEARNING_RATE]
               [--supervised SUPERVISED] [--snapshot SNAPSHOT]
               [--snapshot_gap SNAPSHOT_GAP] [--n_iter N_ITER]
               [--warm_up_iter WARM_UP_ITER] [--user_dim USER_DIM]
               [--nce_sampling NCE_SAMPLING]
               [--triplet_embedding_batch_size TRIPLET_EMBEDDING_BATCH_SIZE]
               [--triplet_embedding_learning_rate_f TRIPLET_EMBEDDING_LEARNING_RATE_F]
               [--triplet_embedding_learning_rate_a TRIPLET_EMBEDDING_LEARNING_RATE_A]
               [--stratified_attribute STRATIFIED_ATTRIBUTE]
```

## Author
Feel free to contact XIE Wei weixie@smu.edu.sg



