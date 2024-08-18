export PYTHONPATH="$PWD"


# (1) train the source model and generate pseudo labels
# tgt_language; dataset; epochs; max_span_len; batch_size; gradient_acc_steps ######
bash generate_pseudo_labels.sh demoLang conll03 8


# (2) train the target model with pseudo labels
# tgt_language; dataset; epochs; pseudo_labels_id; max_span_len; batch_size; gradient_acc_steps
bash train_tgt_model.sh demoLang conll03 5 7
