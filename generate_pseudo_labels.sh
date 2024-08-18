export PYTHONPATH="$PWD"

LANG=${1:-demoLang}
dataname=${2:-conll03}
max_epochs=${3:-10} 

DATA_DIR="data/${dataname}_${LANG}"
PRETRAINED="../PLMs/xlm-roberta-large"
BERT_DIR=${PRETRAINED}

if [[ "$dataname" == "wikiann" ]]; then
  n_class=4
else
  n_class=5
fi
BERT_DROPOUT=0.2
MODEL_DROPOUT=0.2
LR=1e-5
MAXLEN=128
MAXNORM=1.0
max_spanLen=${4:-4} 
batchSize=${5:-8}
grad_acc=${6:-4}
tokenLen_emb_dim=50
spanLen_emb_dim=100
morph_emb_dim=100


use_prune=True
use_spanLen=True
use_morph=True
use_span_weight=True
neg_span_weight=1.0
gpus="0,"



modelName="genpseudo_${LANG}"
idtest=${dataname}_${modelName}
param_name=epoch${max_epochs}_batchsize${batchSize}_lr${LR}_maxlen${MAXLEN}

OUTPUT_DIR="train_logs/${dataname}/${modelName}_new"

mkdir "train_logs"


python generate_pseudo_labels.py \
--dataname $dataname \
--data_dir $DATA_DIR \
--mask_rate 0.25 \
--warmup_steps 50 \
--bert_config_dir $BERT_DIR \
--bert_max_length $MAXLEN \
--batch_size $batchSize \
--gpus=$gpus \
--precision=16 \
--lr $LR \
--val_check_interval 1.0 \
--accumulate_grad_batches $grad_acc \
--default_root_dir $OUTPUT_DIR \
--model_dropout $MODEL_DROPOUT \
--bert_dropout $BERT_DROPOUT \
--max_epochs $max_epochs \
--n_class $n_class \
--max_spanLen $max_spanLen \
--tokenLen_emb_dim $tokenLen_emb_dim \
--modelName $modelName \
--spanLen_emb_dim $spanLen_emb_dim \
--morph_emb_dim $morph_emb_dim \
--use_prune $use_prune \
--use_spanLen $use_spanLen \
--use_morph $use_morph \
--use_span_weight $use_span_weight \
--neg_span_weight $neg_span_weight \
--param_name $param_name \
--gradient_clip_val $MAXNORM \
--optimizer "adamw" | tee train_logs/${dataname}_${modelName}-run_log.log