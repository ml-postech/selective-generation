#!/bin/bash
TAG=NIPSL24
NEXP=1

EPS=0.25
DELTA=0.02

DELTA_P=1e-5

ZU=10000
EXPS=(SSL SL)

FER=true

K=5

DATASETNAMELIST=(nli)
DATASETCONFLIST=(nq_alpaca7B)
# MDLNAMELIST=(gpt2 alpaca7B gpt2-medium gpt2-large llama7B llama13B)
# MDLPATHLIST=(gpt2 ~/data/alpaca/7B gpt2-medium gpt2-large ~/data/llama/7B_hf ~/data/llama/13B_hf)
# MDLNAMELIST=(gpt2)
# MDLPATHLIST=(gpt2)
MDLNAMELIST=(alpaca7B)
MDLPATHLIST=(~/data/models/alpaca/7B)

EMDLNAME=(deberta-v2-xxlarge-mnli)
EMDLPATH=(microsoft/deberta-v2-xxlarge-mnli)

# METHODLIST=(GreedyGen-PACPrec-EM GreedyGen-PACPrec-INC GreedyGen-EPACPrec GreedyGen-NCGPrec)
# METHODLIST=(GreedyGen-NCGPrec GreedyGen-NCGPrec2 GreedyGen-EPACPrec)
METHODLIST=(GreedyGen-SG)
python data/nli/preprocess_u.py alpaca7B nq
python data/nli/preprocess_e.py alpaca7B nq

for i_exp in ${!EXPS[@]};
do
EXP=${EXPS[$i_exp]}
    for i_dataset in ${!DATASETNAMELIST[@]};
    do
	DATASETNAME=${DATASETNAMELIST[$i_dataset]}
	DATASETCONF=${DATASETCONFLIST[$i_dataset]}
	
	for i_model in ${!MDLNAMELIST[@]};
	do
	    MDLNAME=${MDLNAMELIST[$i_model]}
	    MDLPATH=${MDLPATHLIST[$i_model]}
	    for METHOD in ${METHODLIST[*]};
	    do
		echo $TAG, $DATASETNAME, $DATASETCONF, $MDLNAME, $MDLPATH, $METHOD

		python3 main.py \
			--tag ${TAG} \
			--cache_cal_fn "CAL"-${TAG}-${DATASETNAME}-${DATASETCONF}-${MDLNAME}-${METHOD}-${NEXP} \
			--cache_eval_fn "EVAL"-${TAG}-${DATASETNAME}-${DATASETCONF}-${MDLNAME}-${METHOD}-${NEXP} \
			--model_name_or_path ${MDLPATH} \
			--dataset_name data/${DATASETNAME} \
			--dataset_config_name ${DATASETCONF} \
			--preprocessing_num_workers 64 \
			--dataloader_num_workers 64 \
			--per_device_train_batch_size 32 \
			--per_device_eval_batch_size 32 \
			--fp16 \
			--exp_name ${TAG}-${DATASETNAME}-${DATASETCONF}-${MDLNAME}-${METHOD}-EXP-${i_exp} \
			--output_dir snapshots/${TAG}-${DATASETNAME}-${DATASETCONF}-${MDLNAME}-${METHOD}-EXP-${i_exp} \
			--method $METHOD \
			--eps ${EPS} \
			--entail_model_name_or_path ${EMDLPATH} \
			--cache_ent_fn "ENT"-${TAG}-${DATASETNAME}-${DATASETCONF}-${EMDLNAME}-${METHOD}-${NEXP} \
			--cache_ent_eval_fn "ENTEVAL"-${TAG}-${DATASETNAME}-${DATASETCONF}-${EMDLNAME}-${METHOD}-${NEXP} \
			--delta ${DELTA} \
			--delta_p ${DELTA_P} \
			--z_u ${ZU} \
			--fer ${FER} \
			--K ${K} \
			--exp_method ${EXP}
#			--n_cal $N
	    done
	done
    done
done

