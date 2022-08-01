#!/bin/bash

ENCODER_SIZE=$1
ENCODER=google/electra-${ENCODER_SIZE}-discriminator
DATA=./resource/dataset/pen.json
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for SUBSET_TYPE in alg514-fold0 mawps-fold0 draw pen
do
	EXPNAME=${SUBSET_TYPE}_${ENCODER_SIZE}_$2
	EXPERIMENT=./resource/experiments/${SUBSET_TYPE}
	EPOCH=500
	WARMUP=10

	echo -e "\033[33mExp name\033[0m: $EXPNAME"
	echo -e "\033[33mData set\033[0m: $DATA"
	echo -e "\033[33mEncoder \033[0m: $ENCODER"
	echo -e "\033[33mEpoch   \033[0m: $EPOCH"
	echo -e "\033[33mLearner \033[0m: COUNT=1; GPU=1"

	killall -9 -r 'ray::'
	python train_model.py -name $EXPNAME\
	  -data $DATA -exp $EXPERIMENT -model SWAN_P1 -enc $ENCODER \
	  -cpu 1 -gpu 1 -iter $EPOCH -bsz 16 -lr 0.000625 0.00088 0.00125 0.00176 -warmup $WARMUP -win 3
#	python train_model.py -name $EXPNAME\
#	  -data $DATA -exp $EXPERIMENT -model EPT EPTX SWAN_A SWAN_B SWAN_P1 -enc $ENCODER \
#	  -cpu 1 -gpu 1 -iter $EPOCH -bsz 16 -lr 0.000625 0.00088 0.00125 0.00176 -warmup $WARMUP -win 3

	RECENT=$(ls ./runs/pen_${EXPNAME}_* -1dt | head -n 1)
	if [[ "${SUBSET_TYPE}" == *-fold0 ]]
	then
		# Run fold training
		SUBSETS=$(echo $SUBSET_TYPE | cut -d- -f1)
		EXPNAME=${SUBSETS}-folds_${ENCODER_SIZE}_$2
		EXPERIMENT=./resource/experiments/${SUBSETS}
		killall -9 -r 'ray::'
		python train_fold.py -name $EXPNAME -exp ${EXPERIMENT}-fold* -model ${RECENT}/best_*/config.pkl

		# Rename fold files
		RECENT=$(ls ./runs/pen_${EXPNAME}_* -1dt | head -n 1)
		for DIR in `ls -1d ${RECENT}/*`
		do
			if [[ -f $DIR ]]; then continue; fi
			if [[ ! -f $DIR/config.pkl ]]
			then
				mv ${DIR}/checkpoint_*/*.pt ${DIR}/
				cp ${DIR}/params.pkl ${DIR}/config.pkl
			fi

			BASENAME=`basename $DIR`
			if [[ "$BASENAME" == *"fold"* ]]
			then
				echo "$DIR EXISTS"
			else
				NEW_NAME=$BASENAME
				NEW_NAME=$(echo $NEW_NAME | cut -d- -f1)-fold$(($(echo $NEW_NAME | cut -d- -f2 | cut -d_ -f2 | sed 's/^0\+//;s/^$/0/') % 5))
				mv ${DIR} ${RECENT}/${NEW_NAME}
			fi
		done

		# Run fold evaluation
		for i in 0 1 2 3 4
		do
			echo RUN FOR fold$i
			python test_model.py -data $DATA -exp ${EXPERIMENT}-fold${i} -model ${RECENT}/*-fold${i} -ntr 500 -smp 50 -cpu 1 -gpu 0.5
		done
	else
		# Run evaluation
		python test_model.py -data $DATA -exp $EXPERIMENT -model ${RECENT}/best_* -ntr 500 -smp 50 -cpu 1 -gpu 0.5
	fi
done
