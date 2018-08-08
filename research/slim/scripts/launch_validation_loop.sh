#!/bin/bash

TF_DIR=$(dirname "$0")/../

#!/bin/bash

usage()
{
    echo "usage: launch_validation_loop 
            [--checkpoint_path] 
            [--dataset_dir]
            [--dataset_name]
            [--dataset_split_name]
            [--model_name]
            [--eval_dir]
            [--gpu_percentage]"
}


for i in "$@"
do
case $i in
    --checkpoint_path=*)
    CHECKPOINT_PATH="${i#*=}"
    shift # past argument=value
    ;;
    --dataset_dir=*)
    DATASET_DIR="${i#*=}"
    shift # past argument=value
    ;;
    --dataset_name=*)
    DATASET_NAME="${i#*=}"
    shift # past argument=value
    ;;
    --dataset_split_name=*)
    DATASET_SPLIT_NAME="${i#*=}"
    shift # past argument=value
    ;;
    --model_name=*)
    MODEL_NAME="${i#*=}"
    shift # past argument=value
    ;;
    --eval_dir=*)
    EVAL_DIR="${i#*=}"
    shift # past argument=value
    ;;
    --gpu_percentage=*)
    GPU_PERCENTAGE="${i#*=}"
    shift # past argument=value
    ;;
    -h|--help)
    usage
    exit
    ;;
    *)
    usage      # unknown option
    exit
    ;;
esac
done


BATCH=1     # It's not really worth to use larger batches : much more memory, not much faster time
REPLICAS=10 # Uses a smaller number of replica to validate (hopefully it doesn't change between models)

CHECKPOINT_REGEXP='\.ckpt-[0-9]*\.index'

function finish {
    # Delete temporary files
    rm "$TEMP_DIR/*.tmp" 2> /dev/null
    rmdir "$TEMP_DIR"
}
trap finish EXIT
TEMP_DIR=$(mktemp -d -t isbi.XXXXXXXXXX)

PREVIOUS_LIST="$TEMP_DIR/previouslist.tmp"
CURRENT_LIST="$TEMP_DIR/currentlist.tmp"
PREDICTIONS="$TEMP_DIR/predictions.tmp"
rm "$PREVIOUS_LIST" "$CURRENT_LIST" "$PREDICTIONS" 2> /dev/null


while :
do
    ls -1 "$CHECKPOINT_PATH" | grep "$CHECKPOINT_REGEXP" | sort > "$CURRENT_LIST"
    CHANGED=$(diff "$PREVIOUS_LIST" "$CURRENT_LIST")
    if [ "$CHANGED" != "" ]; then
        echo What changed: "$CHANGED" ....
        NEW_CHECKPOINT="$(ls -1 --sort=time "$CHECKPOINT_PATH" | grep "$CHECKPOINT_REGEXP" | head -n 1)"
        echo Found new checkpoint: "$NEW_CHECKPOINT" --- launching validation...
        rm "$NEW_SCORES" 2> /dev/null

        python "$TF_DIR"/eval_image_classifier.py \
            --alsologtostderr \
            --checkpoint_path="$CHECKPOINT_PATH" \
            --dataset_dir="$DATASET_DIR" \
            --dataset_name="$DATASET_NAME" \
            --dataset_split_name="$DATASET_SPLIT_NAME" \
            --model_name="$MODEL_NAME" \
            --eval_dir="$EVAL_DIR" \
            --gpu_percentage="$GPU_PERCENTAGE"
    fi
    cp "$CURRENT_LIST" "$PREVIOUS_LIST"
    sleep 60
done
