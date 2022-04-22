run () {
    docker run \
        -it \
        --rm \
        --init \
        --volume="$PWD:/workspace" \
        -w "/workspace" \
        -p "8889:8888" \
        -p "6006:6006" \
        tensorflow/tensorflow \
        "$@"
}

run tensorboard \
    --logdir=runs \
    --host=0.0.0.0
