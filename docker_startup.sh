sudo docker run --runtime=nvidia -it --rm tensorflow/tensorflow:latest-gpu-py3 \
   python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"

sudo docker run --runtime=nvidia -it --rm tensorflow/tensorflow:latest-gpu-py3 bash