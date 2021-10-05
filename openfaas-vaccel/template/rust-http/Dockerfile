# Fetch the OpenFaaS watchdog
FROM openfaas/of-watchdog:0.8.2 as watchdog

# Image too build Rust code
FROM rust:1.51 as builder

WORKDIR /home/rust

# Copy all the sources
COPY function ./function

ARG HANDLER_PACKAGE

# Build Rust handler
RUN cd function && cargo build --release --bin $HANDLER_PACKAGE

# Use TensorFlow-enabled vAccel image as runner
FROM nubificus/vaccel-tensorflow:latest as runner

ARG HANDLER_PACKAGE

# Install packages and add non-root user
RUN apt-get update && apt-get install -y curl ca-certificates \
    && addgroup --system app && adduser --system app

ENV USER=app

# Copy of-watchdog binary
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

COPY --from=builder /home/rust/function/target/release/$HANDLER_PACKAGE /usr/bin/function
RUN chmod +x /usr/bin/function

# Fetch TensorFlow models
RUN mkdir /opt/resnet50 && \
    wget -q https://tfhub.dev/tensorflow/resnet_50/classification/1?tf-hub-format=compressed -O resnet50.tar && \
    tar xf resnet50.tar -C /opt/resnet50 && \
    wget -q https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt -O /opt/resnet50/ImageNetLabels.txt

COPY launch_vaccel.sh /home/app/
RUN chmod u+x /home/app/launch_vaccel.sh
    
# Set up watchdog for HTTP mode
ENV function_process="/home/app/launch_vaccel.sh"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:3000"

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
