#!/bin/bash

export VACCEL_DEBUG_LEVEL=4
export VACCEL_BACKENDS=/opt/vaccel/lib/libvaccel-vsock.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/vaccel/lib
export LD_PRELOAD=/opt/vaccel/lib/libvaccel-tf-bindings.so

/usr/bin/function
