#!/bin/sh
# onnx2c packs scratch tensors into C unions; its lifetime analysis clobbers
# live data on this graph. Give every member its own storage instead.
sed -i 's/^union tensor_union_/struct tensor_union_/; s/^static union tensor_union_/static struct tensor_union_/' "$1"
