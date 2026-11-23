DIR=$(pwd)

PYBIND_DIR=$(python -m pybind11 --cmakedir 2>/dev/null || echo "")
cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) ${PYBIND_DIR:+-Dpybind11_DIR=$PYBIND_DIR} && make -j$(nproc)
cd /kaolin && rm -rf build *egg* && pip install -e .
# Optional: BundleSDF CUDA ops (required for model-free NeRF path only)
# cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd "${DIR}"
