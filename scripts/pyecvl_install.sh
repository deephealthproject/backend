#!/bin/bash

# Set PyECVL default version
PYECVL_TAG="${1:-0.7.0}"

# PyECVL cloning
echo "Cloning PyECVL"
git clone --recurse-submodules --jobs 2 \
  https://github.com/deephealthproject/pyecvl.git pyecvl_${PYECVL_TAG}
cd pyecvl_${PYECVL_TAG}

git checkout tags/"${PYECVL_TAG}"
PYECVL_ROOT=$(pwd)

# eddl and PyEDDL
echo "Installing PyEDDL"
cd third_party/pyeddl
#git fetch && git checkout ff6b2c123a99734c084038ed465bea7065d70109
PYEDDL_ROOT=$(pwd)
cd third_party/eddl
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=install -DBUILD_SHARED_LIBS=ON -DBUILD_PROTOBUF=ON \
  -DBUILD_TARGET=GPU -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_HPC=OFF ..
cmake --build . --config Release --parallel "$(nproc)"
cmake --build . --target install
export EDDL_WITH_CUDA="true"
export EDDL_DIR="$(pwd)/install"
echo "eddl dir: $EDDL_DIR"
cd "$PYEDDL_ROOT"
python3 setup.py install

# ecvl and PyECVL
echo "Installing PyECVL"
cd "$PYECVL_ROOT"/third_party/ecvl
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=install -DECVL_WITH_DICOM=ON \
  -DECVL_WITH_OPENSLIDE=ON -DECVL_DATASET=ON -DECVL_BUILD_EDDL=ON \
  -DECVL_TESTS=OFF -Deddl_DIR="$EDDL_DIR/lib/cmake/eddl" ..
cmake --build . --config Release --parallel "$(nproc)"
cmake --build . --target install
export ECVL_DIR=$(pwd)/install
cd $PYECVL_ROOT
python3 setup.py install
