#!/bin/bash
export CUR_PATH=`pwd`

# PyECVL clone
git clone --recurse-submodules https://github.com/deephealthproject/pyecvl.git
cd pyecvl
export PYECVL_ROOT=`pwd`

# eddl and PyEDDL
cd third_party/pyeddl
export PYEDDL_ROOT=`pwd`
cd third_party/eddl
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=install -DBUILD_SHARED_LIB=ON -DBUILD_PROTOBUF=ON -DBUILD_TARGET=GPU -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF ..
cmake --build . --config Release --parallel $(nproc)
cmake --build . --target install
export EDDL_WITH_CUDA="true"
export EDDL_BUILD_DIR=`pwd`
export EDDL_DIR=`pwd`/install
echo "eddl dir: $EDDL_DIR"
cd $PYEDDL_ROOT
python3 setup.py install

# ecvl and PyECVL
cd $PYECVL_ROOT
cd third_party/ecvl
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=install -DECVL_WITH_DICOM=ON -DECVL_WITH_OPENSLIDE=ON -DECVL_DATASET_PARSER=ON -DECVL_BUILD_EDDL=ON -Deddl_DIR=$EDDL_BUILD_DIR/cmake ..
cmake --build . --config Release --parallel $(nproc)
cmake --build . --target install
export ECVL_DIR=`pwd`/install
cd $PYECVL_ROOT
python3 setup.py install