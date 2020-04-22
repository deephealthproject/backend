export CUR_PATH=`pwd`

# PyEDDL
git clone --recurse-submodules https://github.com/deephealthproject/pyeddl.git
cd pyeddl
export PYEDDL_ROOT=`pwd`
cd third_party/eddl
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=install -DBUILD_SHARED_LIB=ON -DBUILD_PROTOBUF=ON -DBUILD_TARGET=GPU -DBUILD_TESTS=OFF ..
cmake --build . --config Release
cmake --build . --target install
export EDDL_WITH_CUDA="true"
export EDDL_DIR=`pwd`/install
cd $PYEDDL_ROOT
python3 setup.py install

# PyECVL
cd CUR_PATH
git clone --recurse-submodules https://github.com/deephealthproject/pyecvl.git
cd pyecvl
export PYECVL_ROOT=`pwd`
cd third_party/ecvl
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=install -DECVL_WITH_DICOM=ON -DECVL_WITH_OPENSLIDE=ON -DECVL_DATASET_PARSER=ON -DECVL_BUILD_EDDL=ON -Deddl_dir=$EDDL_DIR/cmake ..
cmake --build . --config Release
cmake --build . --target install
export ECVL_DIR=`pwd`/install
cd $PYEDDL_ROOT
python3 setup.py install