#!/usr/bin/env bash
set -e

# Initialize argument variables with default values:
PIP_CMD=sharepip
INSTALL_ROOT=/share/apps
HADOOP_ROOT=/usr/local/lib/hadoop
JAVA_LIB_DIR=/usr/lib/jvm/current/jre/lib/amd64/server

# Make any necessary directories.
sudo mkdir -p "${INSTALL_ROOT}"/lib/biospark/java
sudo mkdir -p "${INSTALL_ROOT}"/lib/biospark/pyspark
sudo mkdir -p Code/SFile/build


# Copy the bash scripts to the bin directory.
sudo rsync -av `find Scripts/bash -maxdepth 1 -type f` "${INSTALL_ROOT}"/bin
sudo rsync -av `find Scripts/bash/xanthus -maxdepth 1 -type f` "${INSTALL_ROOT}"/bin

# Install the Biospark Hadoop libraries.
sudo rsync -av Code/Hadoop/lib/*.jar "${INSTALL_ROOT}"/lib/biospark/java

# Install the Biospark Protobuf libraries.
cd Code/Protobuf && "${PIP_CMD}" install -U . --install-option="--protoc=/share/apps/bin/protoc" && cd ../..

# Install the Biospark Python packages.
cd Code/Python && "${PIP_CMD}" install -U . --install-option="--deployment=xanthus" && sudo ./setupTabCompletion.py --system-wide && cd ../..

# Install the SFile package.
cd Code/SFile/build && sudo cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_ROOT -DHADOOP_ROOT=$HADOOP_ROOT -DJAVA_LIB_DIR=$JAVA_LIB_DIR .. && sudo make && sudo make install && cd ../../..

# Copy the python scripts to the bin directory.
sudo rsync -av `find Scripts/python -maxdepth 1 -type f` "${INSTALL_ROOT}"/bin

# Copy the pyspark scripts to the lib directory.
sudo rsync -av `find Scripts/pyspark -maxdepth 1 -type f` "${INSTALL_ROOT}"/lib/biospark/pyspark

# Make sure the whole tree is group writable.
sudo chmod -R g+w "${INSTALL_ROOT}"

