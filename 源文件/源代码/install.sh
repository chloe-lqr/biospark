#!/bin/bash
set -e

INSTALL_ROOT=/usr/local

# Make any necessary directories.
mkdir -p $INSTALL_ROOT/lib/biospark/java
mkdir -p $INSTALL_ROOT/lib/biospark/pyspark
mkdir -p Code/SFile/build

# Copy the bash scripts to the bin directory.
rsync -av `find Scripts/bash -maxdepth 1 -type f` $INSTALL_ROOT/bin

# Install the Biospark Hadoop libraries.
rsync -av Code/Hadoop/lib/*.jar $INSTALL_ROOT/lib/biospark/java

# Install the Biospark Protobuf libraries.
cd Code/Protobuf && pip install -U . && cd ../..

# Install the Biospark Avro libraries.
rsync -av Code/Avro/lib/*.zip $INSTALL_ROOT/lib/biospark

# Install the Biospark Python packages.
cd Code/Python && pip install -U . && cd ../..

# Install the SFile package.
cd Code/SFile/build && cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_ROOT .. && make && make install && cd ../../..

# Copy the python scripts to the bin directory.
rsync -av `find Scripts/python -maxdepth 1 -type f` $INSTALL_ROOT/bin

# Copy the pyspark scripts to the lib directory.
rsync -av `find Scripts/pyspark -maxdepth 1 -type f` $INSTALL_ROOT/lib/biospark/pyspark
