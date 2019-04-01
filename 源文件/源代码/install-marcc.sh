#!/bin/bash
set -e

# Initialize argument variables with default values:
PIP_CMD=sharepip
INSTALL_ROOT=/home-2/erober32@jhu.edu/share/apps
HADOOP_ROOT=/home-2/erober32@jhu.edu/share/apps/lib/hadoop
JAVA_LIB_DIR=/cm/shared/apps/java/JDK_1.8.0_45/jre/lib/amd64/server

# Make any necessary directories.
mkdir -p $INSTALL_ROOT/lib/biospark/java
mkdir -p $INSTALL_ROOT/lib/biospark/pyspark
mkdir -p Code/SFile/build


# Copy the bash scripts to the bin directory.
rsync -av `find Scripts/bash -maxdepth 1 -type f` $INSTALL_ROOT/bin
rsync -av `find Scripts/bash/marcc -maxdepth 1 -type f` $INSTALL_ROOT/bin

# Install the Biospark Hadoop libraries.
rsync -av Code/Hadoop/lib/*.jar $INSTALL_ROOT/lib/biospark/java

# Install the Biospark Protobuf libraries.
cd Code/Protobuf && "${PIP_CMD}" install -U . --install-option="--protoc=/home-2/erober32@jhu.edu/usr/bin/protoc" && cd ../..

# install the Biospark and Python packages
cd Code/Python && "${PIP_CMD}" install -U . --install-option="--deployment=marcc" && cd ../..

# Install the SFile package.
cd Code/SFile/build && cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_ROOT -DHADOOP_ROOT=$HADOOP_ROOT -DJAVA_LIB_DIR=$JAVA_LIB_DIR .. && make && make install && cd ../../..

# Copy the python scripts to the bin directory.
rsync -av `find Scripts/python -maxdepth 1 -type f` $INSTALL_ROOT/bin

# Copy the pyspark scripts to the lib directory.
rsync -av `find Scripts/pyspark -maxdepth 1 -type f` $INSTALL_ROOT/lib/biospark/pyspark

# Make sure the whole tree is group writable.
chmod -R g+w $INSTALL_ROOT

