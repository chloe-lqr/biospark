#!/bin/bash

protoc --proto_path=src/protobuf --python_out=lib/robertslab-protobuf-python.zip `find src/protobuf -name "*".proto`

rm -rf tmp
mkdir tmp
for dirname in `find src/protobuf/* -type d`; do
    mkdir -p tmp/${dirname}
    touch tmp/${dirname}/__init__.py
done
echo -e "import pkgutil\n__path__ = pkgutil.extend_path(__path__, __name__)\nprint __path__\nprint __name__" > tmp/src/protobuf/robertslab/__init__.py
cd tmp/src/protobuf
zip ../../../lib/robertslab-protobuf-python.zip `find . -name __init__.py`
cd ../../..
rm -rf tmp
