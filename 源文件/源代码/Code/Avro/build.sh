#!/bin/bash

rm lib/robertslab-avro-python.zip

cd src/avro
zip ../../lib/robertslab-avro-python.zip `find . -name *.avsc`
cd ../..


rm -rf tmp
mkdir -p tmp/robertslab/avro
echo -e "import pkgutil\n__path__ = pkgutil.extend_path(__path__, __name__)\nprint __path__\nprint __name__" > tmp/robertslab/__init__.py
touch tmp/robertslab/avro/__init__.py
cd tmp
zip ../lib/robertslab-avro-python.zip `find . -name __init__.py`
cd ..
rm -rf tmp
