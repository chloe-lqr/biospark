#!/bin/bash

NCBI_DIR=/home/databases/ncbi
MYSQL_USER=ncbi
MYSQL_PASSWORD=ncbi

mkdir -p tmp
cd tmp
wget ftp://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz
tar zxvf taxdump.tar.gz
wget ftp://ftp.ncbi.nih.gov/pub/taxonomy/gi_taxid_prot.dmp.gz
gunzip gi_taxid_prot.dmp.gz
EPWD="${PWD//\//\\/}"
sed "s/DIR/$EPWD/g" ../ncbi_load_taxonomy.sql | mysql -p

# Update the naming for domain usage.
echo "USE ncbi; UPDATE tax_nodes SET rank='domain' WHERE parent_taxid='131567' and rank='superkingdom';" | mysql -u $MYSQL_USER -p$MYSQL_PASSWORD

cd ..
rm -rf tmp

