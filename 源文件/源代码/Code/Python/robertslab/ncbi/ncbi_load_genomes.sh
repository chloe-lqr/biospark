#~/bin/bash

NCBI_DIR=/home/databases/ncbi
MYSQL_USER=ncbi
MYSQL_PASSWORD=ncbi

# Fix the permissions.
echo "Setting permissions"
sudo chown -R erober32:bph-robertslab $NCBI_DIR
sudo chmod -R ug+w $NCBI_DIR

# Download any new databases.
echo "Downloading genomes"
wget -r -N -nH --cut-dirs=2 --directory-prefix=$NCBI_DIR/genomes/Archaea_Bacteria --accept ".faa,.frn,.ptt,.rnt,.rpt" ftp://ftp.ncbi.nlm.nih.gov/genomes/Bacteria/

# Build BLAST databases for any genomes that don't yet have them.
for f in $(find $NCBI_DIR/genomes/Archaea_Bacteria -name "*.faa"|sort)
do
    if [[ ! -e $f.phr || $f.phr -ot $f ]]; then
        echo "Building BLAST database for $f"
        makeblastdb -out $f -in $f -input_type fasta -dbtype prot -parse_seqids
    fi
done

# Download the summary file.
rm -f $NCBI_DIR/genomes/Archaea_Bacteria/summary.txt && wget --directory-prefix=$NCBI_DIR/genomes/Archaea_Bacteria ftp://ftp.ncbi.nlm.nih.gov/genomes/Bacteria/summary.txt

# Load the summary file into the database.
sed "s/NCBI_DIR/${NCBI_DIR//\//\\/}/g" ncbi_load_genomes_bact_arch.sql | mysql -p

# Update the database with the actual filename for each genome.
#for f in $(find $NCBI_DIR/genomes/Archaea_Bacteria -name "NC_*.faa"|sort)
#do
#    GENOMEID=$(echo $f | sed 's/.*uid\([0-9]\+\).*/\1/')
#    ACCESSION=$(echo $f | sed 's/.*\(NC_[0-9]\+\).faa/\1/')
#    echo "Updating record for $f: $GENOMEID $ACCESSION"
#    echo "USE ncbi; UPDATE genomes_arch_bact SET filename='$f' WHERE genomeid='$GENOMEID' AND accession LIKE '$ACCESSION%';" | mysql -u $MYSQL_USER -p$MYSQL_PASSWORD
#done

# Fix the permissions.
echo "Setting permissions"
sudo chown -R erober32:bph-robertslab $NCBI_DIR
sudo chmod -R ug+w $NCBI_DIR

