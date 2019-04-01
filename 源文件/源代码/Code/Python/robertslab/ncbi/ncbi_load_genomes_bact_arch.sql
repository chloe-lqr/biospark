USE ncbi;

TRUNCATE genomes_arch_bact; 

LOAD DATA INFILE 'NCBI_DIR/genomes/Archaea_Bacteria/summary.txt' IGNORE INTO TABLE genomes_arch_bact FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n' IGNORE 1 LINES (accession,genbank_accession,len,taxid,genomeid,name,replicon,create_date,update_date);

