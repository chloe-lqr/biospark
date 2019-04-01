USE ncbi;

TRUNCATE tax_gi_to_taxid; 
TRUNCATE tax_names; 
TRUNCATE tax_nodes;

LOAD DATA INFILE 'DIR/gi_taxid_prot.dmp' INTO TABLE tax_gi_to_taxid FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n' (gi,taxid);
LOAD DATA INFILE 'DIR/names.dmp' INTO TABLE tax_names FIELDS TERMINATED BY '\t|\t' LINES TERMINATED BY '\t|\n' (taxid,name,uniquename,class);
LOAD DATA INFILE 'DIR/nodes.dmp' INTO TABLE tax_nodes FIELDS TERMINATED BY '\t|\t' LINES TERMINATED BY '\t|\n' (taxid,parent_taxid,rank,embl_code,divid,inherit_divid,gcid,inherit_gcid,mit_gcid,inherit_mit_gcid,hidden_node,hidden_subtree,comments);

