CREATE DATABASE IF NOT EXISTS ncbi;

USE ncbi;

CREATE TABLE IF NOT EXISTS tax_gi_to_taxid (
gi          INT UNSIGNED NOT NULL,
taxid       INT UNSIGNED NOT NULL,
            PRIMARY KEY (gi),
            KEY (taxid)
) ENGINE=INNODB CHARSET=UTF8;

CREATE TABLE IF NOT EXISTS tax_names (
taxid       INT UNSIGNED NOT NULL, 
name        VARCHAR(255) NOT NULL, 
uniquename  VARCHAR(255) NOT NULL DEFAULT '',
class       VARCHAR(20) NOT NULL DEFAULT '',
            KEY (taxid),
            KEY (name)
) ENGINE=INNODB CHARSET=UTF8;

CREATE TABLE IF NOT EXISTS tax_nodes (
taxid               INT UNSIGNED NOT NULL,
parent_taxid        INT UNSIGNED NOT NULL,
rank                VARCHAR(20) NOT NULL DEFAULT '',
embl_code           VARCHAR(20) NOT NULL DEFAULT '',
divid               TINYINT UNSIGNED NOT NULL,
inherit_divid       TINYINT UNSIGNED NOT NULL,
gcid                TINYINT UNSIGNED NOT NULL,
inherit_gcid        TINYINT UNSIGNED NOT NULL,
mit_gcid            TINYINT UNSIGNED NOT NULL,
inherit_mit_gcid    TINYINT UNSIGNED NOT NULL,
hidden_node         TINYINT unsigned NOT NULL,
hidden_subtree      TINYINT unsigned NOT NULL,
comments            VARCHAR(255) NOT NULL DEFAULT '',
                    PRIMARY KEY (taxid),
                    KEY (parent_taxid),
                    KEY (rank)
) ENGINE=INNODB DEFAULT CHARSET=UTF8;

CREATE TABLE IF NOT EXISTS genomes_arch_bact (
accession           CHAR(11) NOT NULL, 
replicon            VARCHAR(40) NOT NULL DEFAULT '',
genomeid            INT UNSIGNED NOT NULL, 
taxid               INT UNSIGNED NOT NULL, 
len                 INT UNSIGNED NOT NULL,
name                VARCHAR(255) NOT NULL DEFAULT '',
genbank_accession   VARCHAR(20) NOT NULL DEFAULT '',
create_date         VARCHAR(30) NOT NULL DEFAULT '',
update_date         VARCHAR(30) NOT NULL DEFAULT '',
filename            VARCHAR(1024) NOT NULL DEFAULT '',
                    PRIMARY KEY (accession),
                    KEY (genomeid),
                    KEY (taxid)
) ENGINE=INNODB CHARSET=UTF8;

GRANT SELECT,INSERT,UPDATE,DELETE ON ncbi.* TO ncbi;
