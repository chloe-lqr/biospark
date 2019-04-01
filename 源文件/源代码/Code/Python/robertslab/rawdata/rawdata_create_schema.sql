CREATE DATABASE IF NOT EXISTS rawdata;

USE rawdata;

CREATE TABLE IF NOT EXISTS datasets (
ds_id           INT UNSIGNED NOT NULL,
ds_type         INT UNSIGNED NOT NULL,
deleted         BOOLEAN NOT NULL DEFAULT FALSE,
add_date        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
add_user        VARCHAR(255) NOT NULL,
update_date     DATETIME DEFAULT NULL,
update_user     VARCHAR(255) DEFAULT NULL,
delete_date     DATETIME DEFAULT NULL,
delete_user     VARCHAR(255) DEFAULT NULL,
                PRIMARY KEY (ds_id),
                KEY (ds_type)
) ENGINE=INNODB CHARSET=UTF8;

CREATE TABLE IF NOT EXISTS dataset_types (
ds_type         INT UNSIGNED NOT NULL,
description     VARCHAR(255) NOT NULL,
simulation      BOOLEAN NOT NULL DEFAULT FALSE,
experiment      BOOLEAN NOT NULL DEFAULT FALSE,
                PRIMARY KEY (ds_type),
                KEY (ds_type)
) ENGINE=INNODB CHARSET=UTF8;

TRUNCATE dataset_types;
INSERT INTO dataset_types(ds_type,description,simulation,experiment) VALUES (0,'Molecular dynamics simulation',TRUE,FALSE);
INSERT INTO dataset_types(ds_type,description,simulation,experiment) VALUES (10,'Lattice microbes simulation',TRUE,FALSE);
INSERT INTO dataset_types(ds_type,description,simulation,experiment) VALUES (100,'Microscopy time series',FALSE,TRUE);

CREATE TABLE IF NOT EXISTS dataset_properties (
ds_id           INT UNSIGNED NOT NULL,
name            VARCHAR(255) NOT NULL,
value           VARCHAR(255) NOT NULL,
                KEY (ds_id),
                KEY (name),
                KEY (value)
) ENGINE=INNODB CHARSET=UTF8;

CREATE TABLE IF NOT EXISTS dataset_files (
ds_id           INT UNSIGNED NOT NULL,
filename        VARCHAR(255) NOT NULL,
mimetype        VARCHAR(255) NOT NULL,
size            BIGINT UNSIGNED NOT NULL,
hashcode        VARCHAR(255) NOT NULL,
deleted         BOOLEAN NOT NULL DEFAULT FALSE,
add_date        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
add_user        VARCHAR(255) NOT NULL,
update_date     DATETIME DEFAULT NULL,
update_user     VARCHAR(255) DEFAULT NULL,
delete_date     DATETIME DEFAULT NULL,
delete_user     VARCHAR(255) DEFAULT NULL,
                PRIMARY KEY (ds_id, filename)
) ENGINE=INNODB CHARSET=UTF8;

GRANT SELECT ON rawdata.* TO rawdata IDENTIFIED BY 'rawdata';
