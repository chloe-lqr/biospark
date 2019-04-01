#!/usr/bin/env bash
source ./test_paths.sh

selection="name_CA"
rmsd_script_path="../../../src/rmsd/spark/rmsdSpark.py"
sfmd_file="3d6z_aligned_vmd_wat_ion__CA_only.sfmd"
top_file="3d6z_aligned_vmd_wat_ion.pdb"

top_path="${test_data_dir}/${top_file}"
sfmd_path="${test_data_dir}/${sfmd_file}"

# if the first argument to this test script is set (doesn't matter to what), copy the .sfmd file over to a temporary dir on the hdfs and use that
if [ -n "$1" ]; then
    sfmd_hdfs_path="${hdfs_tmp_dir}/${sfmd_file}"
    hdfs dfs -mkdir -p ${hdfs_tmp_dir}
    hdfs dfs -put ${sfmd_path} ${sfmd_hdfs_path}
else
    sfmd_hdfs_path="${hdfs_data_path}/${sfmd_file}"
fi

${spark_submit_path} ${rmsd_script_path} ${sfmd_hdfs_path} ${top_path} "${selection}" --presliced

if [ -n "$1" ]; then
    hdfs dfs -rm ${sfmd_hdfs_path}
    hdfs dfs -rmdir ${hdfs_tmp_dir}
fi
