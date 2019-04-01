spark_submit_path="spark-submit-robertslab"
test_data_dir="../../testData"

hdfs_data_path="/user/${USER}/test/sfile/md"
hdfs_tmp_leaf=$(mktemp -u "XXXXXX")
hdfs_tmp_dir="/tmp/${hdfs_tmp_leaf}"
