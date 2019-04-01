__all__ = ["dataset"]

# Initialize the database connection
try:
    import MySQLdb as mysql
except ImportError:
    import pymysql as mysql
db = mysql.connect(host="xanthus.bph.jhu.edu", user="rawdata", passwd="rawdata", db="rawdata")

