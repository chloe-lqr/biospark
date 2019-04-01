__all__ = ["genome","taxonomy"]

# Initialize the database connection
try:
    import MySQLdb as mysql
except ImportError:
    import pymysql as mysql
db = mysql.connect(host="xanthus.bph.jhu.edu", user="ncbi", passwd="ncbi", db="ncbi")

