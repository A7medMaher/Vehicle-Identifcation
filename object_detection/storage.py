import configparser
import mysql.connector
import time



def connect():
    config = configparser.ConfigParser()
    config.read('config.ini')

    zzz=mysql.connector.connect(user = config['mysqlDB']['user'],password = config['mysqlDB']['pass'],host = config['mysqlDB']['host'],port=config['mysqlDB']['port'],database = config['mysqlDB']['db'],connect_timeout=5000)

    return zzz