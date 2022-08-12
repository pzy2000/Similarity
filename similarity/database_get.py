#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pymysql
import os
import configparser
from similarity.tools import root_path
import dmPython

# host="localhost", user="root", password="2782680448", database="dbtest", charset='utf8'

keyword = 'database'
read_ini = configparser.ConfigParser()
read_ini.read(os.path.join(root_path, 'config.ini'), encoding='utf-8')


host = read_ini.get(keyword, 'db_host')
user = read_ini.get(keyword, 'db_user')
password = read_ini.get(keyword, 'db_password')
db_name = read_ini.get(keyword, 'db_name')
dm_user = read_ini.get(keyword, 'dm_user')
dm_password = read_ini.get(keyword, 'dm_password')
dm_port = read_ini.get(keyword, 'dm_port')
db_type = read_ini.get(keyword, 'db_type')


class database:
    def __init__(self, host, user, password, database, charset):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset

    def get_colum_by_num(self, num, tablename):  # 通过表名和索引列获取所需要的数据
        if db_type == 'sql':
            db = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database,
                             charset=self.charset)
            cursor = db.cursor()
            sql = 'SHOW COLUMNS FROM %s;' % tablename
        else:
            db = dmPython.connect(user=dm_user, password=dm_password, server=self.host, port=dm_port)
            cursor = db.cursor()
            sql = 'SELECT COLUMN_NAME from all_tab_columns where Table_Name=%s;' % tablename
        cursor.execute(sql)
        name = cursor.fetchall()
        selectname = ''
        for a, item in enumerate(num):
            selectname = selectname + ',' + name[item][0]
        newsql = "select %s from %s" % (selectname.lstrip(','), tablename)
        cursor.execute(newsql)
        matrix = cursor.fetchall()
        matrix = list(matrix)
        for i in matrix:
            matrix[matrix.index(i)] = list(i)
        db.close()
        return matrix

    def get_colum(self, sql):  # 通过sql获取数据
        if db_type == 'sql':
            db = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database,
                             charset=self.charset)
        else:
            db = dmPython.connect(user=dm_user, password=dm_password, server=self.host, port=dm_port)
        cursor = db.cursor()
        cursor.execute(sql)
        matrix = cursor.fetchall()
        db.close()
        return matrix

    def get_data_by_type(self, type):
        if db_type == 'sql':
            db = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database,
                             charset=self.charset)
        else:
            db = dmPython.connect(user=dm_user, password=dm_password, server=self.host, port=dm_port)
        cursor = db.cursor()
        sql = "select match_str,original_code,original_data from ai_original_data where business_type = '%s' " % type
        cursor.execute(sql)
        matrix = cursor.fetchall()
        matrix = list(matrix)
        for i in matrix:
            matrix[matrix.index(i)] = list(i)
        db.close()
        return matrix

    def get_data_by_type_v2(self, num, type, tablename):
        if db_type == 'sql':
            db = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database,
                             charset=self.charset)
        else:
            db = dmPython.connect(user=dm_user, password=dm_password, server=self.host, port=dm_port)
        cursor = db.cursor()
        sql = 'SHOW COLUMNS FROM %s;' % tablename
        cursor.execute(sql)
        name = cursor.fetchall()
        selectname = ''
        for a, item in enumerate(num):
            selectname = selectname + ',' + name[item][0]
        sql = "select %s from %s where business_type = '%s' " % (selectname.lstrip(','), tablename, type)
        cursor.execute(sql)
        matrix = cursor.fetchall()
        matrix = list(matrix)
        for i in matrix:
            matrix[matrix.index(i)] = list(i)
        db.close()
        return matrix


db = database(host=host, user=user, password=password, database=db_name, charset='utf8')

if __name__ == '__main__':
    idx = [2, 3, 4]
    m = db.get_data_by_type_v2(idx, 'model_data', 'ai_original_data')
    print(m)

    for i in m:
        print(i)
    print('-' * 50)
    t = []
    for i in m:
        t.append(i[0].replace('^', ' '))
        print(i[0].replace('^', ' '))
    print(t)

