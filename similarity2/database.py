"""数据库读取相关代码"""
from typing import List, Tuple, Any, Dict, Union

import dmPython
from demo.settings import DEBUG
from similarity2.globals import CONFIG
import pymysql

# 读取config配置
host = CONFIG.get("database", "db_host")
user = CONFIG.get("database", "db_user")
password = CONFIG.get("database", "db_password")
dm_password = CONFIG.get("database", "dm_password")
db_name = CONFIG.get("database", "db_name")
db_port = int(CONFIG.get("database", "db_port"))
dm_name = CONFIG.get("database", "dm_name")
dm_user = CONFIG.get("database", "dm_user")
dm_port = int(CONFIG.get("database", "dm_port"))
db_type = CONFIG.get("database", 'db_type')


class Database:
    """请保证values_list、values_dict方法，最后调用！"""

    @classmethod
    def get_common_database(cls):
        return Database("ai_original_data")

    def __init__(self, tablename):
        self.host = host
        self.user = user
        self.password = password
        self.database = db_name
        self.dm_name = dm_name
        self.port = db_port
        self.charset = 'utf8'
        self.tablename = tablename
        self.projection = None
        if DEBUG:
            print(dict(host=self.host, port=self.port, user=self.user, password=self.password,
                       database=self.database,
                       charset=self.charset))
        if db_type == 'sql' or db_type == 'mysql':
            print('using MYSQL database!')
            self.connection = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                                              database=self.database,
                                              charset=self.charset)
        else:
            print('using BigDream database!')
            self.connection = dmPython.connect(user=dm_user, password=dm_password, server=self.host, port=dm_port)
        self.cursor = None
        self.filter_sql = ""
        self.order_by_sql = ""

    def _execute(self):
        sql = (f" select {self.projection}" if self.projection else "") + \
              (f" from {self.dm_name.upper() + '.' if db_type == 'dm' else ''}{self.tablename}" if self.tablename else "") + \
              (f" where {self.filter_sql}" if self.filter_sql else "") + \
              (f" order by {self.order_by_sql}" if self.order_by_sql else "")
        self.cursor.execute(sql)
        if DEBUG:
            print(f"执行sql:{sql}")
        result = self.cursor.fetchall()
        return result

    def reset(self):
        """重置数据库游标和数据"""
        self.cursor.close()
        self.cursor = None
        self.filter_sql = ""
        self.order_by_sql = ""
        self.projection = None

    def filter(self, **params):
        """
        :param params: 过滤参数，多个参数以and连接
        da  tabase.filter(business_type="item_catalog").all()\n
        等价于 select * from table where business_type='item_catalog'
        """
        self.filter_sql = " and ".join(
            [
                f"{a}=\'{b}\'"
                if type(b) is str
                else f"{a}={b}"
                for a, b in params.items()
            ]
        )
        return self

    def order_by(self, *params):
        """
        :param params: 排序参数
        database.order_by("-id","name").all()\n
        等价于 select * from table order_by id desc, name
        """
        self.order_by_sql = ",".join(
            [
                f"{x} desc"
                if x[0] == '-'
                else {x}
                for x in params
            ]
        )
        return self

    def values_list(self, *projection, flat=False) -> Union[List[Tuple[Any, Any, Any]], List[str]]:
        """
        :param projection: 选择查询的列名
        :param flat: 是否以列表返回，当且仅当选择一列时有效
        :return: Union[List[Tuple[Any,Any,Any]], List[str]]

        查询，并以二维列表返回结果

        >>> db = Database.get_common_database()
        >>> r = db.values_list("match_str","original_code","original_data")
        >>> print(r)
        (
            ("xx","xx","xx"),
            ("xx","xx","xx"),
            ...
        )
        >>> db = Database.get_common_database()
        >>> r = db.values_list("match_str",flat=True)
        >>> print(r)
        ["xx","xx","xx",...]
        """
        if len(projection) == 0:
            self.projection = "*"
        else:
            self.projection = ",".join(projection)
        if flat and len(projection) > 1:
            raise ValueError(f"flat=True requires len(projection) == 1,but got len(projection) == {len(projection)}")
        self.cursor = self.connection.cursor()
        result = self._execute()
        self.reset()
        if flat:
            result = [x[0] for x in result]
        return list(result)

    def values_dict(self, *projection, flat=False) -> List[Dict[str, Any]]:
        """
        :param projection: 选择查询的列名
        :param flat: 是否以列表返回，当且仅当选择一列时有效
        :return: List[Dict[str, Any]]

        查询，以字典格式返回数据

        >>> db = Database.get_common_database()
        >>> r = db.values_dict("match_str","original_code","original_data")
        >>> print(r)
        [
            {"match_str":"xx","original_code":"xxx","original_data":"xxx"},
            {"match_str":"xx","original_code":"xxx","original_data":"xxx"},
            ...
        ]
        """
        self.projection = ",".join(projection)
        self.cursor = self.connection.cursor(cursor=pymysql.cursors.DictCursor)
        result = self._execute()
        self.reset()
        return result
