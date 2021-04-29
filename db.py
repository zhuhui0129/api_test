import pymysql

class DB:
    def __init__(self):
        self.conn = pymysql.connect(host='127.0.0.1',
                               port=3306,
                               user='root',
                               passwd='123456',
                               db='runoob',
                               charset='utf8')  # 如果查询有中文，需要指定测试集编码

        self.cur=self.conn.cursor()

    def __del__(self):
        self.cur.close()  # 关闭游标
        self.conn.close()  # 关闭连接

    def query(self,sql):
        self.cur.execute(sql)
        return self.cur.fetchall()

    def exce(self):
        try:
            self.cur.execute(sql)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(str(e))

    def check_user(self,title):
        result=self.query( "select * from app1_book where title='{}'".format(title) )
        return  True if result else False

db=DB()
res=db.check_user("冲灵剑法")
print(res)