
import mysql.connector as mysqlcon
import cPickle
import time

j = 0
last_time = time.time()

class Database:
    @staticmethod
    def add(state_before, action, state_after, reward, round_id):
        global j
        j += 1
        if (j % 100 == 0) :
            print "saved",j
            print "time", time.time() - last_time
        state_after_bytecode = Database.serialize(state_after)
        state_before_bytecode = Database.serialize(state_before)
        # print "reached : ", action
        db = mysqlcon.connect(host="localhost", user="sergey", passwd="gfgfgjrth1", db="HISTORY", charset='latin1')
        cursor = db.cursor()
        sql = ("""INSERT INTO history(state_before, action, state_after, reward, round_id)
              VALUES (%s, %s, %s, %s, %s)""")
        cursor.execute(sql,(state_before_bytecode, str(action),state_after_bytecode, reward,round_id))
        db.commit()
        db.close()
        return

    @staticmethod
    def getSamples():
        db = mysqlcon.connect(host="localhost", user="sergey", passwd="gfgfgjrth1", db="HISTORY", charset='latin1')
        cursor = db.cursor()
        # entries with reward == 0 not ready to train.
        sql = "SELECT * FROM history WHERE NOT reward=0 ORDER BY RAND() LIMIT 32"
        cursor.execute(sql)
        data = cursor.fetchall()
        tuples = []
        if (len(data) == 0):
            print "empty database"
            return []
        for sample in data:
            result = str(sample[1].decode(encoding="latin1"))
            state_before = Database.deserialize(str(sample[1].decode(encoding="latin1")))
            state_after = Database.deserialize(str(sample[3].decode(encoding="latin1")))
            action = sample[2]
            reward = sample[4]
            tuples.append((state_before,action,state_after,reward))
        db.commit()
        db.close()
        return tuples

    @staticmethod
    def serialize(data):
        s = cPickle.dumps(data)
        return s

    @staticmethod
    def setRewards(round_id, is_positive):
        db = mysqlcon.connect(host="localhost", user="sergey", passwd="gfgfgjrth1", db="HISTORY", charset='latin1')
        cursor = db.cursor()
        if is_positive:
            reward = +1;
        else:
            reward = -1;
        sql = """UPDATE history SET reward=%s WHERE round_id=%s"""
        cursor.execute(sql, (reward, round_id))
        db.commit()
        db.close()

    @staticmethod
    def deserialize(sdata):
        return cPickle.loads(sdata)




