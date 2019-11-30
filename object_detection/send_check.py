import csv
from time import sleep
import storage
import time
import storage
from datetime import datetime
import time
import csv
import mmap
import os

###########################transmision data to db####################################
def update_db(txx,rid,timestr,datestr):

    timebty = bytearray(timestr, 'utf-8')
    datebty = bytearray(datestr, 'utf-8')
    txx = bytearray(txx, 'utf-8')
    rid = bytearray(rid, 'utf-8')
    conn = storage.connect()
    c = conn.cursor(buffered =True)
    c.execute("SELECT * FROM veh_all_bl where veh_no=%s",(txx,))
    row = c.fetchone()
    if not row :
        #print("Not exist")
        c.execute("SELECT MAX(veh_id) FROM  veh_all_bl")
        row1= c.fetchone()
        new_id=row1[0]+1
        #print(row1[0])
        c.execute("INSERT INTO veh_all_bl(veh_id,veh_no) VALUES(%s,%s)",(new_id,txx))
        time.sleep(0.1)
        c.execute("INSERT INTO veh_lpinfotbl(lp_noid,ckpnt_id,lp_time,lp_date) VALUES(%s,%s,%s,%s)",(new_id,rid,timebty,datebty))
    else:
        #print ("exist")
        c.execute("SELECT veh_id FROM veh_all_bl WHERE veh_no =%s",(txx,))
        row2= c.fetchone()
        c.execute("INSERT INTO veh_lpinfotbl(lp_noid,ckpnt_id,lp_time,lp_date) VALUES(%s,%s,%s,%s)",(row2[0],rid,timebty,datebty))
        time.sleep(0.1)
    conn.commit()
    c.close()
#############################send data ###################################
def transmit_no(x,rid,timest,datest):
    try:
        #conn = storage.connect()
        #c = conn.cursor()
        #############################
        print("Sending new data.")
        update_db(x,rid,timest,datest)
        #c.execute("INSERT INTO veh_lpinfotbl(lp_no, lp_date, lp_time,chkpnt_id) VALUES(%s, %s, %s,%s)",(x,datestr,timestr,rid))
        #    inserted=True
        #conn.commit()
        #c.close()
        print("Sending new data is done....")
    except Exception as e:
        print (e)
        sleep(1)
        print("No conncetion to DB. Saving offline")
        z= open('ins.txt','a')
        #z.write(x +'\n')
        z.write(x +',')
        z.write(datest +',')
        z.write(timest +',')
        z.write(rid +'\n')
        z.close()
        print("Saving offline is done")
    finally:
        if check(x):
            #print('Violated')
            return (False)
        else:
            #print('PASS')
            return (True)
#########################resend unsent data########################################
def re_send():
    #conn = storage.connect()
    #c = conn.cursor()
    #############################upload old
    f= open('ins.txt','rt')
    #f.seek(0) #ensure you're at the start of the file..
    first_char = f.read(1) #get the first character
    if  first_char:
        print ("There are unsent data.....Sending")
        #print (first_char)
        f.seek(0)
        reader = csv.reader(f, delimiter = ',', skipinitialspace=True)
        lineData = list()
        #cols = next(reader) skip the first lineData
        cols = reader

        for line in reader:
            if line != []:
                lineData.append(line)
        for i in range(len(lineData)):
            #c.execute("INSERT INTO veh_lpinfotbl(lp_no, lp_date, lp_time,chkpnt_id) VALUES(%s, %s, %s,%s)",(lineData[i][0],lineData[i][1],lineData[i][2],lineData[i][3]))
            update_db(lineData[i][0],lineData[i][3],lineData[i][2],lineData[i][1])
        f.close()
        f = open('ins.txt', 'w') # to empty the file
        f.close()
        print("Sending old data is done.")
    else:
        print("There are no unsent data")


#######################chk if viol table in db was modified#####################################
def chkdbtime():
    try:
        conn = storage.connect()
        c = conn.cursor(buffered=True)
        c.execute("SELECT UPDATE_TIME FROM information_schema.tables WHERE TABLE_SCHEMA = 'db_lp_ckpnt20193' AND TABLE_NAME = 'veh_all_bl'")
        c.execute("SELECT UPDATE_TIME FROM information_schema.tables WHERE TABLE_SCHEMA = 'db_lp_ckpnt20193' AND TABLE_NAME = 'veh_lpinfotbl'")

        y=c.fetchone()
        tdb = y[0] #the data from db is tuple, convert it to date timetuple
        tdb = datetime.strptime(str(tdb), "%Y-%m-%d %H:%M:%S")
        tdb=str(tdb)
        conn.commit()
        c.close()
        return (tdb)
    except Exception as e:
        print (e)
        pass
#######################check if lp is violation#####################################
def check(x):
    fo = open("viol.txt", "r")
    file_contents = fo.read()
    Flag = 0
    for i in file_contents.split('\n'):
        if x == i:
            Flag = 1
    if Flag == 1:
        return True
    else:
        return False
#######################get viol from db#####################################
told =time.strftime('2019-08-29 10:22:22')
def get_viol():
    global told
    try:
        conn = storage.connect()
        c = conn.cursor(buffered=True)
        tnew=chkdbtime()
        if tnew>told:
            print("Updating viol list")
            #c.execute("SELECT veh_no FROM veh_all_bl WHERE bl_flg=%s",(bl,))
            f= open('viol.txt','w')
            c.execute("SELECT veh_no FROM veh_all_bl WHERE bl_flg='1'")
            rows = c.fetchall()
            for row in rows:
                #print(row[0])
                f.write(row[0] +'\n')
            f.close
            conn.commit()
            c.close()
            told=tnew
        else: print("no viol updates")
    except Exception as e:
        print (e)
        pass
#########################fetch viol from db and resend unsent data#################################
def all_a():
    while True:
        try:
            if os.path.getsize("ins.txt") > 0:            
                re_send()           
            else:
                print("Nothing to send")
            get_viol()
            '''
            x='111111'
            if check(x):
                print('True')
            else:
                print('False')
            '''
            time.sleep(5)
        except Exception as e:
            print ("no connection to db")
            pass
##################################################################
