import os
import sys
import time
import base64
import shutil
import datetime
import socketio
import eventlet
import argparse
import threading
import eventlet.wsgi
import numpy as np
import pandas as pd
import configparser

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
from keras.models import model_from_json
import numpy as np
import socket
import pickle


from PIL import Image
from io import BytesIO
from flask import Flask
from threading import Thread
from numpy.lib.function_base import select

from TankEnv import TankEnv # A class of get and save stage send from game environment

tankEnv = TankEnv()

#initialize our client
sio = socketio.Client()

#registering event handler for the client
@sio.event
def connect():
    print("The tank is ready!!! Waiting game starting ...")


@sio.event
def connect_error(data):
    print("The connection to game server failed!")


@sio.event
def disconnect():
    print("Disconnect to game server.")

# called every frame and transfer data from game
@sio.on('radio_tank_1')
def on_message(data):
    tankEnv.get_data(data)
    next_step = tankEnv.next_step()
    if next_step == True:
        action, pos = tankEnv.get_action()
        send_control(action, pos)
    else:
        pass

#send control to game(action:  1 is fire, 0 is move to)
def send_control(action, pos):
    print("Da goi den lenh nay", action, pos)
    sio.emit(
        "radio_tank_1_reply",
        data={
            'action': action.__str__(),
            'pos_x': pos[0].__str__(),
            'pos_y': pos[1].__str__(),
        })

def socket_run():
    config = configparser.ConfigParser()
    config.read('game_server.cfg')
    address = config.get('SOCKET', 'SOCKET_CLIENT_ANDDRESS')
    port = config.get('SOCKET', 'SOCKET_PORT')
    sio.connect(address + port)

from sacd.agent import SacdAgent, SharedSacdAgent
agent = SacdAgent((1606,), 4, log_dir="../logs/", seed=37, log_interval=1e6+7)
### Load weight ####


####################
def predict():    
    while True:
        start = tankEnv.is_game_start()
        if(start == True):
            try:
                s = tankEnv.get_stage()  ##Getting an initial state
                check_game_end = tankEnv.check_game_end()
                while check_game_end != True:
                    try:
                        if tankEnv.can_fire():
                            act = 5
                        else:
                            act = agent.exploit(s)  # Getting an action from the trained 
                        action, pos = tankEnv.nor_action(act)
                        tankEnv.send_action(action, pos) #Send mess to game environment
                        time.sleep(0.2) #Sleep a litter bit
                        #Get next stage
                        s_next = tankEnv.get_stage()
                        s = s_next
                        check_round_end = tankEnv.check_round_end()
                        if(check_round_end == True):
                            time.sleep(3)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print("Finished.")
                    check_game_end = tankEnv.check_game_end()
                result = tankEnv.check_win()
                if result == 1:
                    print("WIN")
                elif result == 2:
                    print("LOSE")
                else:
                    print("ERROR")
            except Exception as e:
                import traceback
                traceback.print_exc()
            print("End game.")
            break
        else:
            print("Waiting game start...")
            continue  

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    try:
        t = time.time()
        # creating thread
        t1 = threading.Thread(target=socket_run)
        t2 = threading.Thread(target=predict)
        # starting thread
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # both threads completely executed
        print("End!")
    except:
        print ("Thearding Error!")
