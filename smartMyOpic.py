import argparse
import os
import random

from envs import MappingEnvironment, LocalISM

import numpy as np

parser = argparse.ArgumentParser()

# General Stuff
parser.add_argument('--experiment', default='runs/myopic', help='folder to put results of experiment in')

# Environment
parser.add_argument('--N', type=int, default=25, help='size of grid')
parser.add_argument('--map_p', type=float, default=.1, help='probability map location is occupied')
parser.add_argument('--prims', action='store_true', help='prims algorithm for filling in map')
parser.add_argument('--episode_length', type=int, default=300, help='length of episode')

# Sensor
parser.add_argument('--sensor_type', default='local', help='local | range')
parser.add_argument('--sensor_span', type=int, default=1, help='span of sensor')
parser.add_argument('--sensor_p', type=float, default=.8, help='probability sensor reading is correct')

parser.add_argument('--seed', type=int, default=random.randint(0, 10000), help='random seed')

opt = parser.parse_args()
print(opt)

random.seed(opt.seed)
np.random.seed(opt.seed)

def stepEntrop(atX, atY):
    p = (obs[atX, atY, 0] + 1) / 2  # wahrscheinlichkeit fue die jeweilige aktion. (x+1)/2 scale back probability to 0-1
    mask = np.ones((3, 3))
    mask[1, 1] = 0
    ent = obs[atX-1:atX+2, atY-1:atY+2, 1]

    expec_entr_atXY = (1 - p) * np.sum(mask * (ent + 1) / 2)  # (x+1)/2 scale back entropy to 0-1

    return expec_entr_atXY



# make experiment path
os.makedirs(opt.experiment, exist_ok=True)
with open(os.path.join(opt.experiment, 'config.txt'), 'w') as f:
    f.write(str(opt))

# Initialize sensor
if opt.sensor_type == 'local':
    ism_proto = lambda x: LocalISM(x, span=opt.sensor_span, p_correct=opt.sensor_p)
else:
    raise Exception('sensor type not supported.')

# Initialize environment
env = MappingEnvironment(ism_proto, N=opt.N, p=opt.map_p, episode_length=opt.episode_length, prims=opt.prims)

# Test
rewards = []
for k in range(1000):
    obs = env.reset()

    done = False
    R = 0
    while not done:
        # Perform a_t according to actor_critic
        expected_greed_ent = 0
        best_greed_action=0
        best_greed_ent=0
        env.render()
        best_ent = 0
        best_action = 0
        best_sum_ent = 0
        a = 0

        for x in range(1,2*opt.N-2):
            for y in range(1,2*opt.N-2):

                go_x = x
                go_y = y
                counter=0
                sum_ent = 0
                done_1 = False

                while not done_1:
                    if (go_x <=opt.N) and (go_y<=opt.N): # left lower quadrant
                        if go_y<=go_x:
                            go_y= go_y+1
                            sum_ent += stepEntrop(go_x,go_y)
                            action= 3 # go up
                        else:
                            go_x = go_x+1
                            sum_ent += stepEntrop(go_x, go_y)
                            action= 1 #go right
                    elif (go_x>opt.N) and (go_y<=opt.N): # right lower quadrant

                        if (go_x-opt.N)<(opt.N-go_y):
                            go_y= go_y+1
                            sum_ent += stepEntrop(go_x, go_y)
                            action= 3 #go up
                        else:
                            go_x = go_x-1
                            sum_ent += stepEntrop(go_x, go_y)
                            action= 0 #go left

                    elif (go_x<=opt.N) and (go_y>opt.N): #left upper quadrant
                        if go_x<=(go_y-opt.N-1):
                            go_x = go_x + 1
                            sum_ent += stepEntrop(go_x, go_y)
                            action= 1 #go right
                        else:
                            go_y = go_y -1
                            sum_ent += stepEntrop(go_x, go_y)
                            action= 2 #go down
                    else: # right lower quadrant
                        if go_x<go_y:
                            go_x = go_x-1
                            sum_ent += stepEntrop(go_x, go_y)
                            action= 0 #go left
                        else:
                            go_y= go_y-1
                            sum_ent += stepEntrop(go_x, go_y)
                            action= 2 #go down
                    counter += 1

                    if (go_x==opt.N) and (go_y==opt.N):
                        done_1= True
                        sum_ent=sum_ent/(counter+1)
                        #print(sum_ent)

                        #if (y==47)and (x==47):
                         #   raise SystemExit(0)
                        if sum_ent > best_sum_ent:
                            #print(y,x,sum_ent)
                            best_sum_ent = sum_ent
                            best_action = action



        for i, (x, y) in enumerate([[1, 0], [-1, 0], [0, 1], [0, -1]]):
            p = (obs[opt.N-1+x, opt.N-1+y, 0]+1)/2 #wahrscheinlichkeit fue die jeweilige aktion. (x+1)/2 scale back probability to 0-1
            mask = np.ones((3, 3))
            mask[1,1] = 0


            ent = obs[opt.N-1-1+x:opt.N-1+2+x, opt.N-1-1+y:opt.N-1+2+y, 1]
            ent_i= (obs[opt.N-1+x, opt.N-1+y, 1]+1)/2


            expected_greed_ent = (1-p) * np.sum(mask * (ent+1)/2)#(x+1)/2 scale back entropy to 0-1
            if expected_greed_ent > best_greed_ent:
                best_greed_ent = expected_greed_ent
                best_greed_action = i

            if (i==best_action):
                expacted_greed_sm_ent=expected_greed_ent


        if(expacted_greed_sm_ent<0.25):
            print("hier",expacted_greed_sm_ent)
            best_action=best_greed_action

        # Receive reward r_t and new state s_t+1
        obs, reward, done, info = env.step(best_action)

        R += reward
    print(R)
    rewards.append(R)

np.save(os.path.join(opt.experiment, 'rewards_test'), rewards)
