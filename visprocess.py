import argparse
import numpy as np
#from environment import Environment
import matplotlib.animation as manimation
import matplotlib.pyplot as plt


from visualization.backpropagation import *
from PIL import Image
from visualization.grad_cam import *
from visualization.model import build_network
from scipy.misc.pilutil import imresize
from scipy.misc.pilutil import imread
from skimage.color import rgb2gray
global load_guided_model
global load_model
global session1



class visprocess():


    def __init__(self, env, args):
        self.frame_width = env.observation_size()
        self.frame_height = env.observation_size()
        self.state_length = 2
        self.num_frames = args.num_frames
        self.env = env
        self.test_dqn_model_path = args.test_dqn_model_path
        self.color = False

    def parse(self):
        parser = argparse.ArgumentParser(description="DQN")
        parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
        parser.add_argument('--video_dir', default=None, help='output video directory')
        parser.add_argument('--do_render', action='store_true', help='whether render environment')
        parser.add_argument('--gbp', action='store_true', help='visualize what the network learned with Guided backpropagation')
        parser.add_argument('--gradCAM', action='store_true', help='visualize what the network learned with GradCAM')
        parser.add_argument('--gbp_GradCAM', action='store_true', help='visualize what the network learned with Guided GradCAM')
        try:
            from argument import add_arguments
            parser = add_arguments(parser)
        except:
            pass
        args = parser.parse_args()
        return args


    def init_saliency_map(self,  agent, history, first_frame=0, prefix='QF_', resolution=300, save_dir='./movies/', env_name="DDDQN"):
        from tensorflow import Graph

        #_, _,load_model_cam,_ = build_network(agent.observation_shape, agent.action_space_n)
        #_, _, load_model_guided_backprop ,_= build_guided_model(agent.observation_shape, agent.action_space_n)
        graph1 = Graph()
        with graph1.as_default():
            session1 = tf.compat.v1.Session()
            with session1.as_default():
                load_model = build_network([agent.frame_width, agent.frame_height, agent.state_length], agent.num_actions)
                load_model.load_weights(self.test_dqn_model_path)
        graph2 = Graph()

        with graph2.as_default():
            session2 = tf.compat.v1.Session()
            with session2.as_default():
                load_guided_model = build_guided_model([agent.frame_width, agent.frame_height, agent.state_length], agent.num_actions)
                load_guided_model.load_weights(self.test_dqn_model_path)

        #load_model_guided_backprop.load_weights(args.load_network_path)
        #load_model_grad_cam.load_weights(args.load_network_path)
        total_frames=len(history['state'])
        backprop_actor = init_guided_backprop(graph1, load_model,"action")
        backprop_critic = init_guided_backprop(graph1, load_model,"value")
        cam_actor = init_grad_cam(graph1, load_model, "conv2d")
        cam_critic = init_grad_cam(graph1, load_model, "conv2d")



        guidedBackprop_actor = init_guided_backprop(graph2, load_guided_model,"action")
        guidedBackprop_critic = init_guided_backprop(graph2, load_guided_model,"value")
        gradCAM_actor = init_grad_cam(graph2, load_guided_model, "conv2d")
        gradCAM_critic = init_grad_cam(graph2, load_guided_model, "conv2d")#, False)
        fig_array = np.zeros((6,2,self.num_frames,self.frame_width,self.frame_height))
        movie= np.zeros((total_frames,self.frame_width,self.frame_height, self.state_length))
        for i in range(self.num_frames):#total_frames): #num_frames
            ix = first_frame+i
            if ix < total_frames: # prevent loop from trying to process a frame ix greater than rollout length
                frame = history['state'][ix].copy()
                action = history['action'][ix]#.copy()
                frame = np.expand_dims(frame, axis=0)
                if ix%10==0:
                    print(ix)

                actor_gbp_heatmap = guided_backprop(session1, graph1, frame, backprop_actor)
                actor_gbp_heatmap = np.asarray(actor_gbp_heatmap)
                history['gradients_actor'].append(actor_gbp_heatmap)

                actor_gbp_heatmap = guided_backprop(session1, graph1, frame, backprop_critic)
                actor_gbp_heatmap = np.asarray(actor_gbp_heatmap)
                history['gradients_critic'].append(actor_gbp_heatmap)

                Cam_heatmap = grad_cam(session1, graph1, cam_actor, frame, action)
                Cam_heatmap = np.asarray(Cam_heatmap)
                history['gradCam_actor'].append(Cam_heatmap)

                gradCam_heatmap = grad_cam(session1, graph1, cam_critic, frame, action)
                gradCam_heatmap = np.asarray(gradCam_heatmap)
                history['gradCam_critic'].append(gradCam_heatmap)

                actor_gbp_heatmap = guided_backprop(session2, graph2, frame, guidedBackprop_actor)
                actor_gbp_heatmap = np.asarray(actor_gbp_heatmap)
                history['gdb_actor'].append(actor_gbp_heatmap)

                critic_gbp_heatmap = guided_backprop(session2, graph2, frame, guidedBackprop_critic)
                critic_gbp_heatmap = np.asarray(critic_gbp_heatmap)
                history['gdb_critic'].append(critic_gbp_heatmap)

                gradCam_heatmap = grad_cam(session2, graph2, gradCAM_actor, frame, action)
                gradCam_heatmap = np.asarray(gradCam_heatmap)
                history['guidedGradCam_actor'].append(gradCam_heatmap)

                gradCam_heatmap = grad_cam(session2, graph2, gradCAM_critic, frame, action)
                gradCam_heatmap = np.asarray(gradCam_heatmap)
                history['guidedGradCam_critic'].append(gradCam_heatmap)

                movie[i,:,:,:]=frame[:,:,:]



        history_gradients_actor = history['gradients_actor'].copy()
        history_gradients_critic = history['gradients_critic'].copy()
        history_gdb_actor = history['gdb_actor'].copy()
        history_gdb_critic = history['gdb_critic'].copy()
        history_gradCam_actor = history['gradCam_actor'].copy()
        history_gradCam_critic = history['gradCam_critic'].copy()
        history_gradCamGuided_actor = history['guidedGradCam_actor'].copy()
        history_gradCamGuided_critic = history['guidedGradCam_critic'].copy()
        fig_array[0,0] = self.normalization(history_gradients_actor, history, "gdb",GDB_actor=1)
        fig_array[0,1] = self.normalization(history_gradients_critic, history, 'gdb')
        fig_array[1,0] = self.normalization(history_gdb_actor, history, "gdb", GDB_actor=1)
        fig_array[1,1] = self.normalization(history_gdb_critic, history, 'gdb')
        fig_array[2,0] = self.normalization(history_gradCam_actor, history, "cam", )
        fig_array[2,1] = movie[:,:,:,0]#self.normalization(history_gradCam_critic, history, "cam")
        fig_array[3,0] = self.normalization(history_gradCam_actor, history, "cam", GDB_actor=1, guided_model=history_gdb_actor)
        fig_array[3,1] = self.normalization(history_gradCam_critic, history, 'cam',guided_model=history_gdb_critic)
        fig_array[4,0] = self.normalization(history_gradCamGuided_actor, history, "cam")
        fig_array[4,1] = movie[:,:,:,1]#self.normalization(history_gradCamGuided_critic, history, "cam")
        fig_array[5,0] = self.normalization(history_gradCamGuided_actor, history, "cam",GDB_actor=1, guided_model=history_gdb_actor)
        fig_array[5,1] = self.normalization(history_gradCamGuided_critic, history, 'cam',guided_model=history_gdb_critic)

        self.make_movie(history, fig_array, first_frame, resolution, save_dir, prefix, env_name)




    def normalization(self, heatmap, history, visu, GDB_actor=0, guided_model=None):
        frame=0
        heatmap=np.asarray(heatmap)
        guided_model = np.asarray(guided_model)
        if guided_model.all() == None:
            if visu == 'gdb':
                print("normal")
                print(heatmap.shape)
                for i in range(heatmap.shape[0]):
                    heatmap_ = heatmap[i, :, :, :, :]
                    heatmap_ -= heatmap_.mean()
                    heatmap[i, :, :, :, :] /= (heatmap_.std() + 1e-5)  #

                heatmap *= 0.1  # 0.1 #0.1
                heatmap = np.clip(heatmap, -1, 1)
                heatmap_pic1 = heatmap[:, 0, :, :, frame]
            if visu == 'cam':
                heatmap_pic1 = heatmap[:, :, :]
        else:
            print(" notnormal")
            for i in range(guided_model.shape[0]):
                guided_model_ = guided_model[i, :, :, :, :]
                guided_model_ -= guided_model_.mean()
                guided_model[i, :, :, :, :] /= (guided_model_.std() + 1e-5)  #

            guided_model *= 0.1  # 0.1
            guided_model = np.clip(guided_model, -1, 1)
            guided_model = guided_model[:, 0, :, :, frame]
            guided_model[guided_model < 0.0] = 0
            heatmap[heatmap < 0.0] = 0
            heatmap_pic1 = (heatmap * guided_model)

        all_unproc_frames = history['state'].copy()
        state=np.asarray(all_unproc_frames)
        #frame = np.zeros((self.num_frames, self.frame_width, self.frame_height, self.state_length))
        #for i in range(len(all_unproc_frames)):
        #    frame[i, :, :, :] = np.asarray(Image.fromarray(all_unproc_frames[i]).resize((84, 84), Image.BILINEAR)) / 255
        if self.color:
            proc_frame1 = self.overlap_color(state, heatmap_pic1)
        else:
            proc_frame1 = self.overlap_BW(state, heatmap_pic1)
        print(proc_frame1.shape)

        return proc_frame1


    def overlap_color(self, frame,gbp_heatmap):
        color_neg = [1.0, 0.0, 0.0]
        color_pos = [0.0, 1.0, 0.0]
        color_chan = np.ones((self.num_frames,self.frame_width,self.frame_height,self.state_length),dtype=gbp_heatmap.dtype)
        alpha = 0.5
        _gbp_heatmap = np.stack(gbp_heatmap, axis=2)
        _gbp_heatmap = [gbp_heatmap for _ in range(3)]
        print(np.asarray(_gbp_heatmap).shape)
        gbp_heatmap=_gbp_heatmap
        gbp_heatmap_pos=np.asarray(gbp_heatmap.copy())
        gbp_heatmap_neg=np.asarray(gbp_heatmap.copy())
        gbp_heatmap_pos[gbp_heatmap_pos<0.0]=0
        gbp_heatmap_neg[gbp_heatmap_neg>=0.0]=0
        gbp_heatmap_neg=-gbp_heatmap_neg
        print(gbp_heatmap_pos.shape, gbp_heatmap_neg.shape)
        gbp_heatmap = color_pos * gbp_heatmap_pos[:,:,:,:] + color_neg * gbp_heatmap_neg[:,:,:,:]
        mixed = alpha * gbp_heatmap + (1.0 - alpha) * frame
        mixed = np.clip(mixed,0,1)



        return mixed

    def overlap_BW(self, frame, gbp_heatmap):
        print(frame.shape)
        alpha = 0.5
        _gbp_heatmap = np.stack(gbp_heatmap, axis=2)
        _gbp_heatmap = np.swapaxes(_gbp_heatmap, 0, 2)
        mixed = alpha * gbp_heatmap + (1.0 - alpha) * frame[:,:,:,0]
        mixed = np.clip(mixed,0,1)



        return _gbp_heatmap

    def make_movie(self, history,fig_array,first_frame,resolution,save_dir,prefix,env_name ):
        movie_title ="{}-{}-{}.mp4".format(prefix, self.num_frames, env_name.lower())
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='test', artist='mateus', comment='atari-video')
        writer = FFMpegWriter(fps=8, metadata=metadata)
        fig = plt.figure(figsize=[6, 6*1.3], dpi=resolution)
        print("fig_array.shape: ",fig_array.shape)
        with writer.saving(fig, save_dir + movie_title, resolution):
            titleListX=["Actor","Critic"]
            titleListY=["Backpropagation", "GradCam", "Guided Backpropagation","Guided GeadCam"]
            for i in range(self.num_frames):#total_frames): #num_frames
                plotColumns = np.shape(fig_array)[1] #fig_array[0,:,0,0,0,0].shape
                plotRows = np.shape(fig_array)[0]#fig_array[:,0,0,0,0,0].shape
                z=0

                for j in range(0, plotRows):
                    for k in range(0, plotColumns):
                        img = fig_array[j,k,i,:,:]
                        ax=fig.add_subplot(plotRows, plotColumns, z+1)
                        #ax.set_ylabel(titleListY[j])
                        #ax.set_xlabel(titleListX[k])
                        plt.imshow(img, cmap='gray')
                        z=z+1

                writer.grab_frame()
                fig.clear()
                if i%100==0:
                    print(i)



    def visgame(self, agent, total_episodes=1):

        history = { 'state': [], 'action': [], 'gradients_actor':[], 'gradients_critic':[],'gradCam_actor':[],'gradCam_critic':[], 'gdb_actor':[],'gdb_critic':[], 'guidedGradCam_actor':[],'guidedGradCam_critic':[] ,'movie_frames':[]}
        rewards = []
        for i in range(total_episodes):
            #print("state:",state.shape)
            #print("orgin_state:",origin_state.shape)
            #prozess_atari_wraper_frames(origin_state, state)
            #agent.init_game_setting()
            state = self.env.reset()
            done = False
            episode_reward = 0.0

            for _ in range(self.num_frames):
                history['state'].append(state)
                if(done==True):
                    state = self.env.reset()
                action = agent.make_action(state, test=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                #agent.save_observation(state)
                #action_state=agent.observations
                history["action"].append(action)
            rewards.append(episode_reward)
        print('Run %d episodes'%(total_episodes))
        print('Mean:', np.mean(rewards))
        self.init_saliency_map(agent, history)

        return history

    def save_observation(observations):
        observations = np.roll(observations, -1, axis=0)
        observation=np.zeros([49,49,2])
        observation[:,:,0]= transform_screen(observations[0,:,:,:])
        observation[:,:,1]= transform_screen(observations[1,:,:,:])
        #observation[:,:,2]= transform_screen(observations[2,:,:,:])
        #observation[:,:,3]= transform_screen(observations[3,:,:,:])
        return observation

