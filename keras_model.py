import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Convolution2D, Flatten, Dense, LeakyReLU, merge
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from keras.layers import Lambda
from keras.utils import plot_model
from agents.agent import Agent

#from keras.backend.tensorflow_backend import set_session



class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)

    def __init__(self, x_axis, y_axis, height, num_actions):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.height = height
        self.num_actions = num_actions

    def build_network(self):
        # Consturct model
        input_frame = Input(shape=(self.x_axis, self.y_axis, self.height))
        action_one_hot = Input(shape=(self.num_actions,))
        conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(input_frame)
        conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv1)
        conv3 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(conv2)
        flat_feature = Flatten()(conv3)
        hidden_feature = Dense(512)(flat_feature)
        lrelu_feature = LeakyReLU()(hidden_feature)
        q_value_prediction = Dense(self.num_actions)(lrelu_feature)

        if self.dueling:
            # Dueling Network
            # Q = Value of state + (Value of Action - Mean of all action value)
            hidden_feature_2 = Dense(512, activation='relu')(flat_feature)
            state_value_prediction = Dense(1)(hidden_feature_2)
            q_value_prediction = merge([q_value_prediction, state_value_prediction],
                                       mode=lambda x: x[0] - K.mean(x[0]) + x[1],
                                       output_shape=(self.num_actions,))

        # select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
        select_q_value_of_action = merge([q_value_prediction, action_one_hot], mode='mul',
                                         output_shape=(self.num_actions,))
        target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), output_shape=lambda_out_shape)(
            select_q_value_of_action)

        model = Model(input=[input_frame, action_one_hot], output=[q_value_prediction, target_q_value])

        # MSE loss on target_q_value only
        model.compile(loss=['mse', 'mse'], loss_weights=[0.0, 1.0], optimizer=Adam(lr=0.00001))  # self.opt)
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model
