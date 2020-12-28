from keras.models import Sequential, Model, load_model
from keras.layers import Input, Convolution2D, Flatten, Dense, LeakyReLU, Multiply, Lambda
from keras.optimizers import RMSprop, Adam
#import keras.backend as K
from keras import backend as K
from keras.layers import Lambda
from keras.utils import plot_model
from keras.layers import Lambda
import tensorflow as tf


def build_network(input_shape, num_actions):
        input_frame = Input(shape=input_shape)
        action_one_hot = Input(shape=(num_actions,))
        conv1 = Convolution2D(32, (8, 8), strides=(4, 4), activation='relu')(input_frame)
        conv2 = Convolution2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        flat_feature = Flatten()(conv3)
        hidden_feature = Dense(512)(flat_feature)
        lrelu_feature = LeakyReLU()(hidden_feature)
        q_value_prediction = Dense(num_actions)(lrelu_feature)

        if True:
            # Dueling Network
            # Q = Value of state + (Value of Action - Mean of all action value)
            #hidden_feature_2 = Dense(512, activation='relu')(flat_feature)
            #state_value_prediction = Dense(1)(hidden_feature_2)
            #q_value_prediction = merge([q_value_prediction, state_value_prediction],
            #                           mode=lambda x: x[0] - K.mean(x[0]) + x[1],
            #                           output_shape=(self.num_actions,))
            value_hidden = Dense(512, activation = 'relu', name = 'value_fc')(hidden_feature)
            value = Dense(1, name = "value")(value_hidden)
            action_hidden = Dense(512, activation = 'relu', name = 'action_fc')(hidden_feature)
            action = Dense(num_actions, name = "action")(action_hidden)
            action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keepdims = True), name = 'action_mean')(action) 
            q_value_prediction = Lambda(lambda x: x[0] + x[1] - x[2], name = 'duel_output')([action, value, action_mean])
        select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
        #select_q_value_of_action = merge([q_value_prediction, action_one_hot], mode='mul',
        #                                 output_shape=(self.num_actions,))

        target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), output_shape=lambda_out_shape)(
            select_q_value_of_action)

        model = Model(inputs=[input_frame, action_one_hot], outputs=[q_value_prediction, target_q_value])

        # MSE loss on target_q_value only
        model.compile(loss=['mse', 'mse'], loss_weights=[0.0, 1.0], optimizer=Adam(lr=0.00001))  # self.opt)
        model.summary()
        #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model

def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)
