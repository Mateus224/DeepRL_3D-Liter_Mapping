To train the agent:
python3 main_keras.py --dueling true --train_dqn


To run the code with a trained agent:
python3 main_keras.py --dueling true --test_dqn_model_path saved_dqn_networks/DDDQN_Litter_Agent_2420000.h5 --test_dqn
