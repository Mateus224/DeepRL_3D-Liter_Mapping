# To train the agent:
```bash
python3 main_keras.py --dueling true --train_dqn
```

# To run the code with a trained agent:
```bash
python3 main_keras.py --dueling true --test_dqn_model_path saved_dqn_networks/DDDQN_Litter_Agent_2420000.h5 --test_dqn
```

# To run the agent with visual explanation:
for visual explanations you need scipy=1.2.0
```bash
python3 main_keras.py --dueling true --test_dqn_model_path saved_dqn_networks/DDDQN_Litter_Agent_2420000.h5 --test_dqn --visualize --num_frames 300
```

