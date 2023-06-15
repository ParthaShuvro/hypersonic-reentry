# hypersonic-reentry(under development)
<Trajectory optimization of hypersonic reentry vehicle>
 
Aerodynamic coefficient taken from CAV-H basic data, same goes for the BC

follow the registration process for gym environment setup

Normalization was used to see if the results somehow improve. but results mostly depend on the reward model and training setup with recommended network types

use stable-baselines3 (https://github.com/DLR-RM/stable-baselines3) for PPO, DDPG, TD3 and sb3-contrib (https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) for TQC.

isacalc package (https://github.com/LukeDeWaal/ISA_Calculator.git) dependency for earth atmosphere parameters

This code refers a good amount of work from the EDLpy(https://github.com/CDNoyes/EDL-Py.git) library
