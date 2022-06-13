# hypersonic-reentry(under development)
<Trajectory optimization of hypersonic reentry vehicle>
 
Entry vehicle: normalization of values yet to be implemented 

Aerodynamic coefficient taken from CAV-H basic data, same goes for the BC

isacalc package (https://github.com/LukeDeWaal/ISA_Calculator.git) dependency for earth atmosphere parameters

This code refers a good amount of work from the EDLpy(https://github.com/CDNoyes/EDL-Py.git) library

 Other dependency pybrain (https://github.com/pybrain/pybrain.git)
 
for DDPG follow (https://github.com/MOCR/DDPG.git) which has version issues which can be bypassed by implementing ```tensorflow.compat.v1```
 
 it would be better to use DDPG baselines from OpenAI (https://github.com/openai/baselines.git)
