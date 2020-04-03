# **Reacher**

## INSTALL
1. Install python dependencies:
  ```bash
  pythonX.Y -m pip install torch numpy matplotlib gym
  ```
  where X and Y are the major and minor sub-versions of the python you're using.

2. Clone version `0.4.0b` of Unity-Technologies ML-Agents with the command:
  ```bash
  git clone -b 0.4.0b https://github.com/Unity-Technologies/ml-agents.git ml-agents
  ```

3. Download and decompress the Reacher unity environment.
  ```bash
  curl -o Reacher_Linux_NoVis.zip https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip
  unzip Reacher_Linux_NoVis.zip
  ```

4. Train the agent.
  ```bash
  pythonX.Y ddpg.py
  ```

5. Good luck!
