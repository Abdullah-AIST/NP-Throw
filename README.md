

# Non-Prehensile Throwing: A Reinforcement Learning Perspective

<!-- [xx](xx)<sup>1</sup>
<sup>1</sup> xx, <sup>2</sup> xxx -->


Submitted to IEEE International Conference of Robotics and Automation (ICRA 2026).

[Paper](npthrow_icra2026.pdf) | [Arxiv](https://abdullah-aist.github.io/NP-Throw/) | [Video](https://www.youtube.com/watch?v=JWAr3b1pHvgv) | [Website](https://abdullah-aist.github.io/NP-Throw/)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)


## Installation

1. Follow IsaacLab installation guide https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html . 

    Our setup is based Ubuntu 22.4, IsaacSim 4.5, and IsaacLab 2.1.0. We use RL_games for training.

2. Activate isaaclab conda env and source the np_throw task -- The task was developed using IsaacLab template (https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/developer-guide/template.html)
    ```
    conda activate env_isaaclab
    python -m pip install -e source/np_throw/
    ```

3. Install other dependencies -- for UR5e control, we use [UR-RTDE](https://sdurobotics.gitlab.io/ur_rtde/) package 
    ```bash
    pip install --user ur_rtde
    ```

## Train & Play & Eval

### Train
* Pretrained weights are included for the four policies.
```bash
# in the root directory of NP-Throw 
python scripts/train.py --task=NPThrow --num_envs 4096 --headless --experiment_name Default --seed 0  
```

### Play
```bash
# in the root directory of NP-Throw  -- Make sure to set training flag to False
python scripts/play.py --task=NPThrow --num_envs 32 --experiment_name Default --seed 0

# Use playZero during enviroment setup for debugging.
# python scripts/zeroAgent.py --task=NPThrow --num_envs 16

```

### Eval
```bash
# in the root directory of NP-Throw  -- Make sure to set training flag to False
# For the environment configuration file, select a target object and the evaluation target.
python scripts/Eval.py --task=NPThrow --num_envs 32 --headless --experiment_name Default --seed 0 --envSeed 0 --targetObject woodBlock

```
### Deploy
1. Process the trajectories using the "processTrajs_real.ipynb" notebook to analyze and generate neccessary trajectories.
2. Deploy based of UR-RTDE package 

```bash
python scripts/deploy.py
```


## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{NP_Throw_ICRA2026,
  title = "Non-Prehensile Throwing: A Reinforcement Learning Perspective",
  author = "{Author One, Author Two}",
  booktitle={ICRA 2026},
  year={2026},
  organization={IEEE}
}
```

## License

This codebase is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en). You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products.


## Acknowledgements
- [IsaacLab](https://github.com/isaac-sim/IsaacLab): We use the `isaaclab` library for the RL training and evaluation.
- [UR-RTDE](https://sdurobotics.gitlab.io/ur_rtde/): We use the `UR-RTDE` package for UR5e real-time control.

## Contact

Feel free to open an issue or discussion if you encounter any problems or have questions about this project.
