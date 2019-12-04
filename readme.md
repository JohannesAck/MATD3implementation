# Implementation of Multi-Agent TD3

This is the implemetation of MATD3, presented in our paper [Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics](https://arxiv.org/pdf/1910.01465.pdf).
Multi-Agent TD3 is an algorithm for multi-agent reinforcement learning, that combines the improvements of [TD3](https://arxiv.org/pdf/1802.09477.pdf) with [MADDPG](https://arxiv.org/pdf/1706.02275.pdf).

The implementation here is based on [maddpg from Ryan Lowe / OpenAI](https://github.com/openai/maddpg), the environments used  are from [multiagent-particle-envs from OpenAI](https://github.com/openai/multiagent-particle-envs)

### Requirements
 - ```python == 3.6```
 - ```TF == 1.12.0```         any 1.x should work
 - ```Gym == 0.10.5```           *this one is important*
 - ```Numpy >= 1.16.2``` 

### Example Useage
To start training on simple_crypto, with an MATD3 team of agents and an MADDPG adversary, use 
```
python train.py --scenario simple_speaker_listener --good-policy matd3 --adv-policy maddpg
```


### Reference
If you use our implementation, please also cite our paper with 
```
@misc{ackermann2019reducing,
    title={Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics},
    author={Johannes Ackermann and Volker Gabler and Takayuki Osa and Masashi Sugiyama},
    year={2019},
    eprint={1910.01465},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
