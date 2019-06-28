# Randomized Value functions via Multiplicative Normalizing Flows
This repo contains code for the paper 

[Randomized Value functions via Multiplicative Normalizing Flows.
Ahmed Touati, Harsh Satija, Joshua Romoff, Joelle Pineau, Pascal Vincent. UAI 2019](https://arxiv.org/abs/1806.02315)
```@article{touati2018randomized,
  title={Randomized value functions via multiplicative normalizing flows},
  author={Touati, Ahmed and Satija, Harsh and Romoff, Joshua and Pineau, Joelle and Vincent, Pascal},
  journal={arXiv preprint arXiv:1806.02315},
  year={2018}
}
```

## Installation

### PyTorch

without cuda:

```conda install pytorch=0.4.0 -c pytorch ```

with cuda:

```conda install pytorch=0.4.1 cuda90 -c pytorch ```

(or cuda92, cuda80, cuda 75. depending on what you have installed)

### Baselines for Atari preprocessing
``` git clone https://github.com/openai/baselines.git ```

``` cd baselines ```

``` pip install -e . ```


## Simple regression as sanity check
```python -m qlearn.commun.local_mnf_toy_regression```

<img src='https://github.com/facebookresearch/RandomizedValueFunctions/blob/master/local_mnf_toy_regression.png' title='regression'>

## Chain env experiments

### DQN
```python -m qlearn.toys.main_nchain --agent DQN --cuda 0 --input-dim 100```

Example of outcome:
```
episode: 5, Avg. reward: 0.107
episode: 6, Avg. reward: 0.107
...
episode: 21, Avg. reward: 0.107
episode: 22, Avg. reward: 0.107
episode: 23, Avg. reward: 0.107
episode: 24, Avg. reward: 0.107
episode: 25, Avg. reward: 0.107
episode: 26, Avg. reward: 0.107
episode: 27, Avg. reward: 0.107
episode: 28, Avg. reward: 0.107
episode: 29, Avg. reward: 0.107
episode: 30, Avg. reward: 0.107
...
```
### MNF DQN

```python -m qlearn.toys.main_nchain --agent MNFDQN --cuda 0 --input-dim 100```

Example of outcome:


```
episode: 5, Avg. reward: 0.0
episode: 6, Avg. reward: 0.0
...
episode: 21, Avg. reward: 0.0
episode: 22, Avg. reward: 0.0
episode: 23, Avg. reward: 0.0
episode: 24, Avg. reward: 10.0
episode: 25, Avg. reward: 10.0
episode: 26, Avg. reward: 10.0
episode: 27, Avg. reward: 10.0
episode: 28, Avg. reward: 10.0
episode: 29, Avg. reward: 10.0
episode: 30, Avg. reward: 10.0
...
```

## Atari experiments
### DQN

```python -m qlearn.atari.train_dqn --env BreakoutNoFrameskip-v4 --log-dir log_dir --save-dir save_dir --print-freq 10 --cuda 0```

Example of outcome

```
                          Options
                          env: BreakoutNoFrameskip-v4
                          seed: 42
                          replay_buffer_size: 1000000
                          lr: 0.0001
                          num_steps: 10000000
                          batch_size: 32
                          learning_freq: 4
                          target_update_freq: 10000
                          learning_starts: 50000
                          double_q: True
                          log_dir: log_dir
                          save_dir: save_dir
                          save_freq: 1000000
                          final_exploration: 0.1
                          final_exploration_frame: 1000000
                          print_freq: 10
                          run_index: None
                          cuda: 0
                          agent: DQN
                          discount: 0.99
                          model: None
WARNING:root:This caffe2 python run does not have GPU support. Will run in CPU only mode.
Writing logs to log_dir/BreakoutNoFrameskip-v4-DQN-seed-42-2019-06-26.1725
Logging to /var/folders/y8/gcb_hv6d7nd6t3ctvrmhf8t9s1xjd_/T/openai-2019-06-26-17-27-50-458572
------------------------------------
| % completion          | 0        |
| episodes              | 290      |
| exploration           | 0.953    |
| FPS                   | 34.4     |
| reward (100 epi mean) | 1.3      |
| total steps           | 51824    |
------------------------------------

ETA: 3 days and 8 hours

Saving model due to mean reward increase: None -> 1.3
------------------------------------
| % completion          | 0        |
| episodes              | 300      |
| exploration           | 0.952    |
| FPS                   | 29.6     |
| reward (100 epi mean) | 1.3      |
| total steps           | 53525    |
------------------------------------

ETA: 3 days and 21 hours
```

### MNF DQN

```python -m qlearn.atari.train_mnf_agent --env BreakoutNoFrameskip-v4 --alpha 0.01 --log-dir log_dir --save-dir save_dir  --print-freq 10 --cuda 0```

Example of outcome:

```
                         Options
                          env: BreakoutNoFrameskip-v4
                          seed: 42
                          replay_buffer_size: 1000000
                          lr: 0.0001
                          num_steps: 10000000
                          batch_size: 32
                          learning_freq: 4
                          target_update_freq: 10000
                          learning_starts: 50000
                          double_q: False
                          log_dir: log_dir
                          save_dir: save_dir
                          save_freq: 1000000
                          print_freq: 10
                          run_index: None
                          cuda: 0
                          agent: MNFDQN
                          discount: 0.99
                          hidden_dim: 50
                          n_hidden: 0
                          n_flows_q: 2
                          n_flows_r: 2
                          alpha: 0.01
                          model: None
WARNING:root:This caffe2 python run does not have GPU support. Will run in CPU only mode.
Writing logs to log_dir/BreakoutNoFrameskip-v4-MNFDQN-seed-42-alpha-0.01-2019-06-26.1730
Logging to /var/folders/y8/gcb_hv6d7nd6t3ctvrmhf8t9s1xjd_/T/openai-2019-06-26-17-32-20-718772
------------------------------------
| % completion          | 0        |
| episodes              | 270      |
| FPS                   | 34.7     |
| reward (100 epi mean) | 1.4      |
| total steps           | 50398    |
------------------------------------

ETA: 3 days and 7 hours

Saving model due to mean reward increase: None -> 1.4
------------------------------------
| % completion          | 0        |
| episodes              | 280      |
| FPS                   | 12.3     |
| reward (100 epi mean) | 1.4      |
| total steps           | 52433    |
------------------------------------
```

### Noisy DQN

```python -m qlearn.atari.train_noisy_agent --env BreakoutNoFrameskip-v4 --log-dir log_dir --save-dir save_dir```

### Bootstrapped DQN

```python -m qlearn.atari.train_bootstrapped_agent --env BreakoutNoFrameskip-v4 --log-dir log_dir --save-dir save_dir```

## License
This repo is CC-BY-NC licensed, as found in the LICENSE file.
