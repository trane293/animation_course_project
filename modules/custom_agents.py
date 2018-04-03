def DDPG_Agent_Network():
    agent = {
            'actions_exploration': {'mu': 0.0,
                                    'sigma': 0.3,
                                    'theta': 0.15,
                                    'type': 'ornstein_uhlenbeck'},
            'critic_network': {'size_t0': 64,
                               'size_t1': 64},
            'critic_optimizer': {'learning_rate': 0.001,
                                 'type': 'adam'},
            'discount': 0.99,
            'entropy_regularization': None,
            'execution': {'distributed_spec': None,
                          'session_config': None,
                          'type': 'single'},
            'memory': {'capacity': 100000,
                       'include_next_states': True,
                       'type': 'replay'},
            'optimizer': {'learning_rate': 0.0001,
                          'type': 'adam'},
            'saver': {'directory': None,
                      'seconds': 600},
            'summarizer': {'directory': None,
                           'labels': [],
                           'seconds': 120},
            'target_sync_frequency': 1,
            'target_update_weight': 0.999,
            'type': 'ddpg_agent',
            'update_mode': {'batch_size': 64,
                            'frequency': 64,
                            'unit': 'timesteps'}
        }

    # Define the networks to be used in DDPG. Although DDPG uses two networks (actor-critic), we'll use the
    # same configuration for both as per the original paper
    network = [
        {'size': 64,
         'type': 'linear'},
        {'layer': 'batch_normalization',
         'type': 'tf_layer'},
        {'name': 'relu',
         'type': 'nonlinearity'},
        {'size': 64,
         'type': 'linear'},
        {'layer': 'batch_normalization',
         'type': 'tf_layer'},
        {'name': 'relu',
         'type': 'nonlinearity'},
        {'activation': None,
         'size': 64,
         'type': 'dense'}
    ]

    return agent, network

def PPO_Agent_Network():
    agent = {
             'baseline': {'sizes': [32, 32],
                          'type': 'mlp'},
             'baseline_mode': 'states',
             'baseline_optimizer': {'num_steps': 5,
                                    'optimizer': {'learning_rate': 0.001,
                                                  'type': 'adam'},
                                    'type': 'multi_step'},
             'discount': 0.99,
             'entropy_regularization': 0.01,
             'execution': None,
             'gae_lambda': None,
             'likelihood_ratio_clipping': 0.2,
             'memory': {'capacity': 5000,
                        'include_next_states': False,
                        'type': 'latest'},
             'optimization_steps': 50,
             'saver': {'directory': None,
                       'seconds': 600},
             'step_optimizer': {'learning_rate': 0.001,
                                'type': 'adam'},
             'subsampling_fraction': 0.1,
             'summarizer': {'directory': None,
                            'labels': [],
                            'seconds': 120},
             'type': 'ppo_agent',
             'update_mode': {'batch_size': 10,
                             'frequency': 10,
                             'unit': 'episodes'}
    }

    network = [
        {'activation': 'tanh',
         'size': 64,
         'type': 'dense'},
        {'activation': 'tanh',
         'size': 64,
         'type': 'dense'}
    ]

    return agent, network


def TRPO_Agent_Network():
    agent = {
         'baseline': None,
         'baseline_mode': None,
         'baseline_optimizer': None,
         'discount': 0.99,
         'entropy_regularization': None,
         'execution': None,
         'gae_lambda': None,
         'learning_rate': 0.01,
         'likelihood_ratio_clipping': None,
         'memory': {'capacity': 5000,
                    'include_next_states': False,
                    'type': 'latest'},
         'saver': {'directory': None,
                   'seconds': 600},
         'summarizer': {'directory': None,
                        'labels': [],
                        'seconds': 120},
         'type': 'trpo_agent',
         'update_mode': {'batch_size': 20,
                         'frequency': 20,
                         'unit': 'episodes'}}

    network = [
        {'activation': 'tanh',
         'size': 64,
         'type': 'dense'},
        {'activation': 'tanh',
         'size': 64,
         'type': 'dense'}
    ]

    return agent, network


def VPG_Agent_Network():
    agent = {
            "type": "vpg_agent",

            "update_mode": {
                "unit": "episodes",
                "batch_size": 20,
                "frequency": 20
            },
            "memory": {
                "type": "latest",
                "include_next_states": False,
                "capacity": 5000
            },

            "optimizer": {
                "type": "adam",
                "learning_rate": 2e-2
            },

            "discount": 0.99,
            "entropy_regularization": None,
            "gae_lambda": None,

            "baseline_mode": "states",
            "baseline": {
                "type": "mlp",
                "sizes": [32, 32]
            },
            "baseline_optimizer": {
                "type": "multi_step",
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 1e-3
                },
                "num_steps": 5
            },

            "saver": {
                "directory": None,
                "seconds": 600
            },
            "summarizer": {
                "directory": None,
                "labels": [],
                "seconds": 120
            },
            "execution": {
                "type": "single",
                "session_config": None,
                "distributed_spec": None
            }
        }

    network = [
        {'activation': 'tanh',
         'size': 64,
         'type': 'dense'},
        {'activation': 'tanh',
         'size': 64,
         'type': 'dense'}
    ]

    return agent, network


def NAF_Agent_Network():
    agent = {
            "type": "naf_agent",

            "update_mode": {
                "unit": "timesteps",
                "batch_size": 64,
                "frequency": 4
            },
            "memory": {
                "type": "replay",
                "capacity": 10000,
                "include_next_states": True
            },

            "optimizer": {
              "type": "adam",
              "learning_rate": 1e-3
            },

            "discount": 0.99,
            "entropy_regularization": None,
            "double_q_model": True,

            "target_sync_frequency": 1000,
            "target_update_weight": 1.0,

            "actions_exploration": {
                "type": "ornstein_uhlenbeck",
                "sigma": 0.2,
                "mu": 0.0,
                "theta": 0.15
            },

            "saver": {
                "directory": None,
                "seconds": 600
            },
            "summarizer": {
                "directory": None,
                "labels": [],
                "seconds": 120
            }
        }

    network = [
        {'activation': 'tanh',
         'size': 64,
         'type': 'dense'},
        {'activation': 'tanh',
         'size': 64,
         'type': 'dense'}
    ]

    return agent, network