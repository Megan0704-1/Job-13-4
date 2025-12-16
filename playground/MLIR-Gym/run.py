# ruff: noqa
import os
import pdb
import random
import numpy as np

import gymnasium

import mlir_env
from mlir_env.utils.adaptors import ForwardAdaptor

import argparse

# e2e test
from scripts.test_e2e import sanity_run

# q learing
from scripts.configs import QConfig
from scripts.q_smoke import train_qtable_from_obs

# gnn train
from scripts.gnn.train import train
from scripts.gnn.eval import evaluate

def make_env(args):
    # create environment
    env = gymnasium.make("mlir_env/mlir-v5", cfg_path=args.cfg_path, ctx_path=args.ctx_path, state_cfg_path=args.state_cfg_path)
    env = ForwardAdaptor(env)
    return env

# ---------- cmd parsers --------- #
parser = argparse.ArgumentParser(description='parse filenames for config and context files to init mlir env')
# common parser requirements
common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--cfg-path', required=True, help='The path to the config file.')
common_parser.add_argument('--ctx-path', required=True, help='The path to the context file.')
common_parser.add_argument('--state-cfg-path', required=True, help='The path to the state config file.')
# subcommand parser
subparsers = parser.add_subparsers(dest="cmd")

# dispatch train function
def dispatch_train(args):
    print(">> Training...")
    env = make_env(args)
    train(env, args.ppo_cfg)
    print("<< Finish Training\n")

# train subcommand
train_parser = subparsers.add_parser("train", parents=[common_parser], help="Run training.")
train_parser.add_argument('--ppo-cfg', required=True, help="PPO config path")
train_parser.set_defaults(func=dispatch_train)

# dispatch eval function
def dispatch_eval(args):
    print(">> Evaluating...")
    env = make_env(args)
    evaluate(env,args.model_path,n_eval_episodes=20,deterministic=True)
    print("<< Finish Evaluating\n")

# eval subcommand
eval_parser = subparsers.add_parser("eval", parents=[common_parser], help="Run evaluation.")
eval_parser.add_argument('--model-path', required=True, help="PPO trained model path")
eval_parser.set_defaults(func=dispatch_eval)

# dispatch q learning function
def dispatch_q_learning(args):
    env = make_env(args)
    _ = train_qtable_from_obs(env, QConfig(episodes=500), args.out_path)
    print("last 10 returns:", [f"{r:+.3f}" for r in curve[-10:]])

# q_smoke subcommand
q_parser = subparsers.add_parser("q-learn", parents=[common_parser], help="Run Q-learning.")
q_parser.add_argument('--out-path', required=True, help='The path to the output log.')
q_parser.set_defaults(func=dispatch_q_learning)

# test subcommand
def dispatch_e2e(args):
    print(">> Testing e2e...")
    env = make_env(args)
    _ = sanity_run(env, seed=17, rounds=17)
    print("<< Finish testing e2e\n")

# e2e subcommand
e2e_parser = subparsers.add_parser("e2e", parents=[common_parser], help="E2E sanity run.")
e2e_parser.set_defaults(func=dispatch_e2e)

# global parser
parser.set_defaults(cmd="e2e", func=dispatch_e2e)
args = parser.parse_args()

if __name__ == "__main__":
    args.func(args)
