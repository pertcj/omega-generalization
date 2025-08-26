"""Example script to train and evaluate a network."""
import argparse

import haiku as hk
import jax.numpy as jnp
import jax
import numpy as np

from omega_generalization.training import constants
from omega_generalization.training import curriculum as curriculum_lib
from omega_generalization.training import training

import logging
logging.basicConfig(
  format='%(asctime)s %(levelname)-8s %(message)s',
  level=logging.INFO,
  datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--sequence_length", default=64, type=int)
parser.add_argument("--hidden_dim", default=256, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--steps", default=100_000, type=int)
parser.add_argument("--optimizer", default="amsgrad", type=str)
parser.add_argument("--max_range_test_length", default=512, type=int)
parser.add_argument("--benchmark", type=str, default="lift", choices=["lift", "acacia"])
parser.add_argument("--train_balance", action="store_true", help="Measure the balance of training data")
parser.add_argument("--test_balance", action="store_true", help="Measure the balance of test data")
parser.add_argument("--use_tensorboard", action="store_true", help="Log experiment to TensorBoard")
args = parser.parse_args()
logging.info(args)

def main() -> None:
  
  architecture_params = {
    'hidden_dim': args.hidden_dim,
  }
  
  # Create the task.
  curriculum = curriculum_lib.UniformCurriculum(
      values=list(range(2, args.sequence_length + 1)))
  if args.benchmark == "lift":
    task = constants.TASK_BUILDERS["ltl"](constants.LIFT_FORMULAS[args.task])
  elif args.benchmark == "acacia":
    task = constants.TASK_BUILDERS["ltl"](constants.ACACIA_FORMULAS[args.task])
  else:
    raise ValueError(f"Unknown benchmark: {args.benchmark}")

  def to_json_serializable(obj):
    if isinstance(obj, jnp.ndarray):
      return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")
  if args.train_balance:
    ratios = task.evaluate_balance(
        rng=jax.random.PRNGKey(args.seed),
        min_length=2,
        max_length=args.sequence_length,
        step=1,
        batch_size=args.batch_size,
        samples=512)
    # save to json
    ratios = {k: to_json_serializable(v) for k, v in ratios.items()}
    import json
    with open(f"data/{args.task}_train_balance.json", "w") as f:
      json.dump(ratios, f, indent=4)
    logging.info(f"Balance ratios saved to {args.task}_train_balance.json")
    exit()
  elif args.test_balance:
    ratios = task.evaluate_balance(
        rng=jax.random.PRNGKey(args.seed),
        min_length=args.sequence_length + 1,
        max_length=args.max_range_test_length,
        step=1,
        batches=8,
        batch_size=64,
        samples=512)
    # save to json
    ratios = {k: to_json_serializable(v) for k, v in ratios.items()}
    import json
    with open(f"data/{args.task}_test_balance.json", "w") as f:
      json.dump(ratios, f, indent=4)
    logging.info(f"Balance ratios saved to {args.task}_test_balance.json")
    exit()

  model = constants.MODEL_BUILDERS["rnn"](
    output_size=task.output_size,
    hidden_size=architecture_params['hidden_dim'],
  )
  eval_model = model

  model = hk.transform(model)
  eval_model = hk.transform(eval_model)

  def loss_fn(output, target):
    loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
    return loss, {}

  def accuracy_fn(output, target):
    mask = task.accuracy_mask(target)
    return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

  # Create the final training parameters.
  training_params = training.ClassicTrainingParams(
      seed=0,
      model_init_seed=args.seed,
      training_steps=args.steps,
      log_frequency=1000,
      l2_lambda=1e-3,
      length_curriculum=curriculum,
      batch_size=args.batch_size,
      task=task,
      model=model,
      task_str=args.task,
      eval_model=eval_model,
      loss_fn=loss_fn,
      learning_rate=args.lr,
      accuracy_fn=accuracy_fn,
      compute_full_range_test=True,
      max_range_test_length=args.max_range_test_length,
      range_test_total_batch_size=512,
      range_test_sub_batch_size=64,
      architecture_params=architecture_params,
      optimizer=args.optimizer,
      use_tensorboard=args.use_tensorboard,)
  
  training_worker = training.TrainingWorker(training_params, use_tqdm=False)
  _, eval_results, _ = training_worker.run()

  # Gather results and print final score.
  accuracies = [r['accuracy'] for r in eval_results]
  score = np.mean(accuracies[args.sequence_length + 1:])
  print(f'OOD accuracy: {score}')
  id_score = np.mean(accuracies[:args.sequence_length + 1])
  print(f"ID accuracy: {id_score}")

if __name__ == '__main__':
  main()
