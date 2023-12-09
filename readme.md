See **model/kmodel.py** for memory implementation

Run for simple training-

  ```shell
  python mytrain.py
  ```

Or, 
to use  multigpu, run the lightning training code:

  ```shell
  python lgt_train.py
  ```

Used sweep from wandb to find the best hyperparameter using grid/random/bayes search.

  ```shell
  python sweep.py
  ```
