### HELPER FUNCTIONS
# utils.data_creation

def set_seed(random_seed: int) -> None:
  """
  Sets the `random_seed` to {random_seed} when given an INT.

  Args:
      `random_seed`: A +- INT value.

  Returns:
    `None`.

  Defaults:
    `235` when {random_seed} is not INT.
  """
  if type(random_seed) != int:
    print(f"Error: `random_seed` must be of type INT. Got {str(type(random_seed)).upper()} instead. Defaulting to 235.")
    set_seed(235)
  else:
    torch.manual_seed(random_seed)
    if torch.cuda.is_available:
      torch.cuda.manual_seed(random_seed)


def tt_split_as_numpy(train: None, test: None,
                      train_size: float = 0.8,
                      test_size: float = 0.2,
                      random_state: int = 235,
                      shuffle: bool = False) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
  """
  Makes use of `sklearn.model_selection.train_test_split` if it is imported.
  Returns 4 ordered values, `X_train, X_test, y_train, y_test`.

  This function **DOES NOT** return `Tensor`s.

  This function is based on the following usage of `train_test_split`:

  ```
  X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      train_size=0.8,
      test_size=0.2,
      random_state=RANDOM_SEED,
      shuffle=SHUFFLE
  )
  ```

  Args:
    - `train`: Dataset with features
    - `test`: Dataset with labels
    - `train_size`: Percentage used for train/test split creation (0.8==80%)
    - `test_size`: Percentage used for train/test split creation (0.2==20%)
    - `random_state`: Random seed for reproducability
    - `shuffle`: Whether or not to shuffle the data when creating the split
  Defaults:
    - `train_size`: 0.8
    - `test_size`: 0.2
    - `random_state`: 235
    - `shuffle`: False
  Returns:
    (ordered) X_train[np.ndarray], X_test[np.ndarray], y_train[np.ndarray], y_test[np.ndarray]
  
  Raises:
  """
  try:
    X_train, X_test, y_train, y_test = train_test_split(
      train, test,
      train_size=train_size,
      test_size=test_size,
      random_state=random_state,
      shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test
  except NameError as err:
    print(f"Error: `train_test_split` is unrecognized. Try importing `train_test_split` from `sklearn.model_selection`.")
    print(f"Error: {err}.")
    print("\nFull error:")
    raise

def tt_split_as_tensor():
  pass
