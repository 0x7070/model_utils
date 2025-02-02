### HELPER FUNCTIONS
# model_utils/info

import torch

def tens_shapes(*tensors: "torch.Tensor") -> None:
  """
  Makes use of `torch.Tensor` if it is imported.

  This function is based on the following operation:

  ```
  tensor.shape
  ```

  Args:
    - `tensors`: Variable number of `torch.Tensor`s
  Returns:
    (ordered) torch.Size()
  """
  for idx, tensor in enumerate(tensors):
    if isinstance(tensor, torch.Tensor):
      print(f"Shape of tensor at arg{idx}: {tensor.shape}")
    else:
      print(f"Error: Cannot call `torch.Tensor.shape` on variable at arg{idx}.")

def tens_op(*tensors: "torch.Tensor",
            func=torch.Tensor.size) -> None:
  """
  For use with:
  ```
  tensors = (X_train, X_test, y_train, y_test)
  tens_op(*tensors, torch.Tensor.size)
  tens_op(X_train, X_test, y_train, y_test, func=len)
  """
  print(f"Operation: {func.__name__}")
  for idx, tensor in enumerate(tensors):
    if isinstance(tensor, torch.Tensor):
      print(f"Result of operation on tensor at arg{idx}: {func(tensor)}")
    else:
      print(f"Error: Cannot call `{func.__name__}` on variable at arg{idx}.")

def likewise_op(*args: "torch.Tensor",
            func=type) -> None:
  """
  Executes `func` on all variables passed into the `args` parameter.

  All variables passed must be of a datatype operable by the provided `func`.
  """
  print(f"Operation: {func.__name__}")
  for idx, var in enumerate(args):
    try:
      print(f"Result of operation on var at arg{idx}: {func(var)}")
    except:
      print(f"Error: Cannot call `{func.__name__}` on variable at arg{idx}.")
