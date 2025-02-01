### HELPER FUNCTIONS
# model_utils/info

def tens_shapes(*tensors: torch.Tensor) -> None:
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
