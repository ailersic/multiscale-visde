import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype
# ruff: noqa: F821, F722

def check_nn_dims(net, in_shapes, out_shapes, error_msg=None):
    """Check the dimensions of a neural network"""

    if error_msg is None:
        error_msg = "check_nn_dims"
    
    n_batch = 42
    test_inputs = [torch.zeros(n_batch, *shape) for shape in in_shapes]
    try:
        test_outputs = net(*test_inputs)
    except Exception as e:
        raise ValueError(f"{error_msg}: Cannot evaluate nn Module with inputs of shape {[t.shape for t in test_inputs]}, got exception {e}")

    if not isinstance(test_outputs, tuple):
        test_outputs = (test_outputs,)
    
    assert len(out_shapes) == len(test_outputs), f"{error_msg}: Expected {len(out_shapes)} outputs, got {len(test_outputs)}"
    for i, shape in enumerate(out_shapes):
        assert test_outputs[i].shape == torch.Size([n_batch, *shape]), f"{error_msg}: Expected output {i} to have shape {torch.Size([n_batch, *shape])}, got {test_outputs[i].shape}"
        
@jaxtyped(typechecker=beartype)
def linterp(y_batch: Float[Tensor, "n_batch dim_y"],
           t_batch: Float[Tensor, "n_batch"],
           t_query: Float[Tensor, "n_query"]
) -> Float[Tensor, "n_query dim_y"]:
    """Linearly interpolate y_batch at t_query"""

    n_batch = t_batch.shape[0]
    n_query = t_query.shape[0]
    dim_y = y_batch.shape[-1]

    y_query = torch.zeros(n_query, dim_y, device=y_batch.device)
    for i in range(n_query):
        t = t_query[i]
        idx = torch.searchsorted(t_batch, t)
        if idx == 0:
            y_query[i] = y_batch[0]
        elif idx == n_batch:
            y_query[i] = y_batch[-1]
        else:
            t0, t1 = t_batch[idx - 1], t_batch[idx]
            y0, y1 = y_batch[idx - 1], y_batch[idx]
            y_query[i] = y0 + (y1 - y0) * (t - t0) / (t1 - t0)
    
    return y_query