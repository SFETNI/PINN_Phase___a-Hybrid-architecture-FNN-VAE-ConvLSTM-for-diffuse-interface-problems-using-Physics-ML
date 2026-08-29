"""Explicit ConvGRU and ConvLSTM recurrent cells.

Both cells are clean, correctly named implementations that can be used
independently in architecture ablations (WP4). Neither imports the historical
``legacy/Modules/ConvLSTMCell.py``.

ConvGRUCell
-----------
The active recurrent architecture in the legacy prototype is GRU-style
(Modules/ConvLSTMCell.py lines 29-91, confirmed by the code comments there).
This class replicates it with the correct name.

    r_t  = sigmoid(W_xr * x + W_hr * h)           # reset gate
    z_t  = sigmoid(W_xz * x + W_hz * h)           # update gate
    h̃_t = tanh(W_xh * x + W_hh * (r_t ⊙ h))     # candidate
    h_t  = (1 - z_t) ⊙ h + z_t ⊙ h̃_t            # new hidden state

State:  h  (hidden only; no cell state)

ConvLSTMCell
------------
A true LSTM cell with input, forget, output, and cell-state gates.

    i_t = sigmoid(W_xi * x + W_hi * h)            # input gate
    f_t = sigmoid(W_xf * x + W_hf * h)            # forget gate
    o_t = sigmoid(W_xo * x + W_ho * h)            # output gate
    g_t = tanh(W_xg * x + W_hg * h)               # cell candidate
    c_t = f_t ⊙ c + i_t ⊙ g_t                    # cell state
    h_t = o_t ⊙ tanh(c_t)                         # hidden state

State:  (h, c)

The historical prototype labelled its GRU-style cell as ConvLSTM. This module
keeps the two recurrent mechanisms separate and names them by implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


_PADDING_MODES = {"zeros", "circular", "reflect", "replicate"}
_GATE_FUSIONS = {"separate", "concat"}


class ConvGRUCell(nn.Module):
    """Single-step 2D Convolutional GRU cell.

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    hidden_channels:
        Number of hidden state channels.
    kernel_size:
        Spatial kernel size for all convolutions (square; must be odd).
    padding_mode:
        Padding mode for all Conv2d layers.  ``"zeros"`` (default) is
        backward-compatible.  ``"circular"`` matches periodic-domain physics
        (periodic Laplacian in losses.py uses ``torch.roll`` for the same
        boundary condition).  Parameter shapes are identical for all modes, so
        existing checkpoints remain loadable regardless of which mode is chosen
        at inference/training time.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding_mode: str = "zeros",
        gate_fusion: str = "separate",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-size padding")
        if padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode must be 'zeros', 'circular', 'reflect', or 'replicate'; "
                f"got {padding_mode!r}"
            )
        if gate_fusion != "separate":
            raise ValueError("ConvGRUCell currently supports only gate_fusion='separate'")
        self.hidden_channels = hidden_channels
        self.padding_mode = padding_mode
        self.gate_fusion = gate_fusion
        padding = kernel_size // 2

        # Gates operating on input
        self.W_xr = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.W_xz = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.W_xh = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        # Gates operating on hidden state
        self.W_hr = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
        self.W_hz = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
        self.W_hh = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """Advance one time step.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, in_channels, H, W)``.
        h:
            Previous hidden state of shape ``(batch, hidden_channels, H, W)``.

        Returns
        -------
        h_new:
            New hidden state of shape ``(batch, hidden_channels, H, W)``.
        """

        r = torch.sigmoid(self.W_xr(x) + self.W_hr(h))
        z = torch.sigmoid(self.W_xz(x) + self.W_hz(h))
        h_tilde = torch.tanh(self.W_xh(x) + self.W_hh(r * h))
        return (1.0 - z) * h + z * h_tilde

    def initial_state(self, batch: int, height: int, width: int, device: torch.device | None = None) -> Tensor:
        """Return a zero hidden state compatible with ``forward``."""
        return torch.zeros(batch, self.hidden_channels, height, width, device=device)


class ConvLSTMCell(nn.Module):
    """Single-step 2D Convolutional LSTM cell.

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    hidden_channels:
        Number of hidden and cell-state channels.
    kernel_size:
        Spatial kernel size for all convolutions (square; must be odd).
    padding_mode:
        Padding mode for all Conv2d layers.  ``"zeros"`` (default) is
        backward-compatible.  ``"circular"`` matches periodic-domain physics
        (periodic Laplacian in losses.py uses ``torch.roll`` for the same
        boundary condition).  Parameter shapes are identical for all modes, so
        existing checkpoints remain loadable regardless of which mode is chosen
        at inference/training time.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding_mode: str = "zeros",
        gate_fusion: str = "separate",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-size padding")
        if padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode must be 'zeros', 'circular', 'reflect', or 'replicate'; "
                f"got {padding_mode!r}"
            )
        if gate_fusion not in _GATE_FUSIONS:
            raise ValueError(
                f"gate_fusion must be 'separate' or 'concat'; got {gate_fusion!r}"
            )
        self.hidden_channels = hidden_channels
        self.padding_mode = padding_mode
        self.gate_fusion = gate_fusion
        padding = kernel_size // 2

        if gate_fusion == "concat":
            self.W_gates = nn.Conv2d(
                in_channels + hidden_channels,
                4 * hidden_channels,
                kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            )
        else:
            self.W_xi = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
            self.W_xf = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
            self.W_xo = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
            self.W_xg = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)

            self.W_hi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
            self.W_hf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
            self.W_ho = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
            self.W_hg = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one time step.

        Parameters
        ----------
        x:
            Input of shape ``(batch, in_channels, H, W)``.
        h:
            Previous hidden state of shape ``(batch, hidden_channels, H, W)``.
        c:
            Previous cell state of shape ``(batch, hidden_channels, H, W)``.

        Returns
        -------
        h_new, c_new:
            Updated hidden and cell states, both of shape
            ``(batch, hidden_channels, H, W)``.
        """

        if self.gate_fusion == "concat":
            gates = self.W_gates(torch.cat((x, h), dim=1))
            i_raw, f_raw, o_raw, g_raw = gates.chunk(4, dim=1)
            i = torch.sigmoid(i_raw)
            f = torch.sigmoid(f_raw)
            o = torch.sigmoid(o_raw)
            g = torch.tanh(g_raw)
        else:
            i = torch.sigmoid(self.W_xi(x) + self.W_hi(h))
            f = torch.sigmoid(self.W_xf(x) + self.W_hf(h))
            o = torch.sigmoid(self.W_xo(x) + self.W_ho(h))
            g = torch.tanh(self.W_xg(x) + self.W_hg(h))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def initial_state(
        self, batch: int, height: int, width: int, device: torch.device | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return zero (h, c) states compatible with ``forward``."""
        zeros = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        return zeros, zeros.clone()


class ConvGRUCell3d(nn.Module):
    """Single-step 3D Convolutional GRU cell using ``nn.Conv3d``.

    Accepts inputs of shape ``(batch, in_channels, D, H, W)`` and returns
    hidden states of shape ``(batch, hidden_channels, D, H, W)``.

    This is the 3D analogue of :class:`ConvGRUCell` and must be used when the
    phase field is a 3D tensor. Using the 2D cell on 3D data silently
    misinterprets the depth axis as a batch or channel dimension.

    State: h only (no cell state, same as :class:`ConvGRUCell`).

    Parameters
    ----------
    padding_mode:
        Padding mode for all Conv3d layers.  ``"zeros"`` (default) is
        backward-compatible.  ``"circular"`` matches periodic-domain physics.
        Callers must select circular padding explicitly for periodic 3D
        benchmarks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding_mode: str = "zeros",
        gate_fusion: str = "separate",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-size padding")
        if padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode must be 'zeros', 'circular', 'reflect', or 'replicate'; "
                f"got {padding_mode!r}"
            )
        if gate_fusion != "separate":
            raise ValueError("ConvGRUCell3d currently supports only gate_fusion='separate'")
        self.hidden_channels = hidden_channels
        self.padding_mode = padding_mode
        self.gate_fusion = gate_fusion
        padding = kernel_size // 2

        self.W_xr = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.W_xz = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.W_xh = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.W_hr = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
        self.W_hz = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
        self.W_hh = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """Advance one time step.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, in_channels, D, H, W)``.
        h:
            Previous hidden state of shape ``(batch, hidden_channels, D, H, W)``.

        Returns
        -------
        h_new:
            New hidden state of shape ``(batch, hidden_channels, D, H, W)``.
        """
        if x.ndim != 5:
            raise ValueError("ConvGRUCell3d requires 5D input (batch, C, D, H, W)")
        r = torch.sigmoid(self.W_xr(x) + self.W_hr(h))
        z = torch.sigmoid(self.W_xz(x) + self.W_hz(h))
        h_tilde = torch.tanh(self.W_xh(x) + self.W_hh(r * h))
        return (1.0 - z) * h + z * h_tilde

    def initial_state(
        self,
        batch: int,
        depth: int,
        height: int,
        width: int,
        device: torch.device | None = None,
    ) -> Tensor:
        """Return a zero hidden state of shape ``(batch, hidden_channels, D, H, W)``."""
        return torch.zeros(batch, self.hidden_channels, depth, height, width, device=device)


class ConvLSTMCell3d(nn.Module):
    """Single-step 3D Convolutional LSTM cell using ``nn.Conv3d``.

    Accepts inputs of shape ``(batch, in_channels, D, H, W)`` and returns
    hidden/cell states of shape ``(batch, hidden_channels, D, H, W)``.

    This is the 3D analogue of :class:`ConvLSTMCell`. Required for learned
    operator-split training on 3D phase-field grids. The 2D cell silently
    treats the depth dimension as part of the (H, W) spatial axes, producing
    incorrect spatial convolutions.

    State: (h, c) — hidden and cell state tensors.

    Parameters
    ----------
    padding_mode:
        Padding mode for all Conv3d layers.  ``"zeros"`` (default) is
        backward-compatible.  ``"circular"`` matches periodic-domain physics.
        Callers must select circular padding explicitly for periodic 3D
        benchmarks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding_mode: str = "zeros",
        gate_fusion: str = "separate",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-size padding")
        if padding_mode not in _PADDING_MODES:
            raise ValueError(
                f"padding_mode must be 'zeros', 'circular', 'reflect', or 'replicate'; "
                f"got {padding_mode!r}"
            )
        if gate_fusion not in _GATE_FUSIONS:
            raise ValueError(
                f"gate_fusion must be 'separate' or 'concat'; got {gate_fusion!r}"
            )
        self.hidden_channels = hidden_channels
        self.padding_mode = padding_mode
        self.gate_fusion = gate_fusion
        padding = kernel_size // 2

        if gate_fusion == "concat":
            self.W_gates = nn.Conv3d(
                in_channels + hidden_channels,
                4 * hidden_channels,
                kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            )
        else:
            self.W_xi = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
            self.W_xf = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
            self.W_xo = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
            self.W_xg = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)

            self.W_hi = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
            self.W_hf = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
            self.W_ho = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
            self.W_hg = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one time step.

        Parameters
        ----------
        x:
            Input of shape ``(batch, in_channels, D, H, W)``.
        h:
            Previous hidden state of shape ``(batch, hidden_channels, D, H, W)``.
        c:
            Previous cell state of shape ``(batch, hidden_channels, D, H, W)``.

        Returns
        -------
        h_new, c_new:
            Updated hidden and cell states, both of shape
            ``(batch, hidden_channels, D, H, W)``.
        """
        if x.ndim != 5:
            raise ValueError("ConvLSTMCell3d requires 5D input (batch, C, D, H, W)")
        if self.gate_fusion == "concat":
            gates = self.W_gates(torch.cat((x, h), dim=1))
            i_raw, f_raw, o_raw, g_raw = gates.chunk(4, dim=1)
            i = torch.sigmoid(i_raw)
            f = torch.sigmoid(f_raw)
            o = torch.sigmoid(o_raw)
            g = torch.tanh(g_raw)
        else:
            i = torch.sigmoid(self.W_xi(x) + self.W_hi(h))
            f = torch.sigmoid(self.W_xf(x) + self.W_hf(h))
            o = torch.sigmoid(self.W_xo(x) + self.W_ho(h))
            g = torch.tanh(self.W_xg(x) + self.W_hg(h))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def initial_state(
        self,
        batch: int,
        depth: int,
        height: int,
        width: int,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return zero ``(h, c)`` states of shape ``(batch, hidden_channels, D, H, W)``."""
        zeros = torch.zeros(batch, self.hidden_channels, depth, height, width, device=device)
        return zeros, zeros.clone()
