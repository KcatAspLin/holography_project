"""Visual-field mappings derived from atlas registration data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor


__all__ = ["AllenAffineVisualFieldMap", "IdentityVisualFieldMap"]


class AllenAffineVisualFieldMap(torch.nn.Module):
    """Affine map from V1 coordinates to visual-field coordinates.

    The repository stores cortical positions in the existing ``space`` coordinate
    system, typically micrometers. This module intentionally does not convert
    units; the supplied affine map must already map from that coordinate system
    to the desired visual-field coordinate system.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        matrix: Sequence[Sequence[float]] | Tensor | None = None,
        offset: Sequence[float] | Tensor | None = None,
    ):
        super().__init__()

        if path is not None:
            if matrix is not None or offset is not None:
                raise ValueError("Specify either path or matrix/offset, not both.")
            matrix, offset = self._load(path)

        if matrix is None:
            raise ValueError("matrix must be provided when path is not provided.")

        matrix = torch.as_tensor(matrix, dtype=torch.get_default_dtype())
        if matrix.ndim != 2 or matrix.shape[0] != 2:
            raise ValueError(
                "matrix must have shape (2, d), mapping d-dimensional V1 space "
                f"to 2-dimensional visual-field coordinates, but got {matrix.shape}."
            )

        if offset is None:
            offset = torch.zeros(2, dtype=matrix.dtype)
        offset = torch.as_tensor(offset, dtype=matrix.dtype)
        if offset.shape != (2,):
            raise ValueError(f"offset must have shape (2,), but got {offset.shape}.")

        self.register_buffer("matrix", matrix, persistent=False)
        self.register_buffer("offset", offset, persistent=False)

    @staticmethod
    def _load(path: str | Path) -> tuple[Tensor, Tensor]:
        path = Path(path)
        if path.suffix.lower() == ".json":
            with path.open() as f:
                data = json.load(f)
            return torch.as_tensor(data["matrix"]), torch.as_tensor(data.get("offset", [0, 0]))

        if path.suffix.lower() == ".csv":
            with path.open(newline="") as f:
                row = next(csv.DictReader(f))
            matrix = [
                [float(row["matrix_00"]), float(row["matrix_01"])],
                [float(row["matrix_10"]), float(row["matrix_11"])],
            ]
            offset = [float(row.get("offset_0", 0.0)), float(row.get("offset_1", 0.0))]
            return torch.as_tensor(matrix), torch.as_tensor(offset)

        raise ValueError(f"Unsupported visual-field map format: {path.suffix}")

    def forward(self, space: Tensor) -> Tensor:
        if space.shape[-1] != self.matrix.shape[1]:
            raise ValueError(
                f"Expected space.shape[-1] == {self.matrix.shape[1]}, "
                f"but got {space.shape[-1]}."
            )
        matrix = self.matrix.to(dtype=space.dtype, device=space.device)
        offset = self.offset.to(dtype=space.dtype, device=space.device)
        return space @ matrix.T + offset


class IdentityVisualFieldMap(AllenAffineVisualFieldMap):
    """Debug helper that treats the first two V1 coordinates as visual field."""

    def __init__(self):
        super().__init__(matrix=[[1.0, 0.0], [0.0, 1.0]], offset=[0.0, 0.0])
