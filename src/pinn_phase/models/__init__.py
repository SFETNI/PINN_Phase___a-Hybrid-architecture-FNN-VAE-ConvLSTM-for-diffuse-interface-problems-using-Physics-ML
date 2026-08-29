"""Neural model implementations for the PINN-Phase architecture families."""

from .cells import ConvGRUCell, ConvLSTMCell, ConvGRUCell3d, ConvLSTMCell3d
from .hybrid_rollout import (
    PINNPhaseHybridRollout,
    PINNPhaseHybridRollout3d,
    PointwiseANNIncrement,
    PointwiseANNIncrement3d,
    RecurrentState,
    detach_recurrent_state,
)
from .explicit_mpf import (
    ExplicitMPFGraph,
    ExplicitMPFGraphExtractor,
    ExplicitMPFHybridRollout,
    MPFRecurrentState,
    MultiPhasePointwiseANN,
    SimpleGraphConditioner,
    detach_mpf_recurrent_state,
)
from .perm_equivariant_mpf import (
    PermEquivariantMPFRollout,
    SharedPhaseEncoder,
)
from .local_rollout import LocalConvGRURollout
from .residual_blend import ResidualBlend
from .factory import ArchitectureFamily, CheckpointIdentity, architecture_family_from_config
from .projection import (
    BOUND_ENFORCEMENT_MODES,
    CLAMP_ONLY,
    DEFAULT_BOUND_ENFORCEMENT_MODE,
    DEFAULT_BOUNDARY_EPS,
    PROJECT_AND_CLAMP,
    UNBOUNDED,
    apply_bound_enforcement,
    projection_diagnostic_stats,
    project_outward_boundary_rates,
)

__all__ = [
    "ConvGRUCell",
    "ConvLSTMCell",
    "ConvGRUCell3d",
    "ConvLSTMCell3d",
    "PINNPhaseHybridRollout",
    "PINNPhaseHybridRollout3d",
    "PointwiseANNIncrement",
    "PointwiseANNIncrement3d",
    "ExplicitMPFGraph",
    "ExplicitMPFGraphExtractor",
    "ExplicitMPFHybridRollout",
    "MPFRecurrentState",
    "MultiPhasePointwiseANN",
    "SimpleGraphConditioner",
    "detach_mpf_recurrent_state",
    "RecurrentState",
    "detach_recurrent_state",
    "PermEquivariantMPFRollout",
    "SharedPhaseEncoder",
    "LocalConvGRURollout",
    "ResidualBlend",
    "projection_diagnostic_stats",
    "project_outward_boundary_rates",
    "apply_bound_enforcement",
    "BOUND_ENFORCEMENT_MODES",
    "CLAMP_ONLY",
    "DEFAULT_BOUND_ENFORCEMENT_MODE",
    "DEFAULT_BOUNDARY_EPS",
    "PROJECT_AND_CLAMP",
    "UNBOUNDED",
    "ArchitectureFamily",
    "CheckpointIdentity",
    "architecture_family_from_config",
]
