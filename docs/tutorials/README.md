# PINN-Phase tutorials

This eight-part curriculum teaches the physics before the release machinery.
Read it in order if PINN-Phase is new to you. Each page states its prerequisites,
marks illustrative calculations, and links to the accepted implementation or
machine record behind its claims. No optional external assets are required.

| Order | Tutorial | Learning objective | Runtime |
|---:|---|---|---|
| 1 | [Phase-Field Foundations for PINN-Phase](01_phase_field_foundations.md) | Understand diffuse interfaces, scalar and explicit MPF states, admissibility, curvature, and topology change. | Reading only |
| 2 | [The Multiphase-Field Operator](02_explicit_mpf_operator.md) | Derive the projected physical rate and calculate a zero-sum example. | Reading only |
| 3 | [Inside PINN-Phase: Local and Spatial Dynamics](03_inside_pinn_phase.md) | Follow the first-generation local and recurrent branches through their tensor shapes. | Reading only |
| 4 | [One Complete PINN-Phase Time Step](04_one_pinn_phase_step.md) | Distinguish bounded rate projection, strict and thresholded state maps, and the auxiliary penalty. | Reading only |
| 5 | [How PINN-Phase Learns from the Physics](05_physics_informed_training.md) | Understand the physical target, mixed loss, interface weighting, reference isolation, and TBPTT. | Reading only |
| 6 | [Symmetry, Phase Labels, and Model Families](06_symmetry_and_model_families.md) | Explain phase-permutation equivariance and place each result in the correct architecture family. | Seconds |
| 7 | [Long-Horizon Evolution and Topology](07_long_horizon_and_topology.md) | Read field, persistence, survivor, extinction, and timing metrics together. | Reading only |
| 8 | [Run, Replay, and Reproduce](08_run_and_reproduce.md) | Run lightweight checks and distinguish recomputation, replay, and provenance. | Seconds to minutes |

Hands-on guides and the N16 case study have distinct roles outside this sequence.
Use the complete [`docs/README.md`](../README.md) reader map to find them.
