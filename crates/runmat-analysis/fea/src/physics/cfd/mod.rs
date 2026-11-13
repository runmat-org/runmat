//! Incompressible finite-volume formulation and shared fluid-domain behavior.

pub mod canonical;
mod diagnostics;
mod domain;
mod interface;
mod recovery;
mod solve;

pub use diagnostics::{assembly_diagnostic, coupling_diagnostics, CfdCouplingEvidence};
pub use domain::{
    realize_velocity_schedule, reynolds_number_for_velocity, validate_authored_boundary_conditions,
    CfdDomainTopology, CfdVelocitySchedule,
};
pub use interface::{
    cell_centered_scalar, connectivity_coverage_ratio as interface_connectivity_coverage_ratio,
    face_count as interface_face_count, graph_edges_for_topology as interface_graph_edges,
    mean_axial_velocity, pressure_drop, resample_scalar_profile,
};
pub use recovery::recover_velocity_pressure;
pub use solve::residual_norms;
