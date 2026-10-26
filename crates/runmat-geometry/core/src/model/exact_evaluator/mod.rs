mod allocation;
mod allocation_scope;
mod api;
mod definition_validation;
mod definition_validation_math;
mod ids;
mod incidence_consistency;
mod nurbs_validation;
mod parameter_range;
mod parameter_frame;
mod portable;
mod registry;
mod surface_differential;
mod validation;

pub use allocation::GeometryAllocationReservation;
pub use allocation_scope::GeometryAllocationScope;
pub use api::*;
pub use ids::*;
pub use incidence_consistency::validate_exact_incidence;
pub use parameter_frame::*;
pub use portable::*;
pub use registry::*;
pub use surface_differential::{surface_principal_curvature, surface_unit_normal};

#[cfg(test)]
pub(crate) mod tests;
