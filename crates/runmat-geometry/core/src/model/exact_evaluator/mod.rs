mod api;
mod definition_validation;
mod definition_validation_math;
mod ids;
mod incidence_consistency;
mod nurbs_validation;
mod portable;
mod registry;
mod surface_differential;
mod validation;

pub use api::*;
pub use ids::*;
pub use incidence_consistency::validate_exact_incidence;
pub use portable::*;
pub use registry::*;
pub use surface_differential::{surface_principal_curvature, surface_unit_normal};

#[cfg(test)]
pub(crate) mod tests;
