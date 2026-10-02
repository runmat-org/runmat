//! Shared adaptive quadrature services for `integral` and legacy `quad`.

mod algebra;
mod adaptive;
mod cooperation;
mod geometry;
mod global_mesh;

pub(crate) use adaptive::{integrate, integrate_array, AdaptiveSimpsonResult, QuadratureEvaluator};
pub(crate) use algebra::{ArrayEstimate, QuadratureArithmeticError};
pub(crate) use cooperation::{QuadratureCooperation, QuadratureIdentity};
pub(crate) use geometry::{midpoint, ScaledEstimate};
pub(crate) use global_mesh::integrate_mesh;
