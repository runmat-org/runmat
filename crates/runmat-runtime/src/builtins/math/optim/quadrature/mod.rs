//! Shared adaptive quadrature services for `integral` and legacy `quad`.

mod algebra;
mod adaptive;
mod cooperation;
mod complex_algebra;
mod complex_array;
mod error_estimate;
mod geometry;
mod mesh;
mod open_algebra;
mod open_rule;

pub(crate) use adaptive::{integrate, AdaptiveSimpsonResult, QuadratureEvaluator};
pub(crate) use algebra::QuadratureArithmeticError;
pub(crate) use complex_algebra::ScaledComplex;
pub(crate) use cooperation::{QuadratureCooperation, QuadratureIdentity};
pub(crate) use geometry::{midpoint, ScaledEstimate};
pub(crate) use open_rule::{integrate_open, OpenDomain, OpenSegment};
