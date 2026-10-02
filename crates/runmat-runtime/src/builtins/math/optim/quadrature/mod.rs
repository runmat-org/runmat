//! Shared adaptive quadrature services for `integral` and legacy `quad`.

mod algebra;
mod adaptive;
mod cooperation;
mod geometry;

pub(crate) use adaptive::{
    integrate, integrate_array_counted, integrate_scaled, AdaptiveSimpsonResult,
    QuadratureEvaluator,
};
pub(crate) use algebra::{AdaptiveEstimate, ArrayEstimate, QuadratureArithmeticError};
pub(crate) use cooperation::{QuadratureCooperation, QuadratureIdentity};
pub(crate) use geometry::{midpoint, ScaledEstimate};
