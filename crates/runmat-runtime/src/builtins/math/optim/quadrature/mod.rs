//! Shared adaptive quadrature services for `integral` and legacy `quad`.

mod adaptive;
mod cooperation;
mod geometry;

pub(crate) use adaptive::{integrate, AdaptiveSimpsonResult, QuadratureEvaluator};
pub(crate) use cooperation::{QuadratureCooperation, QuadratureIdentity};
pub(crate) use geometry::{midpoint, simpson, ScaledEstimate};
