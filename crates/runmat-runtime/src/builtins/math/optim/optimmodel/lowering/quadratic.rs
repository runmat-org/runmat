//! Structural degree-two coefficient workspace over the single original model
//! topology and original variable layout. No finite differences or model copy.
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational;
use runmat_gc::GcHandle;
mod polynomial;
use polynomial::{Monomial, Polynomial, Weight};
mod array;
use array::Array;
mod assembly;
mod binary;
mod engine;
mod functions;
mod proof;
mod structural;
use engine::compile;
use proof::affine_scale;
pub(super) use proof::directional_cost;
mod route;
use route::candidate;
mod preparation;
pub(super) use preparation::{force, from_squares, prepare_or_affine};
pub(super) struct Compiled {
    cost: Polynomial,
    atoms: super::norm_square::Plan,
    facts: std::collections::HashMap<GcHandle, NodeFacts>,
    selection: selection::Selection,
    projection: Projection,
}
pub(super) struct Evidence {
    pub atoms: super::norm_square::Plan,
    pub selection: selection::Selection,
    pub curvature: Curvature,
    pub projection: Projection,
    pub scale: super::proof::ObjectiveScale,
    /// Moved exact scalar algebra only when global PSD failed. Active-set
    /// may still prove convexity on the original equality affine space.
    restricted: Option<Polynomial>,
}
pub(super) enum Projection {
    Affine,
    Scalar,
    Squares(objective::LeastSquares),
}
pub(super) struct Cost {
    pub hessian: clarabel::algebra::CscMatrix<f64>,
    pub linear: Vec<f64>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Curvature {
    Convex,
    Nonconvex,
}
pub(super) struct Prepared {
    pub cost: Cost,
    pub evidence: Evidence,
    pub facts: std::collections::HashMap<GcHandle, NodeFacts>,
}
pub(super) struct Model {
    pub cost: Option<Cost>,
    pub evidence: Evidence,
}
