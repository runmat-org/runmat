//! Original objective syntax selects a numerical family. Recognition never
//! evaluates a callback, forms a Gram matrix, or replaces the original DAG.
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational;
use runmat_gc::GcHandle;
mod assembly;
mod classify;
mod evidence;
mod preparation;
mod scale;
pub(super) mod step;
mod value;
mod weight;
pub(super) use assembly::assemble;
pub(super) use classify::classify;
pub(super) use preparation::compile;
#[derive(Clone, Copy)]
pub(super) enum Scale {
    Multiply,
    Divide,
}
#[derive(Clone)]
pub(super) struct Weight {
    pub proposal: f64,
    pub exact: Rational,
}
pub(super) struct Term {
    pub residual: GcHandle,
    pub weight: Weight,
}
pub(super) enum ModelObjective {
    Affine,
    Equations(super::equation::Evidence),
    ConstantSquares(LeastSquares),
    LeastSquares(LeastSquares),
    Quadratic(crate::builtins::math::optim::common::retained_control::Retained<quadratic::Model>),
}
impl ModelObjective {
    pub fn quadratic(&self) -> Option<&quadratic::Evidence> {
        match self {
            Self::Quadratic(model) => Some(&model.evidence),
            _ => None,
        }
    }
    pub fn take_quadratic_cost(&mut self) -> Option<quadratic::Cost> {
        match self {
            Self::Quadratic(model) => model.cost.take(),
            _ => None,
        }
    }
    pub fn squares(&self) -> Option<&LeastSquares> {
        match self {
            Self::Affine | Self::Quadratic(_) | Self::Equations(_) => None,
            Self::ConstantSquares(value) | Self::LeastSquares(value) => Some(value),
        }
    }
}
pub(super) struct LeastSquares {
    pub constant: Rational,
    pub terms: Vec<Term>,
}
pub(super) enum Classification {
    Unrecognized,
    LeastSquares(LeastSquares),
    Unrepresentable,
}
