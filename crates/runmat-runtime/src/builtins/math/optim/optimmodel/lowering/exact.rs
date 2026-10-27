//! Terminal-only exact evaluation of original finite binary64 constants and
//! recovered rational coordinates. This does not propagate a rational Jacobian
//! through the ordinary numerical compilation pass.
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational;

use runmat_gc::GcHandle;
use std::collections::HashMap;
mod data;
mod functions;
mod power;
use power::evaluate as power;
mod arithmetic;
mod edits;
mod forward;
mod norm_square;
mod pullback;
use data::Data;
pub(super) use forward::{evaluate, evaluate_atomic, evaluate_selected};
#[derive(Clone, Copy)]
pub(super) struct Subset<'a> {
    pub nodes: &'a std::collections::HashSet<GcHandle>,
    pub roots: &'a [GcHandle],
}
pub(super) struct Source<'a> {
    pub source: &'a evaluation::Plan,
    pub layout: &'a layout::Layout,
    pub facts: &'a HashMap<GcHandle, NodeFacts>,
}
pub(super) use pullback::{accumulate, Seed, SparsePlan};
pub(super) enum Coordinates<'a> {
    Origin,
    Point(&'a [Rational]),
}
#[derive(Clone, Copy)]
pub(super) enum Retain {
    Roots,
    ConstantOperands,
    Operands,
}
pub(super) struct Evaluated {
    values: HashMap<GcHandle, Data>,
    retention: Retain,
}
impl Evaluated {
    fn retains_operands(&self) -> bool {
        matches!(self.retention, Retain::Operands)
    }
    pub(super) fn value(
        &self,
        root: GcHandle,
        row: usize,
        policy: Policy,
    ) -> BuiltinResult<Option<Rational>> {
        self.values
            .get(&root)
            .ok_or_else(|| policy.invalid("exact original root is unavailable"))?
            .at(row, policy)
    }
}
