//! Original terminal proofs reuse the source constraint broadcast owner.
pub(super) use super::super::constraint_geometry::Geometry;
use super::*;
impl Geometry {
    pub(super) fn new(
        program: &Program,
        constraint: &formulation::Constraint,
        policy: Policy,
    ) -> BuiltinResult<Self> {
        Self::from_source(&program.source, constraint, policy)
    }
}
