//! Domain acquisition at the existing creation boundary. The ordinary path
//! moves its original model; a subsearch borrows its enclosing domain.
use super::super::{
    common::solver_timing::SolverTiming,
    inputs::Problem,
    invalid,
    linear::{self, Geometry},
    options::Options,
    ScalarCooperation,
};
use crate::BuiltinResult;

pub(in super::super) trait Factory {
    type Geometry: Geometry;
    fn integers<'a>(&'a self, problem: &'a Problem) -> BuiltinResult<&'a [usize]>;
    async fn build(
        self,
        problem: &mut Problem,
        options: &Options,
        timing: &SolverTiming,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self::Geometry>;
}
pub(in super::super) struct Owned;
impl Factory for Owned {
    type Geometry = linear::Domain;
    fn integers<'a>(&'a self, problem: &'a Problem) -> BuiltinResult<&'a [usize]> {
        problem
            .linear
            .as_ref()
            .map(|data| data.integers.as_slice())
            .ok_or_else(|| invalid("creation geometry was already consumed"))
    }
    async fn build(
        self,
        problem: &mut Problem,
        options: &Options,
        timing: &SolverTiming,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<linear::Domain> {
        super::domain::acquire(problem, options, timing, cooperation).await
    }
}

pub(in super::super) struct Borrowed<G>(pub G);
impl<G: Geometry> Factory for Borrowed<G> {
    type Geometry = G;
    fn integers<'a>(&'a self, _problem: &'a Problem) -> BuiltinResult<&'a [usize]> {
        Ok(self.0.integers())
    }
    async fn build(
        self,
        _problem: &mut Problem,
        _options: &Options,
        _timing: &SolverTiming,
        _cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<G> {
        Ok(self.0)
    }
}
