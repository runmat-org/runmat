use crate::*;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::multiobjective_facts::infer(request, entry, OptimSolver::Fminimax)
}
