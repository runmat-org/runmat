//! Immutable compiler cardinality facts derived from canonical owners.
use std::collections::HashMap;

pub(super) fn entrypoint_target_function(
    assembly: &runmat_hir::HirAssembly,
) -> Option<runmat_hir::FunctionId> {
    assembly
        .entrypoints
        .first()
        .map(|entrypoint| entrypoint.target)
}

pub(super) fn mir_local_fact_count_for_entrypoint(
    analysis: &runmat_mir::analysis::AnalysisStore,
    assembly: &runmat_hir::HirAssembly,
) -> usize {
    let function = entrypoint_target_function(assembly)
        .and_then(|function| analysis.function_for_local(function))
        .map(|facts| facts.function);
    analysis.local_value_count(function)
}

pub(super) fn function_output_arities(
    registry: &runmat_vm::FunctionRegistry,
) -> HashMap<runmat_hir::FunctionId, runmat_hir::FunctionOutputArity> {
    registry
        .functions
        .iter()
        .map(|(id, function)| {
            (
                *id,
                runmat_hir::FunctionOutputArity::new(
                    function.output_slots.len(),
                    function.varargout_slot.is_some(),
                ),
            )
        })
        .collect()
}
