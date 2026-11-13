//! Match selected parallel contracts to their actual compiled bodies.

use std::collections::BTreeSet;

use runmat_types::{ParallelManifest, ProgramFunctionId};

use crate::executable::ExecutableUnit;

pub(super) fn validate(
    unit: &ExecutableUnit,
    manifest: &ParallelManifest,
    selected_functions: Option<&BTreeSet<ProgramFunctionId>>,
) -> Result<(), String> {
    // Pure-operation `analysis.regions` use a separate ordinal domain. The
    // compiler-bound parfor/SPMD records and selected manifest are the only
    // relevant authorities here; the worker repeats this check on import.
    runmat_vm::validate_compiled_parallel_contracts(
        unit.bytecode(),
        &unit.bytecode().function_registry,
        manifest,
        selected_functions,
    )
}
