//! Keep complete class registration bodies and their reachable source helpers.

use std::collections::BTreeSet;

use runmat_hir::FunctionId;
use runmat_types::ProgramFunctionId;

use super::{native_product_error, NativeCompilationInput};

pub(super) fn close(
    input: &NativeCompilationInput,
    requested: &BTreeSet<ProgramFunctionId>,
) -> Result<BTreeSet<ProgramFunctionId>, runmat_runtime::RuntimeError> {
    let mut retained = requested
        .iter()
        .map(|function| {
            usize::try_from(function.0)
                .map(FunctionId)
                .map_err(|_| native_product_error("function identity exceeds host indexing".into()))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    for class in &input.mir.classes {
        for method in &class.methods {
            if !input.mir.functions.contains_key(&method.function) {
                return Err(native_product_error(
                    "class method lacks its canonical MIR body".into(),
                ));
            }
            retained.insert(method.function);
        }
    }
    let children = runmat_mir::analysis::function_dependencies(&input.mir, &retained);
    let mut pending = retained.iter().copied().collect::<Vec<FunctionId>>();
    while let Some(function) = pending.pop() {
        for child in children.get(&function).into_iter().flatten() {
            if input.mir.functions.contains_key(child) && retained.insert(*child) {
                pending.push(*child);
            }
        }
    }
    retained
        .into_iter()
        .map(|function| {
            u32::try_from(function.0)
                .map(ProgramFunctionId)
                .map_err(|_| native_product_error("function identity exceeds Native IR".into()))
        })
        .collect()
}
