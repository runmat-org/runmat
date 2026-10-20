//! Keep complete class registration bodies and their reachable source helpers.

use std::collections::BTreeSet;

use runmat_hir::FunctionId;

use super::{native_product_error, NativeCompilationInput};

pub(super) fn close(
    input: &NativeCompilationInput,
    requested: &BTreeSet<FunctionId>,
) -> Result<BTreeSet<FunctionId>, runmat_runtime::RuntimeError> {
    let mut retained = requested.clone();
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
    Ok(retained)
}
