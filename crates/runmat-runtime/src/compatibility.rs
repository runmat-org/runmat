//! Active language compatibility policy for runtime builtin dispatch.

use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::Value;
use runmat_thread_local::runmat_thread_local;
use std::cell::Cell;
use std::collections::HashSet;

runmat_thread_local! {
    static RUNMAT_EXTENSIONS_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Returns whether the current execution may use deliberately classified
/// RunMat-only builtin forms.
pub fn runmat_extensions_enabled() -> bool {
    RUNMAT_EXTENSIONS_ENABLED.with(Cell::get)
}

/// Set the extension policy for subsequent builtin dispatch on this thread.
pub fn set_runmat_extensions_enabled(enabled: bool) {
    RUNMAT_EXTENSIONS_ENABLED.with(|slot| slot.set(enabled));
}

/// Reject a RunMat-only sparse integer result at a MATLAB-compatible language boundary.
pub fn ensure_sparse_integer_extension_enabled(context: &str) -> Result<(), RuntimeError> {
    if runmat_extensions_enabled() {
        return Ok(());
    }
    Err(build_runtime_error(format!(
        "{context}: sparse integer storage is a RunMat extension; enable runmat compatibility mode to use it"
    ))
    .with_identifier("RunMat:compatibility:SparseIntegerExtension")
    .build())
}

/// Enforce the sparse-integer extension policy recursively for a public result.
pub fn ensure_value_compatible(value: &Value, context: &str) -> Result<(), RuntimeError> {
    if value_contains_sparse_integer(value, &mut HashSet::new()) {
        ensure_sparse_integer_extension_enabled(context)?;
    }
    Ok(())
}

fn value_contains_sparse_integer(
    value: &Value,
    visited_handles: &mut HashSet<runmat_gc::GcHandle>,
) -> bool {
    match value {
        Value::SparseTensor(sparse) => sparse.integer_storage().is_some(),
        Value::Cell(cell) => cell
            .data
            .iter()
            .any(|value| value_contains_sparse_integer(value, visited_handles)),
        Value::Struct(struct_value) => struct_value
            .fields
            .values()
            .any(|value| value_contains_sparse_integer(value, visited_handles)),
        Value::Object(object) => object
            .properties
            .values()
            .any(|value| value_contains_sparse_integer(value, visited_handles)),
        Value::HandleObject(handle) => {
            if !visited_handles.insert(handle.target) {
                return false;
            }
            runmat_gc::gc_with_value(&handle.target, |target| {
                value_contains_sparse_integer(target, visited_handles)
            })
            .unwrap_or(false)
        }
        Value::Closure(closure) => closure
            .captures
            .iter()
            .any(|value| value_contains_sparse_integer(value, visited_handles)),
        Value::OutputList(values) => values
            .iter()
            .any(|value| value_contains_sparse_integer(value, visited_handles)),
        _ => false,
    }
}

/// Temporarily replace the extension policy and restore it on drop.
pub fn push_runmat_extensions_enabled(enabled: bool) -> RunMatExtensionsGuard {
    let previous = runmat_extensions_enabled();
    set_runmat_extensions_enabled(enabled);
    RunMatExtensionsGuard { previous }
}

#[must_use]
pub struct RunMatExtensionsGuard {
    previous: bool,
}

impl Drop for RunMatExtensionsGuard {
    fn drop(&mut self) {
        set_runmat_extensions_enabled(self.previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_extension_policy_restores_previous_state() {
        let original = runmat_extensions_enabled();
        {
            let _guard = push_runmat_extensions_enabled(!original);
            assert_eq!(runmat_extensions_enabled(), !original);
        }
        assert_eq!(runmat_extensions_enabled(), original);
    }

    #[test]
    fn sparse_integer_policy_checks_nested_results() {
        use runmat_builtins::{IntegerStorage, SparseTensor, StructValue};

        let sparse = SparseTensor::new_integer(
            1,
            1,
            vec![0, 1],
            vec![0],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("sparse integer");
        let mut fields = StructValue::new();
        fields.insert("x", Value::SparseTensor(sparse));
        let nested = Value::Struct(fields);

        {
            let _compat = push_runmat_extensions_enabled(false);
            let err = ensure_value_compatible(&nested, "load").expect_err("MATLAB mode rejects");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:SparseIntegerExtension")
            );
        }
        {
            let _compat = push_runmat_extensions_enabled(true);
            ensure_value_compatible(&nested, "load").expect("RunMat mode accepts");
        }
    }
}
