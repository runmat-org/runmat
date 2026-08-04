//! Active language compatibility policy for runtime builtin dispatch.

use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{BuiltinExtensionDescriptor, BuiltinExtensionMode, Value};
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

pub const SPARSE_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sparse-integer-storage",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sparse integer storage is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SparseIntegerExtension"),
};

pub const SPARSE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SPARSE_INTEGER_EXTENSION];

pub fn ensure_builtin_extension_enabled(
    extension: &BuiltinExtensionDescriptor,
    context: &str,
) -> Result<(), RuntimeError> {
    let enabled = match extension.mode {
        BuiltinExtensionMode::RunMatOnly => runmat_extensions_enabled(),
    };
    if enabled {
        return Ok(());
    }
    let mut builder = build_runtime_error(format!(
        "{context}: {}; enable runmat compatibility mode to use it",
        extension.description
    ));
    if let Some(identifier) = extension.error_identifier {
        builder = builder.with_identifier(identifier);
    }
    Err(builder.build())
}

/// Reject a RunMat-only sparse integer result at a MATLAB-compatible language boundary.
pub fn ensure_sparse_integer_extension_enabled(context: &str) -> Result<(), RuntimeError> {
    ensure_builtin_extension_enabled(&SPARSE_INTEGER_EXTENSION, context)
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

    #[test]
    fn integer_extensions_are_declared_in_builtin_metadata() {
        let sparse = runmat_builtins::builtin_function_by_name("sparse").expect("sparse builtin");
        assert_eq!(sparse.extensions, &SPARSE_EXTENSIONS);
        let randi = runmat_builtins::builtin_function_by_name("randi").expect("randi builtin");
        assert_eq!(
            randi.extensions,
            &crate::builtins::array::creation::randi::RANDI_EXTENSIONS
        );
        assert_eq!(
            randi.extensions[0].error_identifier,
            Some("RunMat:compatibility:RandiWideIntegerExtension")
        );

        let declared = runmat_builtins::builtin_functions()
            .into_iter()
            .flat_map(|builtin| {
                builtin
                    .extensions
                    .iter()
                    .map(move |extension| (builtin.name, extension.id))
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            declared,
            std::collections::BTreeSet::from([
                ("decomposition", "decomposition-gpu-input"),
                ("decomposition", "decomposition-nonfloating-input"),
                ("db", "db-nonfloating-input"),
                ("eigs", "eigs-gpu-input"),
                ("eigs", "eigs-nonfloating-matrix"),
                ("gpuArray", "gpuarray-dtype-selector"),
                ("gpuArray", "gpuarray-like"),
                ("gpuArray", "gpuarray-size-arguments"),
                ("fread", "fread-like"),
                ("fwrite", "fwrite-gpu-input"),
                ("imshow", "imshow-four-channel-image"),
                ("macd", "macd-nondouble-matrix"),
                ("meshgrid", "meshgrid-complex-axes"),
                ("meshgrid", "meshgrid-like"),
                ("pagefun", "pagefun-host-inputs"),
                ("pagefun", "pagefun-text-callable"),
                ("pskmod", "pskmod-integer-custom-order"),
                ("pskmod", "pskmod-integer-modulation-order"),
                ("pskmod", "pskmod-integer-phase-offset"),
                ("randi", "randi-implicit-prototype"),
                ("randi", "randi-wide-integer-output"),
                ("randperm", "randperm-explicit-double"),
                ("randperm", "randperm-like"),
                ("sawtooth", "sawtooth-gpu-input"),
                ("sawtooth", "sawtooth-nondouble-input"),
                ("sinc", "sinc-nonfloating-input"),
                ("sparse", "sparse-integer-storage"),
                ("square", "square-gpu-input"),
                ("square", "square-nonfloating-input"),
                ("trnd", "trnd-integer-degrees-of-freedom"),
                ("trnd", "trnd-integer-size"),
            ])
        );
    }

    #[test]
    fn generated_builtin_catalog_matches_registered_integer_extensions() {
        let catalog: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/builtins/meta.json"))
                .expect("builtin metadata catalog");
        let builtins = catalog["builtins"].as_array().expect("builtin entries");
        for registered in runmat_builtins::builtin_functions()
            .into_iter()
            .filter(|builtin| !builtin.extensions.is_empty())
        {
            let name = registered.name;
            let exported = builtins
                .iter()
                .find(|entry| entry["name"] == name)
                .unwrap_or_else(|| panic!("exported {name} metadata"));
            assert_eq!(
                exported["extensions"],
                serde_json::to_value(registered.extensions).expect("serialize extensions"),
                "{name}"
            );
        }
    }
}
