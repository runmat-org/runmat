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
                ("DelaunayTri", "delaunaytri-integer-coordinates"),
                ("DelaunayTri", "delaunaytri-integer-topology"),
                ("DelaunayTri.freeBoundary", "delaunaytri-integer-topology"),
                (
                    "DelaunayTri.nearestNeighbor",
                    "delaunaytri-integer-coordinates",
                ),
                (
                    "DelaunayTri.pointLocation",
                    "delaunaytri-integer-coordinates",
                ),
                ("DelaunayTri.pointLocation", "delaunaytri-integer-topology"),
                ("corr", "corr-integer-data"),
                ("corr", "corr-integer-weights"),
                ("cov", "cov-integer-data"),
                ("cov", "cov-logical-data"),
                ("cov", "cov-typed-normalization"),
                ("cov", "cov-vector-weights"),
                ("cummax", "cummax-gpu-nanflag"),
                ("cummin", "cummin-gpu-nanflag"),
                ("cumprod", "cumprod-gpu-nanflag"),
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
                ("freeBoundary", "delaunaytri-integer-topology"),
                ("geomean", "geomean-integer-data"),
                ("geomean", "geomean-typed-integer-control"),
                ("imshow", "imshow-four-channel-image"),
                ("harmmean", "harmmean-integer-data"),
                ("harmmean", "harmmean-typed-integer-control"),
                ("kurtosis", "kurtosis-gpu-all-or-vecdim"),
                ("kurtosis", "kurtosis-integer-data"),
                ("kurtosis", "kurtosis-typed-integer-control"),
                ("macd", "macd-nondouble-matrix"),
                ("mad", "mad-integer-data"),
                ("mad", "mad-typed-integer-control"),
                ("meshgrid", "meshgrid-complex-axes"),
                ("meshgrid", "meshgrid-like"),
                ("movmad", "movmad-gpu-large-window"),
                ("movmax", "movmax-gpu-sample-points"),
                ("movmean", "movmean-gpu-sample-points"),
                ("movmedian", "movmedian-gpu-large-window"),
                ("movmedian", "movmedian-gpu-sample-points"),
                ("movmin", "movmin-gpu-sample-points"),
                ("movprod", "movprod-gpu-sample-points"),
                ("movstd", "movstd-gpu-sample-points"),
                ("movstd", "movstd-typed-integer-control"),
                ("movsum", "movsum-gpu-sample-points"),
                ("movvar", "movvar-gpu-sample-points"),
                ("movvar", "movvar-typed-integer-control"),
                ("nanmax", "nanmax-typed-integer-input"),
                ("nanmean", "nanmean-typed-integer-input"),
                ("nanmedian", "nanmedian-typed-integer-input"),
                ("nanmin", "nanmin-typed-integer-input"),
                ("nanstd", "nanstd-typed-integer-control"),
                ("nansum", "nansum-typed-integer-input"),
                ("nanvar", "nanvar-typed-integer-control"),
                ("nearestNeighbor", "delaunaytri-integer-coordinates"),
                ("pagefun", "pagefun-host-inputs"),
                ("pagefun", "pagefun-text-callable"),
                ("pskmod", "pskmod-integer-custom-order"),
                ("pskmod", "pskmod-integer-modulation-order"),
                ("pskmod", "pskmod-integer-phase-offset"),
                ("pointLocation", "delaunaytri-integer-coordinates"),
                ("pointLocation", "delaunaytri-integer-topology"),
                ("prctile", "prctile-integer-data"),
                ("prctile", "prctile-typed-integer-percentage"),
                ("randi", "randi-implicit-prototype"),
                ("randi", "randi-wide-integer-output"),
                ("randperm", "randperm-explicit-double"),
                ("randperm", "randperm-like"),
                ("rms", "rms-integer-data"),
                ("rmse", "rmse-integer-data"),
                ("rmse", "rmse-integer-weights"),
                ("quantile", "quantile-integer-data"),
                ("quantile", "quantile-typed-integer-probability"),
                ("range", "range-explicit-nanflag"),
                ("range", "range-gpu-all-or-vecdim"),
                ("range", "range-integer-data"),
                ("range", "range-typed-integer-control"),
                ("sawtooth", "sawtooth-gpu-input"),
                ("sawtooth", "sawtooth-nondouble-input"),
                ("sinc", "sinc-nonfloating-input"),
                ("skewness", "skewness-gpu-all-or-vecdim"),
                ("skewness", "skewness-integer-data"),
                ("skewness", "skewness-typed-integer-control"),
                ("sparse", "sparse-integer-storage"),
                ("square", "square-gpu-input"),
                ("square", "square-nonfloating-input"),
                ("std", "std-typed-integer-control"),
                ("tabulate", "tabulate-gpu-input"),
                ("tabulate", "tabulate-integer-data"),
                ("tiedrank", "tiedrank-integer-data"),
                ("trnd", "trnd-integer-degrees-of-freedom"),
                ("trnd", "trnd-integer-size"),
                ("var", "var-typed-integer-control"),
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

    #[test]
    fn generated_builtin_catalog_matches_integer_epic_descriptors() {
        let catalog: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/builtins/meta.json"))
                .expect("builtin metadata catalog");
        let builtins = catalog["builtins"].as_array().expect("builtin entries");
        for name in [
            "gcd", "intmax", "intmin", "linspace", "logspace", "mod", "rem",
        ] {
            let registered = runmat_builtins::builtin_functions()
                .into_iter()
                .find(|builtin| builtin.name == name)
                .unwrap_or_else(|| panic!("registered {name} builtin"));
            let expected = serde_json::to_value(registered.descriptor.expect("public descriptor"))
                .expect("serialize descriptor");
            let mut exported = builtins
                .iter()
                .find(|entry| entry["name"] == name)
                .unwrap_or_else(|| panic!("exported {name} metadata"))
                .clone();
            exported
                .as_object_mut()
                .expect("catalog entry object")
                .remove("name");
            exported
                .as_object_mut()
                .expect("catalog entry object")
                .remove("integer_capabilities");
            assert_eq!(exported, expected, "{name} descriptor");
        }
    }

    #[test]
    fn integer_capability_metadata_is_complete_and_well_formed_for_settled_apis() {
        let registered = runmat_builtins::builtin_functions();
        for builtin in &registered {
            assert!(
                builtin.integer_capabilities.is_empty() || builtin.integer_audit.is_none(),
                "{} cannot carry both integer capabilities and an audit disposition",
                builtin.name
            );
        }
        for name in [
            "DelaunayTri",
            "DelaunayTri.freeBoundary",
            "DelaunayTri.nearestNeighbor",
            "DelaunayTri.pointLocation",
            "circshift",
            "cummax",
            "cummin",
            "cumprod",
            "cumsum",
            "db",
            "factor",
            "factorial",
            "freeBoundary",
            "gcd",
            "histcounts",
            "idivide",
            "imshow",
            "isprime",
            "lcm",
            "max",
            "mean",
            "median",
            "min",
            "mod",
            "mpower",
            "movmax",
            "movmean",
            "movmedian",
            "movmin",
            "movprod",
            "movstd",
            "movsum",
            "movvar",
            "nanmax",
            "nanmean",
            "nanmedian",
            "nanmin",
            "nanstd",
            "nansum",
            "nanvar",
            "nearestNeighbor",
            "nchoosek",
            "num2cell",
            "pskmod",
            "polyint",
            "pointLocation",
            "primes",
            "qammod",
            "rem",
            "std",
            "sum",
            "trnd",
            "var",
            "bounds",
            "prod",
        ] {
            let builtin = registered
                .iter()
                .find(|builtin| builtin.name == name)
                .unwrap_or_else(|| panic!("registered {name} builtin"));
            assert!(
                !builtin.integer_capabilities.is_empty(),
                "{name} integer capabilities"
            );
            for capability in builtin.integer_capabilities {
                assert!(!capability.form.is_empty(), "{name} form");
                assert!(!capability.inputs.is_empty(), "{name} inputs");
                assert!(!capability.notes.is_empty(), "{name} notes");
                for input in capability.inputs {
                    assert!(!input.notes.is_empty(), "{name}:{} notes", input.name);
                    match input.availability {
                        runmat_builtins::BuiltinIntegerInputAvailability::Rejected => {
                            assert!(
                                input.classes.is_empty(),
                                "{name}:{} rejected class mask",
                                input.name
                            );
                            assert_eq!(
                                input.scalar_double,
                                runmat_builtins::BuiltinIntegerScalarDoubleRule::NotApplicable,
                                "{name}:{} rejected scalar-double rule",
                                input.name
                            );
                        }
                        runmat_builtins::BuiltinIntegerInputAvailability::Documented
                        | runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly => {
                            assert!(
                                !input.classes.is_empty(),
                                "{name}:{} accepted class mask",
                                input.name
                            );
                        }
                    }
                    for (index, class) in input.classes.iter().enumerate() {
                        assert!(
                            !input.classes[..index].contains(class),
                            "{name}:{} duplicate {class:?}",
                            input.name
                        );
                    }
                }
            }
            if builtin.integer_capabilities.iter().any(|capability| {
                capability.inputs.iter().any(|input| {
                    input.availability
                        == runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly
                })
            }) {
                assert!(
                    !builtin.extensions.is_empty(),
                    "{name} RunMat-only integer input requires extension metadata"
                );
            }
        }
    }

    #[test]
    fn integer_audit_dispositions_are_complete_and_well_formed() {
        let registered = runmat_builtins::builtin_functions();
        let mut audited = std::collections::BTreeMap::new();
        for builtin in &registered {
            let Some(audit) = builtin.integer_audit else {
                continue;
            };
            if let Some(previous) = audited.insert(builtin.name, audit) {
                assert_eq!(previous, audit, "{} duplicate integer audit", builtin.name);
            }
        }
        assert_eq!(
            audited.keys().copied().collect::<Vec<_>>(),
            ["addlistener", "cancel", "onCleanup"]
        );
        for (name, audit) in audited {
            assert!(!audit.notes.is_empty(), "{name} integer audit notes");
            match audit.kind {
                runmat_builtins::BuiltinIntegerAuditKind::AliasOf => {
                    let canonical = audit
                        .canonical_builtin
                        .unwrap_or_else(|| panic!("{name} alias target"));
                    assert_ne!(canonical, name, "{name} cannot alias itself");
                    let target = registered
                        .iter()
                        .find(|builtin| builtin.name == canonical)
                        .unwrap_or_else(|| panic!("{name} registered alias target {canonical}"));
                    assert!(
                        !target.integer_capabilities.is_empty(),
                        "{name} alias target {canonical} must carry integer capabilities"
                    );
                }
                runmat_builtins::BuiltinIntegerAuditKind::NotApplicable => {
                    assert_eq!(
                        audit.canonical_builtin, None,
                        "{name} inapplicable audit cannot name a canonical builtin"
                    );
                }
            }
        }
    }

    #[test]
    fn generated_builtin_catalog_matches_integer_metadata() {
        let catalog: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/builtins/meta.json"))
                .expect("builtin metadata catalog");
        let builtins = catalog["builtins"].as_array().expect("builtin entries");
        for registered in runmat_builtins::builtin_functions()
            .into_iter()
            .filter(|builtin| {
                !builtin.integer_capabilities.is_empty() || builtin.integer_audit.is_some()
            })
        {
            let name = registered.name;
            let exported = builtins
                .iter()
                .find(|entry| entry["name"] == name)
                .unwrap_or_else(|| panic!("exported {name} metadata"));
            let expected_capabilities = (!registered.integer_capabilities.is_empty()).then(|| {
                serde_json::to_value(registered.integer_capabilities)
                    .expect("serialize integer capabilities")
            });
            assert_eq!(
                exported.get("integer_capabilities"),
                expected_capabilities.as_ref(),
                "{name}"
            );
            let expected_audit = registered
                .integer_audit
                .map(|audit| serde_json::to_value(audit).expect("serialize integer audit"));
            assert_eq!(
                exported.get("integer_audit"),
                expected_audit.as_ref(),
                "{name} integer audit"
            );
        }
    }
}
