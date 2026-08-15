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
    ))
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
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
        let expected = std::collections::BTreeSet::from([
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
            ("abs", "abs-character-input"),
            ("abs", "abs-logical-input"),
            ("abs", "sparse-integer-storage"),
            ("addpath", "addpath-numeric-character-codes"),
            ("all", "all-nanflag"),
            ("allfinite", "allfinite-string-input"),
            ("and", "and-character-input"),
            ("and", "and-complex-input"),
            ("any", "any-nanflag"),
            ("area", "area-linespec"),
            ("array2table", "array2table-gpu-input"),
            ("array2timetable", "array2timetable-gpu-input"),
            ("arrayDatastore", "arraydatastore-gpu-input"),
            ("arrayfun", "arrayfun-gpu-options"),
            ("arrayfun", "arrayfun-host-scalar-expansion"),
            ("arrayfun", "arrayfun-text-callable"),
            ("assert", "assert-complex-condition"),
            ("assert", "assert-unqualified-identifier"),
            ("acos", "acos-character-input"),
            ("acos", "acos-gpu-real-complex-promotion"),
            ("acos", "acos-integer-input"),
            ("acos", "acos-logical-input"),
            ("acosh", "acosh-character-input"),
            ("acosh", "acosh-gpu-real-complex-promotion"),
            ("acosh", "acosh-integer-input"),
            ("acosh", "acosh-logical-input"),
            ("asin", "asin-character-input"),
            ("asin", "asin-gpu-real-complex-promotion"),
            ("asin", "asin-integer-input"),
            ("asin", "asin-logical-input"),
            ("asinh", "asinh-character-input"),
            ("asinh", "asinh-integer-input"),
            ("asinh", "asinh-logical-input"),
            ("atan", "atan-character-input"),
            ("atan", "atan-integer-input"),
            ("atan", "atan-like-output-template"),
            ("atan", "atan-logical-input"),
            ("atan2", "atan2-character-input"),
            ("atan2", "atan2-integer-input"),
            ("atan2", "atan2-logical-input"),
            ("atanh", "atanh-character-input"),
            ("atanh", "atanh-gpu-real-complex-promotion"),
            ("atanh", "atanh-integer-input"),
            ("atanh", "atanh-logical-input"),
            ("any", "any-nanflag"),
            ("bandwidth", "bandwidth-integer-input"),
            ("bandwidth", "bandwidth-logical-input"),
            ("binocdf", "binocdf-integer-probability"),
            ("binocdf", "binocdf-integer-trials"),
            ("binocdf", "binocdf-integer-x"),
            ("binocdf", "binocdf-logical-input"),
            ("cdf", "cdf-integer-parameters"),
            ("cdf", "cdf-integer-x"),
            ("cdf", "cdf-logical-input"),
            ("cdfplot", "cdfplot-gpu-input"),
            ("cdfplot", "cdfplot-integer-input"),
            ("cdfplot", "cdfplot-logical-input"),
            ("cell", "cell-gpu-size"),
            ("mat2cell", "mat2cell-integer-partitions"),
            ("mat2str", "mat2str-integer-precision"),
            ("mat2str", "mat2str-text-input"),
            ("cell", "cell-like"),
            ("char", "char-logical-input"),
            ("char", "char-resident-numeric-input"),
            ("cheb2ord", "cheb2ord-integer-attenuation"),
            ("cheb2ord", "cheb2ord-integer-frequency"),
            ("cheb2ord", "cheb2ord-logical-input"),
            ("cheb2ord", "cheb2ord-resident-input"),
            ("chi2cdf", "chi2cdf-integer-degrees-of-freedom"),
            ("chi2cdf", "chi2cdf-integer-x"),
            ("chi2cdf", "chi2cdf-logical-input"),
            ("chol", "chol-integer-input"),
            ("chol", "chol-logical-input"),
            ("classify", "classify-integer-group"),
            ("classify", "classify-integer-prior"),
            ("classify", "classify-integer-sample"),
            ("classify", "classify-integer-training"),
            ("classify", "classify-logical-predictor"),
            ("classify", "classify-resident-input"),
            ("clf", "clf-all-selector"),
            ("clf", "clf-integer-figure-number"),
            ("clf", "clf-variadic-targets"),
            ("close", "close-integer-figure-number"),
            ("close", "close-variadic-targets"),
            ("colon", "colon-gpu-64-bit-integer"),
            ("colon", "colon-logical-input"),
            ("colon", "colon-zero-imaginary-complex"),
            ("colormap", "colormap-non-uint8-integer-map"),
            ("colororder", "colororder-integer-rgb"),
            ("combinations", "combinations-resident-input"),
            ("compose", "compose-resident-input"),
            ("cond", "cond-integer-matrix"),
            ("cond", "cond-integer-norm-selector"),
            ("cond", "cond-logical-matrix"),
            ("cond", "cond-logical-norm-selector"),
            ("confusionmat", "confusionmat-integer-group"),
            ("confusionmat", "confusionmat-integer-grouphat"),
            ("confusionmat", "confusionmat-integer-order"),
            ("confusionmat", "confusionmat-resident-input"),
            ("conj", "conj-character-input"),
            (
                "containers.Map",
                "containers-map-resident-constructor-input",
            ),
            (
                "containers.Map.isKey",
                "containers-map-resident-iskey-input",
            ),
            (
                "containers.Map.values",
                "containers-map-resident-values-input",
            ),
            (
                "containers.Map.remove",
                "containers-map-resident-remove-input",
            ),
            (
                "containers.Map.subsref",
                "containers-map-resident-subsref-input",
            ),
            (
                "containers.Map.subsasgn",
                "containers-map-resident-subsasgn-input",
            ),
            ("contour", "contour-integer-line-color"),
            ("contourf", "contour-integer-line-color"),
            ("copyobj", "copyobj-integer-handle-aliases"),
            ("cos", "cos-integer-input"),
            ("cos", "cos-logical-input"),
            ("cos", "cos-character-input"),
            ("cos", "cos-like-output"),
            ("cosd", "cosd-integer-input"),
            ("cosd", "cosd-logical-input"),
            ("cosd", "cosd-character-input"),
            ("cosh", "cosh-integer-input"),
            ("cosh", "cosh-logical-input"),
            ("cosh", "cosh-character-input"),
            ("cosineSimilarity", "cosine-similarity-integer-matrix"),
            ("cosineSimilarity", "cosine-similarity-single-matrix"),
            ("cosineSimilarity", "cosine-similarity-logical-matrix"),
            ("cosineSimilarity", "cosine-similarity-resident-input"),
            ("cospi", "cospi-integer-input"),
            ("cospi", "cospi-logical-input"),
            ("cospi", "cospi-character-input"),
            ("cross", "cross-integer-a"),
            ("cross", "cross-integer-b"),
            ("cross", "cross-integer-dim"),
            ("cross", "cross-logical-a"),
            ("cross", "cross-logical-b"),
            ("cross", "cross-logical-dim"),
            ("csvread", "csvread-colon-range"),
            ("csvread", "csvread-resident-control-inputs"),
            ("csvread", "csvread-two-vector-range"),
            ("csvwrite", "csvwrite-bytes-written-output"),
            ("csvwrite", "csvwrite-resident-input"),
            ("ctranspose", "ctranspose-integer-sparse"),
            ("cumtrapz", "cumtrapz-integer-dim"),
            ("cumtrapz", "cumtrapz-integer-spacing"),
            ("cumtrapz", "cumtrapz-integer-y"),
            ("cumtrapz", "cumtrapz-logical-spacing"),
            ("cumtrapz", "cumtrapz-logical-y"),
            ("cumtrapz", "cumtrapz-tensor-spacing"),
            ("cvpartition", "cvpartition-integer-custom-testsets"),
            ("cvpartition", "cvpartition-integer-observation-count"),
            ("cvpartition", "cvpartition-integer-partition-control"),
            ("cvpartition", "cvpartition-integer-stratification"),
            ("cvpartition", "cvpartition-nonlogical-stratify-option"),
            ("daspect", "daspect-numeric-axes-handle-alias"),
            ("dataTipTextRow", "data-tip-text-row-nonvector-value"),
            ("dataTipTextRow", "data-tip-text-row-resident-value"),
            ("datacursormode", "datacursormode-numeric-handle-alias"),
            ("datacursormode", "datacursormode-onoff-aliases"),
            ("datacursormode", "datacursormode-status-output"),
            ("datasample", "datasample-integer-data"),
            ("datasample", "datasample-integer-dim"),
            ("datasample", "datasample-integer-k"),
            ("datasample", "datasample-integer-weights"),
            ("datasample", "datasample-logical-weights"),
            ("datasample", "datasample-numeric-replace"),
            ("datasample", "datasample-resident-input"),
            ("dateshift", "datetime-resident-numeric-input"),
            ("datetime", "datetime-four-five-components"),
            ("datetime", "datetime-implicit-datenum"),
            ("datetime", "datetime-logical-numeric-input"),
            ("datetime", "datetime-resident-numeric-input"),
            ("day", "datetime-logical-numeric-input"),
            ("day", "datetime-resident-numeric-input"),
            ("deal", "deal-resident-input"),
            ("binornd", "binornd-integer-probability"),
            ("binornd", "binornd-integer-size"),
            ("binornd", "binornd-integer-trials"),
            ("binornd", "binornd-logical-input"),
            ("binscatter", "binscatter-gpu-input"),
            ("bitand", "bitand-gpu-assumedtype"),
            ("bitand", "bitand-gpu-undocumented-input"),
            ("bitand", "bitand-single-input"),
            ("bitor", "bitor-gpu-assumedtype"),
            ("bitor", "bitor-gpu-undocumented-input"),
            ("bitor", "bitor-single-input"),
            ("bitshift", "bitshift-gpu-assumedtype"),
            ("bitshift", "bitshift-gpu-undocumented-input"),
            ("bitshift", "bitshift-logical-count-input"),
            ("bitshift", "bitshift-logical-value-input"),
            ("bitshift", "bitshift-single-count-input"),
            ("bitshift", "bitshift-single-value-input"),
            ("blackman", "blackman-logical-length"),
            ("blanks", "blanks-gpu-input"),
            ("blkdiag", "blkdiag-complex-integer-input"),
            ("blkdiag", "blkdiag-sparse-integer-input"),
            ("bootstrp", "bootstrp-gpu-input"),
            ("bootstrp", "bootstrp-integer-data"),
            ("bootstrp", "bootstrp-integer-nboot"),
            ("bootstrp", "bootstrp-integer-weights"),
            ("bootstrp", "bootstrp-logical-nboot"),
            ("bootstrp", "bootstrp-logical-weights"),
            ("bootstrp", "bootstrp-text-callable"),
            ("boxplot", "boxplot-gpu-input"),
            ("boxplot", "boxplot-integer-data"),
            ("boxplot", "boxplot-integer-group"),
            ("boxplot", "boxplot-integer-option"),
            ("boxplot", "boxplot-logical-data"),
            ("boxplot", "boxplot-logical-group"),
            ("boxplot", "boxplot-logical-option"),
            ("bsxfun", "bsxfun-text-callable"),
            ("butter", "butter-gpu-input"),
            ("butter", "butter-integer-cutoff"),
            ("butter", "butter-integer-order"),
            ("butter", "butter-logical-cutoff"),
            ("butter", "butter-logical-order"),
            ("butter", "butter-option-alias"),
            ("butter", "butter-order-above-500"),
            ("butter", "butter-single-cutoff"),
            ("butter", "butter-single-order"),
            ("buttord", "buttord-complex-frequency"),
            ("buttord", "buttord-gpu-input"),
            ("buttord", "buttord-integer-attenuation"),
            ("buttord", "buttord-integer-frequency"),
            ("buttord", "buttord-logical-attenuation"),
            ("buttord", "buttord-logical-frequency"),
            ("cat", "cat-complex-integer-input"),
            ("cat", "cat-like-prototype"),
            ("cat", "cat-resident-dimension"),
            ("categorical", "categorical-gpu-input"),
            ("caxis", "clim-integer-limits"),
            ("clim", "clim-integer-limits"),
            ("contains", "contains-numeric-ignore-case"),
            ("contains", "contains-positional-ignore-case"),
            ("contains", "contains-text-ignore-case"),
            ("corr", "corr-integer-data"),
            ("corr", "corr-integer-weights"),
            ("corrcoef", "corrcoef-integer-data"),
            ("corrcoef", "corrcoef-logical-data"),
            ("corrcov", "corrcov-integer-data"),
            ("corrcov", "corrcov-logical-data"),
            ("cov", "cov-integer-data"),
            ("cov", "cov-logical-data"),
            ("cov", "cov-typed-normalization"),
            ("cov", "cov-vector-weights"),
            ("cov2corr", "cov2corr-gpu-input"),
            ("cov2corr", "cov2corr-integer-data"),
            ("cov2corr", "cov2corr-logical-data"),
            ("cov2corr", "cov2corr-single-data"),
            ("cummax", "cummax-gpu-nanflag"),
            ("cummin", "cummin-gpu-nanflag"),
            ("cumprod", "cumprod-gpu-nanflag"),
            ("decomposition", "decomposition-gpu-input"),
            ("decomposition", "decomposition-integer-check-condition"),
            ("decomposition", "decomposition-integer-rank-tolerance"),
            ("decomposition", "decomposition-nonfloating-input"),
            (
                "decomposition.ctranspose",
                "decomposition-nonfloating-input",
            ),
            ("decomposition.mldivide", "decomposition-nonfloating-input"),
            ("decomposition.mldivide", "decomposition-gpu-input"),
            ("decomposition.mrdivide", "decomposition-nonfloating-input"),
            ("decomposition.mrdivide", "decomposition-gpu-input"),
            ("decomposition.mtimes", "decomposition-nonfloating-input"),
            ("decomposition.mtimes", "decomposition-gpu-input"),
            ("decomposition.mtimes", "decomposition-integer-scale-factor"),
            (
                "decomposition.mtimes",
                "decomposition-complex-integer-scale-factor",
            ),
            ("decomposition.rdivide", "decomposition-nonfloating-input"),
            ("decomposition.rdivide", "decomposition-gpu-input"),
            (
                "decomposition.rdivide",
                "decomposition-integer-scale-factor",
            ),
            (
                "decomposition.rdivide",
                "decomposition-complex-integer-scale-factor",
            ),
            ("decomposition.subsref", "decomposition-nonfloating-input"),
            ("decomposition.times", "decomposition-nonfloating-input"),
            ("decomposition.times", "decomposition-gpu-input"),
            ("decomposition.times", "decomposition-integer-scale-factor"),
            (
                "decomposition.times",
                "decomposition-complex-integer-scale-factor",
            ),
            ("decomposition.uminus", "decomposition-nonfloating-input"),
            ("decomposition.uplus", "decomposition-nonfloating-input"),
            ("dividerand", "dividerand-resident-argument"),
            ("dlmread", "dlmread-colon-spreadsheet-range"),
            ("dlmread", "dlmread-composed-range"),
            ("dlmread", "dlmread-numeric-delimiter"),
            ("dlmread", "dlmread-resident-argument"),
            ("dlmwrite", "dlmwrite-byte-count-output"),
            ("dlmwrite", "dlmwrite-resident-input"),
            ("dot", "dot-integer-data"),
            ("dot", "dot-logical-data"),
            ("double", "double-like-prototype"),
            ("downsample", "downsample-integer-factor"),
            ("downsample", "downsample-integer-phase"),
            ("downsample", "downsample-nd-input"),
            ("dummyvar", "dummyvar-gpu-group"),
            ("dummyvar", "dummyvar-integer-group"),
            ("duration", "duration-gpu-input"),
            ("duration", "duration-short-component-form"),
            ("db", "db-nonfloating-input"),
            ("deconv", "deconv-integer-input"),
            ("deconv", "deconv-logical-input"),
            ("deg2rad", "deg2rad-integer-input"),
            ("deg2rad", "deg2rad-logical-input"),
            ("det", "det-integer-input"),
            ("det", "det-logical-input"),
            (
                "detectImportOptions",
                "detectimportoptions-integer-num-header-lines",
            ),
            ("diag", "diag-explicit-size"),
            ("diag", "diag-like"),
            ("diag", "diag-output-class"),
            ("diag", "diag-trailing-singleton-dimensions"),
            ("diag", "diag-vector-option"),
            ("dictionary", "dictionary-gpu-input"),
            ("diff", "diff-character-input"),
            ("diff", "diff-empty-control"),
            ("diff", "diff-zero-order"),
            ("digits", "digits-default-reset"),
            ("digits", "digits-numeric-text"),
            ("ecdf", "ecdf-integer-censoring"),
            ("ecdf", "ecdf-integer-frequency"),
            ("ecdf", "ecdf-integer-y"),
            ("ecdf", "ecdf-logical-frequency"),
            ("ecdf", "ecdf-logical-y"),
            ("eig", "eig-nonfloating-coefficient"),
            ("eigs", "eigs-gpu-input"),
            ("eigs", "eigs-integer-sigma"),
            ("eigs", "eigs-integer-start-vector"),
            ("eigs", "eigs-nonfloating-matrix"),
            ("empty", "empty-global-call"),
            ("empty", "empty-resident-size"),
            ("empty", "empty-typename"),
            ("encode", "encode-numeric-force-cell-output"),
            ("encode", "encode-resident-force-cell-output"),
            ("endsWith", "endswith-numeric-ignore-case"),
            ("endsWith", "endswith-positional-ignore-case"),
            ("endsWith", "endswith-resident-ignore-case"),
            ("endsWith", "endswith-text-ignore-case"),
            ("envelope", "envelope-integer-control"),
            ("envelope", "envelope-integer-data"),
            ("eraseBetween", "erasebetween-char-matrix"),
            ("eraseBetween", "erasebetween-full-broadcast"),
            ("eraseBetween", "erasebetween-resident-position"),
            ("eraseBetween", "erasebetween-string-cell"),
            ("error", "error-mexception-input"),
            ("error", "error-struct-field-aliases"),
            ("error", "error-unqualified-identifier"),
            ("erasePunctuation", "erase-punctuation-broad-cell-input"),
            ("erasePunctuation", "erase-punctuation-char-matrix-input"),
            ("eraseURLs", "erase-urls-broad-cell-input"),
            ("eraseURLs", "erase-urls-char-matrix-input"),
            ("exist", "exist-nontext-handle-query"),
            ("exist", "exist-search-type-extension"),
            ("exp", "exp-character-input"),
            ("exp", "exp-integer-input"),
            ("exp", "exp-logical-input"),
            ("expm1", "expm1-character-input"),
            ("expm1", "expm1-integer-input"),
            ("expm1", "expm1-logical-input"),
            ("exprnd", "exprnd-integer-mean"),
            ("exprnd", "exprnd-integer-size"),
            ("extractAfter", "extractafter-char-matrix"),
            ("extractAfter", "extractafter-resident-position"),
            ("extractAfter", "extractafter-string-cell"),
            ("extractBefore", "extractbefore-char-matrix"),
            ("extractBefore", "extractbefore-resident-position"),
            ("extractBefore", "extractbefore-string-cell"),
            ("extractBetween", "extractbetween-char-matrix"),
            ("extractBetween", "extractbetween-full-broadcast"),
            ("extractBetween", "extractbetween-resident-position"),
            ("extractBetween", "extractbetween-string-cell"),
            ("extractFileText", "extractfiletext-broad-file-cell"),
            ("extractFileText", "extractfiletext-resident-pages"),
            ("extractHTMLText", "extracthtmltext-broad-cell"),
            ("extractHTMLText", "extracthtmltext-char-matrix"),
            ("eye", "eye-column-size-vector"),
            ("eye", "eye-implicit-prototype"),
            ("eye", "eye-nd-dimensions"),
            ("false", "false-implicit-prototype"),
            ("false", "false-logical-class-option"),
            ("false", "false-resident-size-input"),
            ("false", "false-single-size-input"),
            ("fclose", "fclose-cell-fileids"),
            ("fclose", "fclose-fileid-vector"),
            ("fclose", "fclose-integer-fileid"),
            ("fclose", "fclose-logical-fileid"),
            ("fclose", "fclose-message-output"),
            ("fclose", "fclose-no-input-close-all"),
            ("fclose", "fclose-resident-fileid"),
            ("fclose", "fclose-single-fileid"),
            ("fclose", "fclose-string-all"),
            ("fcontour", "fcontour-integer-line-color"),
            ("fcontour", "fcontour-positional-level-spec"),
            ("fcontour", "fcontour-resident-numeric-input"),
            ("feedback", "feedback-single-system"),
            ("feedback", "feedback-scalar-forward-system"),
            ("feedback", "feedback-scalar-scalar-systems"),
            ("feedback", "feedback-integer-scalar-gain"),
            ("feedback", "feedback-integer-sign"),
            ("feedback", "feedback-logical-numeric-input"),
            ("feedback", "feedback-resident-numeric-input"),
            ("feedback", "feedback-single-numeric-input"),
            ("feval", "feval-at-prefixed-text-target"),
            ("feval", "feval-object-receiver"),
            ("feof", "feof-integer-fileid"),
            ("feof", "feof-logical-fileid"),
            ("feof", "feof-resident-fileid"),
            ("fitclinear", "fitclinear-integer-numeric-arguments"),
            ("fitclinear", "fitclinear-integer-predictors"),
            ("fitclinear", "fitclinear-resident-input"),
            ("fitctree", "fitctree-integer-numeric-arguments"),
            ("fitctree", "fitctree-integer-predictors"),
            ("fitctree", "fitctree-resident-input"),
            ("fitdist", "fitdist-integer-data"),
            ("fitdist", "fitdist-integer-frequency"),
            ("fitdist", "fitdist-resident-fallback"),
            ("fitlm", "fitlm-integer-controls"),
            ("fitlm", "fitlm-integer-predictor-data"),
            ("fitlm", "fitlm-integer-response-data"),
            ("fitlm", "fitlm-integer-selectors"),
            ("fitlm", "fitlm-integer-weights"),
            ("fitlm", "fitlm-resident-fallback"),
            ("fminbnd", "fminbnd-nonfloating-bounds"),
            ("fminbnd", "fminbnd-nonfloating-objective"),
            ("fminbnd", "fminbnd-resident-fallback"),
            ("fminbnd", "fminbnd-typed-option-controls"),
            ("fminunc", "fminunc-nonfloating-callback-output"),
            ("fminunc", "fminunc-nonfloating-initial-point"),
            ("fminunc", "fminunc-resident-fallback"),
            ("fminunc", "fminunc-typed-option-controls"),
            ("fsolve", "fsolve-nonfloating-callback-output"),
            ("fsolve", "fsolve-nonfloating-initial-point"),
            ("fsolve", "fsolve-resident-fallback"),
            ("fsolve", "fsolve-typed-option-controls"),
            ("fspecial", "fspecial-nondouble-parameter"),
            ("fspecial", "fspecial-nondouble-size"),
            ("fspecial", "fspecial-resident-output"),
            ("fspecial", "fspecial-unsharp-filter"),
            ("fsurf", "fsurf-integer-axes-handle"),
            ("fsurf", "fsurf-integer-callback-output"),
            ("fsurf", "fsurf-integer-domain"),
            ("fsurf", "fsurf-integer-mesh-density"),
            ("fsurf", "fsurf-integer-style-property"),
            ("fsurf", "fsurf-logical-numeric-input"),
            ("fsurf", "fsurf-resident-callback-output"),
            ("fsurf", "fsurf-single-numeric-input"),
            ("fzero", "fzero-nonfloating-callback-output"),
            ("fzero", "fzero-nonfloating-initial-point"),
            ("fzero", "fzero-resident-fallback"),
            ("fzero", "fzero-typed-option-controls"),
            ("fft", "fft-wide-integer-controls"),
            ("fft", "fft-wide-integer-data"),
            ("fft2", "fft2-empty-zero-size"),
            ("fft2", "fft2-size-form"),
            ("fft2", "fft2-wide-integer-controls"),
            ("fft2", "fft2-wide-integer-data"),
            ("fftn", "fftn-short-size-vector"),
            ("fftn", "fftn-wide-integer-controls"),
            ("fftn", "fftn-wide-integer-data"),
            ("fftshift", "fftshift-multi-dimension-selector"),
            ("fgetl", "fgetl-integer-fileid"),
            ("fgetl", "fgetl-resident-fileid"),
            ("fgetl", "fgetl-single-fileid"),
            ("fgets", "fgets-integer-fileid"),
            ("fgets", "fgets-integer-nchar"),
            ("fgets", "fgets-resident-fileid"),
            ("fgets", "fgets-resident-nchar"),
            ("fgets", "fgets-single-fileid"),
            ("fgets", "fgets-single-nchar"),
            ("fieldnames", "fieldnames-object-family"),
            ("figure", "figure-integer-property-value"),
            ("figure", "figure-integer-target"),
            ("figure", "figure-next-selector"),
            ("figure", "figure-single-target"),
            ("filewrite", "filewrite-runmat-native"),
            ("fill", "fill-integer-axes-handle"),
            ("fill3", "fill3-integer-axes-handle"),
            ("fillmissing", "fillmissing-aggregate-integer-data"),
            ("fillmissing", "fillmissing-integer-data"),
            ("filloutliers", "filloutliers-integer-data"),
            ("filloutliers", "filloutliers-integer-fill-scalar"),
            ("filloutliers", "filloutliers-numeric-outlier-locations"),
            ("filloutliers", "filloutliers-resident-input"),
            ("filter2", "filter2-convolution-mode"),
            ("filter2", "filter2-integer-gpu-input"),
            ("filter2", "filter2-logical-gpu-input"),
            ("filter2", "filter2-nd-input"),
            ("filter2", "filter2-variadic-options"),
            ("filtfilt", "filtfilt-integer-input"),
            ("filtfilt", "filtfilt-logical-input"),
            ("flip", "flip-dimension-vector"),
            ("flip", "flip-direction-keyword"),
            ("flip", "flip-typed-dimension"),
            ("find", "find-direction-only"),
            ("find", "find-integer-sparse-input"),
            ("findgroups", "findgroups-matrix-as-columns"),
            ("findgroups", "findgroups-resident-input"),
            ("findgroups", "findgroups-table-selector"),
            ("findgroups", "findgroups-timetable-input"),
            ("findobj", "findobj-integer-root-aliases"),
            ("fir1", "fir1-integer-cutoff"),
            ("fir1", "fir1-integer-order"),
            ("fir1", "fir1-integer-window"),
            ("fir1", "fir1-logical-cutoff"),
            ("fir1", "fir1-logical-order"),
            ("fir1", "fir1-logical-window"),
            ("fir1", "fir1-single-cutoff"),
            ("fir1", "fir1-single-order"),
            ("fir1", "fir1-single-window"),
            ("fopen", "fopen-integer-fileid"),
            ("fopen", "fopen-legacy-all"),
            ("fopen", "fopen-resident-fileid"),
            ("fopen", "fopen-single-fileid"),
            ("fprintf", "fprintf-integer-fileid"),
            ("fprintf", "fprintf-integer-format"),
            ("fprintf", "fprintf-numeric-format"),
            ("fprintf", "fprintf-resident-fileid"),
            ("fprintf", "fprintf-resident-format"),
            ("fprintf", "fprintf-single-fileid"),
            ("fprintf", "fprintf-stream-label"),
            ("gather", "gather-recursive-container"),
            ("kron", "kron-nd-input"),
            ("kstest", "kstest-integer-cdf"),
            ("kstest", "kstest-integer-sample"),
            ("kstest", "kstest-resident-input"),
            ("lasso", "lasso-integer-boolean"),
            ("lasso", "lasso-integer-control"),
            ("lasso", "lasso-integer-data"),
            ("lasso", "lasso-integer-parameter"),
            ("lasso", "lasso-resident-input"),
            ("lassoglm", "lassoglm-integer-boolean"),
            ("lassoglm", "lassoglm-integer-control"),
            ("lassoglm", "lassoglm-integer-data"),
            ("lassoglm", "lassoglm-integer-parameter"),
            ("lassoglm", "lassoglm-resident-input"),
            ("lab2rgb", "lab2rgb-explicit-gpu-input"),
            ("ldivide", "ldivide-like-prototype"),
            ("linprog", "linprog-integer-a"),
            ("linprog", "linprog-integer-aeq"),
            ("linprog", "linprog-integer-b"),
            ("linprog", "linprog-integer-beq"),
            ("linprog", "linprog-explicit-gpu-input"),
            ("linprog", "linprog-integer-f"),
            ("linprog", "linprog-integer-lb"),
            ("linprog", "linprog-integer-ub"),
            ("linsolve", "linsolve-explicit-gpu-two-output"),
            ("linsolve", "linsolve-integer-input"),
            ("linsolve", "linsolve-integer-option-control"),
            ("linsolve", "linsolve-logical-input"),
            ("linsolve", "linsolve-rcond-option"),
            ("linsolve", "linsolve-text-transa-option"),
            ("log", "log-character-input"),
            ("log", "log-explicit-real-gpu-complex-promotion"),
            ("log", "log-integer-input"),
            ("log", "log-logical-input"),
            ("log10", "log10-character-input"),
            ("log10", "log10-explicit-real-gpu-complex-promotion"),
            ("log10", "log10-integer-input"),
            ("log10", "log10-logical-input"),
            ("log1p", "log1p-character-input"),
            ("log1p", "log1p-explicit-real-gpu-complex-promotion"),
            ("log1p", "log1p-integer-input"),
            ("log1p", "log1p-logical-input"),
            ("log2", "log2-character-input"),
            ("log2", "log2-integer-input"),
            ("log2", "log2-logical-input"),
            ("logical", "logical-string-array-input"),
            ("logical", "logical-symbolic-constant-input"),
            ("lu", "lu-integer-input"),
            ("lu", "lu-logical-input"),
            ("join", "join-dimension-before-delimiter"),
            ("join", "join-resident-input"),
            ("join", "join-typed-integer-dimension"),
            ("jsonencode", "jsonencode-complex-input"),
            ("jsonencode", "jsonencode-numeric-option"),
            ("jsonencode", "jsonencode-resident-input"),
            ("jsonencode", "jsonencode-sparse-input"),
            ("jsonencode", "jsonencode-text-option"),
            ("jsonencode", "jsonencode-typed-integer-option"),
            ("gpuArray", "gpuarray-dtype-selector"),
            ("gpuArray", "gpuarray-like"),
            ("gpuArray", "gpuarray-size-arguments"),
            ("gpuArray", "gpuarray-text-upload"),
            ("gpuDevice", "gpudevice-logical-selector"),
            ("gpuDevice", "gpudevice-provider-info"),
            ("fread", "fread-integer-fileid"),
            ("fread", "fread-integer-size"),
            ("fread", "fread-integer-skip"),
            ("fread", "fread-like"),
            ("fread", "fread-logical-control"),
            ("fread", "fread-resident-control"),
            ("fread", "fread-single-control"),
            ("freqz", "freqz-integer-coefficients"),
            ("freqz", "freqz-integer-point-count"),
            ("freqz", "freqz-integer-sample-rate"),
            ("freqz", "freqz-logical-coefficients"),
            ("freqz", "freqz-logical-point-count"),
            ("freqz", "freqz-logical-sample-rate"),
            ("freqz", "freqz-resident-input"),
            ("freqz", "freqz-single-point-count"),
            ("freqz", "freqz-single-sample-rate"),
            ("frewind", "frewind-integer-fileid"),
            ("frewind", "frewind-logical-fileid"),
            ("frewind", "frewind-resident-fileid"),
            ("frewind", "frewind-single-fileid"),
            ("full", "full-integer-sparse"),
            ("fwrite", "fwrite-arrow-precision"),
            ("fwrite", "fwrite-gpu-input"),
            ("fwrite", "fwrite-integer-fileid"),
            ("fwrite", "fwrite-integer-skip"),
            ("fwrite", "fwrite-logical-control"),
            ("fwrite", "fwrite-resident-control"),
            ("fwrite", "fwrite-single-control"),
            ("freeBoundary", "delaunaytri-integer-topology"),
            ("gauspuls", "gauspuls-integer-control"),
            ("gauspuls", "gauspuls-integer-time"),
            ("gauspuls", "gauspuls-logical-control"),
            ("gauspuls", "gauspuls-logical-time"),
            ("gauspuls", "gauspuls-resident-input"),
            ("gauspuls", "gauspuls-single-control"),
            ("gamrnd", "gamrnd-integer-scale-parameter"),
            ("gamrnd", "gamrnd-integer-shape-parameter"),
            ("gamrnd", "gamrnd-integer-size"),
            ("gammaln", "gammaln-character-input"),
            ("gammaln", "gammaln-integer-input"),
            ("gammaln", "gammaln-logical-input"),
            ("gradient", "gradient-integer-data"),
            ("gradient", "gradient-integer-spacing"),
            ("gradient", "gradient-logical-data"),
            ("gradient", "gradient-logical-spacing"),
            ("gray2rgb", "gray2rgb-callable"),
            ("griddedInterpolant", "griddedinterpolant-integer-grid"),
            ("griddedInterpolant", "griddedinterpolant-integer-query"),
            ("griddedInterpolant", "griddedinterpolant-integer-values"),
            ("groupcounts", "groupcounts-integer-control"),
            ("groupcounts", "groupcounts-resident-input"),
            ("groupsummary", "groupsummary-integer-control"),
            ("groupsummary", "groupsummary-resident-input"),
            ("grpstats", "grpstats-integer-alpha"),
            ("grpstats", "grpstats-integer-data"),
            ("grpstats", "grpstats-integer-selector"),
            ("grpstats", "grpstats-resident-input"),
            ("geomean", "geomean-integer-data"),
            ("geomean", "geomean-typed-integer-control"),
            ("gca", "gca-figure-argument"),
            ("gca", "gca-integer-figure-alias"),
            ("gca", "gca-struct-output"),
            ("get", "get-integer-handle-alias"),
            ("getenv", "getenv-cell-string-name"),
            ("getenv", "getenv-character-matrix-name"),
            ("getfield", "getfield-indexed-resident-field"),
            ("getfield", "getfield-object-family"),
            ("getfield", "getfield-textual-index"),
            ("getmethod", "getmethod-bound-method-handle"),
            ("getpref", "getpref-group-query"),
            ("hour", "datetime-logical-numeric-input"),
            ("hour", "datetime-resident-numeric-input"),
            ("hour", "hour-typed-legacy-serial-input"),
            ("minute", "datetime-logical-numeric-input"),
            ("minute", "datetime-resident-numeric-input"),
            ("minute", "minute-typed-legacy-serial-input"),
            ("month", "datetime-logical-numeric-input"),
            ("month", "datetime-resident-numeric-input"),
            ("month", "month-typed-legacy-serial-input"),
            ("htmlTree", "htmltree-broad-cell-input"),
            ("htmlTree", "htmltree-cell-object-array"),
            ("icdf", "icdf-integer-input"),
            ("icdf", "icdf-logical-input"),
            ("image", "image-four-channel-cdata"),
            ("imagesc", "imagesc-four-channel-cdata"),
            ("imfilter", "imfilter-fill-keyword"),
            ("imfilter", "imfilter-valid-output-shape"),
            ("imhist", "imhist-typed-integer-bin-count"),
            ("imwrite", "imwrite-single-gif-tiff"),
            ("isdiag", "isdiag-integer-input"),
            ("isgraphics", "isgraphics-integer-handle-alias"),
            ("ishandle", "ishandle-integer-handle-alias"),
            ("ishermitian", "ishermitian-flag-aliases"),
            ("ishermitian", "ishermitian-integer-input"),
            ("ishermitian", "ishermitian-tolerance-form"),
            ("ismissing", "ismissing-resident-input"),
            ("issymmetric", "issymmetric-flag-aliases"),
            ("issymmetric", "issymmetric-integer-input"),
            ("issymmetric", "issymmetric-tolerance-form"),
            ("isstrprop", "isstrprop-resident-input"),
            ("istril", "istril-integer-input"),
            ("istriu", "istriu-integer-input"),
            ("inf", "inf-column-size-vector"),
            ("inf", "inf-resident-size-control"),
            ("input", "input-no-prompt"),
            ("input", "input-swapped-arguments"),
            ("insertAfter", "insertafter-broad-text-containers"),
            ("insertAfter", "insertafter-resident-position"),
            ("insertBefore", "insertbefore-broad-text-containers"),
            ("insertBefore", "insertbefore-resident-position"),
            ("imshow", "imshow-four-channel-image"),
            ("impulse", "impulse-integer-numeric-role"),
            ("impulse", "impulse-logical-numeric-role"),
            ("impulse", "impulse-resident-input"),
            ("impulse", "impulse-single-numeric-role"),
            ("ind2word", "ind2word-nonvector-indices"),
            ("ind2word", "ind2word-resident-indices"),
            ("ind2word", "ind2word-typed-integer-indices"),
            ("int16.empty", "empty-resident-size"),
            ("int32.empty", "empty-resident-size"),
            ("int64.empty", "empty-resident-size"),
            ("int8.empty", "empty-resident-size"),
            ("isoutlier", "isoutlier-gpu-data-variables"),
            ("isoutlier", "isoutlier-gpu-movmedian"),
            ("isoutlier", "isoutlier-gpu-sample-points"),
            ("isoutlier", "isoutlier-integer-data"),
            ("isoutlier", "isoutlier-integer-dimension"),
            ("isoutlier", "isoutlier-integer-threshold"),
            ("isoutlier", "isoutlier-integer-window"),
            ("ismembertol", "ismembertol-gpu-options"),
            ("ismembertol", "ismembertol-gpu-wide-integer-input"),
            ("ismembertol", "ismembertol-host-integer-data"),
            ("ismembertol", "ismembertol-host-logical-data"),
            ("ismembertol", "ismembertol-logical-tolerance-control"),
            ("ismembertol", "ismembertol-typed-tolerance-control"),
            ("issorted", "issorted-gpu-missing-placement"),
            ("issorted", "issorted-gpu-nonvector"),
            ("issortedrows", "issortedrows-gpu-input"),
            ("harmmean", "harmmean-integer-data"),
            ("harmmean", "harmmean-typed-integer-control"),
            ("hamming", "hamming-logical-length"),
            ("hann", "hann-logical-length"),
            ("head", "head-gpu-row-count"),
            ("heatmap", "heatmap-gpu-cdata"),
            ("heaviside", "heaviside-character-input"),
            ("heaviside", "heaviside-gpu-input"),
            ("heaviside", "heaviside-integer-input"),
            ("heaviside", "heaviside-logical-input"),
            ("hilbert", "hilbert-empty-length"),
            ("hilbert", "hilbert-integer-data"),
            ("hilbert", "hilbert-logical-data"),
            ("hilbert", "hilbert-logical-length"),
            ("hilbert", "hilbert-nd-input"),
            ("hilbert", "hilbert-typed-integer-length"),
            ("hilbert", "hilbert-zero-length"),
            ("hist", "hist-integer-centers"),
            ("hist", "hist-integer-data"),
            ("hist", "hist-modern-options"),
            ("histc", "histc-gpu-input"),
            ("histc", "histc-logical"),
            ("histc", "histc-wide-integer"),
            ("histcounts2", "histcounts2-extra-bin-syntax"),
            ("histcounts2", "histcounts2-gpu-input"),
            ("hypot", "hypot-character-input"),
            ("hypot", "hypot-integer-input"),
            ("hypot", "hypot-logical-input"),
            ("ifft", "ifft-wide-integer-controls"),
            ("ifft", "ifft-wide-integer-data"),
            ("ifft2", "ifft2-empty-zero-size"),
            ("ifft2", "ifft2-size-form"),
            ("ifft2", "ifft2-wide-integer-controls"),
            ("ifft2", "ifft2-wide-integer-data"),
            ("ifftn", "ifftn-short-size-vector"),
            ("ifftn", "ifftn-wide-integer-controls"),
            ("ifftn", "ifftn-wide-integer-data"),
            ("ifftn", "ifftn-zero-size-vector"),
            ("ifftshift", "ifftshift-multi-dimension-selector"),
            ("kmeans", "kmeans-integer-cluster-count"),
            ("kmeans", "kmeans-integer-iteration-control"),
            ("kmeans", "kmeans-integer-observation-data"),
            ("kmeans", "kmeans-integer-parallel-toggle"),
            ("kmeans", "kmeans-integer-start-centroids"),
            ("knnsearch", "knnsearch-gpu-fast-distance"),
            ("knnsearch", "knnsearch-gpu-include-ties"),
            ("knnsearch", "knnsearch-gpu-nonexhaustive-search"),
            ("knnsearch", "knnsearch-gpu-sort-indices"),
            ("knnsearch", "knnsearch-integer-floating-control"),
            ("knnsearch", "knnsearch-integer-logical-control"),
            ("knnsearch", "knnsearch-integer-observation-data"),
            ("knnsearch", "knnsearch-integer-structural-control"),
            ("kurtosis", "kurtosis-gpu-all-or-vecdim"),
            ("kurtosis", "kurtosis-integer-data"),
            ("kurtosis", "kurtosis-typed-integer-control"),
            ("int2str", "int2str-logical-input"),
            ("integral", "integral-integer-bound"),
            ("integral", "integral-integer-option"),
            ("integral", "integral-logical-bound"),
            ("integral", "integral-logical-numeric-option"),
            ("interp1", "interp1-integer-query"),
            ("interp1", "interp1-integer-sample"),
            ("interp1", "interp1-integer-extrapolation"),
            ("interp1", "interp1-logical-input"),
            ("interp1q", "interp1q-gpu-input"),
            ("interp1q", "interp1q-integer-data"),
            ("interp1q", "interp1q-integer-query"),
            ("interp1q", "interp1q-logical-input"),
            ("interp2", "interp2-integer-query"),
            ("interp2", "interp2-integer-sample"),
            ("interp2", "interp2-integer-extrapolation"),
            ("interp2", "interp2-logical-input"),
            ("inv", "inv-integer-input"),
            ("inv", "inv-logical-input"),
            ("lhsdesign", "lhsdesign-explicit-gpu-input"),
            ("lhsdesign", "lhsdesign-integer-dimension"),
            ("lhsdesign", "lhsdesign-integer-iterations"),
            ("lhsdesign", "lhsdesign-integer-smooth"),
            ("linkage", "linkage-explicit-gpu-input"),
            ("linkage", "linkage-integer-data"),
            ("linkage", "linkage-integer-distance-parameter"),
            ("macd", "macd-nondouble-matrix"),
            ("mad", "mad-integer-data"),
            ("mad", "mad-typed-integer-control"),
            ("maxk", "maxk-gpu-input"),
            ("meshgrid", "meshgrid-complex-axes"),
            ("meshgrid", "meshgrid-like"),
            ("minus", "minus-like-prototype"),
            ("mink", "mink-gpu-input"),
            ("mnrfit", "mnrfit-integer-control"),
            ("mnrfit", "mnrfit-integer-parameter"),
            ("mnrfit", "mnrfit-integer-x"),
            ("mnrfit", "mnrfit-integer-y"),
            ("mnrfit", "mnrfit-legacy-options"),
            ("mnrfit", "mnrfit-logical-x"),
            ("mnrfit", "mnrfit-resident-input"),
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
            ("mustBeFinite", "mustBeFinite.resident-input"),
            ("mustBeInteger", "mustBeInteger.resident-input"),
            ("mustBeNonNan", "mustBeNonNan.resident-input"),
            ("mustBeNonzero", "mustBeNonzero.resident-input"),
            ("missing", "missing-shaped-array"),
            ("mvnrnd", "mvnrnd-integer-count"),
            ("mvnrnd", "mvnrnd-integer-mu"),
            ("mvnrnd", "mvnrnd-integer-sigma"),
            ("mvnrnd", "mvnrnd-logical-input"),
            ("mvnrnd", "mvnrnd-resident-count"),
            ("nnz", "nnz-dimension-reduction"),
            ("nanmax", "nanmax-typed-integer-input"),
            ("nanmean", "nanmean-typed-integer-input"),
            ("nanmedian", "nanmedian-typed-integer-input"),
            ("nanmin", "nanmin-typed-integer-input"),
            ("nan", "nan-column-size-vector"),
            ("nan", "nan-resident-size-control"),
            ("nanstd", "nanstd-typed-integer-control"),
            ("nansum", "nansum-typed-integer-input"),
            ("nanvar", "nanvar-typed-integer-control"),
            ("nearestNeighbor", "delaunaytri-integer-coordinates"),
            ("norm", "norm-integer-data"),
            ("norm", "norm-logical-data"),
            ("norm", "norm-logical-order"),
            ("norm", "norm-nuclear-order"),
            ("norm", "norm-zero-order"),
            ("normalize", "normalize-integer-data"),
            ("normalize", "normalize-logical-data"),
            ("normalize", "normalize-resident-control"),
            ("normcdf", "normcdf-integer-mu"),
            ("normcdf", "normcdf-integer-sigma"),
            ("normcdf", "normcdf-integer-x"),
            ("normcdf", "normcdf-logical-input"),
            ("norminv", "norminv-integer-mu"),
            ("norminv", "norminv-integer-p"),
            ("norminv", "norminv-integer-sigma"),
            ("norminv", "norminv-logical-input"),
            ("normpdf", "normpdf-integer-mu"),
            ("normpdf", "normpdf-integer-sigma"),
            ("normpdf", "normpdf-integer-x"),
            ("normpdf", "normpdf-logical-input"),
            ("normrnd", "normrnd-integer-mu"),
            ("normrnd", "normrnd-integer-sigma"),
            ("normrnd", "normrnd-integer-size"),
            ("normrnd", "normrnd-logical-input"),
            ("normrnd", "normrnd-resident-size"),
            ("num2str", "num2str-explicit-gpu-input"),
            ("ones", "ones-column-size-vector"),
            ("ones", "ones-resident-size-control"),
            ("onehotdecode", "onehotdecode-explicit-gpu-input"),
            ("onehotdecode", "onehotdecode-integer-logical-encoded-input"),
            ("onehotdecode", "onehotdecode-logical-cell-output"),
            ("onehotdecode", "onehotdecode-outputtype-alias"),
            ("onehotencode", "onehotencode-explicit-gpu-input"),
            ("onehotencode", "onehotencode-outputtype-alias"),
            ("opentoline", "opentoline-integer-column"),
            ("opentoline", "opentoline-integer-line"),
            ("opentoline", "opentoline-resident-position"),
            ("or", "or-character-input"),
            ("or", "or-complex-input"),
            ("ordinal", "ordinal-explicit-gpu-input"),
            ("pagefun", "pagefun-host-inputs"),
            ("pagefun", "pagefun-text-callable"),
            ("pagemtimes", "pagemtimes-integer-input"),
            ("pan", "pan-integer-graphics-target"),
            ("patch", "patch-integer-axes-handle"),
            ("pause", "pause-gpu-input"),
            ("pchip", "pchip-integer-query-points"),
            ("pchip", "pchip-integer-sample-locations"),
            ("pchip", "pchip-integer-sample-values"),
            ("pdf", "pdf-integer-parameters"),
            ("pdf", "pdf-integer-x"),
            ("pdist", "pdist-integer-numeric-control"),
            ("pdist", "pdist-integer-observation-data"),
            ("pdist2", "pdist2-integer-numeric-control"),
            ("pdist2", "pdist2-integer-observation-data"),
            ("pdist2", "pdist2-integer-selection-count"),
            ("peaks", "peaks-explicit-gpu-input"),
            ("peaks", "peaks-typed-integer-grid-size"),
            ("periodogram", "periodogram-integer-numeric-control"),
            ("periodogram", "periodogram-integer-signal"),
            ("perfcurve", "perfcurve-integer-labels"),
            ("perfcurve", "perfcurve-integer-logical-controls"),
            ("perfcurve", "perfcurve-integer-numeric-options"),
            ("perfcurve", "perfcurve-integer-scores"),
            ("pie", "pie-integer-values"),
            ("pinv", "pinv-integer-input"),
            ("pinv", "pinv-integer-tolerance"),
            ("pinv", "pinv-logical-input"),
            ("pinv", "pinv-logical-tolerance"),
            ("plus", "plus-like-prototype"),
            ("pol2cart", "pol2cart-integer-rho"),
            ("pol2cart", "pol2cart-integer-theta"),
            ("pol2cart", "pol2cart-integer-z"),
            ("polarplot", "polarplot-complex-integer-z"),
            ("polyder", "polyder-integer-coefficients"),
            ("polyfit", "polyfit-integer-weights"),
            ("polyfit", "polyfit-integer-x"),
            ("polyfit", "polyfit-integer-y"),
            ("polyval", "polyval-integer-coefficients"),
            ("polyval", "polyval-integer-fit-options"),
            ("polyval", "polyval-integer-points"),
            ("power", "power-like-prototype"),
            ("pow2", "pow2-integer-binary-exponent"),
            ("pow2", "pow2-integer-significand"),
            ("pow2", "pow2-integer-unary-exponent"),
            ("ppval", "ppval-integer-query-points"),
            ("predict", "predict-integer-statistical-controls"),
            ("predict", "predict-integer-statistical-predictors"),
            ("print", "print-integer-figure-handle"),
            ("pskmod", "pskmod-integer-custom-order"),
            ("pskmod", "pskmod-integer-modulation-order"),
            ("pskmod", "pskmod-integer-phase-offset"),
            ("pointLocation", "delaunaytri-integer-coordinates"),
            ("pointLocation", "delaunaytri-integer-topology"),
            ("prctile", "prctile-integer-data"),
            ("prctile", "prctile-typed-integer-percentage"),
            ("pulstran", "pulstran-integer-delay"),
            ("pulstran", "pulstran-integer-parameter"),
            ("pulstran", "pulstran-integer-prototype"),
            ("pulstran", "pulstran-integer-time"),
            ("pulstran", "pulstran-explicit-gpu-input"),
            ("pwelch", "pwelch-integer-numeric-control"),
            ("pwelch", "pwelch-integer-signal"),
            ("pwelch", "pwelch-logical-input"),
            ("randi", "randi-implicit-prototype"),
            ("randi", "randi-wide-integer-output"),
            ("rand", "rand-column-size-vector"),
            ("rand", "rand-resident-size-control"),
            ("randn", "randn-column-size-vector"),
            ("randn", "randn-resident-size-control"),
            ("randperm", "randperm-explicit-double"),
            ("randperm", "randperm-like"),
            ("random", "random-integer-parameters"),
            ("random", "random-integer-size"),
            ("randsample", "randsample-integer-count"),
            ("randsample", "randsample-integer-population"),
            ("randsample", "randsample-integer-range"),
            ("randsample", "randsample-integer-replacement"),
            ("randsample", "randsample-integer-weights"),
            ("rectpuls", "rectpuls-integer-time"),
            ("rectpuls", "rectpuls-integer-width"),
            ("rectpuls", "rectpuls-explicit-gpu-input"),
            ("rms", "rms-integer-data"),
            ("rgb2lab", "rgb2lab-explicit-gpu"),
            ("rmse", "rmse-integer-data"),
            ("rmse", "rmse-integer-weights"),
            ("quantile", "quantile-integer-data"),
            ("quantile", "quantile-typed-integer-probability"),
            ("qqplot", "qqplot-explicit-gpu-input"),
            ("qqplot", "qqplot-integer-pvec"),
            ("qqplot", "qqplot-integer-sample"),
            ("qqplot", "qqplot-logical-input"),
            ("range", "range-explicit-nanflag"),
            ("range", "range-gpu-all-or-vecdim"),
            ("range", "range-integer-data"),
            ("range", "range-typed-integer-control"),
            ("readcell", "readcell-typed-integer-control"),
            ("readmatrix", "readmatrix-like-output"),
            ("readmatrix", "readmatrix-typed-integer-control"),
            ("readtable", "readtable-typed-integer-control"),
            ("readtimetable", "readtimetable-typed-integer-control"),
            ("rdivide", "rdivide-like-prototype"),
            ("rad2deg", "rad2deg-integer-input"),
            ("rad2deg", "rad2deg-logical-input"),
            ("regexprep", "regexprep-explicit-gpu-input"),
            ("removeLongWords", "removelongwords-explicit-gpu-input"),
            ("removeShortWords", "removeshortwords-explicit-gpu-input"),
            ("removeWords", "removewords-explicit-gpu-input"),
            ("replaceBetween", "replacebetween-explicit-gpu-input"),
            ("refline", "refline-explicit-gpu-coefficients"),
            ("refline", "refline-integer-coefficients"),
            ("refline", "refline-logical-coefficients"),
            ("sawtooth", "sawtooth-gpu-input"),
            ("sawtooth", "sawtooth-nondouble-input"),
            ("sinc", "sinc-nonfloating-input"),
            ("skewness", "skewness-gpu-all-or-vecdim"),
            ("skewness", "skewness-integer-data"),
            ("skewness", "skewness-typed-integer-control"),
            ("split", "split-advanced-options"),
            ("split", "split-resident-dimension"),
            ("split", "split-typed-dimension"),
            ("sparse", "sparse-integer-storage"),
            ("square", "square-gpu-input"),
            ("square", "square-nonfloating-input"),
            ("startsWith", "startswith-numeric-ignore-case"),
            ("startsWith", "startswith-positional-ignore-case"),
            ("startsWith", "startswith-resident-ignore-case"),
            ("startsWith", "startswith-text-ignore-case"),
            ("std", "std-typed-integer-control"),
            (
                "spreadsheetImportOptions",
                "spreadsheetimportoptions-typed-integer-location-control",
            ),
            ("strfind", "strfind-nonbinary-force-cell-output"),
            ("strfind", "strfind-resident-force-cell-output"),
            ("strfind", "strfind-typed-force-cell-output"),
            ("tabulate", "tabulate-gpu-input"),
            ("tabulate", "tabulate-integer-data"),
            ("textscan", "textscan-typed-integer-control"),
            ("times", "times-like-prototype"),
            ("tiedrank", "tiedrank-integer-data"),
            ("trnd", "trnd-integer-degrees-of-freedom"),
            ("trnd", "trnd-integer-size"),
            ("uint16.empty", "empty-resident-size"),
            ("uint32.empty", "empty-resident-size"),
            ("uint64.empty", "empty-resident-size"),
            ("uint8.empty", "empty-resident-size"),
            ("var", "var-typed-integer-control"),
            ("xor", "xor-character-input"),
            ("xor", "xor-complex-input"),
            ("lsqcurvefit", "lsqcurvefit-integer-bound"),
            ("lsqcurvefit", "lsqcurvefit-integer-callback-result"),
            ("lsqcurvefit", "lsqcurvefit-integer-option"),
            ("lsqcurvefit", "lsqcurvefit-integer-x0"),
            ("lsqcurvefit", "lsqcurvefit-integer-xdata"),
            ("lsqcurvefit", "lsqcurvefit-integer-ydata"),
            ("lsqcurvefit", "lsqcurvefit-logical-numeric"),
            ("lsqcurvefit", "lsqcurvefit-resident-input"),
            ("null", "null-integer-matrix"),
            ("null", "null-logical-matrix"),
            ("null", "null-resident-tolerance"),
            ("qr", "qr-integer-input"),
            ("qr", "qr-integer-option"),
            ("qr", "qr-logical-input"),
            ("qr", "qr-logical-option"),
            ("qr", "qr-resident-option"),
            ("rank", "rank-integer-input"),
            ("rank", "rank-integer-tolerance"),
            ("rank", "rank-logical-input"),
            ("rank", "rank-logical-tolerance"),
            ("rcond", "rcond-integer-input"),
            ("rcond", "rcond-logical-input"),
            ("ode15s", "ode15s-integer-callback-result"),
            ("ode15s", "ode15s-integer-option"),
            ("ode15s", "ode15s-integer-tspan"),
            ("ode15s", "ode15s-integer-y0"),
            ("ode15s", "ode15s-logical-numeric"),
            ("ode15s", "ode15s-resident-input"),
            ("ode23", "ode23-integer-callback-result"),
            ("ode23", "ode23-integer-option"),
            ("ode23", "ode23-integer-tspan"),
            ("ode23", "ode23-integer-y0"),
            ("ode23", "ode23-logical-numeric"),
            ("ode23", "ode23-resident-input"),
            ("ode45", "ode45-integer-callback-result"),
            ("ode45", "ode45-integer-option"),
            ("ode45", "ode45-integer-tspan"),
            ("ode45", "ode45-integer-y0"),
            ("ode45", "ode45-logical-numeric"),
            ("ode45", "ode45-resident-input"),
            ("optimoptions", "optimoptions-integer-option"),
            ("optimoptions", "optimoptions-resident-option"),
            ("optimset", "optimset-integer-option"),
            ("optimset", "optimset-resident-option"),
            ("quad", "quad-integer-bound"),
            ("quad", "quad-integer-callback-result"),
            ("quad", "quad-integer-tolerance"),
            ("quad", "quad-logical-numeric"),
            ("quad", "quad-resident-input"),
            ("loadobj", "loadobj-plain-payload-passthrough"),
            ("memmapfile", "memmapfile-explicit-gpu-argument"),
            ("memmapfile", "memmapfile-integer-property-controls"),
            ("numel", "numel-dimension-selectors"),
        ]);
        assert_eq!(
            declared.difference(&expected).copied().collect::<Vec<_>>(),
            Vec::<(&str, &str)>::new(),
            "unexpected live extensions"
        );
        assert_eq!(
            expected.difference(&declared).copied().collect::<Vec<_>>(),
            Vec::<(&str, &str)>::new(),
            "missing live extensions"
        );
    }

    #[test]
    fn every_builtin_extension_has_a_structured_compatibility_identifier() {
        for builtin in runmat_builtins::builtin_functions() {
            for extension in builtin.extensions {
                assert!(
                    extension.error_identifier.is_some_and(|identifier| {
                        identifier.starts_with("RunMat:compatibility:")
                    }),
                    "{} extension {} must use the compatibility identifier namespace",
                    builtin.name,
                    extension.id
                );
            }
        }
    }

    #[test]
    fn core_operator_extensions_reject_before_semantic_dispatch_in_matlab_mode() {
        let _matlab_mode = push_runmat_extensions_enabled(false);
        let like_cases = [
            ("plus", "RunMat:compatibility:PlusLikePrototypeExtension"),
            ("minus", "RunMat:compatibility:MinusLikePrototypeExtension"),
            ("times", "RunMat:compatibility:TimesLikePrototypeExtension"),
            (
                "rdivide",
                "RunMat:compatibility:RdivideLikePrototypeExtension",
            ),
            ("power", "RunMat:compatibility:PowerLikePrototypeExtension"),
        ];
        for (name, identifier) in like_cases {
            let builtin = runmat_builtins::builtin_function_by_name(name).expect("builtin");
            let args = [
                Value::Num(2.0),
                Value::Num(1.0),
                Value::String("like".to_string()),
                Value::Num(0.0),
            ];
            let error = futures::executor::block_on((builtin.implementation)(&args))
                .expect_err("MATLAB mode must reject RunMat-only like prototype");
            assert_eq!(error.identifier(), Some(identifier), "{name}");
        }

        for (name, identifier) in [
            ("or", "RunMat:compatibility:OrComplexInputExtension"),
            ("xor", "RunMat:compatibility:XorComplexInputExtension"),
        ] {
            let builtin = runmat_builtins::builtin_function_by_name(name).expect("builtin");
            let args = [Value::Complex(0.0, 1.0), Value::Bool(false)];
            let error = futures::executor::block_on((builtin.implementation)(&args))
                .expect_err("MATLAB mode must reject RunMat-only complex logical input");
            assert_eq!(error.identifier(), Some(identifier), "{name}");
        }

        for (name, identifier) in [
            ("or", "RunMat:compatibility:OrCharacterInputExtension"),
            ("xor", "RunMat:compatibility:XorCharacterInputExtension"),
        ] {
            let builtin = runmat_builtins::builtin_function_by_name(name).expect("builtin");
            let args = [
                Value::CharArray(runmat_builtins::CharArray::new_row("A")),
                Value::Bool(false),
            ];
            let error = futures::executor::block_on((builtin.implementation)(&args))
                .expect_err("MATLAB mode must reject RunMat-only character logical input");
            assert_eq!(error.identifier(), Some(identifier), "{name}");
        }
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
            "DataArray.fill",
            "DataArray.read",
            "DataArray.resize",
            "DataArray.write",
            "DataTransaction.create_array",
            "DataTransaction.fill",
            "DataTransaction.resize",
            "DataTransaction.set_attr",
            "DataTransaction.set_attrs",
            "DataTransaction.write",
            "Dataset.attrs",
            "Dataset.get_attr",
            "Dataset.set_attr",
            "Dataset.set_attrs",
            "DelaunayTri",
            "DelaunayTri.freeBoundary",
            "DelaunayTri.nearestNeighbor",
            "DelaunayTri.pointLocation",
            "circshift",
            "corr",
            "corrcoef",
            "corrcov",
            "cov",
            "cov2corr",
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
            "intersect",
            "ismember",
            "isprime",
            "issorted",
            "lcm",
            "max",
            "maxk",
            "mean",
            "median",
            "min",
            "mink",
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
            "setdiff",
            "setxor",
            "sort",
            "std",
            "sum",
            "trnd",
            "union",
            "var",
            "bounds",
            "prod",
            "accept",
            "area",
            "array2table",
            "atan2",
            "audioread",
            "axes",
            "axis",
            "bagOfNgrams",
            "bagOfWords",
            "bandwidth",
            "bitand",
            "bitor",
            "bitshift",
            "blackman",
            "blanks",
            "blkdiag",
            "bootstrp",
            "boxplot",
            "bsxfun",
            "builtin",
            "butter",
            "buttord",
            "cat",
            "cdf",
            "cdfplot",
            "ceil",
            "cell",
            "cell2mat",
            "cell2struct",
            "cell2table",
            "cellfun",
            "char",
            "cheb2ord",
            "chi2cdf",
            "chol",
            "class",
            "classify",
            "clf",
            "close",
            "colon",
            "colorcube",
            "colormap",
            "colororder",
            "combinations",
            "complex",
            "compose",
            "cond",
            "confusionmat",
            "conj",
            "containers.Map",
            "containers.Map.isKey",
            "containers.Map.keys",
            "containers.Map.remove",
            "containers.Map.subsasgn",
            "containers.Map.subsref",
            "containers.Map.values",
            "contour",
            "contourf",
            "conv",
            "conv2",
            "copyobj",
            "cos",
            "cosd",
            "cosh",
            "cosineSimilarity",
            "cospi",
            "cross",
            "contains",
            "convertCharsToStrings",
            "convertContainedStringsToChars",
            "convertStringsToChars",
            "dbstack",
            "disp",
            "display",
            "data.create",
            "error",
            "errorbar",
            "erf",
            "erfcinv",
            "exp",
            "expm1",
            "exprnd",
            "extractAfter",
            "extractBefore",
            "extractBetween",
            "extractFileText",
            "eye",
            "false",
            "fclose",
            "fcontour",
            "feedback",
            "fea.boundaryCondition",
            "fea.domain",
            "fea.interface",
            "fea.loadCase",
            "fea.material",
            "fea.results",
            "fea.runOptions",
            "fea.trends",
            "feof",
            "fitclinear",
            "fitctree",
            "fitdist",
            "fitlm",
            "fminbnd",
            "fminunc",
            "fsolve",
            "fzero",
            "gradient",
            "gray2rgb",
            "griddedInterpolant",
            "groupcounts",
            "groupsummary",
            "grp2idx",
            "grpstats",
            "gt",
            "hamming",
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
                assert!(
                    !capability.inputs.is_empty()
                        || capability.output_class
                            != runmat_builtins::BuiltinIntegerOutputClassRule::NotApplicable,
                    "{name} inputs or integer output"
                );
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
            [
                "DataArray.chunk_shape",
                "DataArray.codec",
                "DataArray.dtype",
                "DataArray.name",
                "DataArray.rank",
                "DataArray.shape",
                "DataTransaction.abort",
                "DataTransaction.commit",
                "DataTransaction.delete_array",
                "DataTransaction.id",
                "DataTransaction.status",
                "Dataset.array",
                "Dataset.arrays",
                "Dataset.begin",
                "Dataset.has_array",
                "Dataset.id",
                "Dataset.path",
                "Dataset.refresh",
                "Dataset.snapshot",
                "Dataset.version",
                "addDependencyDetails",
                "addEntityDetails",
                "addLemmaDetails",
                "addPartOfSpeechDetails",
                "addSentenceDetails",
                "addTypeDetails",
                "addlistener",
                "addprop",
                "ancestor",
                "append",
                "argsort",
                "cancel",
                "caxis",
                "cellstr",
                "clearCache",
                "commit",
                "data.copy",
                "data.delete",
                "data.exists",
                "data.export",
                "data.import",
                "data.inspect",
                "data.list",
                "data.move",
                "data.open",
                "deblank",
                "erasePunctuation",
                "eraseURLs",
                "exist",
                "extractHTMLText",
                "fea.compare",
                "fea.field",
                "fea.materialAssignment",
                "fea.model",
                "fea.plan",
                "fea.plot",
                "fea.run",
                "fea.step",
                "fea.study",
                "fea.sweep",
                "fea.validate",
                "fieldnames",
                "fileattrib",
                "fileparts",
                "fileread",
                "findElement",
                "findprop",
                "func2str",
                "functions",
                "genvarname",
                "geometry.listRegions",
                "geometry.meshes",
                "getAttribute",
                "getenv",
                "getmethod",
                "hgload",
                "hgsave",
                "htmlTree",
                "input",
                "isStringScalar",
                "isVocabularyWord",
                "iscategorical",
                "iscell",
                "iscellstr",
                "ischar",
                "iscolumn",
                "isdeployed",
                "isdiag",
                "isempty",
                "isenv",
                "isfield",
                "isfile",
                "isfolder",
                "isgraphics",
                "ishandle",
                "ishermitian",
                "isletter",
                "islogical",
                "ismatrix",
                "ismethod",
                "isobject",
                "isordinal",
                "ispref",
                "isrow",
                "isscalar",
                "isspace",
                "isstable",
                "isstring",
                "issymmetric",
                "istable",
                "istimetable",
                "istril",
                "istriu",
                "isvalid",
                "isvarname",
                "isvector",
                "jsondecode",
                "legend",
                "lower",
                "matches",
                "matlab.metadata.DynamicProperty.delete",
                "memoize",
                "mustBeFile",
                "mustBeFolder",
                "mustBeNonzeroLengthText",
                "mustBeText",
                "mustBeTextScalar",
                "mustBeValidVariableName",
                "namedargs2cell",
                "normalizeWords",
                "notify",
                "onCleanup",
                "openfig",
                "pattern",
                "readWordEmbedding",
                "readline",
                "regexp",
                "regexpPattern",
                "regexpi",
                "replace",
                "rethrow",
                "reverse",
                "splitlines",
                "strip",
                "upper",
            ]
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
