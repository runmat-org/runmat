use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, HANN_ERROR_ARG_COUNT, HANN_ERROR_INTERNAL, HANN_ERROR_INVALID_LENGTH,
    HANN_ERROR_INVALID_OPTION, HANN_ERROR_UNKNOWN_OPTION, HANN_LOGICAL_LENGTH_EXTENSION,
};
use runmat_macros::runtime_builtin;

use crate::builtins::math::signal::common::window_provider::{try_generate, CosineWindow};
use crate::builtins::math::signal::common::{
    is_scalar_logical_window_length, parse_window_controls_view, window_allocation_error,
    window_options_from_length, window_tensor, SignalCooperation, WindowArgError,
};
use crate::builtins::math::trigonometry::pi_helpers::cospi_real;

const BUILTIN_NAME: &str = "hann";

fn hann_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    hann_error_with_message(error.message, error)
}

fn hann_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    hann_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn hann_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hann_map_window_error(error: WindowArgError) -> RuntimeError {
    match error {
        WindowArgError::Text(source) => {
            let mut error = hann_error(&HANN_ERROR_INVALID_OPTION);
            error.source = Some(Box::new(source));
            error
        }
        WindowArgError::TooManyOptions => hann_error(&HANN_ERROR_ARG_COUNT),
        WindowArgError::InvalidLength => hann_error(&HANN_ERROR_INVALID_LENGTH),
        WindowArgError::InvalidOptionType => hann_error(&HANN_ERROR_INVALID_OPTION),
        WindowArgError::UnknownOption(option) => {
            hann_error_with_detail(&HANN_ERROR_UNKNOWN_OPTION, format!("'{option}'"))
        }
        WindowArgError::Allocation(cause) => {
            window_allocation_error(BUILTIN_NAME, &HANN_ERROR_INTERNAL, cause)
        }
        WindowArgError::TensorBuild(detail) => hann_error_with_detail(&HANN_ERROR_INTERNAL, detail),
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::hann")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "hann",
    op_kind: GpuOpKind::Custom("window"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("hann_window")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Generates the Hann window directly on the active provider when supported; only a typed unsupported hook falls back to host construction.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::hann")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "hann",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "hann materialises a new window vector and is not currently fused.",
};

#[runtime_builtin(
    name = "hann",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::hann"
)]
async fn hann_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    n: runmat_value::Value,
    varargin: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<runmat_value::Value> {
    hann_in_runtime(runtime.as_ref(), n, varargin).await
}

#[cfg(test)]
async fn hann_builtin(
    n: runmat_value::Value,
    varargin: Vec<runmat_value::Value>,
) -> crate::BuiltinResult<runmat_value::Value> {
    hann_in_runtime(
        None,
        n,
        crate::call::BuiltinArguments::from_borrowed(&varargin, None)?,
    )
    .await
}

async fn hann_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    n: runmat_value::Value,
    varargin: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<runmat_value::Value> {
    let controls =
        parse_window_controls_view(varargin.view(), true).map_err(hann_map_window_error)?;
    if is_scalar_logical_window_length(&n) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HANN_LOGICAL_LENGTH_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let options = window_options_from_length(n, controls).map_err(hann_map_window_error)?;
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME);
    cooperation.completed(0).await?;
    if let Some(value) = try_generate(CosineWindow::Hann, options)? {
        return Ok(value);
    }
    window_tensor(
        options,
        |idx, total| {
            let denom = (total - 1) as f64;
            0.5 - 0.5 * cospi_real(2.0 * idx as f64 / denom)
        },
        &mut cooperation,
    )
    .await?
    .map_err(hann_map_window_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::HANN_CATALOG_ENTRY;
    use runmat_value::{IntValue, Value};

    #[test]
    fn registered_hann_binding_uses_invocation_cancellation() {
        let source = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        source.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(hann_builtin_registered(
            Some(runtime),
            Value::Num(8193.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("cancelled invocation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }

    #[test]
    fn hann_descriptor_signatures_and_errors() {
        let descriptor = HANN_CATALOG_ENTRY.descriptor;
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"w = hann(n)"));
        assert!(labels.contains(&"w = hann(n, sampling)"));
        assert!(labels.contains(&"w = hann(n, precision)"));
        assert!(labels.contains(&"w = hann(n, sampling, precision)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.HANN.INVALID_LENGTH"));
    }

    #[test]
    fn hann_integer_metadata_and_logical_extension_are_registered() {
        assert_eq!(HANN_CATALOG_ENTRY.integer_capabilities.len(), 1);
        assert_eq!(HANN_CATALOG_ENTRY.extensions.len(), 1);

        let integer = block_on(super::hann_builtin(
            Value::Int(IntValue::U64(4)),
            Vec::new(),
        ))
        .expect("documented integer length");
        assert_eq!(test_support::gather(integer).unwrap().shape(), vec![4, 1]);

        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(super::hann_builtin(Value::Bool(true), Vec::new()))
            .expect_err("strict mode rejects logical length");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:HannLogicalLengthExtension")
        );
    }

    #[test]
    fn hann_scalar_logical_array_uses_the_same_extension_as_bool() {
        let _guard = test_support::accel_test_lock();
        let logical = |bit| {
            Value::LogicalArray(runmat_value::LogicalArray::new(vec![bit], vec![1, 1]).unwrap())
        };
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(hann_builtin(logical(1), Vec::new())).unwrap_err();
            assert_eq!(
                error.identifier(),
                HANN_LOGICAL_LENGTH_EXTENSION.error_identifier
            );
        }
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for (bit, shape) in [(0, vec![0, 1]), (1, vec![1, 1])] {
            let result = block_on(hann_builtin(logical(bit), Vec::new())).unwrap();
            assert_eq!(test_support::gather(result).unwrap().shape(), shape);
        }
        let nonscalar =
            Value::LogicalArray(runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap());
        let error = block_on(hann_builtin(nonscalar, Vec::new())).unwrap_err();
        assert_eq!(error.identifier(), HANN_ERROR_INVALID_LENGTH.identifier);
    }

    #[test]
    fn hann_rejects_extra_or_misordered_options() {
        let extra = block_on(super::hann_builtin(
            Value::Num(4.0),
            vec![
                Value::from("periodic"),
                Value::from("single"),
                Value::from("double"),
            ],
        ))
        .expect_err("too many options");
        assert_eq!(extra.identifier(), Some("RunMat:hann:ArgumentCount"));

        let misordered = block_on(super::hann_builtin(
            Value::Num(4.0),
            vec![Value::from("single"), Value::from("periodic")],
        ))
        .expect_err("sampling must precede output type");
        assert_eq!(misordered.identifier(), Some("RunMat:hann:UnknownOption"));
    }

    #[test]
    fn hann_returns_expected_values() {
        let _guard = test_support::accel_test_lock();
        let t = test_support::gather(
            block_on(hann_builtin(Value::Num(8.0), Vec::new())).expect("hann"),
        )
        .expect("gather hann");
        let expected = [
            0.0,
            0.1882550990706332,
            0.6112604669781572,
            0.9504844339512095,
            0.9504844339512095,
            0.6112604669781573,
            0.1882550990706333,
            0.0,
        ];
        assert_eq!(t.shape(), vec![8, 1]);
        for (got, want) in t.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn hann_handles_zero_and_one_lengths() {
        let _guard = test_support::accel_test_lock();
        let zero = test_support::gather(
            block_on(hann_builtin(Value::Num(0.0), Vec::new())).expect("hann(0)"),
        )
        .expect("gather hann(0)");
        assert_eq!(zero.shape(), vec![0, 1]);
        assert!(zero.materialize_f64().is_empty());

        let one = test_support::gather(
            block_on(hann_builtin(Value::Num(1.0), Vec::new())).expect("hann(1)"),
        )
        .expect("gather hann(1)");
        assert_eq!(one.shape(), vec![1, 1]);
        assert_eq!(one.materialize_f64(), vec![1.0]);
    }

    #[test]
    fn hann_rejects_invalid_lengths() {
        let _guard = test_support::accel_test_lock();
        assert!(block_on(hann_builtin(Value::Num(-1.0), Vec::new())).is_err());
        let rounded = test_support::gather(
            block_on(hann_builtin(Value::Num(2.5), Vec::new())).expect("hann rounded"),
        )
        .expect("gather hann rounded");
        assert_eq!(rounded.shape(), vec![3, 1]);
        assert!(block_on(hann_builtin(
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Vec::new()
        ))
        .is_err());
    }

    #[test]
    fn hann_supports_periodic_and_single_overloads() {
        let _guard = test_support::accel_test_lock();
        let periodic = test_support::gather(
            block_on(hann_builtin(Value::Num(4.0), vec![Value::from("periodic")]))
                .expect("hann periodic"),
        )
        .expect("gather hann periodic");
        assert_eq!(periodic.shape(), vec![4, 1]);
        assert!((periodic.materialize_f64()[1] - 0.5).abs() < 1e-12);

        let single = test_support::gather(
            block_on(hann_builtin(Value::Num(4.0), vec![Value::from("single")]))
                .expect("hann single"),
        )
        .expect("gather hann single");
        assert_eq!(single.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    #[test]
    fn hann_gpu_matches_cpu() {
        test_support::with_test_provider(|_| {
            let value = block_on(hann_builtin(Value::Num(8.0), Vec::new())).expect("hann gpu");
            let tensor = test_support::gather(value).expect("gather");
            assert_eq!(tensor.shape(), vec![8, 1]);
            assert!((tensor.materialize_f64()[3] - 0.9504844339512095).abs() < 1e-12);

            let periodic = block_on(hann_builtin(Value::Num(4.0), vec![Value::from("periodic")]))
                .expect("hann periodic gpu");
            let periodic = test_support::gather(periodic).expect("gather periodic");
            assert_eq!(periodic.shape(), vec![4, 1]);
            assert!((periodic.materialize_f64()[1] - 0.5).abs() < 1e-12);

            let periodic_one =
                block_on(hann_builtin(Value::Num(1.0), vec![Value::from("periodic")]))
                    .expect("hann periodic len1 gpu");
            let periodic_one = test_support::gather(periodic_one).expect("gather periodic len1");
            assert_eq!(periodic_one.materialize_f64(), vec![1.0]);
        });
    }
}
