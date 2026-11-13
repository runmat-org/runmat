use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, BLACKMAN_ERROR_ARG_COUNT, BLACKMAN_ERROR_INTERNAL,
    BLACKMAN_ERROR_INVALID_LENGTH, BLACKMAN_ERROR_INVALID_OPTION, BLACKMAN_ERROR_UNKNOWN_OPTION,
    BLACKMAN_LOGICAL_LENGTH_EXTENSION,
};
use runmat_macros::runtime_builtin;

use crate::builtins::math::signal::common::window_provider::{try_generate, CosineWindow};
use crate::builtins::math::signal::common::{
    is_scalar_logical_window_length, parse_window_controls_view, window_allocation_error,
    window_options_from_length, window_tensor, SignalCooperation, WindowArgError,
};
use crate::builtins::math::trigonometry::pi_helpers::cospi_real;

const BUILTIN_NAME: &str = "blackman";

fn blackman_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    blackman_error_with_message(error.message, error)
}

fn blackman_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    blackman_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn blackman_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn blackman_map_window_error(error: WindowArgError) -> RuntimeError {
    match error {
        WindowArgError::Text(source) => {
            let mut error = blackman_error(&BLACKMAN_ERROR_INVALID_OPTION);
            error.source = Some(Box::new(source));
            error
        }
        WindowArgError::TooManyOptions => blackman_error(&BLACKMAN_ERROR_ARG_COUNT),
        WindowArgError::InvalidLength => blackman_error(&BLACKMAN_ERROR_INVALID_LENGTH),
        WindowArgError::InvalidOptionType => blackman_error(&BLACKMAN_ERROR_INVALID_OPTION),
        WindowArgError::UnknownOption(option) => {
            blackman_error_with_detail(&BLACKMAN_ERROR_UNKNOWN_OPTION, format!("'{option}'"))
        }
        WindowArgError::Allocation(cause) => {
            window_allocation_error(BUILTIN_NAME, &BLACKMAN_ERROR_INTERNAL, cause)
        }
        WindowArgError::TensorBuild(detail) => {
            blackman_error_with_detail(&BLACKMAN_ERROR_INTERNAL, detail)
        }
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::blackman")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "blackman",
    op_kind: GpuOpKind::Custom("window"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("blackman_window")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Generates the Blackman window directly on the active provider when supported; only a typed unsupported hook falls back to host construction.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::blackman")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "blackman",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "blackman materialises a new window vector and is not currently fused.",
};

#[runtime_builtin(
    name = "blackman",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::blackman"
)]
async fn blackman_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    n: runmat_value::Value,
    varargin: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<runmat_value::Value> {
    blackman_in_runtime(runtime.as_ref(), n, varargin).await
}

#[cfg(test)]
async fn blackman_builtin(
    n: runmat_value::Value,
    varargin: Vec<runmat_value::Value>,
) -> crate::BuiltinResult<runmat_value::Value> {
    blackman_in_runtime(
        None,
        n,
        crate::call::BuiltinArguments::from_borrowed(&varargin, None)?,
    )
    .await
}

async fn blackman_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    n: runmat_value::Value,
    varargin: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<runmat_value::Value> {
    let controls =
        parse_window_controls_view(varargin.view(), true).map_err(blackman_map_window_error)?;
    if is_scalar_logical_window_length(&n) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BLACKMAN_LOGICAL_LENGTH_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let options = window_options_from_length(n, controls).map_err(blackman_map_window_error)?;
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME);
    cooperation.completed(0).await?;
    if let Some(value) = try_generate(CosineWindow::Blackman, options)? {
        return Ok(value);
    }
    window_tensor(
        options,
        |idx, total| {
            let denom = (total - 1) as f64;
            let first = 2.0 * idx as f64 / denom;
            let second = 4.0 * idx as f64 / denom;
            0.42 - 0.5 * cospi_real(first) + 0.08 * cospi_real(second)
        },
        &mut cooperation,
    )
    .await?
    .map_err(blackman_map_window_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        BLACKMAN_CATALOG_ENTRY, BLACKMAN_EXTENSIONS, BLACKMAN_INTEGER_CAPABILITIES,
    };
    use runmat_value::{IntValue, NumericDType, Value};

    #[test]
    fn registered_blackman_binding_uses_invocation_cancellation() {
        let source = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        source.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(blackman_builtin_registered(
            Some(runtime),
            Value::Num(8193.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("cancelled invocation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }

    #[test]
    fn blackman_returns_expected_values() {
        let _guard = test_support::accel_test_lock();
        let t = test_support::gather(
            block_on(blackman_builtin(Value::Num(8.0), Vec::new())).expect("blackman"),
        )
        .expect("gather blackman");
        let expected = [
            -1.3877787807814457e-17,
            0.09045342435412812,
            0.45918295754596355,
            0.9203636180999081,
            0.9203636180999083,
            0.45918295754596383,
            0.09045342435412818,
            -1.3877787807814457e-17,
        ];
        assert_eq!(t.shape(), vec![8, 1]);
        for (got, want) in t.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn blackman_descriptor_signatures_and_errors() {
        let descriptor = BLACKMAN_CATALOG_ENTRY.descriptor;
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"w = blackman(n)"));
        assert!(labels.contains(&"w = blackman(n, sampling)"));
        assert!(labels.contains(&"w = blackman(n, precision)"));
        assert!(labels.contains(&"w = blackman(n, sampling, precision)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.BLACKMAN.INVALID_LENGTH"));
        assert_eq!(BLACKMAN_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            BLACKMAN_INTEGER_CAPABILITIES[0].inputs[0].classes,
            crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
        );
        assert_eq!(BLACKMAN_EXTENSIONS, [BLACKMAN_LOGICAL_LENGTH_EXTENSION]);
    }

    #[test]
    fn blackman_handles_zero_and_one_lengths() {
        let _guard = test_support::accel_test_lock();
        let zero = test_support::gather(
            block_on(blackman_builtin(Value::Num(0.0), Vec::new())).expect("blackman(0)"),
        )
        .expect("gather blackman(0)");
        assert_eq!(zero.shape(), vec![0, 1]);
        assert!(zero.materialize_f64().is_empty());

        let one = test_support::gather(
            block_on(blackman_builtin(Value::Num(1.0), Vec::new())).expect("blackman(1)"),
        )
        .expect("gather blackman(1)");
        assert_eq!(one.shape(), vec![1, 1]);
        assert_eq!(one.materialize_f64(), vec![1.0]);
    }

    #[test]
    fn blackman_rejects_invalid_lengths() {
        let _guard = test_support::accel_test_lock();
        assert!(block_on(blackman_builtin(Value::Num(-1.0), Vec::new())).is_err());
        let rounded = test_support::gather(
            block_on(blackman_builtin(Value::Num(2.5), Vec::new())).expect("blackman rounded"),
        )
        .expect("gather blackman rounded");
        assert_eq!(rounded.shape(), vec![3, 1]);
        assert!(block_on(blackman_builtin(
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Vec::new()
        ))
        .is_err());
    }

    #[test]
    fn blackman_accepts_every_integer_length_class_and_selects_output_precision() {
        let _guard = test_support::accel_test_lock();
        for length in [
            IntValue::I8(5),
            IntValue::I16(5),
            IntValue::I32(5),
            IntValue::I64(5),
            IntValue::U8(5),
            IntValue::U16(5),
            IntValue::U32(5),
            IntValue::U64(5),
        ] {
            let output = test_support::gather(
                block_on(blackman_builtin(Value::Int(length.clone()), Vec::new()))
                    .expect("integer length"),
            )
            .expect("gather default output");
            assert_eq!(output.shape(), vec![5, 1]);
            assert_eq!(output.numeric_dtype(), NumericDType::F64);

            let single = test_support::gather(
                block_on(blackman_builtin(
                    Value::Int(length),
                    vec![Value::from("single")],
                ))
                .expect("integer length with single output"),
            )
            .expect("gather single output");
            assert_eq!(single.shape(), vec![5, 1]);
            assert_eq!(single.numeric_dtype(), NumericDType::F32);
        }
    }

    #[test]
    fn blackman_logical_length_is_mode_gated_and_argument_count_precedes_the_gate() {
        let _guard = test_support::accel_test_lock();
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(blackman_builtin(Value::Bool(true), Vec::new()))
                .expect_err("logical length must be gated");
            assert_eq!(
                error.identifier(),
                BLACKMAN_LOGICAL_LENGTH_EXTENSION.error_identifier
            );
            let error = block_on(blackman_builtin(
                Value::Bool(true),
                vec![
                    Value::from("symmetric"),
                    Value::from("double"),
                    Value::from("single"),
                ],
            ))
            .expect_err("argument count must reject first");
            assert_eq!(error.identifier(), BLACKMAN_ERROR_ARG_COUNT.identifier);
        }

        {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let output = test_support::gather(
                block_on(blackman_builtin(Value::Bool(true), Vec::new()))
                    .expect("RunMat mode accepts logical length"),
            )
            .expect("gather logical-length output");
            assert_eq!(output.shape(), vec![1, 1]);
            assert_eq!(output.numeric_dtype(), NumericDType::F64);
        }
    }

    #[test]
    fn blackman_scalar_logical_array_uses_the_same_extension_as_bool() {
        let _guard = test_support::accel_test_lock();
        let logical = |bit| {
            Value::LogicalArray(runmat_value::LogicalArray::new(vec![bit], vec![1, 1]).unwrap())
        };
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(blackman_builtin(logical(1), Vec::new())).unwrap_err();
            assert_eq!(
                error.identifier(),
                BLACKMAN_LOGICAL_LENGTH_EXTENSION.error_identifier
            );
        }
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for (bit, shape) in [(0, vec![0, 1]), (1, vec![1, 1])] {
            let result = block_on(blackman_builtin(logical(bit), Vec::new())).unwrap();
            assert_eq!(test_support::gather(result).unwrap().shape(), shape);
        }
        let nonscalar =
            Value::LogicalArray(runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap());
        let error = block_on(blackman_builtin(nonscalar, Vec::new())).unwrap_err();
        assert_eq!(error.identifier(), BLACKMAN_ERROR_INVALID_LENGTH.identifier);
    }

    #[test]
    fn blackman_rejects_reversed_duplicate_and_excess_options() {
        let _guard = test_support::accel_test_lock();
        for options in [
            vec![Value::from("single"), Value::from("periodic")],
            vec![Value::from("periodic"), Value::from("periodic")],
        ] {
            let error = block_on(blackman_builtin(Value::Num(5.0), options))
                .expect_err("invalid option order");
            assert_eq!(error.identifier(), BLACKMAN_ERROR_UNKNOWN_OPTION.identifier);
        }
        let error = block_on(blackman_builtin(
            Value::Num(5.0),
            vec![
                Value::from("symmetric"),
                Value::from("double"),
                Value::from("single"),
            ],
        ))
        .expect_err("too many options");
        assert_eq!(error.identifier(), BLACKMAN_ERROR_ARG_COUNT.identifier);
    }

    #[test]
    fn blackman_uses_cospi_coefficient_accuracy() {
        let _guard = test_support::accel_test_lock();
        let output = test_support::gather(
            block_on(blackman_builtin(Value::Int(IntValue::U8(5)), Vec::new()))
                .expect("five-point window"),
        )
        .expect("gather five-point window");
        let improved = 0.42 - 0.5 * cospi_real(0.5) + 0.08 * cospi_real(1.0);
        let legacy =
            0.42 - 0.5 * (0.5 * std::f64::consts::PI).cos() + 0.08 * std::f64::consts::PI.cos();
        assert_ne!(improved.to_bits(), legacy.to_bits());
        assert_eq!(output.materialize_f64()[1].to_bits(), improved.to_bits());
    }

    #[test]
    fn blackman_supports_periodic_and_single_overloads() {
        let _guard = test_support::accel_test_lock();
        let periodic = test_support::gather(
            block_on(blackman_builtin(
                Value::Num(4.0),
                vec![Value::from("periodic")],
            ))
            .expect("blackman periodic"),
        )
        .expect("gather blackman periodic");
        assert_eq!(periodic.shape(), vec![4, 1]);
        assert!((periodic.materialize_f64()[1] - 0.34).abs() < 1e-12);

        let single = test_support::gather(
            block_on(blackman_builtin(
                Value::Num(4.0),
                vec![Value::from("single")],
            ))
            .expect("blackman single"),
        )
        .expect("gather blackman single");
        assert_eq!(single.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    #[test]
    fn blackman_gpu_matches_cpu() {
        test_support::with_test_provider(|_| {
            let value =
                block_on(blackman_builtin(Value::Num(8.0), Vec::new())).expect("blackman gpu");
            let tensor = test_support::gather(value).expect("gather");
            assert_eq!(tensor.shape(), vec![8, 1]);
            assert!((tensor.materialize_f64()[3] - 0.9203636180999081).abs() < 1e-12);

            let periodic_one = block_on(blackman_builtin(
                Value::Num(1.0),
                vec![Value::from("periodic")],
            ))
            .expect("blackman periodic len1 gpu");
            let periodic_one = test_support::gather(periodic_one).expect("gather periodic len1");
            assert_eq!(periodic_one.materialize_f64(), vec![1.0]);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn blackman_integer_length_generates_current_window_with_wgpu_active() {
        let _guard = test_support::accel_test_lock();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let result = block_on(blackman_builtin(
            Value::Int(IntValue::U16(5)),
            vec![Value::from("periodic")],
        ))
        .expect("WGPU Blackman window");
        let output = test_support::gather(result).expect("gather WGPU window");
        assert_eq!(output.shape(), vec![5, 1]);
        let expected = 0.42 - 0.5 * cospi_real(2.0 / 5.0) + 0.08 * cospi_real(4.0 / 5.0);
        assert!((output.materialize_f64()[1] - expected).abs() < 1.0e-14);
    }
}
