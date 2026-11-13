//! MATLAB-compatible `tf` transfer-function constructor and SISO operator methods for RunMat.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    DEFAULT_DISCRETE_SAMPLE_TIME_EXTENSION, EXPLICIT_GPU_INPUT_EXTENSION,
    INTEGER_COEFFICIENT_EXTENSION, INTEGER_SAMPLE_TIME_EXTENSION, LOGICAL_COEFFICIENT_EXTENSION,
    TF_ERROR_DENOMINATOR_INVALID, TF_ERROR_INVALID_ARGUMENT, TF_ERROR_INVALID_COEFFICIENTS,
    TF_ERROR_INVALID_OPTION, TF_ERROR_INVALID_SAMPLE_TIME, TF_ERROR_INVALID_VARIABLE,
    VARIABLE_SHORTHAND_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{
    is_discrete_variable, parse_coefficients, scalar_f64, scalar_text, validate_sample_time,
    validate_variable, validate_variable_domain, TfModel, TfOptions,
};
use crate::{build_runtime_error, dispatcher, BuiltinResult, RuntimeError};

pub(in crate::builtins) mod methods;

const BUILTIN_NAME: &str = "tf";
const DEFAULT_VARIABLE: &str = "s";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::tf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tf",
    op_kind: GpuOpKind::Custom("transfer-function-constructor"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Object construction runs on the host. gpuArray coefficient inputs are gathered before storing the transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::tf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Transfer-function construction is metadata-only and terminates numeric fusion chains.",
};

fn tf_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    tf_error_with_message(error.message, error)
}

fn tf_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    tf_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn tf_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "tf",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPLICIT_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    match (args.len(), args.get(0), args.get(1)) {
        (1, Some(variable), _) if is_text_value(variable) => {
            ensure_variable_shorthand(variable, false)?;
            variable_model(variable, None)?.to_value("tf")
        }
        (2, Some(variable), Some(sample_time)) if is_text_value(variable) => {
            ensure_variable_shorthand(variable, true)?;
            let sample_time = admitted_sample_time(sample_time).await?;
            variable_model(variable, Some(&Value::Num(sample_time)))?.to_value("tf")
        }
        (_, Some(numerator), Some(denominator)) => {
            preflight_coefficient_extensions(numerator)?;
            preflight_coefficient_extensions(denominator)?;
            let admitted_options = TfAdmittedOptions::parse(args.view().range(2..)?)?;
            let host_sample_times = admitted_options.preflight_host().await?;
            let host_numerator = preparse_host_coefficients(numerator, "numerator").await?;
            let host_denominator = preparse_host_coefficients(denominator, "denominator").await?;
            if host_denominator.as_ref().is_some_and(|coefficients| {
                coefficients
                    .iter()
                    .all(|value| value.re == 0.0 && value.im == 0.0)
            }) {
                return Err(tf_error_with_detail(
                    &TF_ERROR_DENOMINATOR_INVALID,
                    "denominator coefficients must not all be zero",
                ));
            }
            let options = admitted_options.resolve(host_sample_times).await?;
            let numerator = resolve_coefficients(host_numerator, numerator, "numerator").await?;
            let denominator =
                resolve_coefficients(host_denominator, denominator, "denominator").await?;
            TfModel::new(
                numerator,
                denominator,
                TfOptions {
                    variable: options.variable,
                    sample_time: options.sample_time,
                },
            )?
            .to_value("tf")
        }
        (0, _, _) => Err(tf_error_with_detail(
            &TF_ERROR_INVALID_ARGUMENT,
            "expected tf('s'), tf(num, den), or tf(num, den, ...)",
        )),
        _ => Err(tf_error_with_detail(
            &TF_ERROR_INVALID_ARGUMENT,
            "unsupported tf invocation",
        )),
    }
}

fn preflight_coefficient_extensions(value: &Value) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_logical_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOGICAL_COEFFICIENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_COEFFICIENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn admitted_coefficient(value: &Value, role: &str) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_COEFFICIENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let gathered = dispatcher::gather_if_needed_async(value).await?;
    ensure_exact_integer_boundary(
        &gathered,
        &INTEGER_COEFFICIENT_EXTENSION,
        &TF_ERROR_INVALID_COEFFICIENTS,
        role,
    )?;
    Ok(gathered)
}

fn ensure_exact_integer_boundary(
    value: &Value,
    extension: &'static runmat_builtins::BuiltinExtensionDescriptor,
    error: &'static BuiltinErrorDescriptor,
    role: &str,
) -> BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64, value_has_native_integer_class,
    };
    if value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
        if !native_integer_value_is_exact_f64(value) {
            let mut failure = tf_error_with_detail(
                error,
                format!("integer {role} values must be exactly representable as double"),
            );
            failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(failure);
        }
    }
    Ok(())
}

async fn preparse_host_coefficients(
    value: &Value,
    role: &str,
) -> BuiltinResult<Option<Vec<num_complex::Complex64>>> {
    if matches!(value, Value::GpuTensor(_)) {
        return Ok(None);
    }
    Ok(Some(
        parse_coefficients(role, admitted_coefficient(value, role).await?, BUILTIN_NAME).await?,
    ))
}

async fn resolve_coefficients(
    parsed_host: Option<Vec<num_complex::Complex64>>,
    value: &Value,
    role: &str,
) -> BuiltinResult<Vec<num_complex::Complex64>> {
    match parsed_host {
        Some(coefficients) => Ok(coefficients),
        None => {
            parse_coefficients(role, admitted_coefficient(value, role).await?, BUILTIN_NAME).await
        }
    }
}

async fn admitted_sample_time(value: &Value) -> BuiltinResult<f64> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_SAMPLE_TIME_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let gathered = dispatcher::gather_if_needed_async(value).await?;
    ensure_exact_integer_boundary(
        &gathered,
        &INTEGER_SAMPLE_TIME_EXTENSION,
        &TF_ERROR_INVALID_SAMPLE_TIME,
        "sample time",
    )?;
    parse_sample_time(&gathered)
}

fn ensure_variable_shorthand(value: &Value, has_sample_time: bool) -> BuiltinResult<()> {
    let variable = scalar_text(value, "variable", BUILTIN_NAME)?;
    let variable = variable.trim();
    if !matches!(variable, "s" | "z") {
        crate::compatibility::ensure_builtin_extension_enabled(
            &VARIABLE_SHORTHAND_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !has_sample_time && is_discrete_variable(variable) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DEFAULT_DISCRETE_SAMPLE_TIME_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

#[derive(Clone)]
struct TfConstructorOptions {
    variable: String,
    sample_time: f64,
    variable_explicit: bool,
    sample_time_explicit: bool,
}

enum TfOption<'a> {
    Variable(String),
    SampleTime(&'a Value),
}

struct TfAdmittedOptions<'a> {
    options: Vec<TfOption<'a>>,
}

impl<'a> TfAdmittedOptions<'a> {
    fn parse(rest: crate::call::FunctionArgumentView<'a>) -> BuiltinResult<Self> {
        let mut options = Vec::new();
        match (rest.len(), rest.first()) {
            (0, _) => {}
            (1, Some(sample_time)) => options.push(TfOption::SampleTime(sample_time)),
            _ => {
                if !rest.len().is_multiple_of(2) {
                    return Err(tf_error_with_detail(
                        &TF_ERROR_INVALID_ARGUMENT,
                        "optional arguments must be name-value pairs or a scalar sample time",
                    ));
                }
                for pair in rest.chunks_exact(2) {
                    let name = scalar_text(&pair[0], "option name", "tf").map_err(|error| {
                        tf_error_with_detail(&TF_ERROR_INVALID_OPTION, error.message())
                    })?;
                    let lowered = name.trim().to_ascii_lowercase();
                    options.push(match lowered.as_str() {
                        "variable" => {
                            let variable = parse_variable(&pair[1])?;
                            if !matches!(variable.as_str(), "s" | "z") {
                                crate::compatibility::ensure_builtin_extension_enabled(
                                    &VARIABLE_SHORTHAND_EXTENSION,
                                    BUILTIN_NAME,
                                )?;
                            }
                            TfOption::Variable(variable)
                        }
                        "ts" | "sampletime" => {
                            TfOption::SampleTime(pair.get(1).expect("paired sample time"))
                        }
                        _ => {
                            return Err(tf_error_with_detail(
                                &TF_ERROR_INVALID_OPTION,
                                format!("unsupported option '{name}'"),
                            ));
                        }
                    });
                }
            }
        }
        Ok(Self { options })
    }

    async fn preflight_host(&self) -> BuiltinResult<Vec<Option<f64>>> {
        let mut values = Vec::with_capacity(self.options.len());
        for option in &self.options {
            values.push(match option {
                TfOption::Variable(_) => None,
                TfOption::SampleTime(value) if matches!(*value, Value::GpuTensor(_)) => None,
                TfOption::SampleTime(value) => Some(admitted_sample_time(value).await?),
            });
        }
        Ok(values)
    }

    async fn resolve(self, preflight: Vec<Option<f64>>) -> BuiltinResult<TfConstructorOptions> {
        let mut options = TfConstructorOptions {
            variable: DEFAULT_VARIABLE.to_string(),
            sample_time: 0.0,
            variable_explicit: false,
            sample_time_explicit: false,
        };
        for (option, admitted) in self.options.into_iter().zip(preflight) {
            match option {
                TfOption::Variable(variable) => {
                    options.variable = variable;
                    options.variable_explicit = true;
                }
                TfOption::SampleTime(value) => {
                    options.sample_time = match admitted {
                        Some(sample_time) => sample_time,
                        None => admitted_sample_time(value).await?,
                    };
                    options.sample_time_explicit = true;
                }
            }
        }
        if options.sample_time != 0.0 && !options.variable_explicit {
            options.variable = "z".to_string();
        }

        if options.variable_explicit
            && is_discrete_variable(&options.variable)
            && !options.sample_time_explicit
        {
            options.sample_time = 1.0;
        }
        validate_variable_domain(&options.variable, options.sample_time, "tf").map_err(|err| {
            let identifier = err.identifier();
            if identifier == TF_ERROR_INVALID_SAMPLE_TIME.identifier {
                tf_error_with_detail(&TF_ERROR_INVALID_SAMPLE_TIME, err.message())
            } else {
                tf_error_with_detail(&TF_ERROR_INVALID_VARIABLE, err.message())
            }
        })?;
        Ok(options)
    }
}

fn parse_sample_time(value: &Value) -> BuiltinResult<f64> {
    let sample_time = scalar_f64(value, "sample time", "tf").map_err(|_| {
        tf_error_with_detail(
            &TF_ERROR_INVALID_SAMPLE_TIME,
            format!("expected -1 or a non-negative scalar, got {value:?}"),
        )
    })?;
    if let Err(err) = validate_sample_time(sample_time, "tf") {
        let _ = err;
        return Err(tf_error(&TF_ERROR_INVALID_SAMPLE_TIME));
    }
    Ok(sample_time)
}

fn parse_variable(value: &Value) -> BuiltinResult<String> {
    let variable = scalar_text(value, "Variable", "tf")?;
    validate_variable(&variable, "tf").map_err(|_| {
        tf_error_with_detail(
            &TF_ERROR_INVALID_VARIABLE,
            "must be one of 's', 'p', 'z', 'q', 'z^-1', or 'q^-1'",
        )
    })
}

fn is_text_value(value: &Value) -> bool {
    match value {
        Value::String(runmat_value::StringElement::Text(_)) => true,
        Value::StringArray(array) => array.data().len() == 1,
        Value::CharArray(array) => array.rows == 1,
        _ => false,
    }
}

fn variable_model(value: &Value, sample_time: Option<&Value>) -> BuiltinResult<TfModel> {
    let variable = parse_variable(value)?;
    match variable.as_str() {
        "s" | "p" => {
            if let Some(sample_time) = sample_time {
                let parsed = parse_sample_time(sample_time)?;
                if parsed != 0.0 {
                    return Err(tf_error_with_detail(
                        &TF_ERROR_INVALID_SAMPLE_TIME,
                        "continuous transfer-function variables require Ts = 0",
                    ));
                }
            }
            TfModel::continuous_variable(variable)
        }
        "z" | "q" | "z^-1" | "q^-1" => {
            let sample_time = match sample_time {
                Some(value) => parse_sample_time(value)?,
                None => 1.0,
            };
            if sample_time != -1.0 && sample_time <= 0.0 {
                return Err(tf_error_with_detail(
                    &TF_ERROR_INVALID_SAMPLE_TIME,
                    "discrete transfer-function variables require a positive sample time or -1",
                ));
            }
            TfModel::discrete_variable(variable, sample_time)
        }
        _ => unreachable!("validated variable"),
    }
}

#[cfg(test)]
mod tests {
    use super::methods::*;
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{TF_DESCRIPTOR, TF_ERROR_INVALID_COEFFICIENTS};
    use runmat_value::{CharArray, IntValue, IntegerStorage, Tensor};

    fn run_tf(numerator: Value, denominator: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let mut args = vec![numerator, denominator];
        args.extend(rest);
        block_on(tf_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn run_tf_args(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(tf_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn property<'a>(value: &'a Value, name: &str) -> &'a Value {
        let Value::Object(object) = value else {
            panic!("expected object, got {value:?}");
        };
        object
            .properties()
            .get(name)
            .unwrap_or_else(|| panic!("missing property {name}"))
    }

    fn tensor_property(value: &Value, name: &str) -> Vec<f64> {
        match property(value, name) {
            Value::Tensor(tensor) => tensor.materialize_f64().clone(),
            other => panic!("expected tensor property {name}, got {other:?}"),
        }
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    #[test]
    fn exponent_parser_reads_typed_integer_scalar_storage_exactly() {
        assert_eq!(
            integer_exponent(&Value::Int(IntValue::I64(-3))).unwrap(),
            -3
        );

        let exponent =
            Tensor::new_integer(IntegerStorage::I16(vec![-2]), vec![1, 1]).expect("typed exponent");
        assert_eq!(integer_exponent(&Value::Tensor(exponent)).unwrap(), -2);

        let too_large =
            Tensor::new_integer(IntegerStorage::U64(vec![i64::MAX as u64 + 1]), vec![1, 1])
                .expect("wide exponent");
        assert!(integer_exponent(&Value::Tensor(too_large)).is_err());
    }

    #[test]
    fn exponent_parser_rejects_unrepresentable_double_boundary() {
        assert!(integer_exponent(&Value::Num(i64::MAX as f64)).is_err());
        assert!(integer_exponent(&Value::Num((i64::MAX as f64) + 1024.0)).is_err());
    }

    #[test]
    fn tf_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"s = tf('s')"));
        assert!(labels.contains(&"z = tf('z', Ts)"));
        assert!(labels.contains(&"sys = tf(numerator, denominator)"));
        assert!(labels.contains(&"sys = tf(numerator, denominator, Ts)"));
        assert!(labels.contains(&"sys = tf(numerator, denominator, \"Variable\", variableName)"));
        assert!(labels.contains(&"sys = tf(numerator, denominator, name, value, ...)"));
    }

    #[test]
    fn tf_constructs_continuous_siso_object() {
        let sys = run_tf(
            Value::Num(20.0),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect("tf");

        let Value::Object(object) = &sys else {
            panic!("expected object");
        };
        assert_eq!(object.class_name().display_name(), "tf");
        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("s"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.0));
        match property(&sys, "Numerator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 1]);
                assert_eq!(tensor.materialize_f64(), vec![20.0]);
            }
            other => panic!("expected numerator tensor, got {other:?}"),
        }
        match property(&sys, "Denominator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 2]);
                assert_eq!(tensor.materialize_f64(), vec![1.0, 5.0]);
            }
            other => panic!("expected denominator tensor, got {other:?}"),
        }
    }

    #[test]
    fn tf_accepts_continuous_variable_constructor() {
        let sys = run_tf_args(vec![Value::from("s")]).expect("tf('s')");
        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("s"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.0));
        assert_eq!(tensor_property(&sys, "Numerator"), vec![1.0, 0.0]);
        assert_eq!(tensor_property(&sys, "Denominator"), vec![1.0]);
    }

    #[test]
    fn tf_accepts_discrete_variable_constructor() {
        let sys = run_tf_args(vec![Value::from("z"), Value::Num(0.2)]).expect("tf('z', Ts)");
        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.2));
        assert_eq!(tensor_property(&sys, "Numerator"), vec![1.0, 0.0]);
        assert_eq!(tensor_property(&sys, "Denominator"), vec![1.0]);
    }

    #[test]
    fn tf_rejects_continuous_variable_constructor_with_positive_sample_time() {
        let err = run_tf_args(vec![Value::from("s"), Value::Num(0.2)])
            .expect_err("tf('s', positive Ts) should fail");
        assert_eq!(err.identifier(), TF_ERROR_INVALID_SAMPLE_TIME.identifier);
    }

    #[test]
    fn tf_arithmetic_builds_polynomial_transfer_functions() {
        let s = run_tf_args(vec![Value::from("s")]).expect("tf('s')");
        let s_squared = block_on(tf_power(s.clone(), Value::Num(2.0))).expect("s^2");
        let quadratic = block_on(tf_plus(
            block_on(tf_plus(
                block_on(tf_mtimes(Value::Num(0.4), s_squared)).expect("0.4*s^2"),
                block_on(tf_mtimes(Value::Num(1.8), s.clone())).expect("1.8*s"),
            ))
            .expect("sum terms"),
            Value::Num(1.0),
        ))
        .expect("add constant");
        let g = block_on(tf_mrdivide(Value::Num(2.5), quadratic)).expect("2.5/poly");

        assert_eq!(tensor_property(&g, "Numerator"), vec![2.5]);
        assert_eq!(tensor_property(&g, "Denominator"), vec![0.4, 1.8, 1.0]);
    }

    #[test]
    fn tf_normalizes_column_coefficients_to_rows() {
        let sys = run_tf(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![3, 1]).unwrap()),
            Vec::new(),
        )
        .expect("tf");

        match property(&sys, "Numerator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 2]);
                assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0]);
            }
            other => panic!("expected numerator tensor, got {other:?}"),
        }
        match property(&sys, "Denominator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 3]);
                assert_eq!(tensor.materialize_f64(), vec![1.0, 3.0, 2.0]);
            }
            other => panic!("expected denominator tensor, got {other:?}"),
        }
    }

    #[test]
    fn tf_accepts_discrete_sample_time() {
        let sys = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, -0.5], vec![1, 2]).unwrap()),
            vec![Value::Num(0.1)],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.1));
    }

    #[test]
    fn tf_admitted_resident_coefficients_and_sample_time_download_once_each() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let numerator = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
            let sample_time = Tensor::new(vec![0.25], vec![1, 1]).unwrap();
            let numerator =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &numerator).unwrap();
            let sample_time =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &sample_time)
                    .unwrap();
            provider.reset_telemetry();
            let model = run_tf(
                Value::GpuTensor(numerator),
                Value::Num(1.0),
                vec![Value::GpuTensor(sample_time)],
            )
            .expect("resident tf inputs");
            assert_eq!(tensor_property(&model, "Numerator"), vec![1.0, 2.0]);
            assert_eq!(property(&model, "Ts"), &Value::Num(0.25));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 24);
        });
    }

    #[test]
    fn tf_variable_alias_option_is_gated_before_resident_sample_time_download() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let sample_time = Tensor::new(vec![0.25], vec![1, 1]).unwrap();
            let sample_time =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &sample_time)
                    .unwrap();
            provider.reset_telemetry();
            let error = run_tf(
                Value::Num(1.0),
                Value::Num(1.0),
                vec![
                    Value::from("Ts"),
                    Value::GpuTensor(sample_time),
                    Value::from("Variable"),
                    Value::from("p"),
                ],
            )
            .expect_err("alias is a RunMat-only extension");
            assert_eq!(
                error.identifier(),
                VARIABLE_SHORTHAND_EXTENSION.error_identifier
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn tf_nontext_option_name_uses_catalog_invalid_option_error() {
        let error = run_tf(
            Value::Num(1.0),
            Value::Num(2.0),
            vec![Value::Num(7.0), Value::Num(0.1)],
        )
        .expect_err("nontext option name");
        assert_eq!(error.identifier(), TF_ERROR_INVALID_OPTION.identifier);
    }

    #[test]
    fn tf_zero_host_denominator_preflights_before_resident_numerator() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let numerator = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
            )
            .unwrap();
            provider.reset_telemetry();
            let error = run_tf(
                Value::GpuTensor(numerator),
                Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
                vec![],
            )
            .expect_err("zero denominator");
            assert_eq!(error.identifier(), TF_ERROR_DENOMINATOR_INVALID.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn tf_invalid_host_coefficient_preflights_before_resident_sample_time() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let sample_time = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![0.25], vec![1, 1]).unwrap(),
            )
            .unwrap();
            provider.reset_telemetry();
            let error = run_tf(
                Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap()),
                Value::Num(1.0),
                vec![Value::GpuTensor(sample_time)],
            )
            .expect_err("matrix coefficients are unsupported");
            assert_eq!(error.identifier(), TF_ERROR_INVALID_COEFFICIENTS.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn tf_invalid_host_sample_time_preflights_before_resident_numerator() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let numerator = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            provider.reset_telemetry();
            let error = run_tf(
                Value::GpuTensor(numerator),
                Value::Num(1.0),
                vec![Value::Num(-2.0)],
            )
            .expect_err("invalid host sample time");
            assert_eq!(error.identifier(), TF_ERROR_INVALID_SAMPLE_TIME.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn tf_typed_integer_coefficients_and_sample_time_cross_double_boundary_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sys = run_tf(
            integer_tensor(IntegerStorage::U64(vec![1, 2]), vec![1, 2]),
            integer_tensor(IntegerStorage::I16(vec![1, 3, 2]), vec![1, 3]),
            vec![integer_tensor(IntegerStorage::U8(vec![1]), vec![1, 1])],
        )
        .expect("tf");

        assert_eq!(tensor_property(&sys, "Numerator"), vec![1.0, 2.0]);
        assert_eq!(tensor_property(&sys, "Denominator"), vec![1.0, 3.0, 2.0]);
        assert_eq!(property(&sys, "Ts"), &Value::Num(1.0));
        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
    }

    #[test]
    fn tf_integer_roles_and_shorthand_extensions_follow_compatibility_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let coefficients = run_tf(
            integer_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
            Value::Num(1.0),
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(
            coefficients.identifier(),
            INTEGER_COEFFICIENT_EXTENSION.error_identifier
        );
        let sample_time = run_tf(
            Value::Num(1.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I8(1))],
        )
        .unwrap_err();
        assert_eq!(
            sample_time.identifier(),
            INTEGER_SAMPLE_TIME_EXTENSION.error_identifier
        );
        let shorthand = run_tf_args(vec![Value::from("p")]).unwrap_err();
        assert_eq!(
            shorthand.identifier(),
            VARIABLE_SHORTHAND_EXTENSION.error_identifier
        );
        let implicit_discrete = run_tf_args(vec![Value::from("z")]).unwrap_err();
        assert_eq!(
            implicit_discrete.identifier(),
            DEFAULT_DISCRETE_SAMPLE_TIME_EXTENSION.error_identifier
        );
    }

    #[test]
    fn tf_accepts_unspecified_discrete_sample_time_and_rejects_inexact_integer_coefficients() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let discrete = run_tf_args(vec![Value::from("z"), Value::Num(-1.0)]).expect("tf('z', -1)");
        assert_eq!(property(&discrete, "Ts"), &Value::Num(-1.0));

        let inexact = run_tf(
            integer_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]),
            Value::Num(1.0),
            Vec::new(),
        )
        .unwrap_err();
        assert!(inexact.message().contains("exactly representable"));
        assert_eq!(
            inexact.identifier(),
            TF_ERROR_INVALID_COEFFICIENTS.identifier
        );
        assert_eq!(inexact.gpu_gather_retry, crate::GpuGatherRetry::Never);

        let sample_time = run_tf(
            Value::Num(1.0),
            Value::Num(1.0),
            vec![integer_tensor(
                IntegerStorage::U64(vec![9_007_199_254_740_993]),
                vec![1, 1],
            )],
        )
        .unwrap_err();
        assert_eq!(
            sample_time.identifier(),
            TF_ERROR_INVALID_SAMPLE_TIME.identifier
        );
        assert_eq!(sample_time.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }

    #[test]
    fn tf_positional_zero_sample_time_remains_continuous() {
        let sys = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![1, 2]).unwrap()),
            vec![Value::Num(0.0)],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("s"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.0));
    }

    #[test]
    fn tf_accepts_variable_name_value_option() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sys = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![Value::from("Variable"), Value::from("p")],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("p"))
        );
    }

    #[test]
    fn tf_explicit_discrete_variable_defaults_positive_sample_time() {
        let sys = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![Value::from("Variable"), Value::from("z")],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(1.0));
    }

    #[test]
    fn tf_rejects_explicit_continuous_variable_with_positive_sample_time() {
        let err = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![
                Value::from("Variable"),
                Value::from("s"),
                Value::from("Ts"),
                Value::Num(0.5),
            ],
        )
        .expect_err("continuous variable with positive Ts should fail");

        assert_eq!(err.identifier(), TF_ERROR_INVALID_VARIABLE.identifier);
    }

    #[test]
    fn tf_rejects_explicit_discrete_variable_with_zero_sample_time() {
        let err = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![
                Value::from("Variable"),
                Value::from("z"),
                Value::from("Ts"),
                Value::Num(0.0),
            ],
        )
        .expect_err("discrete variable with zero Ts should fail");

        assert_eq!(err.identifier(), TF_ERROR_INVALID_SAMPLE_TIME.identifier);
    }

    #[test]
    fn tf_accepts_explicit_discrete_variable_with_positive_sample_time() {
        let sys = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![
                Value::from("Variable"),
                Value::from("z"),
                Value::from("Ts"),
                Value::Num(0.5),
            ],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.5));
    }

    #[test]
    fn tf_rejects_zero_denominator() {
        let err = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect_err("zero denominator should fail");
        assert!(err.message().contains("must not all be zero"));
        assert_eq!(err.identifier(), TF_ERROR_DENOMINATOR_INVALID.identifier);
    }

    #[test]
    fn tf_rejects_matrix_coefficients() {
        let err = run_tf(
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect_err("matrix numerator should fail");
        assert!(err
            .message()
            .contains("numerator coefficients must be a vector"));
        assert_eq!(err.identifier(), TF_ERROR_INVALID_COEFFICIENTS.identifier);
    }
}
