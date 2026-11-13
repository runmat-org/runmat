//! MATLAB-compatible `ss` state-space model constructor for RunMat.
use runmat_types::MemberAccess;

use std::collections::HashMap;

use runmat_builtins::{
    BuiltinErrorDescriptor, SS_ERROR_INTERNAL, SS_ERROR_INVALID_ARGUMENT,
    SS_ERROR_INVALID_DIMENSIONS, SS_ERROR_INVALID_MATRIX, SS_ERROR_INVALID_OPTION,
    SS_ERROR_INVALID_SAMPLE_TIME, SS_ERROR_UNSUPPORTED_INPUT, SS_EXPLICIT_GPU_EXTENSION,
    SS_INTEGER_MATRIX_EXTENSION, SS_INTEGER_SAMPLE_TIME_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, Tensor, Value};

use crate::builtins::common::{
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::control::tf_model::state_matrix_dimensions;
use crate::{build_runtime_error, dispatcher, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "ss";
const SS_CLASS: runmat_types::StaticClassIdentity = runmat_types::StaticClassIdentity::new("ss");

static SS_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(SS_CLASS);

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::ss")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ss",
    op_kind: GpuOpKind::Custom("state-space-model-constructor"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Object construction runs on the host. gpuArray matrix inputs are gathered before validating and storing the state-space metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::ss")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ss",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "State-space construction is metadata-only and terminates numeric fusion chains.",
};

fn ss_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ss_error_with_message(error.message, error)
}

fn ss_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    ss_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn ss_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ensure_ss_class_registered() {
    SS_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in [
            "A",
            "B",
            "C",
            "D",
            "Ts",
            "InputDelay",
            "OutputDelay",
            "StateName",
            "InputName",
            "OutputName",
        ] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Public,
                    default_value: None,
                },
            );
        }

        let methods: HashMap<runmat_types::MethodName, crate::class_registry::RuntimeMethod> =
            HashMap::new();
        crate::class_registry::register_builtin_class(
            SS_CLASS,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: SS_CLASS.into(),
                parent: None,
                properties,
                methods,
            },
        );
    });
}

#[runtime_builtin(
    name = "ss",
    binding_variant = "default",
    builtin_path = "crate::builtins::control::ss",
    context = true
)]
pub(crate) async fn ss_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    a: Value,
    b: Value,
    c: Value,
    d: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let matrices = [&a, &b, &c, &d];
    if matrices
        .iter()
        .any(|value| crate::builtins::common::validation::value_contains_explicit_gpu(value))
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SS_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    for value in matrices {
        if crate::builtins::common::validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &SS_INTEGER_MATRIX_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    let admitted_options = SsAdmittedOptions::parse(rest.view())?;
    let host_sample_times = admitted_options.preflight_host().await?;
    let host_a = preparse_host_matrix(&a, "A").await?;
    let host_b = preparse_host_matrix(&b, "B").await?;
    let host_c = preparse_host_matrix(&c, "C").await?;
    let host_d = preparse_host_matrix(&d, "D").await?;
    if let (Some(a), Some(b), Some(c), Some(d)) = (&host_a, &host_b, &host_c, &host_d) {
        validate_state_space_dimensions(a, b, c, d)?;
    }
    let sample_time = admitted_options.resolve(host_sample_times).await?;
    let a = resolve_matrix(host_a, &a, "A").await?;
    let b = resolve_matrix(host_b, &b, "B").await?;
    let c = resolve_matrix(host_c, &c, "C").await?;
    let d = resolve_matrix(host_d, &d, "D").await?;

    validate_state_space_dimensions(&a, &b, &c, &d)?;

    let state_count = a.rows;
    let input_count = b.cols;
    let output_count = c.rows;

    ensure_ss_class_registered();
    let fields = [
        ("A", a.into_value()),
        ("B", b.into_value()),
        ("C", c.into_value()),
        ("D", d.into_value()),
        ("Ts", Value::Num(sample_time)),
        ("InputDelay", zero_tensor_value(vec![input_count, 1])?),
        ("OutputDelay", zero_tensor_value(vec![output_count, 1])?),
        ("StateName", empty_name_cell_value(state_count, 1)?),
        ("InputName", empty_name_cell_value(input_count, 1)?),
        ("OutputName", empty_name_cell_value(output_count, 1)?),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime.as_ref(),
        SS_CLASS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}

async fn admitted_matrix(value: &Value, role: &str) -> BuiltinResult<Value> {
    let gathered = dispatcher::gather_if_needed_async(value).await?;
    ensure_ss_integer_boundary(
        &gathered,
        &SS_INTEGER_MATRIX_EXTENSION,
        &SS_ERROR_INVALID_MATRIX,
        role,
    )
    .await?;
    Ok(gathered)
}

async fn preparse_host_matrix(value: &Value, role: &str) -> BuiltinResult<Option<RealMatrix>> {
    if matches!(value, Value::GpuTensor(_)) {
        return Ok(None);
    }
    Ok(Some(RealMatrix::parse(
        role,
        admitted_matrix(value, role).await?,
    )?))
}

async fn resolve_matrix(
    parsed_host: Option<RealMatrix>,
    value: &Value,
    role: &str,
) -> BuiltinResult<RealMatrix> {
    match parsed_host {
        Some(matrix) => Ok(matrix),
        None => RealMatrix::parse(role, admitted_matrix(value, role).await?),
    }
}

async fn admitted_sample_time(value: &Value) -> BuiltinResult<f64> {
    admit_ss_integer_extension(value, &SS_INTEGER_SAMPLE_TIME_EXTENSION)?;
    let gathered = dispatcher::gather_if_needed_async(value).await?;
    ensure_ss_integer_boundary(
        &gathered,
        &SS_INTEGER_SAMPLE_TIME_EXTENSION,
        &SS_ERROR_INVALID_SAMPLE_TIME,
        "sample-time",
    )
    .await?;
    parse_sample_time(&gathered)
}

fn admit_ss_integer_extension(
    value: &Value,
    extension: &'static runmat_builtins::BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
    }
    Ok(())
}

async fn ensure_ss_integer_boundary(
    value: &Value,
    extension: &'static runmat_builtins::BuiltinExtensionDescriptor,
    error: &'static BuiltinErrorDescriptor,
    role: &str,
) -> BuiltinResult<()> {
    admit_ss_integer_extension(value, extension)?;
    if crate::builtins::common::validation::value_has_native_integer_class(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        let identifier = error
            .identifier
            .expect("ss precision error has an identifier");
        return Err(build_runtime_error(format!(
            "ss: integer {role} values must be exactly representable as double"
        ))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(identifier)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    Ok(())
}

struct SsAdmittedOptions<'a> {
    sample_times: Vec<&'a Value>,
}

impl<'a> SsAdmittedOptions<'a> {
    fn parse(rest: crate::call::FunctionArgumentView<'a>) -> BuiltinResult<Self> {
        let mut sample_times = Vec::new();
        match rest.len() {
            0 => {}
            1 => sample_times.push(rest.first().expect("one ss option")),
            _ => {
                if !rest.len().is_multiple_of(2) {
                    return Err(ss_error_with_detail(
                        &SS_ERROR_INVALID_ARGUMENT,
                        "optional arguments must be name-value pairs or a scalar sample time",
                    ));
                }
                for index in (0..rest.len()).step_by(2) {
                    let name_value = rest.get(index).expect("validated ss option name");
                    let sample_value = rest.get(index + 1).expect("validated ss option value");
                    let name = scalar_text(name_value, "option name")?;
                    let lowered = name.trim().to_ascii_lowercase();
                    match lowered.as_str() {
                        "ts" | "sampletime" => sample_times.push(sample_value),
                        _ => {
                            return Err(ss_error_with_detail(
                                &SS_ERROR_INVALID_OPTION,
                                format!("unsupported option '{name}'"),
                            ));
                        }
                    }
                }
            }
        }
        Ok(Self { sample_times })
    }

    async fn preflight_host(&self) -> BuiltinResult<Vec<Option<f64>>> {
        let mut values = Vec::with_capacity(self.sample_times.len());
        for value in &self.sample_times {
            values.push(if matches!(*value, Value::GpuTensor(_)) {
                None
            } else {
                Some(admitted_sample_time(value).await?)
            });
        }
        Ok(values)
    }

    async fn resolve(self, preflight: Vec<Option<f64>>) -> BuiltinResult<f64> {
        let mut sample_time = 0.0;
        for (value, admitted) in self.sample_times.into_iter().zip(preflight) {
            sample_time = match admitted {
                Some(sample_time) => sample_time,
                None => admitted_sample_time(value).await?,
            };
        }
        Ok(sample_time)
    }
}

fn parse_sample_time(value: &Value) -> BuiltinResult<f64> {
    let sample_time = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        other => {
            return Err(ss_error_with_detail(
                &SS_ERROR_INVALID_SAMPLE_TIME,
                format!("expected non-negative scalar, got {other:?}"),
            ))
        }
    };
    if !sample_time.is_finite() || sample_time < 0.0 {
        return Err(ss_error(&SS_ERROR_INVALID_SAMPLE_TIME));
    }
    Ok(sample_time)
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<runmat_value::Utf8Text> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.clone()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("nonmissing scalar text")
                .clone())
        }
        Value::CharArray(array) if array.rows == 1 => {
            Ok(runmat_value::decode_utf16_code_units(array.data())
                .map_err(|source| {
                    let mut error =
                        ss_error_with_detail(&SS_ERROR_INVALID_OPTION, source.to_string());
                    error.source = Some(Box::new(source));
                    error
                })?
                .into())
        }
        other => Err(ss_error_with_detail(
            &SS_ERROR_INVALID_OPTION,
            format!("{context} must be a string scalar or character vector, got {other:?}"),
        )),
    }
}

#[derive(Clone)]
struct RealMatrix {
    tensor: Tensor,
    rows: usize,
    cols: usize,
}

impl RealMatrix {
    fn parse(label: &str, value: Value) -> BuiltinResult<Self> {
        let tensor = match value {
            Value::Tensor(tensor) => tensor::integer_tensor_to_f64(tensor).map_err(|err| {
                ss_error_with_detail(
                    &SS_ERROR_INTERNAL,
                    format!("failed to normalize {label}: {err}"),
                )
            })?,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|err| {
                ss_error_with_detail(&SS_ERROR_INTERNAL, format!("failed to build tensor: {err}"))
            })?,
            Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|err| {
                ss_error_with_detail(&SS_ERROR_INTERNAL, format!("failed to build tensor: {err}"))
            })?,
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                return Err(ss_error_with_detail(
                    &SS_ERROR_UNSUPPORTED_INPUT,
                    format!(
                        "{label} must be finite real numeric data; complex input is unsupported"
                    ),
                ));
            }
            other => {
                return Err(ss_error_with_detail(
                    &SS_ERROR_UNSUPPORTED_INPUT,
                    format!("{label} must be a finite real numeric matrix, got {other:?}"),
                ));
            }
        };

        if tensor.shape().len() > 2 {
            return Err(ss_error_with_detail(
                &SS_ERROR_INVALID_DIMENSIONS,
                format!(
                    "{label} must be a 2-D matrix, got shape {:?}",
                    tensor.shape()
                ),
            ));
        }
        let values = tensor::tensor_values_f64_cow(&tensor);
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ss_error_with_detail(
                &SS_ERROR_UNSUPPORTED_INPUT,
                format!("{label} must contain only finite real values"),
            ));
        }

        let (rows, cols) = state_matrix_dimensions(&tensor);
        Ok(Self { rows, cols, tensor })
    }

    fn into_value(self) -> Value {
        Value::Tensor(self.tensor)
    }
}

fn validate_state_space_dimensions(
    a: &RealMatrix,
    b: &RealMatrix,
    c: &RealMatrix,
    d: &RealMatrix,
) -> BuiltinResult<()> {
    if a.rows != a.cols {
        return Err(ss_error_with_detail(
            &SS_ERROR_INVALID_DIMENSIONS,
            format!("A must be square, got {}x{}", a.rows, a.cols),
        ));
    }

    let state_count = a.rows;
    if b.rows != state_count {
        return Err(ss_error_with_detail(
            &SS_ERROR_INVALID_DIMENSIONS,
            format!(
                "B must have {} rows to match A, got {}x{}",
                state_count, b.rows, b.cols
            ),
        ));
    }
    if c.cols != state_count {
        return Err(ss_error_with_detail(
            &SS_ERROR_INVALID_DIMENSIONS,
            format!(
                "C must have {} columns to match A, got {}x{}",
                state_count, c.rows, c.cols
            ),
        ));
    }
    if d.rows != c.rows || d.cols != b.cols {
        return Err(ss_error_with_detail(
            &SS_ERROR_INVALID_DIMENSIONS,
            format!(
                "D must have shape {}x{} to match C outputs and B inputs, got {}x{}",
                c.rows, b.cols, d.rows, d.cols
            ),
        ));
    }

    Ok(())
}

fn zero_tensor_value(shape: Vec<usize>) -> BuiltinResult<Value> {
    let len = shape.iter().product();
    Tensor::new(vec![0.0; len], shape)
        .map(Value::Tensor)
        .map_err(|err| {
            ss_error_with_detail(&SS_ERROR_INTERNAL, format!("failed to build tensor: {err}"))
        })
}

fn empty_name_cell_value(rows: usize, cols: usize) -> BuiltinResult<Value> {
    let len = rows * cols;
    let values = (0..len)
        .map(|_| Value::CharArray(CharArray::new_row("")))
        .collect();
    CellArray::new(values, rows, cols)
        .map(Value::Cell)
        .map_err(|err| {
            ss_error_with_detail(
                &SS_ERROR_INTERNAL,
                format!("failed to build cell array: {err}"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::SS_DESCRIPTOR;
    use runmat_value::{IntValue, IntegerStorage};

    fn run_ss(a: Value, b: Value, c: Value, d: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(ss_builtin(
            None,
            a,
            b,
            c,
            d,
            crate::builtins::common::test_support::builtin_arguments(rest),
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

    fn assert_tensor(value: &Value, shape: &[usize], data: &[f64]) {
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), shape);
                assert_eq!(tensor.materialize_f64(), data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn ss_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"sys = ss(A, B, C, D)"));
        assert!(labels.contains(&"sys = ss(A, B, C, D, Ts)"));
        assert!(labels.contains(&"sys = ss(A, B, C, D, \"Ts\", Ts)"));
        assert!(labels.contains(&"sys = ss(A, B, C, D, name, value, ...)"));
    }

    #[test]
    fn ss_constructs_continuous_state_space_object() {
        let sys = run_ss(
            Value::Tensor(Tensor::new(vec![0.0, -2.0, 1.0, -3.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Num(0.0),
            Vec::new(),
        )
        .expect("ss");

        let Value::Object(object) = &sys else {
            panic!("expected object");
        };
        assert_eq!(object.class_name().display_name(), "ss");
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.0));
        assert_tensor(property(&sys, "A"), &[2, 2], &[0.0, -2.0, 1.0, -3.0]);
        assert_tensor(property(&sys, "B"), &[2, 1], &[0.0, 1.0]);
        assert_tensor(property(&sys, "C"), &[1, 2], &[1.0, 0.0]);
        assert_tensor(property(&sys, "D"), &[1, 1], &[0.0]);
        assert_tensor(property(&sys, "InputDelay"), &[1, 1], &[0.0]);
        assert_tensor(property(&sys, "OutputDelay"), &[1, 1], &[0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn ss_preserves_dense_rank_zero_and_rank_one_matrix_projection() {
        let rank_zero = || Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap());
        let system = run_ss(
            rank_zero(),
            rank_zero(),
            rank_zero(),
            rank_zero(),
            Vec::new(),
        )
        .expect("rank-zero state-space matrices");
        assert_tensor(property(&system, "A"), &[], &[1.0]);

        let error = run_ss(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Vec::new(),
        )
        .expect_err("rank-one A is a one-by-two matrix");
        assert!(error.message().contains("A must be square, got 1x2"));
    }

    #[test]
    fn ss_preserves_matrix_orientation_for_mimo_systems() {
        let sys = run_ss(
            Value::Num(-1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.1, 0.2, 0.3], vec![2, 2]).unwrap()),
            Vec::new(),
        )
        .expect("ss");

        assert_tensor(property(&sys, "A"), &[1, 1], &[-1.0]);
        assert_tensor(property(&sys, "B"), &[1, 2], &[1.0, 2.0]);
        assert_tensor(property(&sys, "C"), &[2, 1], &[3.0, 4.0]);
        assert_tensor(property(&sys, "D"), &[2, 2], &[0.0, 0.1, 0.2, 0.3]);
        assert_tensor(property(&sys, "InputDelay"), &[2, 1], &[0.0, 0.0]);
        assert_tensor(property(&sys, "OutputDelay"), &[2, 1], &[0.0, 0.0]);
    }

    #[test]
    fn ss_accepts_discrete_sample_time() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sys = run_ss(
            Value::Int(IntValue::I32(1)),
            Value::Int(IntValue::I32(2)),
            Value::Int(IntValue::I32(3)),
            Value::Int(IntValue::I32(4)),
            vec![Value::Num(0.25)],
        )
        .expect("ss");

        assert_eq!(property(&sys, "Ts"), &Value::Num(0.25));
    }

    #[test]
    fn ss_typed_integer_matrices_and_sample_time_cross_double_boundary_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        fn mirrorless_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
            let tensor = Tensor::new_integer(storage, shape).unwrap();
            Value::Tensor(tensor)
        }

        let sys = run_ss(
            mirrorless_integer_tensor(IntegerStorage::I16(vec![0, -2, 1, -3]), vec![2, 2]),
            mirrorless_integer_tensor(IntegerStorage::U16(vec![0, 1]), vec![2, 1]),
            mirrorless_integer_tensor(IntegerStorage::I8(vec![1, 0]), vec![1, 2]),
            mirrorless_integer_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
            vec![Value::from("Ts"), {
                let ts = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).unwrap();
                Value::Tensor(ts)
            }],
        )
        .expect("ss");

        assert_tensor(property(&sys, "A"), &[2, 2], &[0.0, -2.0, 1.0, -3.0]);
        assert_tensor(property(&sys, "B"), &[2, 1], &[0.0, 1.0]);
        assert_tensor(property(&sys, "C"), &[1, 2], &[1.0, 0.0]);
        assert_tensor(property(&sys, "D"), &[1, 1], &[0.0]);
        assert_eq!(property(&sys, "Ts"), &Value::Num(1.0));
        assert!(
            matches!(property(&sys, "A"), Value::Tensor(tensor) if tensor.integer_storage().is_none())
        );
    }

    #[test]
    fn ss_gates_typed_integer_matrices_in_strict_mode() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run_ss(
            Value::Int(IntValue::I32(1)),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
            Vec::new(),
        )
        .expect_err("typed-integer ss matrices must be gated in strict mode");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:SsIntegerMatrixInputExtension")
        );
    }

    #[test]
    fn ss_rejects_integer_matrix_values_that_round_at_double_boundary() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = run_ss(
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
            Vec::new(),
        )
        .expect_err("inexact typed-integer ss matrices must not round silently");
        assert!(error.message().contains("exactly representable as double"));
        assert_eq!(error.identifier(), SS_ERROR_INVALID_MATRIX.identifier);
    }

    #[test]
    fn ss_rejects_inexact_integer_sample_time_with_owning_identifier() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = run_ss(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
            vec![Value::Int(IntValue::U64((1_u64 << 53) + 1))],
        )
        .expect_err("inexact integer sample time must not round silently");
        assert!(error.message().contains("exactly representable as double"));
        assert_eq!(error.identifier(), SS_ERROR_INVALID_SAMPLE_TIME.identifier);
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }

    #[test]
    fn ss_accepts_exact_wide_integer_matrix_and_sample_time() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let exactly_representable = 1_u64 << 53;
        let sys = run_ss(
            Value::Int(IntValue::U64(exactly_representable)),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
            vec![Value::Int(IntValue::U64(exactly_representable))],
        )
        .expect("exact wide integer values are permitted");
        assert_tensor(
            property(&sys, "A"),
            &[1, 1],
            &[exactly_representable as f64],
        );
        assert_eq!(
            property(&sys, "Ts"),
            &Value::Num(exactly_representable as f64)
        );
    }

    #[test]
    fn ss_accepts_sample_time_name_value_options() {
        let sys = run_ss(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
            vec![Value::from("SampleTime"), Value::Num(0.5)],
        )
        .expect("ss");

        assert_eq!(property(&sys, "Ts"), &Value::Num(0.5));
    }

    #[test]
    fn ss_rejects_nonsquare_a_matrix() {
        let err = run_ss(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .expect_err("nonsquare A should fail");
        assert!(err.message().contains("A must be square"));
        assert_eq!(err.identifier(), SS_ERROR_INVALID_DIMENSIONS.identifier);
    }

    #[test]
    fn ss_rejects_b_row_mismatch() {
        let err = run_ss(
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .expect_err("B mismatch should fail");
        assert!(err.message().contains("B must have 2 rows"));
        assert_eq!(err.identifier(), SS_ERROR_INVALID_DIMENSIONS.identifier);
    }

    #[test]
    fn ss_rejects_d_shape_mismatch() {
        let err = run_ss(
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect_err("D mismatch should fail");
        assert!(err.message().contains("D must have shape 1x1"));
        assert_eq!(err.identifier(), SS_ERROR_INVALID_DIMENSIONS.identifier);
    }

    #[test]
    fn ss_rejects_invalid_sample_time() {
        let err = run_ss(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(0.0),
            vec![Value::Num(-0.1)],
        )
        .expect_err("negative Ts should fail");
        assert_eq!(err.identifier(), SS_ERROR_INVALID_SAMPLE_TIME.identifier);
    }

    #[test]
    fn ss_rejects_complex_inputs() {
        let err = run_ss(
            Value::Complex(1.0, 1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(0.0),
            Vec::new(),
        )
        .expect_err("complex A should fail");
        assert!(err.message().contains("complex input is unsupported"));
        assert_eq!(err.identifier(), SS_ERROR_UNSUPPORTED_INPUT.identifier);
    }

    #[test]
    fn ss_gpu_matrix_input_gathers_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let sys = run_ss(
                Value::GpuTensor(handle),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
                Vec::new(),
            )
            .expect("ss");

            assert_tensor(property(&sys, "A"), &[1, 1], &[1.0]);
        });
    }

    #[test]
    fn ss_admitted_resident_matrix_and_sample_time_download_once_each() {
        test_support::with_test_provider(|provider| {
            let matrix = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let sample_time = Tensor::new(vec![0.25], vec![1, 1]).unwrap();
            let matrix =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &matrix).unwrap();
            let sample_time =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &sample_time)
                    .unwrap();
            provider.reset_telemetry();
            let model = run_ss(
                Value::GpuTensor(matrix),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
                vec![Value::GpuTensor(sample_time)],
            )
            .expect("resident ss inputs");
            assert_tensor(property(&model, "A"), &[1, 1], &[1.0]);
            assert_eq!(property(&model, "Ts"), &Value::Num(0.25));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 16);
        });
    }

    #[test]
    fn ss_invalid_option_preflights_before_resident_matrix_download() {
        test_support::with_test_provider(|provider| {
            let matrix = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &matrix).unwrap();
            provider.reset_telemetry();
            let error = run_ss(
                Value::GpuTensor(resident),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
                vec![Value::from("Unsupported"), Value::Num(0.1)],
            )
            .expect_err("unsupported option");
            assert_eq!(error.identifier(), SS_ERROR_INVALID_OPTION.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn ss_nontext_option_name_uses_catalog_invalid_option_error() {
        let error = run_ss(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
            vec![Value::Num(7.0), Value::Num(0.1)],
        )
        .expect_err("nontext option name");
        assert_eq!(error.identifier(), SS_ERROR_INVALID_OPTION.identifier);
    }

    #[test]
    fn ss_later_invalid_host_matrix_preflights_before_resident_a() {
        test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            provider.reset_telemetry();
            let error = run_ss(
                Value::GpuTensor(resident),
                Value::from("not a numeric matrix"),
                Value::Num(1.0),
                Value::Num(1.0),
                vec![],
            )
            .expect_err("invalid B");
            assert_eq!(error.identifier(), SS_ERROR_UNSUPPORTED_INPUT.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn ss_all_host_dimension_error_preflights_before_resident_sample_time() {
        test_support::with_test_provider(|provider| {
            let sample_time = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![0.25], vec![1, 1]).unwrap(),
            )
            .unwrap();
            provider.reset_telemetry();
            let error = run_ss(
                Value::Num(1.0),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                Value::Num(1.0),
                Value::Num(1.0),
                vec![Value::GpuTensor(sample_time)],
            )
            .expect_err("inconsistent host dimensions");
            assert_eq!(error.identifier(), SS_ERROR_INVALID_DIMENSIONS.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn ss_invalid_host_sample_time_preflights_before_resident_matrix() {
        test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            provider.reset_telemetry();
            let error = run_ss(
                Value::GpuTensor(resident),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
                vec![Value::Num(-1.0)],
            )
            .expect_err("invalid host sample time");
            assert_eq!(error.identifier(), SS_ERROR_INVALID_SAMPLE_TIME.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn ss_inexact_host_integer_matrix_preflights_before_resident_matrix() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            provider.reset_telemetry();
            let error = run_ss(
                Value::GpuTensor(resident),
                Value::Int(IntValue::U64((1_u64 << 53) + 1)),
                Value::Num(1.0),
                Value::Num(1.0),
                vec![],
            )
            .expect_err("inexact host B preflights before A download");
            assert_eq!(error.identifier(), SS_ERROR_INVALID_MATRIX.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn ss_inexact_host_integer_sample_time_preflights_before_resident_matrix() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let resident = crate::builtins::common::gpu_helpers::upload_tensor(
                provider,
                &Tensor::new(vec![1.0], vec![1, 1]).unwrap(),
            )
            .unwrap();
            provider.reset_telemetry();
            let error = run_ss(
                Value::GpuTensor(resident),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
                vec![Value::Int(IntValue::U64((1_u64 << 53) + 1))],
            )
            .expect_err("inexact host Ts preflights before A download");
            assert_eq!(error.identifier(), SS_ERROR_INVALID_SAMPLE_TIME.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }
}
