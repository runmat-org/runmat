//! MATLAB-compatible `atan2` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{broadcast::BroadcastPlan, gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_binary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "atan2";

const ATAN2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Quadrant-aware inverse tangent result.",
}];

const ATAN2_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real single/double y-coordinate; integer, logical, and character forms are RunMat-only extensions.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real single/double x-coordinate; integer, logical, and character forms are RunMat-only extensions.",
    },
];

const ATAN2_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Z = atan2(Y, X)",
    inputs: &ATAN2_INPUTS,
    outputs: &ATAN2_OUTPUT,
}];

const ATAN2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.INVALID_INPUT",
    identifier: Some("RunMat:atan2:InvalidInput"),
    when: "An input cannot be interpreted as supported real numeric data.",
    message: "atan2: invalid input",
};

const ATAN2_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:atan2:ComplexUnsupported"),
    when: "At least one operand is complex.",
    message: "atan2: complex inputs are not supported",
};

const ATAN2_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.SIZE_MISMATCH",
    identifier: Some("RunMat:atan2:SizeMismatch"),
    when: "Input operands are not broadcast-compatible.",
    message: "atan2: size mismatch",
};

const ATAN2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.INTERNAL",
    identifier: Some("RunMat:atan2:Internal"),
    when: "Internal gather/conversion/allocation/provider flow failed.",
    message: "atan2: internal error",
};

const ATAN2_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN2.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:atan2:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "atan2: too many output arguments",
};

const ATAN2_ERRORS: [BuiltinErrorDescriptor; 5] = [
    ATAN2_ERROR_INVALID_INPUT,
    ATAN2_ERROR_COMPLEX_UNSUPPORTED,
    ATAN2_ERROR_SIZE_MISMATCH,
    ATAN2_ERROR_INTERNAL,
    ATAN2_ERROR_TOO_MANY_OUTPUTS,
];

const ATAN2_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan2-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan2 with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Atan2IntegerInputExtension"),
};
const ATAN2_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan2-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan2 with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Atan2LogicalInputExtension"),
};
const ATAN2_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan2-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan2 with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Atan2CharacterInputExtension"),
};
const ATAN2_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ATAN2_INTEGER_INPUT_EXTENSION,
    ATAN2_LOGICAL_INPUT_EXTENSION,
    ATAN2_CHARACTER_INPUT_EXTENSION,
];
const ATAN2_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented domain is real single/double; RunMat mode additionally accepts every real integer class for Y.",
    },
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented domain is real single/double; RunMat mode additionally accepts every real integer class for X.",
    },
];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Z = atan2(integer_Y, integer_X)",
        inputs: &ATAN2_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer operands cross one explicit binary64 atan2 boundary after compatible-size expansion. Resident inputs gather exactly and the floating result returns to the owning provider.",
    }];

pub const ATAN2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ATAN2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ATAN2_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::atan2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "atan2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_atan2",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can implement elem_atan2 to keep the computation on device; the runtime gathers operands to the host when the hook is unavailable or broadcasting is required.",
};

fn atan2_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn atan2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::atan2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "atan2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let y = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let x = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            let negative_zero = match ctx.scalar_ty {
                ScalarType::F32 => format!("bitcast<u32>({x}) == 0x80000000u"),
                ScalarType::F64 => {
                    format!("bitcast<u64>({x}) == 0x8000000000000000u")
                }
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!(
                "select(atan2({y}, {x}), select({y}, 0.0, {negative_zero}), ({y} == 0.0) && ({x} == 0.0))"
            ))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits MATLAB-compatible atan2(y, x), returning positive zero for either signed-zero numerator when the denominator is negative zero while preserving the numerator sign for a positive-zero denominator; providers may override via elem_atan2 for standalone execution.",
};

#[runtime_builtin(
    name = "atan2",
    category = "math/trigonometry",
    summary = "Quadrant-aware inverse tangent atan2(y, x).",
    keywords = "atan2,inverse tangent,quadrant,gpu",
    accel = "binary",
    type_resolver(numeric_binary_type),
    descriptor(crate::builtins::math::trigonometry::atan2::ATAN2_DESCRIPTOR),
    extensions(ATAN2_EXTENSIONS),
    integer_capabilities(crate::builtins::math::trigonometry::atan2::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::trigonometry::atan2"
)]
async fn atan2_builtin(y: Value, x: Value) -> BuiltinResult<Value> {
    super::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    ensure_atan2_input_extensions(&y)?;
    ensure_atan2_input_extensions(&x)?;
    match (y, x) {
        (Value::GpuTensor(yh), Value::GpuTensor(xh)) => atan2_gpu_pair(yh, xh).await,
        (Value::GpuTensor(yh), other) => {
            let provider = runmat_accelerate_api::provider_for_handle(&yh).ok_or_else(|| {
                atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, "GPU input has no owning provider")
            })?;
            let gathered = gpu_helpers::gather_tensor_async(&yh).await?;
            let output = atan2_host(Value::Tensor(gathered), other)?;
            super::inverse_helpers::upload_value(provider, output, BUILTIN_NAME)
        }
        (other, Value::GpuTensor(xh)) => {
            let provider = runmat_accelerate_api::provider_for_handle(&xh).ok_or_else(|| {
                atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, "GPU input has no owning provider")
            })?;
            let gathered = gpu_helpers::gather_tensor_async(&xh).await?;
            let output = atan2_host(other, Value::Tensor(gathered))?;
            super::inverse_helpers::upload_value(provider, output, BUILTIN_NAME)
        }
        (lhs, rhs) => atan2_host(lhs, rhs),
    }
}

async fn atan2_gpu_pair(y: GpuTensorHandle, x: GpuTensorHandle) -> BuiltinResult<Value> {
    let owner = runmat_accelerate_api::provider_for_handle(&y).ok_or_else(|| {
        atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, "GPU input has no owning provider")
    })?;
    let nonfloating = |handle: &GpuTensorHandle| {
        runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
    };
    if y.device_id == x.device_id && y.shape == x.shape && !nonfloating(&y) && !nonfloating(&x) {
        if let Ok(handle) = owner.elem_atan2(&y, &x).await {
            return Ok(Value::GpuTensor(handle));
        }
    }
    let host_y = gpu_helpers::gather_tensor_async(&y).await?;
    let host_x = gpu_helpers::gather_tensor_async(&x).await?;
    let output = atan2_host(Value::Tensor(host_y), Value::Tensor(host_x))?;
    super::inverse_helpers::upload_value(owner, output, BUILTIN_NAME)
}

fn atan2_host(y: Value, x: Value) -> BuiltinResult<Value> {
    let tensor_y = value_into_atan2_tensor(y)?;
    let tensor_x = value_into_atan2_tensor(x)?;
    compute_atan2_tensor(&tensor_y, &tensor_x)
}

fn compute_atan2_tensor(y: &Tensor, x: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&y.shape, &x.shape)
        .map_err(|e| atan2_error_with_detail(&ATAN2_ERROR_SIZE_MISMATCH, e))?;
    let y_dtype = y.numeric_dtype();
    let x_dtype = x.numeric_dtype();
    let output_f32 = y_dtype == NumericDType::F32 || x_dtype == NumericDType::F32;
    let both_f32 = y_dtype == NumericDType::F32 && x_dtype == NumericDType::F32;
    if plan.is_empty() {
        let storage = if output_f32 {
            NumericStorage::F32(Vec::new())
        } else {
            NumericStorage::F64(Vec::new())
        };
        let empty = Tensor::from_numeric_storage(storage, plan.output_shape().to_vec())
            .map_err(|e| atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, e))?;
        return Ok(tensor::tensor_into_value(empty));
    }
    let y_data = tensor::tensor_values_f64_cow(y);
    let x_data = tensor::tensor_values_f64_cow(x);
    let storage = if output_f32 {
        let mut out = vec![0.0f32; plan.len()];
        for (out_index, idx_y, idx_x) in plan.iter() {
            out[out_index] = if both_f32 {
                matlab_atan2_f32(y_data[idx_y] as f32, x_data[idx_x] as f32)
            } else {
                matlab_atan2_f64(y_data[idx_y], x_data[idx_x]) as f32
            };
        }
        NumericStorage::F32(out)
    } else {
        let mut out = vec![0.0f64; plan.len()];
        for (out_index, idx_y, idx_x) in plan.iter() {
            out[out_index] = matlab_atan2_f64(y_data[idx_y], x_data[idx_x]);
        }
        NumericStorage::F64(out)
    };
    let tensor = Tensor::from_numeric_storage(storage, plan.output_shape().to_vec())
        .map_err(|e| atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn matlab_atan2_f64(y: f64, x: f64) -> f64 {
    if y == 0.0 && x == 0.0 && x.is_sign_negative() {
        0.0
    } else {
        y.atan2(x)
    }
}

fn matlab_atan2_f32(y: f32, x: f32) -> f32 {
    if y == 0.0 && x == 0.0 && x.is_sign_negative() {
        0.0
    } else {
        y.atan2(x)
    }
}

fn ensure_atan2_input_extensions(value: &Value) -> BuiltinResult<()> {
    super::inverse_helpers::ensure_input_extensions(
        value,
        BUILTIN_NAME,
        &ATAN2_INTEGER_INPUT_EXTENSION,
        &ATAN2_LOGICAL_INPUT_EXTENSION,
        &ATAN2_CHARACTER_INPUT_EXTENSION,
    )
}

fn value_into_atan2_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::CharArray(chars) => {
            let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
            Tensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|e| atan2_error_with_detail(&ATAN2_ERROR_INTERNAL, e))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(atan2_error(&ATAN2_ERROR_COMPLEX_UNSUPPORTED))
        }
        Value::GpuTensor(_) => Err(atan2_error_with_detail(
            &ATAN2_ERROR_INTERNAL,
            "internal error converting GPU tensor",
        )),
        other => tensor::value_into_tensor_for("atan2", other)
            .map_err(|e| atan2_error_with_detail(&ATAN2_ERROR_INVALID_INPUT, e)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{CharArray, IntegerStorage, LogicalArray, Tensor, Value};
    use std::f64::consts::PI;

    const EPS: f64 = 1e-12;

    fn atan2_builtin(y: Value, x: Value) -> BuiltinResult<Value> {
        block_on(super::atan2_builtin(y, x))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn atan2_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = ATAN2_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Z = atan2(Y, X)"));
    }

    #[test]
    fn atan2_type_preserves_tensor_shape() {
        let out = numeric_binary_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn atan2_type_scalar_returns_num() {
        let out = numeric_binary_type(&[Type::Num, Type::Int], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_scalar_pair() {
        let result = atan2_builtin(Value::Num(1.0), Value::Num(1.0)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v - PI / 4.0).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_quadrant_detection() {
        let result = atan2_builtin(Value::Num(-1.0), Value::Num(-1.0)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v + 3.0 * PI / 4.0).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_matrix_vs_scalar_broadcast() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = atan2_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    (1.0f64).atan2(2.0),
                    (2.0f64).atan2(2.0),
                    (3.0f64).atan2(2.0),
                    (4.0f64).atan2(2.0),
                ];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_row_vector_broadcast() {
        let y = Tensor::new(vec![1.0, -1.0, 2.0, -2.0], vec![2, 2]).unwrap();
        let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("row broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    (1.0f64).atan2(1.0),
                    (-1.0f64).atan2(1.0),
                    (2.0f64).atan2(1.0),
                    (-2.0f64).atan2(1.0),
                ];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn atan2_typed_integer_tensors_read_exact_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let y = Tensor::new_integer(IntegerStorage::I16(vec![1, -1, 2, -2]), vec![2, 2]).unwrap();
        let x = Tensor::new_integer(IntegerStorage::I16(vec![1, 1]), vec![1, 2]).unwrap();

        let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    (1.0f64).atan2(1.0),
                    (-1.0f64).atan2(1.0),
                    (2.0f64).atan2(1.0),
                    (-2.0f64).atan2(1.0),
                ];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn atan2_scalar_fast_path_reads_typed_integer_without_double_mirror() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let y = Tensor::new_integer(IntegerStorage::I16(vec![1]), vec![1, 1]).unwrap();
        let x = Tensor::new_integer(IntegerStorage::I16(vec![1]), vec![1, 1]).unwrap();

        let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.atan2(1.0)).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_char_input() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("A".chars().collect(), 1, 1).unwrap();
        let result = atan2_builtin(Value::CharArray(chars), Value::Num(100.0)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v - (65.0f64).atan2(100.0)).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_logical_input() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let x = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]).unwrap();
        let result =
            atan2_builtin(Value::LogicalArray(logical), Value::Tensor(x)).expect("logical atan2");
        match result {
            Value::Tensor(t) => {
                let expected = [
                    1.0f64.atan2(1.0),
                    0.0f64.atan2(1.0),
                    0.0f64.atan2(-1.0),
                    1.0f64.atan2(-1.0),
                ];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_zero_zero_is_zero() {
        let result = atan2_builtin(Value::Num(0.0), Value::Num(0.0)).expect("atan2");
        match result {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_signed_zero_behaviour() {
        let neg_zero = f64::from_bits(0x8000_0000_0000_0000);
        let Value::Num(positive_zero_case) =
            atan2_builtin(Value::Num(0.0), Value::Num(neg_zero)).expect("atan2")
        else {
            panic!("expected numeric result");
        };
        assert_eq!(positive_zero_case.to_bits(), 0.0f64.to_bits());

        let Value::Num(negative_zero_pair) =
            atan2_builtin(Value::Num(neg_zero), Value::Num(neg_zero)).expect("atan2")
        else {
            panic!("expected numeric result");
        };
        assert_eq!(negative_zero_pair.to_bits(), 0.0f64.to_bits());

        let Value::Num(neg_zero_result) =
            atan2_builtin(Value::Num(neg_zero), Value::Num(0.0)).expect("atan2")
        else {
            panic!("expected numeric result");
        };
        assert_eq!(
            neg_zero_result.to_bits(),
            f64::from_bits(0x8000_0000_0000_0000).to_bits(),
            "expected negative zero, got {neg_zero_result}"
        );
    }

    #[test]
    fn atan2_mixed_single_double_computes_in_double_and_returns_single() {
        let y = Tensor::from_numeric_storage(NumericStorage::F32(vec![1.4e32_f32]), vec![1, 1])
            .unwrap();
        let x = Tensor::new(vec![-5.305e32], vec![1, 1]).unwrap();
        let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
        let Value::Tensor(tensor) = result else {
            panic!("expected native-single scalar tensor");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        let expected = matlab_atan2_f64(1.4e32_f32 as f64, -5.305e32) as f32;
        assert_eq!(tensor.as_f32_slice(), Some([expected].as_slice()));
    }

    #[test]
    fn atan2_nonfloating_inputs_are_independently_gated() {
        let integer = atan2_builtin(
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            Value::Num(1.0),
        )
        .expect_err("integer input must be gated");
        assert_eq!(
            integer.identifier(),
            ATAN2_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let logical = atan2_builtin(Value::Bool(true), Value::Num(1.0))
            .expect_err("logical input must be gated");
        assert_eq!(
            logical.identifier(),
            ATAN2_LOGICAL_INPUT_EXTENSION.error_identifier
        );
        let chars = CharArray::new_row("A");
        let character = atan2_builtin(Value::CharArray(chars), Value::Num(1.0))
            .expect_err("character input must be gated");
        assert_eq!(
            character.identifier(),
            ATAN2_CHARACTER_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn atan2_integer_extension_covers_all_eight_classes_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let values = [
            runmat_value::IntValue::I8(i8::MAX),
            runmat_value::IntValue::I16(i16::MAX),
            runmat_value::IntValue::I32(i32::MAX),
            runmat_value::IntValue::I64(i64::MAX),
            runmat_value::IntValue::U8(u8::MAX),
            runmat_value::IntValue::U16(u16::MAX),
            runmat_value::IntValue::U32(u32::MAX),
            runmat_value::IntValue::U64(u64::MAX),
        ];
        for value in values {
            let expected = matlab_atan2_f64(value.to_f64(), 1.0);
            let result = atan2_builtin(Value::Int(value), Value::Num(1.0)).expect("atan2");
            let Value::Num(actual) = result else {
                panic!("expected double scalar result");
            };
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn atan2_rejects_excess_outputs() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let error = atan2_builtin(Value::Num(1.0), Value::Num(1.0))
            .expect_err("second output must be rejected");
        assert_eq!(error.identifier(), ATAN2_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_empty_tensor_result() {
        let y = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let x = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 3]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_complex_input_errors() {
        let err = atan2_builtin(Value::Complex(1.0, 1.0), Value::Num(1.0)).unwrap_err();
        assert_eq!(err.identifier(), ATAN2_ERROR_COMPLEX_UNSUPPORTED.identifier);
        let message = error_message(err);
        assert!(message.to_ascii_lowercase().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_dimension_mismatch_errors() {
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let x = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let err = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).unwrap_err();
        assert_eq!(err.identifier(), ATAN2_ERROR_SIZE_MISMATCH.identifier);
        let message = error_message(err);
        assert!(
            message.to_ascii_lowercase().contains("size"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let neg_zero = f64::from_bits(0x8000_0000_0000_0000);
            let y = Tensor::new(
                vec![1.0, 1.0, -1.0, -1.0, 0.0, neg_zero, neg_zero],
                vec![1, 7],
            )
            .unwrap();
            let x = Tensor::new(
                vec![1.0, -1.0, 1.0, -1.0, neg_zero, neg_zero, 0.0],
                vec![1, 7],
            )
            .unwrap();
            let hy = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload y");
            let hx = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &x.materialize_f64(),
                    shape: &x.shape,
                })
                .expect("upload x");
            let result =
                atan2_builtin(Value::GpuTensor(hy), Value::GpuTensor(hx)).expect("gpu atan2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 7]);
            let expected = [
                (1.0f64).atan2(1.0),
                (1.0f64).atan2(-1.0),
                (-1.0f64).atan2(1.0),
                (-1.0f64).atan2(-1.0),
                0.0,
                0.0,
                neg_zero,
            ];
            for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS);
            }
            let values = gathered.materialize_f64();
            assert_eq!(values[4].to_bits(), 0.0f64.to_bits());
            assert_eq!(values[5].to_bits(), 0.0f64.to_bits());
            assert_eq!(values[6].to_bits(), neg_zero.to_bits());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_gpu_host_mix_falls_back() {
        test_support::with_test_provider(|provider| {
            let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let hy = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload y");
            let result = atan2_builtin(Value::GpuTensor(hy), Value::Num(2.0)).expect("atan2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            let expected = [(1.0f64).atan2(2.0), (2.0f64).atan2(2.0)];
            for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS);
            }
        });
    }

    #[test]
    fn atan2_gpu_host_mix_reads_typed_integer_rhs_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let hy = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload y");
            let x = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).unwrap();

            let result = atan2_builtin(Value::GpuTensor(hy), Value::Tensor(x)).expect("atan2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert!((gathered.materialize_f64()[0] - 1.0f64.atan2(1.0)).abs() < EPS);
            assert!((gathered.materialize_f64()[1] - 2.0f64.atan2(2.0)).abs() < EPS);
        });
    }

    #[test]
    fn atan2_gpu_host_mix_reads_typed_integer_lhs_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let x = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let hx = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &x.materialize_f64(),
                    shape: &x.shape,
                })
                .expect("upload x");
            let y = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).unwrap();

            let result = atan2_builtin(Value::Tensor(y), Value::GpuTensor(hx)).expect("atan2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert!((gathered.materialize_f64()[0] - 1.0f64.atan2(1.0)).abs() < EPS);
            assert!((gathered.materialize_f64()[1] - 2.0f64.atan2(2.0)).abs() < EPS);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn atan2_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let neg_zero = f64::from_bits(0x8000_0000_0000_0000);
        let y = Tensor::new(vec![0.0, neg_zero, neg_zero, 1.0, -1.0, 2.0], vec![2, 3]).unwrap();
        let x = Tensor::new(vec![neg_zero, neg_zero, 0.0, 1.0, 1.0, -1.0], vec![2, 3]).unwrap();
        let cpu = atan2_host(Value::Tensor(y.clone()), Value::Tensor(x.clone())).unwrap();
        let hy = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &y.materialize_f64(),
                shape: &y.shape,
            })
            .unwrap();
        let hx = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &x.materialize_f64(),
                shape: &x.shape,
            })
            .unwrap();
        let gpu = block_on(atan2_gpu_pair(hy, hx)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape, gathered.shape);
                let (absolute_tolerance, relative_tolerance) =
                    test_support::gpu_transcendental_tolerances(
                        runmat_accelerate_api::provider().unwrap().precision(),
                    );
                for (actual, expect) in gathered
                    .materialize_f64()
                    .iter()
                    .zip(ct.materialize_f64().iter())
                {
                    assert!(
                        test_support::floats_match(
                            *actual,
                            *expect,
                            absolute_tolerance,
                            relative_tolerance,
                        ),
                        "{actual} vs {expect}"
                    );
                }
                let values = gathered.materialize_f64();
                assert_eq!(values[0].to_bits(), 0.0f64.to_bits());
                assert_eq!(values[1].to_bits(), 0.0f64.to_bits());
                assert_eq!(values[2].to_bits(), neg_zero.to_bits());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
