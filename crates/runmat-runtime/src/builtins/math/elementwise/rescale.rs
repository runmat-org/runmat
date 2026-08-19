//! MATLAB-compatible `rescale` builtin.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Type,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, Tensor, Value};

use crate::builtins::common::{
    broadcast::{align_shape, compute_strides, BroadcastPlan},
    gpu_helpers, map_control_flow_with_builtin,
    random_args::keyword_of,
    spec::{
        BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ProviderHook,
        ReductionNaN, ResidencyPolicy, ScalarType,
    },
    tensor,
};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "rescale";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::rescale")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("range-scale"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction { name: "reduce_min" },
        ProviderHook::Reduction { name: "reduce_max" },
        ProviderHook::Custom("rescale"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat computes documented semantics on host for now, then uploads the result back to the active provider when any operand was a gpuArray. Providers can add a composed rescale hook to avoid the host round trip.",
};

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scaled output array.",
}];

const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input numeric or logical array.",
};

const INPUT_L: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "l",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Lower bound for the output interval.",
};

const INPUT_U: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "u",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Upper bound for the output interval.",
};

const INPUT_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Name-value option name: \"InputMin\" or \"InputMax\".",
};

const INPUT_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Name-value option value.",
};

const INPUTS_A: [BuiltinParamDescriptor; 1] = [INPUT_A];
const INPUTS_A_L_U: [BuiltinParamDescriptor; 3] = [INPUT_A, INPUT_L, INPUT_U];
const INPUTS_A_NV: [BuiltinParamDescriptor; 3] = [INPUT_A, INPUT_NAME, INPUT_VALUE];
const INPUTS_A_L_U_NV: [BuiltinParamDescriptor; 5] =
    [INPUT_A, INPUT_L, INPUT_U, INPUT_NAME, INPUT_VALUE];

const RESCALE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "R = rescale(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = rescale(A, l, u)",
        inputs: &INPUTS_A_L_U,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = rescale(A, Name, Value)",
        inputs: &INPUTS_A_NV,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = rescale(A, l, u, Name, Value)",
        inputs: &INPUTS_A_L_U_NV,
        outputs: &OUTPUT_R,
    },
];

const RESCALE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESCALE.INVALID_ARGUMENT",
    identifier: Some("RunMat:rescale:InvalidArgument"),
    when: "Arguments, name-value pairs, interval bounds, or input range bounds are invalid.",
    message: "rescale: invalid argument",
};

const RESCALE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESCALE.INVALID_INPUT",
    identifier: Some("RunMat:rescale:InvalidInput"),
    when: "Input values cannot be converted to real numeric or logical arrays.",
    message: "rescale: invalid input",
};

const RESCALE_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESCALE.SIZE_MISMATCH",
    identifier: Some("RunMat:rescale:SizeMismatch"),
    when: "A bound array cannot be implicitly expanded with the input array.",
    message: "rescale: size mismatch",
};

const RESCALE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESCALE.INTERNAL",
    identifier: Some("RunMat:rescale:Internal"),
    when: "Tensor construction, allocation, or GPU gather fails internally.",
    message: "rescale: internal error",
};

const RESCALE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    RESCALE_ERROR_INVALID_ARGUMENT,
    RESCALE_ERROR_INVALID_INPUT,
    RESCALE_ERROR_SIZE_MISMATCH,
    RESCALE_ERROR_INTERNAL,
];

pub const RESCALE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RESCALE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RESCALE_ERRORS,
};

const RESCALE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target accepts numeric arrays; non-single integer data produces double output after a checked binary64 normalization boundary.",
    }];
const RESCALE_INTEGER_BOUND_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "l/u/InputMin/InputMax",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed interval and input-range bounds are accepted numeric operands and cross the same checked computation boundary.",
    }];
pub const RESCALE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "R = rescale(integer_A, ...)",
        inputs: &RESCALE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Authoritative integer samples are checked before materialization; range scaling is an intentional binary64 boundary and output is double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = rescale(A, integer_l, integer_u, integer_InputMin, integer_InputMax)",
        inputs: &RESCALE_INTEGER_BOUND_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Bounds retain exact storage through admission and are materialized only at the named range-scaling boundary; A controls single-versus-double output.",
    },
];

fn rescale_type(args: &[Type], ctx: &runmat_builtins::ResolveContext) -> Type {
    numeric_unary_type(args, ctx)
}

#[runtime_builtin(
    name = "rescale",
    category = "math/elementwise",
    summary = "Scale the range of array elements.",
    keywords = "rescale,normalize,range,InputMin,InputMax,gpu",
    type_resolver(rescale_type),
    descriptor(crate::builtins::math::elementwise::rescale::RESCALE_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::math::elementwise::rescale::RESCALE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::elementwise::rescale"
)]
async fn rescale_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let raw = parse_rescale_args(&rest)?;
    ensure_rescale_integer_boundary(&value, "input").await?;
    ensure_rescale_integer_boundary(&raw.lower, "lower bound").await?;
    ensure_rescale_integer_boundary(&raw.upper, "upper bound").await?;
    if let Some(bound) = raw.input_min.as_ref() {
        ensure_rescale_integer_boundary(bound, "InputMin").await?;
    }
    if let Some(bound) = raw.input_max.as_ref() {
        ensure_rescale_integer_boundary(bound, "InputMax").await?;
    }
    let input = input_tensor(value).await?;
    let lower = bound_tensor(raw.lower, "lower bound").await?;
    let upper = bound_tensor(raw.upper, "upper bound").await?;
    let input_min = match raw.input_min {
        Some(value) => bound_tensor(value, "InputMin").await?,
        None => BoundOperand::host(scalar_tensor(default_input_min(&input.tensor))),
    };
    let input_max = match raw.input_max {
        Some(value) => bound_tensor(value, "InputMax").await?,
        None => BoundOperand::host(scalar_tensor(default_input_max(&input.tensor))),
    };

    compute_rescale(input, lower, upper, input_min, input_max)
}

#[derive(Debug)]
struct ParsedRescaleArgs {
    lower: Value,
    upper: Value,
    input_min: Option<Value>,
    input_max: Option<Value>,
}

struct RescaleInput {
    tensor: Tensor,
    output_dtype: NumericDType,
    gpu_source: Option<GpuTensorHandle>,
}

struct BoundOperand {
    tensor: Tensor,
    gpu_source: Option<GpuTensorHandle>,
}

impl BoundOperand {
    fn host(tensor: Tensor) -> Self {
        Self {
            tensor,
            gpu_source: None,
        }
    }

    fn was_gpu(&self) -> bool {
        self.gpu_source.is_some()
    }
}

#[derive(Debug)]
struct OperandBroadcast {
    shape: Vec<usize>,
    strides: Vec<usize>,
}

fn parse_rescale_args(rest: &[Value]) -> BuiltinResult<ParsedRescaleArgs> {
    let mut parsed = ParsedRescaleArgs {
        lower: Value::Num(0.0),
        upper: Value::Num(1.0),
        input_min: None,
        input_max: None,
    };
    if rest.is_empty() {
        return Ok(parsed);
    }

    if keyword_of(&rest[0]).is_some() {
        parse_name_value_pairs(rest, 0, &mut parsed)?;
        return Ok(parsed);
    }

    if rest.len() < 2 {
        return Err(rescale_invalid_argument(
            "output interval requires both lower and upper bounds",
        ));
    }
    parsed.lower = rest[0].clone();
    parsed.upper = rest[1].clone();
    parse_name_value_pairs(rest, 2, &mut parsed)?;
    Ok(parsed)
}

fn parse_name_value_pairs(
    rest: &[Value],
    start: usize,
    parsed: &mut ParsedRescaleArgs,
) -> BuiltinResult<()> {
    let remaining = rest.len().saturating_sub(start);
    if !remaining.is_multiple_of(2) {
        return Err(rescale_invalid_argument(
            "name-value arguments must appear in pairs",
        ));
    }
    let mut idx = start;
    while idx < rest.len() {
        let name = keyword_of(&rest[idx])
            .ok_or_else(|| rescale_invalid_argument("expected name-value option name"))?;
        let value = rest[idx + 1].clone();
        match name.as_str() {
            "inputmin" => parsed.input_min = Some(value),
            "inputmax" => parsed.input_max = Some(value),
            _ => {
                return Err(rescale_invalid_argument(format!(
                    "unknown option '{}'",
                    name
                )));
            }
        }
        idx += 2;
    }
    Ok(())
}

async fn input_tensor(value: Value) -> BuiltinResult<RescaleInput> {
    let (value, gpu_source) = gather_value(value).await?;
    match value {
        Value::Tensor(tensor) => {
            let output_dtype = if tensor.numeric_dtype() == NumericDType::F32 {
                NumericDType::F32
            } else {
                NumericDType::F64
            };
            Ok(RescaleInput {
                tensor,
                output_dtype,
                gpu_source,
            })
        }
        Value::LogicalArray(logical) => {
            let data = logical
                .data
                .iter()
                .map(|&flag| if flag != 0 { 1.0 } else { 0.0 })
                .collect();
            let tensor = Tensor::new(data, logical.shape).map_err(rescale_internal)?;
            Ok(RescaleInput {
                tensor,
                output_dtype: NumericDType::F64,
                gpu_source,
            })
        }
        Value::Num(n) => Ok(RescaleInput {
            tensor: scalar_tensor(n),
            output_dtype: NumericDType::F64,
            gpu_source,
        }),
        Value::Int(i) => Ok(RescaleInput {
            tensor: scalar_tensor(int_to_f64(&i)),
            output_dtype: NumericDType::F64,
            gpu_source,
        }),
        Value::Bool(flag) => Ok(RescaleInput {
            tensor: scalar_tensor(if flag { 1.0 } else { 0.0 }),
            output_dtype: NumericDType::F64,
            gpu_source,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(rescale_invalid_input(
            "complex inputs are not supported by rescale",
        )),
        other => Err(rescale_invalid_input(format!(
            "expected real numeric or logical input, got {other:?}"
        ))),
    }
}

async fn bound_tensor(value: Value, label: &str) -> BuiltinResult<BoundOperand> {
    let (value, gpu_source) = gather_value(value).await?;
    let tensor = match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::LogicalArray(logical) => {
            let data = logical
                .data
                .iter()
                .map(|&flag| if flag != 0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::new(data, logical.shape).map_err(rescale_internal)
        }
        Value::Num(n) => Ok(scalar_tensor(n)),
        Value::Int(i) => Ok(scalar_tensor(int_to_f64(&i))),
        Value::Bool(flag) => Ok(scalar_tensor(if flag { 1.0 } else { 0.0 })),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(rescale_invalid_input(format!("{label} must be real")))
        }
        other => Err(rescale_invalid_input(format!(
            "{label} must be numeric or logical, got {other:?}"
        ))),
    }?;
    Ok(BoundOperand { tensor, gpu_source })
}

async fn gather_value(value: Value) -> BuiltinResult<(Value, Option<GpuTensorHandle>)> {
    match value {
        Value::GpuTensor(handle) => {
            provider_for_gpu(&handle)?;
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            Ok((Value::Tensor(tensor), Some(handle)))
        }
        other => Ok((other, None)),
    }
}

fn compute_rescale(
    input: RescaleInput,
    lower: BoundOperand,
    upper: BoundOperand,
    input_min: BoundOperand,
    input_max: BoundOperand,
) -> BuiltinResult<Value> {
    let should_upload = input.gpu_source.is_some()
        || lower.was_gpu()
        || upper.was_gpu()
        || input_min.was_gpu()
        || input_max.was_gpu();
    let output_source = gpu_helpers::select_resident_output_source(
        [
            input.gpu_source.clone(),
            lower.gpu_source.clone(),
            upper.gpu_source.clone(),
            input_min.gpu_source.clone(),
            input_max.gpu_source.clone(),
        ]
        .into_iter()
        .flatten(),
        BUILTIN_NAME,
    )?;
    let output_shape = broadcast_output_shape(
        &input.tensor.shape,
        &[
            &lower.tensor.shape,
            &upper.tensor.shape,
            &input_min.tensor.shape,
            &input_max.tensor.shape,
        ],
    )?;
    let len = checked_element_count(&output_shape)?;
    if len == 0 {
        let tensor = Tensor::new_with_dtype(Vec::new(), output_shape, input.output_dtype)
            .map_err(rescale_internal)?;
        return output_value(tensor, should_upload, output_source.as_ref());
    }

    let input_bc = OperandBroadcast::new(&input.tensor.shape, output_shape.len());
    let lower_bc = OperandBroadcast::new(&lower.tensor.shape, output_shape.len());
    let upper_bc = OperandBroadcast::new(&upper.tensor.shape, output_shape.len());
    let input_min_bc = OperandBroadcast::new(&input_min.tensor.shape, output_shape.len());
    let input_max_bc = OperandBroadcast::new(&input_max.tensor.shape, output_shape.len());
    let input_values = tensor::tensor_values_f64_cow(&input.tensor);
    let lower_values = tensor::tensor_values_f64_cow(&lower.tensor);
    let upper_values = tensor::tensor_values_f64_cow(&upper.tensor);
    let input_min_values = tensor::tensor_values_f64_cow(&input_min.tensor);
    let input_max_values = tensor::tensor_values_f64_cow(&input_max.tensor);

    let mut out = Vec::with_capacity(len);
    for linear in 0..len {
        let a = input_values[input_bc.index(linear, &output_shape)];
        let lo = lower_values[lower_bc.index(linear, &output_shape)];
        let hi = upper_values[upper_bc.index(linear, &output_shape)];
        let in_min = input_min_values[input_min_bc.index(linear, &output_shape)];
        let in_max = input_max_values[input_max_bc.index(linear, &output_shape)];

        if lo >= hi && !lo.is_nan() && !hi.is_nan() {
            return Err(rescale_invalid_argument(
                "lower output bounds must be less than upper output bounds",
            ));
        }
        if in_min > in_max && !in_min.is_nan() && !in_max.is_nan() {
            return Err(rescale_invalid_argument(
                "InputMin must be less than or equal to InputMax",
            ));
        }

        let scaled = scale_one(a, lo, hi, in_min, in_max);
        out.push(cast_output(scaled, input.output_dtype));
    }

    let tensor =
        Tensor::new_with_dtype(out, output_shape, input.output_dtype).map_err(rescale_internal)?;
    output_value(tensor, should_upload, output_source.as_ref())
}

fn broadcast_output_shape(input_shape: &[usize], shapes: &[&[usize]]) -> BuiltinResult<Vec<usize>> {
    let mut output_shape = input_shape.to_vec();
    for shape in shapes {
        let plan =
            BroadcastPlan::new(&output_shape, shape).map_err(|err| rescale_size_mismatch(&err))?;
        output_shape = plan.output_shape().to_vec();
    }
    Ok(output_shape)
}

impl OperandBroadcast {
    fn new(shape: &[usize], rank: usize) -> Self {
        let padded = align_shape(shape, rank);
        let strides = compute_strides(&padded);
        Self {
            shape: padded,
            strides,
        }
    }

    fn index(&self, mut linear: usize, output_shape: &[usize]) -> usize {
        let mut offset = 0usize;
        for (dim, &out_extent) in output_shape.iter().enumerate() {
            let coord = if out_extent == 0 {
                0
            } else {
                linear % out_extent
            };
            if out_extent != 0 {
                linear /= out_extent;
            }
            let in_extent = self.shape[dim];
            let mapped = if in_extent <= 1 { 0 } else { coord };
            offset += mapped * self.strides[dim];
        }
        offset
    }
}

fn scale_one(value: f64, lower: f64, upper: f64, input_min: f64, input_max: f64) -> f64 {
    if input_min == input_max {
        if lower.is_infinite() || upper.is_infinite() {
            return f64::NAN;
        }
        return lower;
    }

    let clipped = if value < input_min {
        input_min
    } else if value > input_max {
        input_max
    } else {
        value
    };
    lower + ((clipped - input_min) / (input_max - input_min)) * (upper - lower)
}

fn default_input_min(tensor: &Tensor) -> f64 {
    let mut min = f64::NAN;
    let values = tensor::tensor_values_f64_cow(tensor);
    for &value in values.iter() {
        if value.is_nan() {
            continue;
        }
        if min.is_nan() || value < min {
            min = value;
        }
    }
    min
}

fn default_input_max(tensor: &Tensor) -> f64 {
    let mut max = f64::NAN;
    let values = tensor::tensor_values_f64_cow(tensor);
    for &value in values.iter() {
        if value.is_nan() {
            continue;
        }
        if max.is_nan() || value > max {
            max = value;
        }
    }
    max
}

fn cast_output(value: f64, dtype: NumericDType) -> f64 {
    if dtype == NumericDType::F32 {
        value as f32 as f64
    } else {
        value
    }
}

fn scalar_tensor(value: f64) -> Tensor {
    Tensor::new(vec![value], vec![1, 1]).expect("scalar tensor shape is valid")
}

fn int_to_f64(value: &IntValue) -> f64 {
    match value {
        IntValue::I8(v) => *v as f64,
        IntValue::I16(v) => *v as f64,
        IntValue::I32(v) => *v as f64,
        IntValue::I64(v) => *v as f64,
        IntValue::U8(v) => *v as f64,
        IntValue::U16(v) => *v as f64,
        IntValue::U32(v) => *v as f64,
        IntValue::U64(v) => *v as f64,
    }
}

fn output_value(
    tensor: Tensor,
    upload_to_gpu: bool,
    source: Option<&GpuTensorHandle>,
) -> BuiltinResult<Value> {
    if upload_to_gpu {
        let source =
            source.ok_or_else(|| rescale_internal("missing GPU source for rescale output"))?;
        return gpu_helpers::restore_class_preserving_value(
            source,
            rescale_tensor_into_value(tensor),
            BUILTIN_NAME,
        );
    }
    Ok(rescale_tensor_into_value(tensor))
}

fn provider_for_gpu(
    handle: &GpuTensorHandle,
) -> BuiltinResult<&'static dyn runmat_accelerate_api::AccelProvider> {
    gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| rescale_internal("no active GPU provider for rescale input"))
}

async fn ensure_rescale_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        return Err(rescale_invalid_input(format!(
            "integer {role} values must be exactly representable as double"
        )));
    }
    Ok(())
}

fn rescale_tensor_into_value(tensor: Tensor) -> Value {
    if tensor.numeric_dtype() == NumericDType::F64 && tensor.len() == 1 {
        Value::Num(
            tensor
                .as_f64_slice()
                .expect("double tensor has double storage")[0],
        )
    } else {
        Value::Tensor(tensor)
    }
}

fn checked_element_count(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            rescale_internal(format!(
                "output shape {:?} exceeds supported element count",
                shape
            ))
        })
    })
}

fn rescale_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rescale_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    rescale_error(&RESCALE_ERROR_INVALID_ARGUMENT, detail)
}

fn rescale_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    rescale_error(&RESCALE_ERROR_INVALID_INPUT, detail)
}

fn rescale_size_mismatch(detail: impl AsRef<str>) -> RuntimeError {
    rescale_error(&RESCALE_ERROR_SIZE_MISMATCH, detail)
}

fn rescale_internal(detail: impl AsRef<str>) -> RuntimeError {
    rescale_error(&RESCALE_ERROR_INTERNAL, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, LogicalArray, NumericStorage};

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn single_tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_with_dtype(data, shape, NumericDType::F32).unwrap())
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
    }

    fn values(value: Value) -> (Vec<f64>, Vec<usize>, NumericDType) {
        match value {
            Value::Num(n) => (vec![n], vec![1, 1], NumericDType::F64),
            Value::Tensor(t) => {
                let dtype = t.numeric_dtype();
                let data = t.materialize_f64();
                (data, t.shape, dtype)
            }
            other => panic!("expected numeric output, got {other:?}"),
        }
    }

    #[test]
    fn rescale_operand_indexing_appends_trailing_singletons() {
        let operand = OperandBroadcast::new(&[2, 1], 3);
        assert_eq!(
            (0..6)
                .map(|linear| operand.index(linear, &[2, 1, 3]))
                .collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 0, 1]
        );
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            if e.is_nan() {
                assert!(a.is_nan(), "index {idx}: expected NaN, got {a}");
            } else {
                assert!((a - e).abs() < 1e-12, "index {idx}: expected {e}, got {a}");
            }
        }
    }

    #[tokio::test]
    async fn rescales_vector_to_unit_interval() {
        let result = rescale_builtin(tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]), vec![])
            .await
            .unwrap();
        let (data, shape, dtype) = values(result);
        assert_eq!(shape, vec![1, 5]);
        assert_eq!(dtype, NumericDType::F64);
        assert_close(&data, &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[tokio::test]
    async fn supports_custom_output_interval() {
        let result = rescale_builtin(
            tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
            vec![Value::Num(-1.0), Value::Num(1.0)],
        )
        .await
        .unwrap();
        let (data, shape, _) = values(result);
        assert_eq!(shape, vec![1, 5]);
        assert_close(&data, &[-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[tokio::test]
    async fn clips_with_input_min_and_input_max() {
        let result = rescale_builtin(
            tensor(vec![-30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 70.0], vec![1, 7]),
            vec![
                Value::from("InputMin"),
                Value::Num(1.0),
                Value::from("InputMax"),
                Value::Num(5.0),
            ],
        )
        .await
        .unwrap();
        let (data, shape, _) = values(result);
        assert_eq!(shape, vec![1, 7]);
        assert_close(&data, &[0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
    }

    #[tokio::test]
    async fn supports_columnwise_range_vectors() {
        let a = tensor(vec![0.4, 0.5, 0.9, 0.2, -4.0, -5.0, 9.0, 1.0], vec![4, 2]);
        let result = rescale_builtin(
            a,
            vec![
                Value::from("InputMin"),
                tensor(vec![0.2, -5.0], vec![1, 2]),
                Value::from("InputMax"),
                tensor(vec![0.9, 9.0], vec![1, 2]),
            ],
        )
        .await
        .unwrap();
        let (data, shape, _) = values(result);
        assert_eq!(shape, vec![4, 2]);
        assert_close(
            &data,
            &[
                0.28571428571428575,
                0.4285714285714286,
                1.0,
                0.0,
                0.07142857142857142,
                0.0,
                1.0,
                0.42857142857142855,
            ],
        );
    }

    #[tokio::test]
    async fn broadcasts_output_bounds_with_input_range_vectors() {
        let a = tensor(vec![0.4, 0.5, 0.9, 0.2, -4.0, -5.0, 9.0, 1.0], vec![4, 2]);
        let result = rescale_builtin(
            a,
            vec![
                tensor(vec![0.0, -1.0], vec![1, 2]),
                Value::Num(1.0),
                Value::from("InputMin"),
                tensor(vec![0.2, -5.0], vec![1, 2]),
                Value::from("InputMax"),
                tensor(vec![0.9, 9.0], vec![1, 2]),
            ],
        )
        .await
        .unwrap();
        let (data, shape, _) = values(result);
        assert_eq!(shape, vec![4, 2]);
        assert_close(
            &data,
            &[
                0.28571428571428575,
                0.4285714285714286,
                1.0,
                0.0,
                -0.8571428571428572,
                -1.0,
                1.0,
                -0.1428571428571429,
            ],
        );
    }

    #[tokio::test]
    async fn constant_input_returns_lower_bound() {
        let result = rescale_builtin(
            tensor(vec![7.0, 7.0, 7.0], vec![1, 3]),
            vec![Value::Num(-2.0), Value::Num(2.0)],
        )
        .await
        .unwrap();
        let (data, shape, _) = values(result);
        assert_eq!(shape, vec![1, 3]);
        assert_close(&data, &[-2.0, -2.0, -2.0]);
    }

    #[tokio::test]
    async fn constant_input_with_infinite_interval_returns_nan() {
        let result = rescale_builtin(
            tensor(vec![7.0, 7.0], vec![1, 2]),
            vec![Value::Num(0.0), Value::Num(f64::INFINITY)],
        )
        .await
        .unwrap();
        let (data, _, _) = values(result);
        assert!(data.iter().all(|value| value.is_nan()));
    }

    #[tokio::test]
    async fn mixed_nan_uses_non_nan_default_range_and_preserves_nan_element() {
        let result = rescale_builtin(tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]), vec![])
            .await
            .unwrap();
        let (data, _, _) = values(result);
        assert_close(&data, &[0.0, f64::NAN, 1.0]);
    }

    #[tokio::test]
    async fn all_nan_default_range_returns_nan_values() {
        let result = rescale_builtin(tensor(vec![f64::NAN, f64::NAN], vec![1, 2]), vec![])
            .await
            .unwrap();
        let (data, _, _) = values(result);
        assert!(data.iter().all(|value| value.is_nan()));
    }

    #[tokio::test]
    async fn explicit_nan_input_bound_propagates() {
        let result = rescale_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            vec![Value::from("InputMin"), Value::Num(f64::NAN)],
        )
        .await
        .unwrap();
        let (data, _, _) = values(result);
        assert!(data.iter().all(|value| value.is_nan()));
    }

    #[tokio::test]
    async fn preserves_empty_shape() {
        let result = rescale_builtin(tensor(Vec::new(), vec![0, 3]), vec![])
            .await
            .unwrap();
        let (data, shape, dtype) = values(result);
        assert!(data.is_empty());
        assert_eq!(shape, vec![0, 3]);
        assert_eq!(dtype, NumericDType::F64);
    }

    #[tokio::test]
    async fn logical_input_maps_to_double_numeric() {
        let logical = Value::LogicalArray(LogicalArray::new(vec![0, 1, 1], vec![1, 3]).unwrap());
        let result = rescale_builtin(logical, vec![]).await.unwrap();
        let (data, shape, dtype) = values(result);
        assert_eq!(shape, vec![1, 3]);
        assert_eq!(dtype, NumericDType::F64);
        assert_close(&data, &[0.0, 1.0, 1.0]);
    }

    #[tokio::test]
    async fn single_input_preserves_native_single_storage() {
        let result = rescale_builtin(single_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]), vec![])
            .await
            .unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected single tensor");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        match tensor.into_numeric_storage().unwrap() {
            NumericStorage::F32(values) => assert_eq!(values, vec![0.0, 0.5, 1.0]),
            storage => panic!("expected native single storage, got {storage:?}"),
        }
    }

    #[tokio::test]
    async fn scalar_uint64_uses_unsigned_value_without_i64_saturation() {
        let above_i64 = (1_u64 << 63) + 4096;
        let result = rescale_builtin(
            Value::Int(IntValue::U64(above_i64)),
            vec![
                Value::from("InputMin"),
                Value::Int(IntValue::U64(above_i64 - 4096)),
                Value::from("InputMax"),
                Value::Int(IntValue::U64(above_i64 + 4096)),
            ],
        )
        .await
        .unwrap();
        let (data, shape, _) = values(result);
        assert_eq!(shape, vec![1, 1]);
        assert_close(&data, &[0.5]);
    }

    #[tokio::test]
    async fn integer_inputs_promote_to_double_output() {
        let result = rescale_builtin(
            integer_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]),
            vec![],
        )
        .await
        .unwrap();
        let (data, shape, dtype) = values(result);
        assert_eq!(shape, vec![1, 3]);
        assert_eq!(dtype, NumericDType::F64);
        assert_close(&data, &[0.0, 0.5, 1.0]);

        let result = rescale_builtin(
            integer_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]),
            vec![
                integer_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
                integer_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
                Value::from("InputMin"),
                integer_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
                Value::from("InputMax"),
                integer_tensor(IntegerStorage::I16(vec![3]), vec![1, 1]),
            ],
        )
        .await
        .unwrap();
        let (data, shape, dtype) = values(result);
        assert_eq!(shape, vec![1, 3]);
        assert_eq!(dtype, NumericDType::F64);
        assert_close(&data, &[-1.0, 0.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_input_returns_gpu_value_and_matches_cpu() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result =
                block_on(rescale_builtin(Value::GpuTensor(handle), vec![])).expect("rescale gpu");
            let Value::GpuTensor(_) = result else {
                panic!("expected gpuArray output, got {result:?}");
            };
            let gathered = crate::builtins::common::test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_close(&gathered.materialize_f64(), &[0.0, 0.5, 1.0]);
        });
    }

    #[tokio::test]
    async fn rejects_unknown_name_value_option() {
        let err = rescale_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            vec![Value::from("Range"), Value::Num(1.0)],
        )
        .await
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:rescale:InvalidArgument"));
    }

    #[tokio::test]
    async fn rejects_mismatched_bound_shapes() {
        let err = rescale_builtin(
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            vec![tensor(vec![0.0, 0.0, 0.0], vec![1, 3]), Value::Num(1.0)],
        )
        .await
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:rescale:SizeMismatch"));
    }

    #[tokio::test]
    async fn rejects_invalid_output_interval() {
        let err = rescale_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            vec![Value::Num(1.0), Value::Num(1.0)],
        )
        .await
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:rescale:InvalidArgument"));
    }

    #[tokio::test]
    async fn rejects_complex_input() {
        let err = rescale_builtin(Value::Complex(1.0, 2.0), vec![])
            .await
            .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:rescale:InvalidInput"));
    }
}
