//! MATLAB-compatible `bsxfun` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, LogicalArray, NumericDType, NumericScalar, NumericStorage, Tensor,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::user_functions::resolve_semantic_function_by_name;
use crate::{
    build_runtime_error, call_feval_async_with_outputs, gather_if_needed_async, BuiltinResult,
    RuntimeError,
};

const BUILTIN_NAME: &str = "bsxfun";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::bsxfun")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("host-binary-map"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Executes arbitrary binary callbacks on the host after gathering gpuArray inputs; use direct elementwise operators for provider-native GPU execution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::bsxfun")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Callback execution happens through feval; fusion planners should treat bsxfun as a fusion barrier.",
};

const BSXFUN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise callback result after singleton expansion.",
}];

const BSXFUN_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Binary function handle; callable text is a RunMat-only extension.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left scalar or array input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right scalar or array input.",
    },
];

const BSXFUN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = bsxfun(fun, A, B)",
    inputs: &BSXFUN_INPUTS,
    outputs: &BSXFUN_OUTPUT,
}];

const BSXFUN_ERROR_INVALID_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.INVALID_FUNCTION",
    identifier: Some("RunMat:bsxfun:InvalidFunction"),
    when: "The first input cannot be used as a binary function.",
    message: "bsxfun: first input must be a function handle or callable name",
};

const BSXFUN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.INVALID_INPUT",
    identifier: Some("RunMat:bsxfun:InvalidInput"),
    when: "Input arrays use unsupported types for bsxfun scalar expansion.",
    message: "bsxfun: inputs must be numeric, logical, complex, or character arrays",
};

const BSXFUN_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.SIZE_MISMATCH",
    identifier: Some("RunMat:bsxfun:SizeMismatch"),
    when: "Input arrays are not compatible for singleton expansion.",
    message: "bsxfun: input sizes are not compatible for singleton expansion",
};

const BSXFUN_ERROR_FUNCTION_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.FUNCTION_ERROR",
    identifier: Some("RunMat:bsxfun:FunctionError"),
    when: "The binary function errors or does not return scalar-uniform values.",
    message: "bsxfun: callback execution error",
};

const BSXFUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BSXFUN.INTERNAL",
    identifier: Some("RunMat:bsxfun:Internal"),
    when: "Internal output allocation or GPU gather fails.",
    message: "bsxfun: internal error",
};

const BSXFUN_ERRORS: [BuiltinErrorDescriptor; 5] = [
    BSXFUN_ERROR_INVALID_FUNCTION,
    BSXFUN_ERROR_INVALID_INPUT,
    BSXFUN_ERROR_SIZE_MISMATCH,
    BSXFUN_ERROR_FUNCTION_ERROR,
    BSXFUN_ERROR_INTERNAL,
];

const BSXFUN_TEXT_CALLABLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bsxfun-text-callable",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bsxfun with a text callback name is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BsxfunTextCallableExtension"),
};

const BSXFUN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BSXFUN_TEXT_CALLABLE_EXTENSION];

const BSXFUN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "Every integer class is accepted when the selected binary callback accepts that class; mixed-class and scalar-double rules are inherited from the callback.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "Every integer class is accepted when the selected binary callback accepts that class; singleton expansion does not change callback class rules.",
    },
];

pub const BSXFUN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = bsxfun(fun,A,B) with integer A or B",
        inputs: &BSXFUN_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Input and output class, mixed-class admission, saturation, and errors are inherited from fun; RunMat preserves exact homogeneous callback results after singleton expansion. Public gpuArray input is currently gathered for host callback execution.",
    }];

pub const BSXFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BSXFUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BSXFUN_ERRORS,
};

#[runtime_builtin(
    name = "bsxfun",
    category = "math/elementwise",
    summary = "Apply a binary function with singleton expansion.",
    keywords = "bsxfun,binary singleton expansion,implicit expansion,function handle",
    accel = "host",
    descriptor(crate::builtins::math::elementwise::bsxfun::BSXFUN_DESCRIPTOR),
    extensions(crate::builtins::math::elementwise::bsxfun::BSXFUN_EXTENSIONS),
    integer_capabilities(crate::builtins::math::elementwise::bsxfun::BSXFUN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::bsxfun"
)]
async fn bsxfun_builtin(function: Value, left: Value, right: Value) -> BuiltinResult<Value> {
    validate_function(&function)?;
    if matches!(
        function,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BSXFUN_TEXT_CALLABLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&left, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&right, BUILTIN_NAME)?;
    let left = gather_if_needed_async(&left)
        .await
        .map_err(|flow| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, flow.to_string()))?;
    let right = gather_if_needed_async(&right)
        .await
        .map_err(|flow| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, flow.to_string()))?;
    let left = ArrayInput::from_value(left)?;
    let right = ArrayInput::from_value(right)?;
    let plan = BsxfunBroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_SIZE_MISMATCH, err))?;
    let output_hint = CallbackOutputHint::for_function(&function, &left, &right);

    let mut collector = UniformCollector::Pending;
    for (_, left_idx, right_idx) in plan.iter() {
        let args = [left.value_at(left_idx)?, right.value_at(right_idx)?];
        let mut value = call_feval_async_with_outputs(function.clone(), &args, 1)
            .await
            .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_FUNCTION_ERROR, err.message()))?;
        value = gather_if_needed_async(&value)
            .await
            .map_err(|flow| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, flow.to_string()))?;
        if output_hint == CallbackOutputHint::Logical {
            value = logicalize_callback_output(value)?;
        }
        collector.push(&value)?;
    }

    collector.finish(plan.output_shape(), output_hint)
}

fn validate_function(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_) => Ok(()),
        other => Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_INVALID_FUNCTION,
            format!("got {other:?}"),
        )),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CallbackOutputHint {
    None,
    Logical,
    Numeric(NumericDType),
}

impl CallbackOutputHint {
    fn for_function(value: &Value, left: &ArrayInput, right: &ArrayInput) -> Self {
        let Some(name) = callback_function_name(value) else {
            return Self::None;
        };
        if resolve_semantic_function_by_name(&name).is_some() {
            return Self::None;
        }
        match name.as_str() {
            "eq" | "ne" | "gt" | "ge" | "lt" | "le" | "and" | "or" | "xor" => Self::Logical,
            "plus" | "minus" | "times" | "min" | "max" | "mod" | "rem" => {
                infer_numeric_callback_dtype(left, right)
                    .map(Self::Numeric)
                    .unwrap_or(Self::None)
            }
            _ => Self::None,
        }
    }
}

fn infer_numeric_callback_dtype(left: &ArrayInput, right: &ArrayInput) -> Option<NumericDType> {
    let left_dtype = left.data.numeric_dtype()?;
    let right_dtype = right.data.numeric_dtype()?;
    if left_dtype == right_dtype {
        return Some(left_dtype);
    }
    if left_dtype == NumericDType::F32 || right_dtype == NumericDType::F32 {
        return Some(NumericDType::F32);
    }
    if numeric_dtype_is_integer(left_dtype)
        && right_dtype == NumericDType::F64
        && right.data.is_scalar()
    {
        return Some(left_dtype);
    }
    if numeric_dtype_is_integer(right_dtype)
        && left_dtype == NumericDType::F64
        && left.data.is_scalar()
    {
        return Some(right_dtype);
    }
    None
}

fn numeric_dtype_is_integer(dtype: NumericDType) -> bool {
    !matches!(dtype, NumericDType::F64 | NumericDType::F32)
}

fn callback_function_name(value: &Value) -> Option<String> {
    match value {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => normalize_callback_name(name),
        Value::String(name) => normalize_callback_name(name),
        Value::StringArray(array) if array.data.len() == 1 => {
            normalize_callback_name(array.data.first()?)
        }
        Value::CharArray(array) if array.rows == 1 => {
            let name: String = array.data.iter().collect();
            normalize_callback_name(&name)
        }
        _ => None,
    }
}

fn normalize_callback_name(name: &str) -> Option<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.strip_prefix('@').unwrap_or(trimmed).to_string())
}

fn logicalize_callback_output(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => Ok(value),
        Value::Num(value) => Ok(Value::Bool(value != 0.0)),
        Value::Int(value) => Ok(Value::Bool(!value.is_zero())),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => Ok(Value::Bool(
            !tensor
                .numeric_value_at(0)
                .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL))?
                .is_zero(),
        )),
        other => Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            format!("logical callback must return scalar logical values (got {other:?})"),
        )),
    }
}

struct ArrayInput {
    data: ArrayData,
    shape: Vec<usize>,
}

impl ArrayInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        let data = ArrayData::from_value(value)?;
        let shape = data.shape_vec();
        Ok(Self { data, shape })
    }

    fn value_at(&self, idx: usize) -> BuiltinResult<Value> {
        self.data.value_at(idx)
    }
}

enum ArrayData {
    Tensor(Tensor),
    Logical(LogicalArray),
    Complex(ComplexTensor),
    Char(CharArray),
    Scalar(Value),
}

impl ArrayData {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => Ok(Self::Tensor(tensor)),
            Value::LogicalArray(array) => Ok(Self::Logical(array)),
            Value::ComplexTensor(tensor) => Ok(Self::Complex(tensor)),
            Value::CharArray(array) => Ok(Self::Char(array)),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => {
                Ok(Self::Scalar(value))
            }
            other => Err(bsxfun_error_with_detail(
                &BSXFUN_ERROR_INVALID_INPUT,
                format!("unsupported input type {other:?}"),
            )),
        }
    }

    fn shape_vec(&self) -> Vec<usize> {
        match self {
            Self::Tensor(tensor) => normalize_shape(&tensor.shape),
            Self::Logical(array) => normalize_shape(&array.shape),
            Self::Complex(tensor) => normalize_shape(&tensor.shape),
            Self::Char(array) => vec![array.rows, array.cols],
            Self::Scalar(_) => vec![1, 1],
        }
    }

    fn numeric_dtype(&self) -> Option<NumericDType> {
        match self {
            Self::Tensor(tensor) => Some(tensor.numeric_dtype()),
            Self::Scalar(Value::Num(_)) => Some(NumericDType::F64),
            Self::Scalar(Value::Int(value)) => {
                Some(NumericScalar::from(value.clone()).numeric_dtype())
            }
            _ => None,
        }
    }

    fn is_scalar(&self) -> bool {
        self.shape_vec().iter().copied().product::<usize>() == 1
    }

    fn value_at(&self, idx: usize) -> BuiltinResult<Value> {
        match self {
            Self::Tensor(tensor) => tensor
                .numeric_value_at(idx)
                .map(numeric_scalar_value)
                .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL)),
            Self::Logical(array) => array
                .data
                .get(idx)
                .map(|bit| Value::Bool(*bit != 0))
                .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL)),
            Self::Complex(tensor) => tensor
                .materialize_f64()
                .get(idx)
                .copied()
                .map(|(re, im)| Value::Complex(re, im))
                .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL)),
            Self::Char(array) => char_value_at(array, idx),
            Self::Scalar(value) => Ok(value.clone()),
        }
    }
}

fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn numeric_scalar_value(value: NumericScalar) -> Value {
    match value {
        NumericScalar::F64(value) => Value::Num(value),
        NumericScalar::F32(value) => Value::Tensor(
            Tensor::from_f32(vec![value], vec![1, 1])
                .expect("one native-single scalar has a valid shape"),
        ),
        value => Value::Int(
            value
                .into_int_value()
                .expect("non-floating numeric scalar is integer"),
        ),
    }
}

fn char_value_at(array: &CharArray, idx: usize) -> BuiltinResult<Value> {
    if array.rows == 0 || array.cols == 0 {
        let empty = CharArray::new(Vec::new(), 0, 0)
            .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err))?;
        return Ok(Value::CharArray(empty));
    }
    let row = idx % array.rows;
    let col = idx / array.rows;
    let data_idx = row * array.cols + col;
    let ch = array
        .data
        .get(data_idx)
        .copied()
        .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL))?;
    let scalar = CharArray::new(vec![ch], 1, 1)
        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err))?;
    Ok(Value::CharArray(scalar))
}

struct BsxfunBroadcastPlan {
    output_shape: Vec<usize>,
    left_shape: Vec<usize>,
    right_shape: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
    len: usize,
}

impl BsxfunBroadcastPlan {
    fn new(left_shape: &[usize], right_shape: &[usize]) -> Result<Self, String> {
        let rank = left_shape.len().max(right_shape.len());
        let left_shape = crate::builtins::common::broadcast::align_shape(left_shape, rank);
        let right_shape = crate::builtins::common::broadcast::align_shape(right_shape, rank);
        let mut output_shape = Vec::with_capacity(rank);

        for dim in 0..rank {
            let left = left_shape[dim];
            let right = right_shape[dim];
            if left == right {
                output_shape.push(left);
            } else if left == 1 {
                output_shape.push(right);
            } else if right == 1 {
                output_shape.push(left);
            } else {
                return Err(format!(
                    "broadcast: non-singleton dimension mismatch (dimension {}: {} vs {})",
                    dim + 1,
                    left,
                    right
                ));
            }
        }

        let len = checked_element_count(&output_shape)?;
        let left_strides = checked_strides(&left_shape)?;
        let right_strides = checked_strides(&right_shape)?;

        Ok(Self {
            output_shape,
            left_shape,
            right_shape,
            left_strides,
            right_strides,
            len,
        })
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        (0..self.len).map(|idx| {
            (
                idx,
                broadcast_input_index(
                    idx,
                    &self.output_shape,
                    &self.left_shape,
                    &self.left_strides,
                ),
                broadcast_input_index(
                    idx,
                    &self.output_shape,
                    &self.right_shape,
                    &self.right_strides,
                ),
            )
        })
    }

    fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
}

fn checked_element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().copied().try_fold(1usize, |acc, extent| {
        acc.checked_mul(extent)
            .ok_or_else(|| "broadcast: output size exceeds platform limits".to_string())
    })
}

fn checked_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| "broadcast: input size exceeds platform limits".to_string())?;
    }
    Ok(strides)
}

fn broadcast_input_index(
    output_idx: usize,
    output_shape: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
) -> usize {
    let mut remaining = output_idx;
    let mut offset = 0usize;
    for dim in 0..output_shape.len() {
        let extent = output_shape[dim];
        let coord = remaining % extent;
        remaining /= extent;

        if input_shape[dim] > 1 {
            offset += coord * input_strides[dim];
        }
    }
    offset
}

enum UniformCollector {
    Pending,
    Numeric {
        dtype: NumericDType,
        values: Vec<NumericScalar>,
    },
    Logical(Vec<u8>),
    Complex(Vec<(f64, f64)>),
    Char(Vec<char>),
}

impl UniformCollector {
    fn push(&mut self, value: &Value) -> BuiltinResult<()> {
        let classified = classify_value(value)?;
        match self {
            Self::Pending => match classified {
                ClassifiedValue::Logical(bit) => {
                    *self = Self::Logical(vec![u8::from(bit)]);
                    Ok(())
                }
                ClassifiedValue::Numeric(value) => {
                    *self = Self::Numeric {
                        dtype: numeric_scalar_dtype(value),
                        values: vec![value],
                    };
                    Ok(())
                }
                ClassifiedValue::Complex(value) => {
                    *self = Self::Complex(vec![value]);
                    Ok(())
                }
                ClassifiedValue::Char(value) => {
                    *self = Self::Char(vec![value]);
                    Ok(())
                }
            },
            Self::Numeric { dtype, values } => match classified {
                ClassifiedValue::Numeric(value) if numeric_scalar_dtype(value) == *dtype => {
                    values.push(value);
                    Ok(())
                }
                other => Err(inconsistent_callback_class(dtype.class_name(), &other)),
            },
            Self::Logical(bits) => match classified {
                ClassifiedValue::Logical(bit) => {
                    bits.push(u8::from(bit));
                    Ok(())
                }
                other => Err(inconsistent_callback_class("logical", &other)),
            },
            Self::Complex(values) => match classified {
                ClassifiedValue::Complex(value) => {
                    values.push(value);
                    Ok(())
                }
                other => Err(inconsistent_callback_class("complex double", &other)),
            },
            Self::Char(chars) => match classified {
                ClassifiedValue::Char(ch) => {
                    chars.push(ch);
                    Ok(())
                }
                other => Err(inconsistent_callback_class("char", &other)),
            },
        }
    }

    fn finish(self, shape: &[usize], output_hint: CallbackOutputHint) -> BuiltinResult<Value> {
        match self {
            Self::Pending => {
                let len = shape.iter().copied().product();
                match output_hint {
                    CallbackOutputHint::Logical => LogicalArray::new(vec![0; len], shape.to_vec())
                        .map(Value::LogicalArray)
                        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
                    CallbackOutputHint::Numeric(dtype) => Tensor::from_numeric_storage(
                        NumericStorage::zeros(dtype, len),
                        shape.to_vec(),
                    )
                    .map(Value::Tensor)
                    .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
                    CallbackOutputHint::None => Tensor::new(vec![0.0; len], shape.to_vec())
                        .map(Value::Tensor)
                        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
                }
            }
            Self::Numeric { dtype, values } => {
                let mut storage = NumericStorage::zeros(dtype, values.len());
                for (index, value) in values.into_iter().enumerate() {
                    storage
                        .set_value(index, value)
                        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err))?;
                }
                Tensor::from_numeric_storage(storage, shape.to_vec())
                    .map(Value::Tensor)
                    .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err))
            }
            Self::Logical(bits) => LogicalArray::new(bits, shape.to_vec())
                .map(Value::LogicalArray)
                .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
            Self::Complex(values) => ComplexTensor::new(values, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err)),
            Self::Char(chars) => finish_char_output(chars, shape),
        }
    }
}

#[derive(Debug, PartialEq)]
enum ClassifiedValue {
    Logical(bool),
    Numeric(NumericScalar),
    Complex((f64, f64)),
    Char(char),
}

fn numeric_scalar_dtype(value: NumericScalar) -> NumericDType {
    match value {
        NumericScalar::F64(_) => NumericDType::F64,
        NumericScalar::F32(_) => NumericDType::F32,
        NumericScalar::I8(_) => NumericDType::I8,
        NumericScalar::I16(_) => NumericDType::I16,
        NumericScalar::I32(_) => NumericDType::I32,
        NumericScalar::I64(_) => NumericDType::I64,
        NumericScalar::U8(_) => NumericDType::U8,
        NumericScalar::U16(_) => NumericDType::U16,
        NumericScalar::U32(_) => NumericDType::U32,
        NumericScalar::U64(_) => NumericDType::U64,
    }
}

fn classified_value_class(value: &ClassifiedValue) -> &'static str {
    match value {
        ClassifiedValue::Logical(_) => "logical",
        ClassifiedValue::Numeric(value) => numeric_scalar_dtype(*value).class_name(),
        ClassifiedValue::Complex(_) => "complex double",
        ClassifiedValue::Char(_) => "char",
    }
}

fn inconsistent_callback_class(expected: &str, actual: &ClassifiedValue) -> RuntimeError {
    bsxfun_error_with_detail(
        &BSXFUN_ERROR_FUNCTION_ERROR,
        format!(
            "callback output class must be consistent (expected {expected}, got {})",
            classified_value_class(actual)
        ),
    )
}

fn classify_value(value: &Value) -> BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(value) => Ok(ClassifiedValue::Logical(*value)),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(ClassifiedValue::Logical(array.data[0] != 0))
        }
        Value::Num(value) => Ok(ClassifiedValue::Numeric(NumericScalar::F64(*value))),
        Value::Int(value) => Ok(ClassifiedValue::Numeric(NumericScalar::from(
            value.clone(),
        ))),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .numeric_value_at(0)
            .map(ClassifiedValue::Numeric)
            .ok_or_else(|| bsxfun_error(&BSXFUN_ERROR_INTERNAL)),
        Value::Complex(re, im) => Ok(ClassifiedValue::Complex((*re, *im))),
        Value::ComplexTensor(tensor) if tensor::is_scalar_complex_tensor(tensor) => {
            let value = tensor::complex_tensor_value_complex64(tensor, 0);
            Ok(ClassifiedValue::Complex((value.re, value.im)))
        }
        Value::CharArray(array) if array.rows * array.cols == 1 => {
            Ok(ClassifiedValue::Char(array.data.first().copied().unwrap_or('\0')))
        }
        other => Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            format!("callback must return scalar numeric, logical, complex, or character values (got {other:?})"),
        )),
    }
}

fn finish_char_output(chars: Vec<char>, shape: &[usize]) -> BuiltinResult<Value> {
    let normalized_shape = normalize_shape(shape);
    if normalized_shape.len() > 2 {
        return Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            "character callback outputs must form a 2-D char array",
        ));
    }
    let rows = normalized_shape.first().copied().unwrap_or(1);
    let cols = normalized_shape.get(1).copied().unwrap_or(1);
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        bsxfun_error_with_detail(
            &BSXFUN_ERROR_INTERNAL,
            "character output size exceeds platform limits",
        )
    })?;
    if expected != chars.len() {
        return Err(bsxfun_error_with_detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            "callback returned the wrong number of characters",
        ));
    }

    let mut row_major = vec!['\0'; expected];
    for col in 0..cols {
        for row in 0..rows {
            let col_major_idx = row + col * rows;
            let row_major_idx = row * cols + col;
            row_major[row_major_idx] = chars[col_major_idx];
        }
    }

    CharArray::new(row_major, rows, cols)
        .map(Value::CharArray)
        .map_err(|err| bsxfun_error_with_detail(&BSXFUN_ERROR_INTERNAL, err))
}

fn bsxfun_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn bsxfun_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage};
    use std::sync::Arc;

    fn call(function: Value, left: Value, right: Value) -> BuiltinResult<Value> {
        block_on(bsxfun_builtin(function, left, right))
    }

    #[cfg(feature = "wgpu")]
    fn register_wgpu_provider_available() -> bool {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_ok()
            && runmat_accelerate_api::provider().is_some()
    }

    #[test]
    fn bsxfun_descriptor_records_integer_semantics_and_text_extension() {
        assert_eq!(BSXFUN_EXTENSIONS.len(), 1);
        assert_eq!(BSXFUN_EXTENSIONS[0].id, "bsxfun-text-callable");
        assert_eq!(BSXFUN_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(BSXFUN_INTEGER_CAPABILITIES[0].inputs.len(), 2);
        assert_eq!(
            BSXFUN_INTEGER_CAPABILITIES[0].backend,
            BuiltinIntegerBackendRule::GatherFallback
        );
    }

    #[test]
    fn bsxfun_callable_text_is_gated_but_function_handles_are_ordinary() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(Value::from("@plus"), Value::Num(1.0), Value::Num(2.0))
            .expect_err("strict mode must reject callable text");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:BsxfunTextCallableExtension")
        );
        call(
            Value::FunctionHandle("plus".to_string()),
            Value::Num(1.0),
            Value::Num(2.0),
        )
        .expect("function handle is documented");
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = call(Value::from("@plus"), Value::Num(1.0), Value::Num(2.0))
            .expect("RunMat mode admits callable text");
        let Value::Tensor(result) = result else {
            panic!("expected double tensor");
        };
        assert_eq!(result.materialize_f64(), vec![3.0]);
    }

    #[test]
    fn bsxfun_plus_preserves_all_exact_integer_classes() {
        let cases = [
            (
                IntegerStorage::I8(vec![i8::MAX, -5]),
                IntegerStorage::I8(vec![1]),
                IntegerStorage::I8(vec![i8::MAX, -4]),
            ),
            (
                IntegerStorage::I16(vec![i16::MAX, -5]),
                IntegerStorage::I16(vec![1]),
                IntegerStorage::I16(vec![i16::MAX, -4]),
            ),
            (
                IntegerStorage::I32(vec![i32::MAX, -5]),
                IntegerStorage::I32(vec![1]),
                IntegerStorage::I32(vec![i32::MAX, -4]),
            ),
            (
                IntegerStorage::I64(vec![i64::MAX, -5]),
                IntegerStorage::I64(vec![1]),
                IntegerStorage::I64(vec![i64::MAX, -4]),
            ),
            (
                IntegerStorage::U8(vec![u8::MAX, 5]),
                IntegerStorage::U8(vec![1]),
                IntegerStorage::U8(vec![u8::MAX, 6]),
            ),
            (
                IntegerStorage::U16(vec![u16::MAX, 5]),
                IntegerStorage::U16(vec![1]),
                IntegerStorage::U16(vec![u16::MAX, 6]),
            ),
            (
                IntegerStorage::U32(vec![u32::MAX, 5]),
                IntegerStorage::U32(vec![1]),
                IntegerStorage::U32(vec![u32::MAX, 6]),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
                IntegerStorage::U64(vec![1]),
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_994]),
            ),
        ];
        for (left, right, expected) in cases {
            let left = Tensor::new_integer(left, vec![2, 1]).expect("left");
            let right = Tensor::new_integer(right, vec![1, 1]).expect("right");
            let result = call(
                Value::FunctionHandle("plus".to_string()),
                Value::Tensor(left),
                Value::Tensor(right),
            )
            .expect("integer bsxfun plus");
            let Value::Tensor(result) = result else {
                panic!("expected typed tensor, got {result:?}");
            };
            assert_eq!(result.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn bsxfun_preserves_native_single_and_integer_predicate_outputs() {
        let single = Tensor::from_f32(vec![1.25, 2.5], vec![2, 1]).expect("single");
        let one = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single scalar");
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(single),
            Value::Tensor(one),
        )
        .expect("single plus");
        let Value::Tensor(result) = result else {
            panic!("expected single tensor");
        };
        assert_eq!(
            result.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![2.25, 3.5])
        );

        let wide = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_995]),
            vec![2, 1],
        )
        .expect("wide");
        let threshold =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_994]), vec![1, 1])
                .expect("threshold");
        let result = call(
            Value::FunctionHandle("gt".to_string()),
            Value::Tensor(wide),
            Value::Tensor(threshold),
        )
        .expect("exact integer predicate");
        let Value::LogicalArray(result) = result else {
            panic!("expected logical output");
        };
        assert_eq!(result.data, vec![0, 1]);
    }

    #[test]
    fn bsxfun_empty_known_callbacks_preserve_output_class() {
        let empty = Tensor::new_integer(IntegerStorage::U64(Vec::new()), vec![0, 1])
            .expect("empty integer");
        let scalar = Value::Int(IntValue::U64(9_007_199_254_740_993));
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(empty),
            scalar,
        )
        .expect("empty integer plus");
        let Value::Tensor(result) = result else {
            panic!("expected typed empty tensor");
        };
        assert_eq!(
            result.into_numeric_storage().expect("uint64 storage"),
            NumericStorage::U64(Vec::new())
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn bsxfun_integer_wgpu_fallback_gathers_exactly_and_returns_host_storage() {
        let _guard = crate::builtins::common::test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("provider");
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2, 1],
        )
        .expect("source");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
            .expect("integer upload");
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::GpuTensor(handle),
            Value::Int(IntValue::U64(1)),
        )
        .expect("gather fallback");
        let Value::Tensor(result) = result else {
            panic!("current fallback must return host tensor, got {result:?}");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_994, u64::MAX]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_plus_expands_row_and_column_vectors() {
        let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap();
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(column),
            Value::Tensor(row),
        )
        .expect("bsxfun plus");

        let Value::Tensor(tensor) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(tensor.shape, vec![3, 2]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_nd_expansion_uses_matlab_trailing_dimensions() {
        let left =
            Tensor::new((1..=24).map(|value| value as f64).collect(), vec![2, 3, 4]).unwrap();
        let right = Tensor::new(vec![100.0, 200.0, 300.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(left),
            Value::Tensor(right),
        )
        .expect("bsxfun nd trailing expansion");

        let Value::Tensor(tensor) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(tensor.shape, vec![2, 3, 4]);
        assert_eq!(tensor.materialize_f64()[0], 101.0);
        assert_eq!(tensor.materialize_f64()[1], 102.0);
        assert_eq!(tensor.materialize_f64()[2], 203.0);
        assert_eq!(tensor.materialize_f64()[6], 107.0);
        assert_eq!(tensor.materialize_f64()[8], 209.0);
        assert_eq!(tensor.materialize_f64()[23], 324.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_rejects_front_padded_nd_shapes() {
        let left =
            Tensor::new((1..=24).map(|value| value as f64).collect(), vec![2, 3, 4]).unwrap();
        let right = Tensor::new((1..=12).map(|value| value as f64).collect(), vec![3, 4]).unwrap();
        let err = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(left),
            Value::Tensor(right),
        )
        .expect_err("expected MATLAB trailing-dimension mismatch");
        assert_eq!(err.identifier(), BSXFUN_ERROR_SIZE_MISMATCH.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_gt_collects_logical_outputs() {
        let column = Tensor::new(vec![8.0, 17.0, 20.0, 24.0], vec![4, 1]).unwrap();
        let row = Tensor::new(vec![0.0, 10.0, 21.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("gt".to_string()),
            Value::Tensor(column),
            Value::Tensor(row),
        )
        .expect("bsxfun gt");

        let Value::LogicalArray(array) = result else {
            panic!("expected logical result, got {result:?}");
        };
        assert_eq!(array.shape, vec![4, 3]);
        assert_eq!(array.data, vec![1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_does_not_logicalize_shadowed_callback_names() {
        let _resolver =
            crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
                (name == "gt").then_some(17)
            })));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, _args, _requested_outputs| Box::pin(async move { Ok(Value::Num(2.0)) }),
        )));
        let result = call(
            Value::FunctionHandle("gt".to_string()),
            Value::Num(10.0),
            Value::Num(1.0),
        )
        .expect("bsxfun shadowed gt");

        let Value::Tensor(tensor) = result else {
            panic!("expected numeric result, got {result:?}");
        };
        assert_eq!(tensor.shape, vec![1, 1]);
        assert_eq!(tensor.materialize_f64(), vec![2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_handles_complex_callback_results() {
        let left = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -1.0)], vec![2, 1]).unwrap();
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::ComplexTensor(left),
            Value::Num(10.0),
        )
        .expect("bsxfun complex");

        let Value::ComplexTensor(tensor) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(tensor.materialize_f64(), vec![(11.0, 2.0), (13.0, -1.0)]);
    }

    #[test]
    fn bsxfun_callback_classifier_reads_typed_complex_integer_storage_exactly() {
        let storage = runmat_builtins::IntegerComplexStorage::new(
            runmat_builtins::IntegerStorage::I16(vec![8]),
            runmat_builtins::IntegerStorage::I16(vec![-3]),
        )
        .expect("storage");
        let complex = ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex");

        assert_eq!(
            classify_value(&Value::ComplexTensor(complex)).expect("classify"),
            ClassifiedValue::Complex((8.0, -3.0))
        );
    }

    #[test]
    fn bsxfun_rejects_typed_complex_integer_inputs_before_callback_dispatch() {
        let complex = ComplexTensor::new_integer(
            runmat_builtins::IntegerComplexStorage::new(
                runmat_builtins::IntegerStorage::U64(vec![u64::MAX]),
                runmat_builtins::IntegerStorage::U64(vec![1]),
            )
            .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");

        let err = call(
            Value::FunctionHandle("plus".to_string()),
            Value::ComplexTensor(complex),
            Value::Num(1.0),
        )
        .expect_err("typed complex integer input must reject");
        assert!(err.message().contains("complex numbers with integer types"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_invokes_bound_function_for_each_broadcasted_pair() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, args, _requested_outputs| {
                let args = args.to_vec();
                Box::pin(async move {
                    let a = match &args[0] {
                        Value::Num(value) => *value,
                        _ => 0.0,
                    };
                    let b = match &args[1] {
                        Value::Num(value) => *value,
                        _ => 0.0,
                    };
                    Ok(Value::Num(a + 2.0 * b))
                })
            },
        )));
        let column = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
        let result = call(
            Value::BoundFunctionHandle {
                name: "scaled_add".to_string(),
                function: 7,
            },
            Value::Tensor(column),
            Value::Tensor(row),
        )
        .expect("bsxfun closure");

        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![21.0, 22.0, 41.0, 42.0, 61.0, 62.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_singleton_dimension_can_expand_to_zero() {
        let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(empty),
            Value::Tensor(row),
        )
        .expect("bsxfun empty");

        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![0, 3]);
        assert!(tensor.materialize_f64().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_empty_logical_callback_preserves_logical_type() {
        let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = call(
            Value::FunctionHandle("gt".to_string()),
            Value::Tensor(empty),
            Value::Tensor(row),
        )
        .expect("bsxfun empty logical");

        let Value::LogicalArray(array) = result else {
            panic!("expected logical result, got {result:?}");
        };
        assert_eq!(array.shape, vec![0, 3]);
        assert!(array.data.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_rejects_incompatible_sizes() {
        let left = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = call(
            Value::FunctionHandle("plus".to_string()),
            Value::Tensor(left),
            Value::Tensor(right),
        )
        .expect_err("expected size mismatch");
        assert_eq!(err.identifier(), BSXFUN_ERROR_SIZE_MISMATCH.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bsxfun_rejects_non_scalar_callback_output() {
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |_function, _args, _requested_outputs| {
                Box::pin(async move {
                    Ok(Value::Tensor(
                        Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
                    ))
                })
            },
        )));
        let err = call(
            Value::BoundFunctionHandle {
                name: "bad".to_string(),
                function: 9,
            },
            Value::Num(1.0),
            Value::Num(2.0),
        )
        .expect_err("expected callback output error");
        assert_eq!(err.identifier(), BSXFUN_ERROR_FUNCTION_ERROR.identifier);
    }
}
