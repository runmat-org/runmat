//! MATLAB-compatible integer bitwise function builtins.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BITAND_NAME: &str = "bitand";
const BITOR_NAME: &str = "bitor";
const BITSHIFT_NAME: &str = "bitshift";
const IDIVIDE_NAME: &str = "idivide";
const SWAPBYTES_NAME: &str = "swapbytes";

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bitwise numeric result.",
}];

const BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right integer-valued input.",
    },
];

const BITSHIFT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer-valued input.",
    },
    BuiltinParamDescriptor {
        name: "K",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Shift count; positive shifts left and negative shifts right.",
    },
];

const IDIVIDE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer dividend.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer divisor.",
    },
];

const IDIVIDE_INPUTS_ROUNDING: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer dividend.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer divisor.",
    },
    BuiltinParamDescriptor {
        name: "rounding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"fix\""),
        description: "Rounding mode: \"fix\", \"floor\", \"ceil\", or \"round\".",
    },
];

const SWAPBYTES_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric scalar or array whose element byte order is reversed.",
}];

const BITAND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = bitand(A, B)",
    inputs: &BINARY_INPUTS,
    outputs: &OUTPUT,
}];

const BITOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = bitor(A, B)",
    inputs: &BINARY_INPUTS,
    outputs: &OUTPUT,
}];

const BITSHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = bitshift(A, K)",
    inputs: &BITSHIFT_INPUTS,
    outputs: &OUTPUT,
}];

const IDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = idivide(A, B)",
        inputs: &IDIVIDE_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = idivide(A, B, rounding)",
        inputs: &IDIVIDE_INPUTS_ROUNDING,
        outputs: &OUTPUT,
    },
];

const SWAPBYTES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = swapbytes(X)",
    inputs: &SWAPBYTES_INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BITWISE.INVALID_INPUT",
    identifier: Some("RunMat:bitwise:InvalidInput"),
    when: "Inputs are not finite integer-valued numeric, logical, or gatherable gpuArray values.",
    message: "bitwise operation: invalid input",
};

const ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BITWISE.SIZE_MISMATCH",
    identifier: Some("RunMat:bitwise:SizeMismatch"),
    when: "Input shapes are not compatible for implicit expansion.",
    message: "bitwise operation: array sizes are not compatible",
};

const ERROR_DIVIDE_BY_ZERO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IDIVIDE.DIVIDE_BY_ZERO",
    identifier: Some("RunMat:idivide:DivideByZero"),
    when: "The divisor contains zero.",
    message: "idivide: division by zero",
};

const ERROR_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IDIVIDE.OVERFLOW",
    identifier: Some("RunMat:idivide:Overflow"),
    when: "A rounded quotient cannot be represented in the dividend integer class.",
    message: "idivide: quotient overflows output class",
};

const ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERROR_INVALID_INPUT,
    ERROR_SIZE_MISMATCH,
    ERROR_DIVIDE_BY_ZERO,
    ERROR_OVERFLOW,
];

pub const BITAND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITAND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const BITOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const BITSHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BITSHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const IDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const SWAPBYTES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SWAPBYTES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "bitand",
    category = "logical/bit",
    summary = "Compute bitwise AND for integer-valued scalars and arrays.",
    keywords = "bitand,bitwise,and,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITAND_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitand_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary_bitwise(BITAND_NAME, lhs, rhs, |a, b| a & b).await
}

#[runtime_builtin(
    name = "bitor",
    category = "logical/bit",
    summary = "Compute bitwise OR for integer-valued scalars and arrays.",
    keywords = "bitor,bitwise,or,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITOR_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitor_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    binary_bitwise(BITOR_NAME, lhs, rhs, |a, b| a | b).await
}

#[runtime_builtin(
    name = "bitshift",
    category = "logical/bit",
    summary = "Shift integer-valued scalars and arrays left or right by bit counts.",
    keywords = "bitshift,bitwise,shift,integer,uint32",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::BITSHIFT_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn bitshift_builtin(value: Value, shift: Value) -> BuiltinResult<Value> {
    let left = bit_buffer_from(BITSHIFT_NAME, value).await?;
    let shifts = shift_buffer_from(shift).await?;
    let plan = BroadcastPlan::new(&left.shape, &shifts.shape)
        .map_err(|err| error_with_detail(BITSHIFT_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, idx_a, idx_b) in plan.iter() {
        data.push(apply_shift(left.data[idx_a], shifts.data[idx_b]));
    }
    value_from_bits(data, plan.output_shape().to_vec(), left.kind, BITSHIFT_NAME)
}

#[runtime_builtin(
    name = "idivide",
    category = "math/elementwise",
    summary = "Integer division with MATLAB-compatible rounding modes.",
    keywords = "idivide,integer,division,rounding",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::IDIVIDE_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn idivide_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !(2..=3).contains(&args.len()) {
        return Err(error_with_detail(
            IDIVIDE_NAME,
            &ERROR_INVALID_INPUT,
            "expected two integer inputs and an optional rounding mode",
        ));
    }
    let mut iter = args.into_iter();
    let left = idivide_buffer_from(iter.next().expect("A")).await?;
    let right = idivide_buffer_from(iter.next().expect("B")).await?;
    let output_class = idivide_output_class(&left, &right)?;
    let rounding = match iter.next() {
        Some(value) => RoundingMode::parse(&value)?,
        None => RoundingMode::Fix,
    };
    let plan = BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|err| error_with_detail(IDIVIDE_NAME, &ERROR_SIZE_MISMATCH, err))?;
    let mut out = Vec::with_capacity(plan.len());
    for (_, idx_a, idx_b) in plan.iter() {
        let divisor = right.data[idx_b];
        if divisor == 0 {
            return Err(error_with_detail(
                IDIVIDE_NAME,
                &ERROR_DIVIDE_BY_ZERO,
                "divisor contains zero",
            ));
        }
        let quotient = rounded_integer_divide(left.data[idx_a], divisor, rounding);
        out.push(quotient);
    }
    value_from_integer_data(
        out,
        plan.output_shape().to_vec(),
        output_class,
        IDIVIDE_NAME,
    )
}

#[runtime_builtin(
    name = "swapbytes",
    category = "math/elementwise",
    summary = "Reverse byte order of numeric values.",
    keywords = "swapbytes,byte order,endian,numeric",
    accel = "gather",
    descriptor(crate::builtins::logical::bit::integer::SWAPBYTES_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::bit::integer"
)]
async fn swapbytes_builtin(value: Value) -> BuiltinResult<Value> {
    let gathered = gpu_helpers::gather_value_async(&value)
        .await
        .map_err(|err| error_with_detail(SWAPBYTES_NAME, &ERROR_INVALID_INPUT, err.message()))?;
    match gathered {
        Value::Num(value) => Ok(Value::Num(f64::from_bits(value.to_bits().swap_bytes()))),
        Value::Int(value) => Ok(Value::Int(swap_int_value(value))),
        Value::Tensor(tensor) => swap_tensor_bytes(tensor),
        other => Err(error_with_detail(
            SWAPBYTES_NAME,
            &ERROR_INVALID_INPUT,
            format!("unsupported input {other:?}"),
        )),
    }
}

async fn binary_bitwise(
    name: &'static str,
    lhs: Value,
    rhs: Value,
    op: impl Fn(u32, u32) -> u32,
) -> BuiltinResult<Value> {
    let left = bit_buffer_from(name, lhs).await?;
    let right = bit_buffer_from(name, rhs).await?;
    let plan = BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|err| error_with_detail(name, &ERROR_SIZE_MISMATCH, err))?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, idx_a, idx_b) in plan.iter() {
        data.push(op(left.data[idx_a], right.data[idx_b]));
    }
    value_from_bits(
        data,
        plan.output_shape().to_vec(),
        combine_binary_output_kind(left.kind, right.kind),
        name,
    )
}

fn apply_shift(value: u32, shift: i32) -> u32 {
    let amount = shift.unsigned_abs().min(32);
    if shift >= 0 {
        value.checked_shl(amount).unwrap_or(0)
    } else {
        value.checked_shr(amount).unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputKind {
    Double,
    UInt8,
    UInt16,
    UInt32,
}

struct BitBuffer {
    data: Vec<u32>,
    shape: Vec<usize>,
    kind: OutputKind,
}

struct ShiftBuffer {
    data: Vec<i32>,
    shape: Vec<usize>,
}

async fn bit_buffer_from(name: &'static str, value: Value) -> BuiltinResult<BitBuffer> {
    match value {
        Value::Num(value) => Ok(BitBuffer {
            data: vec![double_to_u32(name, value)?],
            shape: vec![1, 1],
            kind: OutputKind::Double,
        }),
        Value::Bool(value) => Ok(BitBuffer {
            data: vec![if value { 1 } else { 0 }],
            shape: vec![1, 1],
            kind: OutputKind::Double,
        }),
        Value::Int(value) => Ok(BitBuffer {
            data: vec![int_to_u32(&value)],
            shape: vec![1, 1],
            kind: int_output_kind(&value),
        }),
        Value::Tensor(tensor) => tensor_to_bit_buffer(name, tensor),
        Value::LogicalArray(array) => Ok(BitBuffer {
            data: array.data.into_iter().map(|v| u32::from(v != 0)).collect(),
            shape: array.shape,
            kind: OutputKind::Double,
        }),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))?;
            tensor_to_bit_buffer(name, tensor)
        }
        other => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("{name}: unsupported input {other:?}"),
        )),
    }
}

async fn shift_buffer_from(value: Value) -> BuiltinResult<ShiftBuffer> {
    match value {
        Value::Num(value) => Ok(ShiftBuffer {
            data: vec![double_to_i32_shift(value)?],
            shape: vec![1, 1],
        }),
        Value::Int(value) => Ok(ShiftBuffer {
            data: vec![value.to_i64().clamp(i32::MIN as i64, i32::MAX as i64) as i32],
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor_to_shift_buffer(tensor),
        Value::LogicalArray(array) => Ok(ShiftBuffer {
            data: array.data.into_iter().map(|v| i32::from(v != 0)).collect(),
            shape: array.shape,
        }),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(BITSHIFT_NAME, &ERROR_INVALID_INPUT, err))?;
            tensor_to_shift_buffer(tensor)
        }
        other => Err(error_with_detail(
            BITSHIFT_NAME,
            &ERROR_INVALID_INPUT,
            format!("bitshift: unsupported shift input {other:?}"),
        )),
    }
}

fn tensor_to_shift_buffer(tensor: Tensor) -> BuiltinResult<ShiftBuffer> {
    let data = tensor
        .data
        .into_iter()
        .map(double_to_i32_shift)
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(ShiftBuffer {
        data,
        shape: tensor.shape,
    })
}

fn tensor_to_bit_buffer(name: &'static str, tensor: Tensor) -> BuiltinResult<BitBuffer> {
    let kind = match tensor.dtype {
        NumericDType::U8 => OutputKind::UInt8,
        NumericDType::U16 => OutputKind::UInt16,
        NumericDType::U32 => OutputKind::UInt32,
        NumericDType::F32 | NumericDType::F64 => OutputKind::Double,
    };
    let data = tensor
        .data
        .into_iter()
        .map(|value| double_to_u32(name, value))
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(BitBuffer {
        data,
        shape: tensor.shape,
        kind,
    })
}

fn value_from_bits(
    data: Vec<u32>,
    shape: Vec<usize>,
    kind: OutputKind,
    name: &'static str,
) -> BuiltinResult<Value> {
    let data = data
        .into_iter()
        .map(|value| normalize_output_bits(value, kind))
        .collect::<Vec<_>>();
    if data.len() == 1 && tensor::element_count(&shape) == 1 {
        return Ok(match kind {
            OutputKind::UInt8 => Value::Int(IntValue::U8(data[0] as u8)),
            OutputKind::UInt16 => Value::Int(IntValue::U16(data[0] as u16)),
            OutputKind::UInt32 => Value::Int(IntValue::U32(data[0])),
            OutputKind::Double => Value::Num(data[0] as f64),
        });
    }

    let dtype = match kind {
        OutputKind::UInt8 => NumericDType::U8,
        OutputKind::UInt16 => NumericDType::U16,
        OutputKind::UInt32 => NumericDType::U32,
        OutputKind::Double => NumericDType::F64,
    };
    Tensor::new_with_dtype(
        data.into_iter().map(|value| value as f64).collect(),
        shape,
        dtype,
    )
    .map(Value::Tensor)
    .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))
}

fn normalize_output_bits(value: u32, kind: OutputKind) -> u32 {
    match kind {
        OutputKind::UInt8 => u32::from(value as u8),
        OutputKind::UInt16 => u32::from(value as u16),
        OutputKind::UInt32 | OutputKind::Double => value,
    }
}

fn combine_binary_output_kind(left: OutputKind, right: OutputKind) -> OutputKind {
    match (left, right) {
        (OutputKind::Double, _) | (_, OutputKind::Double) => OutputKind::Double,
        (OutputKind::UInt32, _) | (_, OutputKind::UInt32) => OutputKind::UInt32,
        (OutputKind::UInt16, _) | (_, OutputKind::UInt16) => OutputKind::UInt16,
        (OutputKind::UInt8, OutputKind::UInt8) => OutputKind::UInt8,
    }
}

fn double_to_u32(name: &'static str, value: f64) -> BuiltinResult<u32> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=u32::MAX as f64).contains(&value) {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("{name}: input values must be finite nonnegative integers no larger than uint32 max"),
        ));
    }
    Ok(value as u32)
}

fn double_to_i32_shift(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(error_with_detail(
            BITSHIFT_NAME,
            &ERROR_INVALID_INPUT,
            "bitshift: shift counts must be finite integers",
        ));
    }
    Ok(value.clamp(i32::MIN as f64, i32::MAX as f64) as i32)
}

fn int_to_u32(value: &IntValue) -> u32 {
    match value {
        IntValue::I8(value) => *value as i32 as u32,
        IntValue::I16(value) => *value as i32 as u32,
        IntValue::I32(value) => *value as u32,
        IntValue::I64(value) => *value as u32,
        IntValue::U8(value) => *value as u32,
        IntValue::U16(value) => *value as u32,
        IntValue::U32(value) => *value,
        IntValue::U64(value) => *value as u32,
    }
}

fn int_output_kind(value: &IntValue) -> OutputKind {
    match value {
        IntValue::U8(_) => OutputKind::UInt8,
        IntValue::U16(_) => OutputKind::UInt16,
        IntValue::U32(_) | IntValue::U64(_) => OutputKind::UInt32,
        IntValue::I8(_) | IntValue::I16(_) | IntValue::I32(_) | IntValue::I64(_) => {
            OutputKind::Double
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerClass {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntegerClass {
    fn from_int(value: &IntValue) -> Self {
        match value {
            IntValue::I8(_) => Self::I8,
            IntValue::I16(_) => Self::I16,
            IntValue::I32(_) => Self::I32,
            IntValue::I64(_) => Self::I64,
            IntValue::U8(_) => Self::U8,
            IntValue::U16(_) => Self::U16,
            IntValue::U32(_) => Self::U32,
            IntValue::U64(_) => Self::U64,
        }
    }

    fn from_dtype(dtype: NumericDType) -> BuiltinResult<Self> {
        match dtype {
            NumericDType::U8 => Ok(Self::U8),
            NumericDType::U16 => Ok(Self::U16),
            NumericDType::U32 => Ok(Self::U32),
            NumericDType::F32 | NumericDType::F64 => Err(error_with_detail(
                IDIVIDE_NAME,
                &ERROR_INVALID_INPUT,
                "dense inputs must use an integer class",
            )),
        }
    }

    fn range(self) -> (i128, i128) {
        match self {
            Self::I8 => (i8::MIN as i128, i8::MAX as i128),
            Self::I16 => (i16::MIN as i128, i16::MAX as i128),
            Self::I32 => (i32::MIN as i128, i32::MAX as i128),
            Self::I64 => (i64::MIN as i128, i64::MAX as i128),
            Self::U8 => (0, u8::MAX as i128),
            Self::U16 => (0, u16::MAX as i128),
            Self::U32 => (0, u32::MAX as i128),
            Self::U64 => (0, u64::MAX as i128),
        }
    }

    fn dtype(self) -> Option<NumericDType> {
        match self {
            Self::U8 => Some(NumericDType::U8),
            Self::U16 => Some(NumericDType::U16),
            Self::U32 => Some(NumericDType::U32),
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U64 => None,
        }
    }

    fn scalar_value(self, value: i128, name: &'static str) -> BuiltinResult<Value> {
        self.validate(value, name)?;
        Ok(Value::Int(match self {
            Self::I8 => IntValue::I8(value as i8),
            Self::I16 => IntValue::I16(value as i16),
            Self::I32 => IntValue::I32(value as i32),
            Self::I64 => IntValue::I64(value as i64),
            Self::U8 => IntValue::U8(value as u8),
            Self::U16 => IntValue::U16(value as u16),
            Self::U32 => IntValue::U32(value as u32),
            Self::U64 => IntValue::U64(value as u64),
        }))
    }

    fn validate(self, value: i128, name: &'static str) -> BuiltinResult<()> {
        let (min, max) = self.range();
        if (min..=max).contains(&value) {
            Ok(())
        } else {
            Err(error_with_detail(
                name,
                &ERROR_OVERFLOW,
                format!("value {value} is outside output class range"),
            ))
        }
    }
}

struct IntegerBuffer {
    data: Vec<i128>,
    shape: Vec<usize>,
    class: IntegerClass,
}

struct IdivideBuffer {
    data: Vec<i128>,
    shape: Vec<usize>,
    class: Option<IntegerClass>,
}

async fn integer_buffer_from(name: &'static str, value: Value) -> BuiltinResult<IntegerBuffer> {
    match value {
        Value::Int(value) => Ok(IntegerBuffer {
            data: vec![int_value_to_i128(&value)],
            shape: vec![1, 1],
            class: IntegerClass::from_int(&value),
        }),
        Value::Tensor(tensor) => tensor_to_integer_buffer(name, tensor),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))?;
            tensor_to_integer_buffer(name, tensor)
        }
        other => Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            format!("unsupported integer input {other:?}"),
        )),
    }
}

fn tensor_to_integer_buffer(name: &'static str, tensor: Tensor) -> BuiltinResult<IntegerBuffer> {
    let class = IntegerClass::from_dtype(tensor.dtype)?;
    let (min, max) = class.range();
    let data = tensor
        .data
        .into_iter()
        .map(|value| {
            if !value.is_finite() || value.fract() != 0.0 {
                return Err(error_with_detail(
                    name,
                    &ERROR_INVALID_INPUT,
                    "integer tensor values must be finite integers",
                ));
            }
            let integer = value as i128;
            if !(min..=max).contains(&integer) {
                return Err(error_with_detail(
                    name,
                    &ERROR_INVALID_INPUT,
                    "integer tensor value is outside its dtype range",
                ));
            }
            Ok(integer)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(IntegerBuffer {
        data,
        shape: tensor.shape,
        class,
    })
}

async fn idivide_buffer_from(value: Value) -> BuiltinResult<IdivideBuffer> {
    match value {
        Value::Num(value) => {
            let integer = scalar_double_integer(value)?;
            Ok(IdivideBuffer {
                data: vec![integer],
                shape: vec![1, 1],
                class: None,
            })
        }
        other => {
            let buffer = integer_buffer_from(IDIVIDE_NAME, other).await?;
            Ok(IdivideBuffer {
                data: buffer.data,
                shape: buffer.shape,
                class: Some(buffer.class),
            })
        }
    }
}

fn idivide_output_class(
    left: &IdivideBuffer,
    right: &IdivideBuffer,
) -> BuiltinResult<IntegerClass> {
    match (left.class, right.class) {
        (Some(lhs), Some(rhs)) if lhs == rhs => Ok(lhs),
        (Some(class), None) | (None, Some(class)) => {
            if matches!(class, IntegerClass::I64 | IntegerClass::U64) {
                return Err(error_with_detail(
                    IDIVIDE_NAME,
                    &ERROR_INVALID_INPUT,
                    "scalar double operands are not supported with int64 or uint64",
                ));
            }
            Ok(class)
        }
        (Some(_), Some(_)) => Err(error_with_detail(
            IDIVIDE_NAME,
            &ERROR_INVALID_INPUT,
            "integer inputs must have matching classes unless one input is a scalar double",
        )),
        (None, None) => Err(error_with_detail(
            IDIVIDE_NAME,
            &ERROR_INVALID_INPUT,
            "at least one input must be an integer class",
        )),
    }
}

fn scalar_double_integer(value: f64) -> BuiltinResult<i128> {
    if value.is_finite() && value.fract() == 0.0 {
        Ok(value as i128)
    } else {
        Err(error_with_detail(
            IDIVIDE_NAME,
            &ERROR_INVALID_INPUT,
            "scalar double operands must be finite integer-valued values",
        ))
    }
}

fn value_from_integer_data(
    data: Vec<i128>,
    shape: Vec<usize>,
    class: IntegerClass,
    name: &'static str,
) -> BuiltinResult<Value> {
    for &value in &data {
        class.validate(value, name)?;
    }
    if data.len() == 1 && tensor::element_count(&shape) == 1 {
        return class.scalar_value(data[0], name);
    }
    let dtype = class.dtype().ok_or_else(|| {
        error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "dense signed integer output is not representable by the current tensor value model",
        )
    })?;
    Tensor::new_with_dtype(
        data.into_iter().map(|value| value as f64).collect(),
        shape,
        dtype,
    )
    .map(Value::Tensor)
    .map_err(|err| error_with_detail(name, &ERROR_INVALID_INPUT, err))
}

fn int_value_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => *value as i128,
        IntValue::I16(value) => *value as i128,
        IntValue::I32(value) => *value as i128,
        IntValue::I64(value) => *value as i128,
        IntValue::U8(value) => *value as i128,
        IntValue::U16(value) => *value as i128,
        IntValue::U32(value) => *value as i128,
        IntValue::U64(value) => *value as i128,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundingMode {
    Fix,
    Floor,
    Ceil,
    Round,
}

impl RoundingMode {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let Some(keyword) = keyword_of(value) else {
            return Err(error_with_detail(
                IDIVIDE_NAME,
                &ERROR_INVALID_INPUT,
                "rounding mode must be text",
            ));
        };
        match keyword.as_str() {
            "fix" => Ok(Self::Fix),
            "floor" => Ok(Self::Floor),
            "ceil" => Ok(Self::Ceil),
            "round" => Ok(Self::Round),
            _ => Err(error_with_detail(
                IDIVIDE_NAME,
                &ERROR_INVALID_INPUT,
                format!("unsupported rounding mode '{keyword}'"),
            )),
        }
    }
}

fn rounded_integer_divide(dividend: i128, divisor: i128, mode: RoundingMode) -> i128 {
    let quotient = dividend / divisor;
    let remainder = dividend % divisor;
    if remainder == 0 {
        return quotient;
    }
    match mode {
        RoundingMode::Fix => quotient,
        RoundingMode::Floor => {
            if (dividend < 0) != (divisor < 0) {
                quotient - 1
            } else {
                quotient
            }
        }
        RoundingMode::Ceil => {
            if (dividend < 0) == (divisor < 0) {
                quotient + 1
            } else {
                quotient
            }
        }
        RoundingMode::Round => round_quotient_away_from_zero(dividend, divisor),
    }
}

fn round_quotient_away_from_zero(dividend: i128, divisor: i128) -> i128 {
    let sign = if (dividend < 0) == (divisor < 0) {
        1
    } else {
        -1
    };
    let numerator = dividend.unsigned_abs();
    let denominator = divisor.unsigned_abs();
    let mut quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder.saturating_mul(2) >= denominator {
        quotient += 1;
    }
    (quotient as i128) * sign
}

fn swap_int_value(value: IntValue) -> IntValue {
    match value {
        IntValue::I8(value) => IntValue::I8(value),
        IntValue::I16(value) => IntValue::I16(value.swap_bytes()),
        IntValue::I32(value) => IntValue::I32(value.swap_bytes()),
        IntValue::I64(value) => IntValue::I64(value.swap_bytes()),
        IntValue::U8(value) => IntValue::U8(value),
        IntValue::U16(value) => IntValue::U16(value.swap_bytes()),
        IntValue::U32(value) => IntValue::U32(value.swap_bytes()),
        IntValue::U64(value) => IntValue::U64(value.swap_bytes()),
    }
}

fn swap_tensor_bytes(tensor: Tensor) -> BuiltinResult<Value> {
    let dtype = tensor.dtype;
    let data = tensor
        .data
        .into_iter()
        .map(|value| swap_tensor_scalar(value, dtype))
        .collect::<BuiltinResult<Vec<_>>>()?;
    Tensor::new_with_dtype(data, tensor.shape, dtype)
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(SWAPBYTES_NAME, &ERROR_INVALID_INPUT, err))
}

fn swap_tensor_scalar(value: f64, dtype: NumericDType) -> BuiltinResult<f64> {
    Ok(match dtype {
        NumericDType::F64 => f64::from_bits(value.to_bits().swap_bytes()),
        NumericDType::F32 => f32::from_bits((value as f32).to_bits().swap_bytes()) as f64,
        NumericDType::U8 => {
            validate_unsigned_scalar(value, u8::MAX as f64)?;
            value
        }
        NumericDType::U16 => {
            validate_unsigned_scalar(value, u16::MAX as f64)?;
            f64::from((value as u16).swap_bytes())
        }
        NumericDType::U32 => {
            validate_unsigned_scalar(value, u32::MAX as f64)?;
            (value as u32).swap_bytes() as f64
        }
    })
}

fn validate_unsigned_scalar(value: f64, max: f64) -> BuiltinResult<()> {
    if value.is_finite() && value.fract() == 0.0 && (0.0..=max).contains(&value) {
        Ok(())
    } else {
        Err(error_with_detail(
            SWAPBYTES_NAME,
            &ERROR_INVALID_INPUT,
            "integer tensor values must be finite and within dtype range",
        ))
    }
}

fn error_with_detail(
    name: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let message = format!("{}: {}", error.message, detail);
    let mut builder = build_runtime_error(message).with_builtin(name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn bitand_double_scalars_return_double() {
        let out = block_on(bitand_builtin(Value::Num(6.0), Value::Num(3.0))).expect("bitand");
        assert_eq!(out, Value::Num(2.0));
    }

    #[test]
    fn bitwise_uint32_scalars_preserve_uint32() {
        let out = block_on(bitor_builtin(
            Value::Int(IntValue::U32(0b0101)),
            Value::Int(IntValue::U32(0b0011)),
        ))
        .expect("bitor");
        assert_eq!(out, Value::Int(IntValue::U32(0b0111)));
    }

    #[test]
    fn bitand_broadcasts_tensor_and_scalar() {
        let tensor =
            Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::U32)
                .unwrap();
        let out = block_on(bitand_builtin(
            Value::Tensor(tensor),
            Value::Int(IntValue::U32(1)),
        ))
        .expect("bitand");
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.dtype, NumericDType::U32);
                assert_eq!(t.data, vec![1.0, 0.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn binary_bitwise_mixed_unsigned_width_is_commutative() {
        let forward = block_on(bitor_builtin(
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U32(256)),
        ))
        .expect("bitor");
        let reverse = block_on(bitor_builtin(
            Value::Int(IntValue::U32(256)),
            Value::Int(IntValue::U8(1)),
        ))
        .expect("bitor");

        assert_eq!(forward, Value::Int(IntValue::U32(257)));
        assert_eq!(reverse, Value::Int(IntValue::U32(257)));
    }

    #[test]
    fn bitshift_supports_positive_and_negative_counts() {
        assert_eq!(
            block_on(bitshift_builtin(
                Value::Int(IntValue::U32(3)),
                Value::Num(2.0)
            ))
            .expect("left shift"),
            Value::Int(IntValue::U32(12))
        );
        assert_eq!(
            block_on(bitshift_builtin(
                Value::Int(IntValue::U32(8)),
                Value::Num(-1.0)
            ))
            .expect("right shift"),
            Value::Int(IntValue::U32(4))
        );
    }

    #[test]
    fn bitshift_preserves_integer_width() {
        assert_eq!(
            block_on(bitshift_builtin(
                Value::Int(IntValue::U8(255)),
                Value::Num(1.0)
            ))
            .expect("left shift"),
            Value::Int(IntValue::U8(254))
        );

        let tensor =
            Tensor::new_with_dtype(vec![255.0, 128.0], vec![1, 2], NumericDType::U8).unwrap();
        let out = block_on(bitshift_builtin(Value::Tensor(tensor), Value::Num(1.0)))
            .expect("tensor shift");
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.dtype, NumericDType::U8);
                assert_eq!(t.data, vec![254.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn bitwise_rejects_fractional_double() {
        let err = block_on(bitand_builtin(Value::Num(1.5), Value::Num(1.0)))
            .expect_err("fractional inputs should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn idivide_preserves_scalar_integer_class_and_rounding_modes() {
        assert_eq!(
            block_on(idivide_builtin(vec![
                Value::Int(IntValue::I16(-7)),
                Value::Int(IntValue::I16(3)),
            ]))
            .expect("idivide fix"),
            Value::Int(IntValue::I16(-2))
        );
        assert_eq!(
            block_on(idivide_builtin(vec![
                Value::Int(IntValue::I16(-7)),
                Value::Int(IntValue::I16(3)),
                Value::String("floor".to_string()),
            ]))
            .expect("idivide floor"),
            Value::Int(IntValue::I16(-3))
        );
        assert_eq!(
            block_on(idivide_builtin(vec![
                Value::Int(IntValue::I16(-7)),
                Value::Int(IntValue::I16(3)),
                Value::String("ceil".to_string()),
            ]))
            .expect("idivide ceil"),
            Value::Int(IntValue::I16(-2))
        );
        assert_eq!(
            block_on(idivide_builtin(vec![
                Value::Int(IntValue::I16(5)),
                Value::Int(IntValue::I16(2)),
                Value::String("round".to_string()),
            ]))
            .expect("idivide round"),
            Value::Int(IntValue::I16(3))
        );
    }

    #[test]
    fn idivide_broadcasts_uint_tensor_and_preserves_dtype() {
        let lhs =
            Tensor::new_with_dtype(vec![9.0, 10.0, 11.0], vec![1, 3], NumericDType::U16).unwrap();
        let out = block_on(idivide_builtin(vec![
            Value::Tensor(lhs),
            Value::Int(IntValue::U16(3)),
        ]))
        .expect("idivide tensor");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.dtype, NumericDType::U16);
        assert_eq!(tensor.data, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn idivide_allows_scalar_double_with_non64_integer_class() {
        assert_eq!(
            block_on(idivide_builtin(vec![
                Value::Num(10.0),
                Value::Int(IntValue::U16(3)),
            ]))
            .expect("double dividend"),
            Value::Int(IntValue::U16(3))
        );
        assert_eq!(
            block_on(idivide_builtin(vec![
                Value::Int(IntValue::I32(-7)),
                Value::Num(3.0),
                Value::String("floor".to_string()),
            ]))
            .expect("double divisor"),
            Value::Int(IntValue::I32(-3))
        );
    }

    #[test]
    fn idivide_rejects_zero_mixed_class_and_invalid_double_inputs() {
        let zero = block_on(idivide_builtin(vec![
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U8(0)),
        ]))
        .expect_err("zero divisor should fail");
        assert_eq!(zero.identifier(), ERROR_DIVIDE_BY_ZERO.identifier);

        let mixed = block_on(idivide_builtin(vec![
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U16(1)),
        ]))
        .expect_err("mixed class should fail");
        assert_eq!(mixed.identifier(), ERROR_INVALID_INPUT.identifier);

        let two_doubles = block_on(idivide_builtin(vec![Value::Num(4.0), Value::Num(2.0)]))
            .expect_err("two doubles should fail");
        assert_eq!(two_doubles.identifier(), ERROR_INVALID_INPUT.identifier);

        let int64_double = block_on(idivide_builtin(vec![
            Value::Int(IntValue::I64(4)),
            Value::Num(2.0),
        ]))
        .expect_err("int64 plus double should fail");
        assert_eq!(int64_double.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn swapbytes_preserves_integer_scalar_classes() {
        assert_eq!(
            block_on(swapbytes_builtin(Value::Int(IntValue::U16(0x1234)))).expect("swapbytes"),
            Value::Int(IntValue::U16(0x3412))
        );
        assert_eq!(
            block_on(swapbytes_builtin(Value::Int(IntValue::I32(0x01020304)))).expect("swapbytes"),
            Value::Int(IntValue::I32(0x04030201))
        );
    }

    #[test]
    fn swapbytes_preserves_tensor_dtype() {
        let tensor = Tensor::new_with_dtype(
            vec![0x1234_u16 as f64, 0x00ff_u16 as f64],
            vec![1, 2],
            NumericDType::U16,
        )
        .unwrap();
        let out = block_on(swapbytes_builtin(Value::Tensor(tensor))).expect("swapbytes");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.dtype, NumericDType::U16);
        assert_eq!(tensor.data, vec![0x3412_u16 as f64, 0xff00_u16 as f64]);
    }

    #[test]
    fn swapbytes_reinterprets_floating_point_bytes() {
        let value = 1.25_f64;
        let out = block_on(swapbytes_builtin(Value::Num(value))).expect("swapbytes");
        assert_eq!(
            out,
            Value::Num(f64::from_bits(value.to_bits().swap_bytes()))
        );
    }
}
