//! MATLAB-compatible integer bitwise function builtins.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BITAND_NAME: &str = "bitand";
const BITOR_NAME: &str = "bitor";
const BITSHIFT_NAME: &str = "bitshift";

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

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_SIZE_MISMATCH];

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
}
