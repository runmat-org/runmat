//! MATLAB-compatible `lcm` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, IntegerStorage, NumericDType, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "lcm";

const LCM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Least common multiples of A and B.",
}];

const LCM_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real positive integer scalar, vector, or array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real positive integer scalar, vector, or array.",
    },
];

const LCM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "L = lcm(A, B)",
    inputs: &LCM_INPUTS,
    outputs: &LCM_OUTPUT,
}];

const LCM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LCM.INVALID_INPUT",
    identifier: Some("RunMat:lcm:InvalidInput"),
    when: "Inputs are not real positive integer numeric values, or integer class mixing is unsupported.",
    message: "lcm: invalid input",
};

const LCM_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LCM.SIZE_MISMATCH",
    identifier: Some("RunMat:lcm:SizeMismatch"),
    when: "Inputs are neither the same size nor scalar-expandable.",
    message: "lcm: input sizes are not compatible",
};

const LCM_ERROR_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LCM.OVERFLOW",
    identifier: Some("RunMat:lcm:Overflow"),
    when: "The least common multiple cannot be represented in the output numeric class.",
    message: "lcm: result overflows output type",
};

const LCM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LCM.INTERNAL",
    identifier: Some("RunMat:lcm:Internal"),
    when: "GPU gather or tensor construction fails.",
    message: "lcm: internal error",
};

const LCM_ERRORS: [BuiltinErrorDescriptor; 4] = [
    LCM_ERROR_INVALID_INPUT,
    LCM_ERROR_SIZE_MISMATCH,
    LCM_ERROR_OVERFLOW,
    LCM_ERROR_INTERNAL,
];

pub const LCM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LCM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LCM_ERRORS,
};

fn lcm_type(args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    if args.iter().all(|ty| matches!(ty, Type::Int)) {
        Type::Int
    } else if args.iter().all(|ty| matches!(ty, Type::Num | Type::Int)) {
        Type::Num
    } else {
        Type::tensor()
    }
}

#[runtime_builtin(
    name = "lcm",
    category = "math/discrete",
    summary = "Compute least common multiples for positive integer inputs.",
    keywords = "lcm,least common multiple,integer,number theory,discrete",
    accel = "gather",
    type_resolver(lcm_type),
    descriptor(crate::builtins::math::discrete::lcm::LCM_DESCRIPTOR),
    builtin_path = "crate::builtins::math::discrete::lcm"
)]
async fn lcm_builtin(left: Value, right: Value) -> BuiltinResult<Value> {
    let left = LcmInput::from_value(left).await?;
    let right = LcmInput::from_value(right).await?;
    let output_kind = resolve_output_kind(&left, &right)?;
    let plan = SameSizeOrScalarPlan::new(&left, &right)?;
    let mut out = Vec::with_capacity(plan.len());
    for (left_idx, right_idx) in plan.iter() {
        out.push(lcm_u128(left.data[left_idx], right.data[right_idx]));
    }
    value_from_lcms(out, plan.output_shape, output_kind)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericClass {
    Double,
    Single,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl NumericClass {
    fn is_float(self) -> bool {
        matches!(self, Self::Double | Self::Single)
    }

    fn is_integer(self) -> bool {
        !self.is_float()
    }

    fn max_value(self) -> u128 {
        match self {
            Self::Double | Self::Single => u128::MAX,
            Self::I8 => i8::MAX as u128,
            Self::I16 => i16::MAX as u128,
            Self::I32 => i32::MAX as u128,
            Self::I64 => i64::MAX as u128,
            Self::U8 => u8::MAX as u128,
            Self::U16 => u16::MAX as u128,
            Self::U32 => u32::MAX as u128,
            Self::U64 => u64::MAX as u128,
        }
    }

    fn tensor_dtype(self) -> Option<NumericDType> {
        match self {
            Self::Double => Some(NumericDType::F64),
            Self::Single => Some(NumericDType::F32),
            Self::U8 => Some(NumericDType::U8),
            Self::U16 => Some(NumericDType::U16),
            Self::U32 => Some(NumericDType::U32),
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U64 => None,
        }
    }

    fn from_integer_storage(storage: &IntegerStorage) -> Self {
        match storage {
            IntegerStorage::I8(_) => Self::I8,
            IntegerStorage::I16(_) => Self::I16,
            IntegerStorage::I32(_) => Self::I32,
            IntegerStorage::I64(_) => Self::I64,
            IntegerStorage::U8(_) => Self::U8,
            IntegerStorage::U16(_) => Self::U16,
            IntegerStorage::U32(_) => Self::U32,
            IntegerStorage::U64(_) => Self::U64,
        }
    }
}

struct LcmInput {
    data: Vec<u128>,
    shape: Vec<usize>,
    class: NumericClass,
    native_integer_storage: bool,
}

#[derive(Clone, Copy)]
struct LcmOutput {
    class: NumericClass,
    native_integer_storage: bool,
}

impl LcmInput {
    fn is_scalar(&self) -> bool {
        self.data.len() == 1 && element_count(&self.shape) == 1
    }

    async fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(value) => Ok(Self {
                data: vec![positive_integer_from_f64(value)?],
                shape: vec![1, 1],
                class: NumericClass::Double,
                native_integer_storage: false,
            }),
            Value::Int(value) => Self::from_int_value(value),
            Value::Tensor(tensor) => Self::from_tensor(tensor),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|err| error_with_detail(&LCM_ERROR_INTERNAL, err))?;
                Self::from_tensor(tensor)
            }
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(error_with_detail(
                &LCM_ERROR_INVALID_INPUT,
                "inputs must be real",
            )),
            Value::Bool(_) | Value::LogicalArray(_) => Err(error_with_detail(
                &LCM_ERROR_INVALID_INPUT,
                "logical inputs are not numeric integer classes for lcm",
            )),
            other => Err(error_with_detail(
                &LCM_ERROR_INVALID_INPUT,
                format!("unsupported input type {other:?}"),
            )),
        }
    }

    fn from_int_value(value: IntValue) -> BuiltinResult<Self> {
        let (data, class) = match value {
            IntValue::I8(value) => (
                positive_integer_from_i128(i128::from(value))?,
                NumericClass::I8,
            ),
            IntValue::I16(value) => (
                positive_integer_from_i128(i128::from(value))?,
                NumericClass::I16,
            ),
            IntValue::I32(value) => (
                positive_integer_from_i128(i128::from(value))?,
                NumericClass::I32,
            ),
            IntValue::I64(value) => (
                positive_integer_from_i128(i128::from(value))?,
                NumericClass::I64,
            ),
            IntValue::U8(value) => (
                positive_integer_from_u128(u128::from(value))?,
                NumericClass::U8,
            ),
            IntValue::U16(value) => (
                positive_integer_from_u128(u128::from(value))?,
                NumericClass::U16,
            ),
            IntValue::U32(value) => (
                positive_integer_from_u128(u128::from(value))?,
                NumericClass::U32,
            ),
            IntValue::U64(value) => (
                positive_integer_from_u128(u128::from(value))?,
                NumericClass::U64,
            ),
        };
        Ok(Self {
            data: vec![data],
            shape: vec![1, 1],
            class,
            native_integer_storage: true,
        })
    }

    fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        if let Some(storage) = tensor.integer_storage() {
            let class = NumericClass::from_integer_storage(storage);
            let data = storage
                .exact_values()
                .into_iter()
                .map(positive_integer_from_int_value)
                .collect::<BuiltinResult<Vec<_>>>()?;
            return Ok(Self {
                data,
                shape: tensor.shape,
                class,
                native_integer_storage: true,
            });
        }
        let class = match tensor.dtype {
            NumericDType::F64 => NumericClass::Double,
            NumericDType::F32 => NumericClass::Single,
            NumericDType::I8 => NumericClass::I8,
            NumericDType::I16 => NumericClass::I16,
            NumericDType::I32 => NumericClass::I32,
            NumericDType::I64 => NumericClass::I64,
            NumericDType::U8 => NumericClass::U8,
            NumericDType::U16 => NumericClass::U16,
            NumericDType::U32 => NumericClass::U32,
            NumericDType::U64 => NumericClass::U64,
        };
        let data = tensor
            .data
            .into_iter()
            .map(positive_integer_from_f64)
            .collect::<BuiltinResult<Vec<_>>>()?;
        Ok(Self {
            data,
            shape: tensor.shape,
            class,
            native_integer_storage: false,
        })
    }
}

fn positive_integer_from_int_value(value: IntValue) -> BuiltinResult<u128> {
    match value {
        IntValue::I8(value) => positive_integer_from_i128(i128::from(value)),
        IntValue::I16(value) => positive_integer_from_i128(i128::from(value)),
        IntValue::I32(value) => positive_integer_from_i128(i128::from(value)),
        IntValue::I64(value) => positive_integer_from_i128(i128::from(value)),
        IntValue::U8(value) => positive_integer_from_u128(u128::from(value)),
        IntValue::U16(value) => positive_integer_from_u128(u128::from(value)),
        IntValue::U32(value) => positive_integer_from_u128(u128::from(value)),
        IntValue::U64(value) => positive_integer_from_u128(u128::from(value)),
    }
}

struct SameSizeOrScalarPlan {
    output_shape: Vec<usize>,
    len: usize,
    left_scalar: bool,
    right_scalar: bool,
}

impl SameSizeOrScalarPlan {
    fn new(left: &LcmInput, right: &LcmInput) -> BuiltinResult<Self> {
        let left_scalar = left.is_scalar();
        let right_scalar = right.is_scalar();
        let output_shape = if left.shape == right.shape {
            left.shape.clone()
        } else if left_scalar {
            right.shape.clone()
        } else if right_scalar {
            left.shape.clone()
        } else {
            return Err(error_with_detail(
                &LCM_ERROR_SIZE_MISMATCH,
                "inputs must be the same size or one input must be scalar",
            ));
        };
        Ok(Self {
            len: element_count(&output_shape),
            output_shape,
            left_scalar,
            right_scalar,
        })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.len).map(|idx| {
            (
                if self.left_scalar { 0 } else { idx },
                if self.right_scalar { 0 } else { idx },
            )
        })
    }
}

fn resolve_output_kind(left: &LcmInput, right: &LcmInput) -> BuiltinResult<LcmOutput> {
    let (class, native_integer_storage) = match (left.class, right.class) {
        (a, b) if a == b => (
            a,
            left.native_integer_storage || right.native_integer_storage,
        ),
        (NumericClass::Double, NumericClass::Single)
        | (NumericClass::Single, NumericClass::Double) => (NumericClass::Single, false),
        (integer, NumericClass::Double) if integer.is_integer() && right.is_scalar() => {
            (integer, left.native_integer_storage)
        }
        (NumericClass::Double, integer) if integer.is_integer() && left.is_scalar() => {
            (integer, right.native_integer_storage)
        }
        (a, b) if a.is_integer() && b.is_integer() => {
            return Err(error_with_detail(
                &LCM_ERROR_INVALID_INPUT,
                "integer inputs must have the same class",
            ));
        }
        _ => {
            return Err(error_with_detail(
                &LCM_ERROR_INVALID_INPUT,
                "integer inputs can only be paired with the same class or a double scalar",
            ));
        }
    };
    Ok(LcmOutput {
        class,
        native_integer_storage,
    })
}

fn value_from_lcms(data: Vec<u128>, shape: Vec<usize>, output: LcmOutput) -> BuiltinResult<Value> {
    let class = output.class;
    for &value in &data {
        if value > class.max_value() {
            return Err(error_with_detail(
                &LCM_ERROR_OVERFLOW,
                "result exceeds output class range",
            ));
        }
    }

    if data.len() == 1 && element_count(&shape) == 1 {
        let value = data[0];
        return match class {
            NumericClass::Double => Ok(Value::Num(value as f64)),
            NumericClass::Single => Ok(Value::Num((value as f32) as f64)),
            NumericClass::I8 => Ok(Value::Int(IntValue::I8(value as i8))),
            NumericClass::I16 => Ok(Value::Int(IntValue::I16(value as i16))),
            NumericClass::I32 => Ok(Value::Int(IntValue::I32(value as i32))),
            NumericClass::I64 => Ok(Value::Int(IntValue::I64(value as i64))),
            NumericClass::U8 => Ok(Value::Int(IntValue::U8(value as u8))),
            NumericClass::U16 => Ok(Value::Int(IntValue::U16(value as u16))),
            NumericClass::U32 => Ok(Value::Int(IntValue::U32(value as u32))),
            NumericClass::U64 => Ok(Value::Int(IntValue::U64(value as u64))),
        };
    }

    if output.native_integer_storage {
        return Tensor::new_integer(integer_storage_from_lcms(data, class), shape)
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(&LCM_ERROR_INTERNAL, err));
    }
    let dtype = class
        .tensor_dtype()
        .expect("floating LCM output classes have a tensor dtype");
    Tensor::new_with_dtype(
        data.into_iter()
            .map(|value| match class {
                NumericClass::Single => (value as f32) as f64,
                _ => value as f64,
            })
            .collect(),
        shape,
        dtype,
    )
    .map(Value::Tensor)
    .map_err(|err| error_with_detail(&LCM_ERROR_INTERNAL, err))
}

fn integer_storage_from_lcms(data: Vec<u128>, class: NumericClass) -> IntegerStorage {
    match class {
        NumericClass::I8 => IntegerStorage::I8(data.into_iter().map(|value| value as i8).collect()),
        NumericClass::I16 => {
            IntegerStorage::I16(data.into_iter().map(|value| value as i16).collect())
        }
        NumericClass::I32 => {
            IntegerStorage::I32(data.into_iter().map(|value| value as i32).collect())
        }
        NumericClass::I64 => {
            IntegerStorage::I64(data.into_iter().map(|value| value as i64).collect())
        }
        NumericClass::U8 => IntegerStorage::U8(data.into_iter().map(|value| value as u8).collect()),
        NumericClass::U16 => {
            IntegerStorage::U16(data.into_iter().map(|value| value as u16).collect())
        }
        NumericClass::U32 => {
            IntegerStorage::U32(data.into_iter().map(|value| value as u32).collect())
        }
        NumericClass::U64 => {
            IntegerStorage::U64(data.into_iter().map(|value| value as u64).collect())
        }
        NumericClass::Double | NumericClass::Single => {
            unreachable!("integer storage requires an integer output class")
        }
    }
}

fn positive_integer_from_i128(value: i128) -> BuiltinResult<u128> {
    if value <= 0 {
        return Err(error_with_detail(
            &LCM_ERROR_INVALID_INPUT,
            "inputs must be positive integers",
        ));
    }
    Ok(value as u128)
}

fn positive_integer_from_u128(value: u128) -> BuiltinResult<u128> {
    if value == 0 {
        return Err(error_with_detail(
            &LCM_ERROR_INVALID_INPUT,
            "inputs must be positive integers",
        ));
    }
    Ok(value)
}

fn positive_integer_from_f64(value: f64) -> BuiltinResult<u128> {
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 {
        return Err(error_with_detail(
            &LCM_ERROR_INVALID_INPUT,
            "inputs must be finite positive integers",
        ));
    }
    if value >= u64::MAX as f64 {
        return Err(error_with_detail(
            &LCM_ERROR_INVALID_INPUT,
            "input is too large",
        ));
    }
    Ok(value as u128)
}

fn lcm_u128(left: u128, right: u128) -> u128 {
    left / gcd_u128(left, right) * right
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let rem = left % right;
        left = right;
        right = rem;
    }
    left
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(BUILTIN_NAME);
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
    fn lcm_double_array_and_scalar() {
        let input = Tensor::new(vec![5.0, 17.0, 10.0, 60.0], vec![2, 2]).unwrap();
        let out = block_on(lcm_builtin(Value::Tensor(input), Value::Num(45.0))).expect("lcm");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert_eq!(tensor.dtype, NumericDType::F64);
                assert_eq!(tensor.data, vec![45.0, 765.0, 90.0, 180.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lcm_preserves_unsigned_integer_class() {
        let left = Tensor::new_with_dtype(vec![255.0, 511.0, 15.0], vec![1, 3], NumericDType::U16)
            .unwrap();
        let right =
            Tensor::new_with_dtype(vec![15.0, 127.0, 1023.0], vec![1, 3], NumericDType::U16)
                .unwrap();
        let out = block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right))).expect("lcm");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.dtype, NumericDType::U16);
                assert_eq!(tensor.data, vec![255.0, 64897.0, 5115.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lcm_integer_array_accepts_double_scalar_and_keeps_integer_class() {
        let left =
            Tensor::new_with_dtype(vec![6.0, 10.0, 21.0], vec![1, 3], NumericDType::U32).unwrap();
        let out = block_on(lcm_builtin(Value::Tensor(left), Value::Num(15.0))).expect("lcm");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.dtype, NumericDType::U32);
                assert_eq!(tensor.data, vec![30.0, 30.0, 105.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lcm_preserves_all_native_integer_tensor_classes() {
        let cases = [
            (
                IntegerStorage::I8(vec![6, 10]),
                IntegerStorage::I8(vec![30, 30]),
            ),
            (
                IntegerStorage::I16(vec![6, 10]),
                IntegerStorage::I16(vec![30, 30]),
            ),
            (
                IntegerStorage::I32(vec![6, 10]),
                IntegerStorage::I32(vec![30, 30]),
            ),
            (
                IntegerStorage::I64(vec![6, 10]),
                IntegerStorage::I64(vec![30, 30]),
            ),
            (
                IntegerStorage::U8(vec![6, 10]),
                IntegerStorage::U8(vec![30, 30]),
            ),
            (
                IntegerStorage::U16(vec![6, 10]),
                IntegerStorage::U16(vec![30, 30]),
            ),
            (
                IntegerStorage::U32(vec![6, 10]),
                IntegerStorage::U32(vec![30, 30]),
            ),
            (
                IntegerStorage::U64(vec![6, 10]),
                IntegerStorage::U64(vec![30, 30]),
            ),
        ];

        for (input, expected) in cases {
            let input = Tensor::new_integer(input, vec![1, 2]).expect("native integer tensor");
            let out = block_on(lcm_builtin(Value::Tensor(input), Value::Num(15.0)))
                .expect("integer tensor with double scalar");
            match out {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape, vec![1, 2]);
                    assert_eq!(tensor.integer_storage(), Some(&expected));
                }
                other => panic!("expected tensor, got {other:?}"),
            }
        }
    }

    #[test]
    fn lcm_uses_exact_native_uint64_tensor_values() {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_995]),
            vec![1, 2],
        )
        .expect("native uint64 tensor");
        let out = block_on(lcm_builtin(Value::Tensor(input), Value::Num(1.0)))
            .expect("lcm with identity");
        match out {
            Value::Tensor(tensor) => assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    9_007_199_254_740_993,
                    9_007_199_254_740_995,
                ]))
            ),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lcm_scalar_integer_output_preserves_width() {
        let out = block_on(lcm_builtin(
            Value::Int(IntValue::U8(12)),
            Value::Int(IntValue::U8(18)),
        ))
        .expect("lcm");
        assert_eq!(out, Value::Int(IntValue::U8(36)));
    }

    #[test]
    fn lcm_rejects_zero_negative_fractional_and_complex() {
        for value in [
            Value::Num(0.0),
            Value::Num(-2.0),
            Value::Num(2.5),
            Value::Complex(2.0, 0.0),
        ] {
            let err = block_on(lcm_builtin(value, Value::Num(3.0))).expect_err("invalid input");
            assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn lcm_rejects_unrepresentable_double_u64_boundary_before_casting() {
        let err = block_on(lcm_builtin(Value::Num(u64::MAX as f64), Value::Num(3.0)))
            .expect_err("unrepresentable u64 boundary should fail before cast");
        assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
        assert!(err.to_string().contains("input is too large"));

        let tensor = Tensor::new(vec![3.0, u64::MAX as f64], vec![1, 2]).unwrap();
        let err = block_on(lcm_builtin(Value::Tensor(tensor), Value::Num(3.0)))
            .expect_err("unrepresentable tensor entry should fail before cast");
        assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
        assert!(err.to_string().contains("input is too large"));
    }

    #[test]
    fn lcm_rejects_mismatched_shapes_and_integer_classes() {
        let left = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let right = Tensor::new(vec![5.0, 7.0, 11.0], vec![1, 3]).unwrap();
        let err = block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right)))
            .expect_err("shape mismatch");
        assert_eq!(err.identifier(), LCM_ERROR_SIZE_MISMATCH.identifier);

        let left = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let right = Tensor::new(vec![5.0, 7.0, 11.0], vec![1, 3]).unwrap();
        let err = block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right)))
            .expect_err("implicit expansion is not supported for lcm");
        assert_eq!(err.identifier(), LCM_ERROR_SIZE_MISMATCH.identifier);

        let err = block_on(lcm_builtin(
            Value::Int(IntValue::U8(2)),
            Value::Int(IntValue::U16(4)),
        ))
        .expect_err("integer class mismatch");
        assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);

        let single_scalar =
            Tensor::new_with_dtype(vec![3.0], vec![1, 1], NumericDType::F32).unwrap();
        let err = block_on(lcm_builtin(
            Value::Int(IntValue::U16(2)),
            Value::Tensor(single_scalar),
        ))
        .expect_err("integer plus single scalar is not permitted");
        assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn lcm_rejects_integer_overflow() {
        let err = block_on(lcm_builtin(
            Value::Int(IntValue::U8(200)),
            Value::Int(IntValue::U8(201)),
        ))
        .expect_err("overflow");
        assert_eq!(err.identifier(), LCM_ERROR_OVERFLOW.identifier);
    }

    #[test]
    fn lcm_single_output_rounds_through_single_precision() {
        let left =
            Tensor::new_with_dtype(vec![16_777_217.0, 3.0], vec![1, 2], NumericDType::F32).unwrap();
        let right = Tensor::new_with_dtype(vec![1.0, 5.0], vec![1, 2], NumericDType::F32).unwrap();
        let out = block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right))).expect("lcm");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.dtype, NumericDType::F32);
                assert_eq!(tensor.data, vec![(16_777_217_u128 as f32) as f64, 15.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}
