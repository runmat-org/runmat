//! MATLAB-compatible `factor` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, NumericScalar, NumericStorage, Tensor, Value};

use super::integer_number_theory::prime_factors;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "factor";

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of prime factors in ascending order.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real nonnegative integer scalar.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "F = factor(n)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTOR.INVALID_INPUT",
    identifier: Some("RunMat:factor:InvalidInput"),
    when: "Input is not a real nonnegative integer scalar in a supported host numeric class.",
    message: "factor: input must be a real nonnegative integer scalar",
};
const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FACTOR.INTERNAL",
    identifier: Some("RunMat:factor:Internal"),
    when: "The result tensor cannot be constructed.",
    message: "factor: result construction failed",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const FACTOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Input must be a real nonnegative integer-valued scalar.",
}];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "F = factor(n)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Returns a same-class row of prime factors; interactive GPU input is unsupported.",
    }];

fn factor_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::tensor()
}

#[runtime_builtin(
    name = "factor",
    category = "math/discrete",
    summary = "Return the prime factors of a nonnegative integer scalar.",
    keywords = "factor,prime factors,integer,number theory,discrete",
    type_resolver(factor_type),
    descriptor(crate::builtins::math::discrete::factor::FACTOR_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::discrete::factor::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::discrete::factor"
)]
async fn factor_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(invalid());
    }
    let (n, class) = parse(value)?;
    result_value(prime_factors(n), class)
}

#[derive(Clone, Copy)]
enum Class {
    F64,
    F32,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

fn parse(value: Value) -> BuiltinResult<(u64, Class)> {
    match value {
        Value::Num(value) => parse_float(value)
            .map(|n| (n, Class::F64))
            .ok_or_else(invalid),
        Value::Int(value) => parse_int(value),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            let class = class_from_dtype(tensor.numeric_dtype());
            parse_scalar(tensor.numeric_value_at(0).ok_or_else(invalid)?, class)
        }
        _ => Err(invalid()),
    }
}

fn parse_scalar(value: NumericScalar, class: Class) -> BuiltinResult<(u64, Class)> {
    let n = match value {
        NumericScalar::F64(value) => parse_float(value),
        NumericScalar::F32(value) => parse_float(f64::from(value)),
        NumericScalar::I8(value) => u64::try_from(value).ok(),
        NumericScalar::I16(value) => u64::try_from(value).ok(),
        NumericScalar::I32(value) => u64::try_from(value).ok(),
        NumericScalar::I64(value) => u64::try_from(value).ok(),
        NumericScalar::U8(value) => Some(u64::from(value)),
        NumericScalar::U16(value) => Some(u64::from(value)),
        NumericScalar::U32(value) => Some(u64::from(value)),
        NumericScalar::U64(value) => Some(value),
    };
    n.map(|n| (n, class)).ok_or_else(invalid)
}

fn parse_int(value: IntValue) -> BuiltinResult<(u64, Class)> {
    let class = match value {
        IntValue::I8(_) => Class::I8,
        IntValue::I16(_) => Class::I16,
        IntValue::I32(_) => Class::I32,
        IntValue::I64(_) => Class::I64,
        IntValue::U8(_) => Class::U8,
        IntValue::U16(_) => Class::U16,
        IntValue::U32(_) => Class::U32,
        IntValue::U64(_) => Class::U64,
    };
    parse_scalar(NumericScalar::from(value), class)
}

fn parse_float(value: f64) -> Option<u64> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < u64::MAX as f64)
        .then_some(value as u64)
}

fn class_from_dtype(dtype: NumericDType) -> Class {
    match dtype {
        NumericDType::F64 => Class::F64,
        NumericDType::F32 => Class::F32,
        NumericDType::I8 => Class::I8,
        NumericDType::I16 => Class::I16,
        NumericDType::I32 => Class::I32,
        NumericDType::I64 => Class::I64,
        NumericDType::U8 => Class::U8,
        NumericDType::U16 => Class::U16,
        NumericDType::U32 => Class::U32,
        NumericDType::U64 => Class::U64,
    }
}

fn result_value(values: Vec<u64>, class: Class) -> BuiltinResult<Value> {
    macro_rules! cast {
        ($variant:ident, $ty:ty) => {
            NumericStorage::$variant(values.iter().map(|&value| value as $ty).collect())
        };
    }
    let storage = match class {
        Class::F64 => cast!(F64, f64),
        Class::F32 => cast!(F32, f32),
        Class::I8 => cast!(I8, i8),
        Class::I16 => cast!(I16, i16),
        Class::I32 => cast!(I32, i32),
        Class::I64 => cast!(I64, i64),
        Class::U8 => cast!(U8, u8),
        Class::U16 => cast!(U16, u16),
        Class::U32 => cast!(U32, u32),
        Class::U64 => NumericStorage::U64(values),
    };
    let len = storage.len();
    Tensor::from_numeric_storage(storage, vec![1, len])
        .map(Value::Tensor)
        .map_err(|detail| error(&ERROR_INTERNAL, detail))
}

fn invalid() -> RuntimeError {
    error(&ERROR_INVALID_INPUT, ERROR_INVALID_INPUT.message)
}

fn error(descriptor: &'static BuiltinErrorDescriptor, detail: impl Into<String>) -> RuntimeError {
    build_runtime_error(detail)
        .with_builtin(NAME)
        .with_identifier(descriptor.identifier.expect("descriptor identifier"))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(factor_builtin(value, Vec::new()))
    }

    #[test]
    fn handles_special_values_and_large_semiprime() {
        for value in [0.0, 1.0] {
            let Value::Tensor(out) = call(Value::Num(value)).unwrap() else {
                panic!()
            };
            assert_eq!(out.as_f64_slice(), Some([value].as_slice()));
        }
        let n = 4_294_967_291u64 * 4_294_967_279u64;
        let Value::Tensor(out) = call(Value::Int(IntValue::U64(n))).unwrap() else {
            panic!()
        };
        assert_eq!(
            out.integer_storage(),
            Some(&IntegerStorage::U64(vec![4_294_967_279, 4_294_967_291]))
        );
    }

    #[test]
    fn preserves_all_numeric_classes() {
        let inputs = [
            NumericStorage::F64(vec![12.0]),
            NumericStorage::F32(vec![12.0]),
            NumericStorage::I8(vec![12]),
            NumericStorage::I16(vec![12]),
            NumericStorage::I32(vec![12]),
            NumericStorage::I64(vec![12]),
            NumericStorage::U8(vec![12]),
            NumericStorage::U16(vec![12]),
            NumericStorage::U32(vec![12]),
            NumericStorage::U64(vec![12]),
        ];
        for storage in inputs {
            let dtype = storage.numeric_dtype();
            let input = Tensor::from_numeric_storage(storage, vec![1, 1]).unwrap();
            let Value::Tensor(out) = call(Value::Tensor(input)).unwrap() else {
                panic!()
            };
            assert_eq!(out.numeric_dtype(), dtype);
            assert_eq!(out.shape, vec![1, 3]);
        }
    }

    #[test]
    fn rejects_invalid_inputs() {
        for value in [
            Value::Num(-1.0),
            Value::Num(1.5),
            Value::Num(u64::MAX as f64),
            Value::Complex(2.0, 0.0),
        ] {
            assert!(call(value).is_err());
        }
    }

    #[test]
    fn public_number_theory_builtins_are_registered() {
        assert!(runmat_builtins::builtin_function_by_name("factor").is_some());
        assert!(runmat_builtins::builtin_function_by_name("isprime").is_some());
    }
}
