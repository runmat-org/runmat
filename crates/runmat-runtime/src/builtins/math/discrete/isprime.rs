//! MATLAB-compatible `isprime` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, LogicalArray, NumericStorage, Type, Value,
};
use runmat_macros::runtime_builtin;

use super::integer_number_theory::is_prime;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "isprime";

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical array indicating which input elements are prime.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real array of nonnegative integer values.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isprime(A)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISPRIME.INVALID_INPUT",
    identifier: Some("RunMat:isprime:InvalidInput"),
    when: "Input is not a real host numeric array of nonnegative integer values.",
    message: "isprime: input values must be real nonnegative integers",
};
const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISPRIME.INTERNAL",
    identifier: Some("RunMat:isprime:Internal"),
    when: "The logical result array cannot be constructed.",
    message: "isprime: result construction failed",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const ISPRIME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every element must be a real nonnegative integer value.",
}];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = isprime(A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Logical output preserves input shape including empties; interactive GPU input is unsupported.",
    }];

fn isprime_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::logical()
}

#[runtime_builtin(
    name = "isprime",
    category = "math/discrete",
    summary = "Test which nonnegative integer values are prime.",
    keywords = "isprime,prime test,integer,number theory,discrete",
    type_resolver(isprime_type),
    descriptor(crate::builtins::math::discrete::isprime::ISPRIME_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::discrete::isprime::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::discrete::isprime"
)]
async fn isprime_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(invalid());
    }
    evaluate(value)
}

fn evaluate(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => Ok(Value::Bool(test_float(value)?)),
        Value::Int(value) => Ok(Value::Bool(test_int(value)?)),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let storage = tensor
                .into_numeric_storage()
                .map_err(|detail| error(&ERROR_INTERNAL, detail))?;
            let data = test_storage(storage)?;
            LogicalArray::new(data, shape)
                .map(Value::LogicalArray)
                .map_err(|detail| error(&ERROR_INTERNAL, detail))
        }
        _ => Err(invalid()),
    }
}

fn test_storage(storage: NumericStorage) -> BuiltinResult<Vec<u8>> {
    macro_rules! unsigned {
        ($values:expr) => {
            $values
                .into_iter()
                .map(|value| u8::from(is_prime(u64::from(value))))
                .collect()
        };
    }
    macro_rules! signed {
        ($values:expr) => {{
            let values = $values;
            if values.iter().any(|&value| value < 0) {
                return Err(invalid());
            }
            values
                .into_iter()
                .map(|value| u8::from(is_prime(value as u64)))
                .collect()
        }};
    }
    Ok(match storage {
        NumericStorage::F64(values) => values
            .into_iter()
            .map(test_float)
            .collect::<BuiltinResult<Vec<_>>>()?
            .into_iter()
            .map(u8::from)
            .collect(),
        NumericStorage::F32(values) => values
            .into_iter()
            .map(|value| test_float(f64::from(value)))
            .collect::<BuiltinResult<Vec<_>>>()?
            .into_iter()
            .map(u8::from)
            .collect(),
        NumericStorage::I8(values) => signed!(values),
        NumericStorage::I16(values) => signed!(values),
        NumericStorage::I32(values) => signed!(values),
        NumericStorage::I64(values) => signed!(values),
        NumericStorage::U8(values) => unsigned!(values),
        NumericStorage::U16(values) => unsigned!(values),
        NumericStorage::U32(values) => unsigned!(values),
        NumericStorage::U64(values) => unsigned!(values),
    })
}

fn test_float(value: f64) -> BuiltinResult<bool> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value >= u64::MAX as f64 {
        return Err(invalid());
    }
    Ok(is_prime(value as u64))
}

fn test_int(value: IntValue) -> BuiltinResult<bool> {
    let value = match value {
        IntValue::I8(value) => u64::try_from(value).ok(),
        IntValue::I16(value) => u64::try_from(value).ok(),
        IntValue::I32(value) => u64::try_from(value).ok(),
        IntValue::I64(value) => u64::try_from(value).ok(),
        IntValue::U8(value) => Some(u64::from(value)),
        IntValue::U16(value) => Some(u64::from(value)),
        IntValue::U32(value) => Some(u64::from(value)),
        IntValue::U64(value) => Some(value),
    }
    .ok_or_else(invalid)?;
    Ok(is_prime(value))
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
    use runmat_builtins::Tensor;

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(isprime_builtin(value, Vec::new()))
    }

    #[test]
    fn classifies_scalars_and_full_width_integers() {
        assert_eq!(call(Value::Num(2.0)).unwrap(), Value::Bool(true));
        assert_eq!(
            call(Value::Int(IntValue::U64(u64::MAX))).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            call(Value::Int(IntValue::U64(18_446_744_073_709_551_557))).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn supports_every_numeric_storage_class_and_preserves_shape() {
        let storages = [
            NumericStorage::F64(vec![2.0, 4.0]),
            NumericStorage::F32(vec![2.0, 4.0]),
            NumericStorage::I8(vec![2, 4]),
            NumericStorage::I16(vec![2, 4]),
            NumericStorage::I32(vec![2, 4]),
            NumericStorage::I64(vec![2, 4]),
            NumericStorage::U8(vec![2, 4]),
            NumericStorage::U16(vec![2, 4]),
            NumericStorage::U32(vec![2, 4]),
            NumericStorage::U64(vec![2, 4]),
        ];
        for storage in storages {
            let input = Tensor::from_numeric_storage(storage, vec![2, 1]).unwrap();
            let Value::LogicalArray(out) = call(Value::Tensor(input)).unwrap() else {
                panic!()
            };
            assert_eq!(out.shape, vec![2, 1]);
            assert_eq!(out.data, vec![1, 0]);
        }
    }

    #[test]
    fn supports_empty_and_rejects_invalid_or_gpu_values() {
        let input = Tensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::LogicalArray(out) = call(Value::Tensor(input)).unwrap() else {
            panic!()
        };
        assert_eq!(out.shape, vec![0, 3]);
        for value in [
            Value::Num(-1.0),
            Value::Num(2.5),
            Value::Num(u64::MAX as f64),
            Value::Complex(2.0, 0.0),
        ] {
            assert!(call(value).is_err());
        }
    }
}
