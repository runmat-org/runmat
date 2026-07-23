//! MATLAB-compatible `primes` builtin.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, IntegerStorage, NumericDType, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "primes";
const MAX_PRIMES_LIMIT: u64 = 10_000_000;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of prime numbers less than or equal to n.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar real integer upper bound.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = primes(n)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRIMES.INVALID_INPUT",
    identifier: Some("RunMat:primes:InvalidInput"),
    when: "Input is not a scalar real integer value.",
    message: "primes: invalid input",
};

const ERROR_LIMIT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRIMES.LIMIT",
    identifier: Some("RunMat:primes:Limit"),
    when: "Requested sieve would exceed RunMat's safety limit.",
    message: "primes: requested limit is too large",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PRIMES.INTERNAL",
    identifier: Some("RunMat:primes:Internal"),
    when: "Internal tensor construction or GPU gather failed.",
    message: "primes: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INVALID_INPUT, ERROR_LIMIT, ERROR_INTERNAL];

pub const PRIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn primes_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::tensor()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputKind {
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

#[derive(Clone, Copy, Debug)]
struct PrimeRequest {
    limit: u64,
    output: OutputKind,
}

#[runtime_builtin(
    name = "primes",
    category = "math/discrete",
    summary = "Return all prime numbers less than or equal to a scalar bound.",
    keywords = "primes,prime,numbers,discrete,number theory",
    type_resolver(primes_type),
    descriptor(crate::builtins::math::discrete::primes::PRIMES_DESCRIPTOR),
    builtin_path = "crate::builtins::math::discrete::primes"
)]
async fn primes_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "expected exactly one input argument",
        ));
    }
    let request = parse_prime_request(value).await?;
    if request.limit > MAX_PRIMES_LIMIT {
        return Err(error_with_detail(
            &ERROR_LIMIT,
            format!("n must be <= {MAX_PRIMES_LIMIT}"),
        ));
    }
    primes_value(request)
}

async fn parse_prime_request(value: Value) -> BuiltinResult<PrimeRequest> {
    match value {
        Value::Num(n) => scalar_from_f64(n, OutputKind::F64),
        Value::Int(i) => scalar_from_int(i),
        Value::Tensor(tensor) => scalar_from_tensor(tensor),
        Value::GpuTensor(handle) => scalar_from_gpu(handle).await,
        Value::Bool(_) | Value::LogicalArray(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "n must be numeric, not logical",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(error_with_detail(&ERROR_INVALID_INPUT, "n must be real"))
        }
        other => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            format!("unsupported input type {other:?}"),
        )),
    }
}

async fn scalar_from_gpu(handle: GpuTensorHandle) -> BuiltinResult<PrimeRequest> {
    let len = handle
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| error_with_detail(&ERROR_INVALID_INPUT, "n must be a scalar value"))?;
    if len != 1 {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "n must be a scalar value",
        ));
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
    scalar_from_tensor(tensor)
}

fn scalar_from_tensor(tensor: Tensor) -> BuiltinResult<PrimeRequest> {
    if tensor.data.len() != 1 {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "n must be a scalar value",
        ));
    }
    if let Some(storage) = tensor.integer_storage() {
        let value = storage
            .value_at(0)
            .expect("integer storage length matches tensor data length");
        return scalar_from_int(value);
    }
    let output = match tensor.dtype {
        NumericDType::F32 => OutputKind::F32,
        NumericDType::U8 => OutputKind::U8,
        NumericDType::U16 => OutputKind::U16,
        NumericDType::U32 => OutputKind::U32,
        NumericDType::F64 => OutputKind::F64,
    };
    scalar_from_f64(tensor.data[0], output)
}

fn scalar_from_int(value: IntValue) -> BuiltinResult<PrimeRequest> {
    let (limit, output) = match value {
        IntValue::I8(v) => (signed_limit(i64::from(v)), OutputKind::I8),
        IntValue::I16(v) => (signed_limit(i64::from(v)), OutputKind::I16),
        IntValue::I32(v) => (signed_limit(i64::from(v)), OutputKind::I32),
        IntValue::I64(v) => (signed_limit(v), OutputKind::I64),
        IntValue::U8(v) => (u64::from(v), OutputKind::U8),
        IntValue::U16(v) => (u64::from(v), OutputKind::U16),
        IntValue::U32(v) => (u64::from(v), OutputKind::U32),
        IntValue::U64(v) => (v, OutputKind::U64),
    };
    Ok(PrimeRequest { limit, output })
}

fn signed_limit(value: i64) -> u64 {
    if value < 2 {
        0
    } else {
        value as u64
    }
}

fn scalar_from_f64(value: f64, output: OutputKind) -> BuiltinResult<PrimeRequest> {
    if !value.is_finite() {
        return Err(error_with_detail(&ERROR_INVALID_INPUT, "n must be finite"));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 0.0 {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "n must be an integer value",
        ));
    }
    let limit = if rounded < 2.0 {
        0
    } else if rounded > u64::MAX as f64 {
        return Err(error_with_detail(&ERROR_INVALID_INPUT, "n is too large"));
    } else {
        rounded as u64
    };
    Ok(PrimeRequest { limit, output })
}

fn primes_value(request: PrimeRequest) -> BuiltinResult<Value> {
    let primes = sieve_primes(request.limit);
    let shape = vec![1, primes.len()];
    let tensor = match request.output {
        OutputKind::F64 => Tensor::new_with_dtype(
            primes.iter().map(|&prime| prime as f64).collect(),
            shape,
            NumericDType::F64,
        ),
        OutputKind::F32 => Tensor::new_with_dtype(
            primes.iter().map(|&prime| prime as f64).collect(),
            shape,
            NumericDType::F32,
        ),
        OutputKind::I8 => Tensor::new_integer(
            IntegerStorage::I8(primes.iter().map(|&prime| prime as i8).collect()),
            shape,
        ),
        OutputKind::I16 => Tensor::new_integer(
            IntegerStorage::I16(primes.iter().map(|&prime| prime as i16).collect()),
            shape,
        ),
        OutputKind::I32 => Tensor::new_integer(
            IntegerStorage::I32(primes.iter().map(|&prime| prime as i32).collect()),
            shape,
        ),
        OutputKind::I64 => Tensor::new_integer(
            IntegerStorage::I64(primes.iter().map(|&prime| prime as i64).collect()),
            shape,
        ),
        OutputKind::U8 => Tensor::new_integer(
            IntegerStorage::U8(primes.iter().map(|&prime| prime as u8).collect()),
            shape,
        ),
        OutputKind::U16 => Tensor::new_integer(
            IntegerStorage::U16(primes.iter().map(|&prime| prime as u16).collect()),
            shape,
        ),
        OutputKind::U32 => Tensor::new_integer(
            IntegerStorage::U32(primes.iter().map(|&prime| prime as u32).collect()),
            shape,
        ),
        OutputKind::U64 => Tensor::new_integer(IntegerStorage::U64(primes), shape),
    };
    tensor
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))
}

fn sieve_primes(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let limit = limit as usize;
    let mut composite = vec![false; limit + 1];
    let mut p = 2usize;
    while p <= limit / p {
        if !composite[p] {
            let mut multiple = p * p;
            while multiple <= limit {
                composite[multiple] = true;
                multiple += p;
            }
        }
        p += 1;
    }
    (2..=limit)
        .filter(|&candidate| !composite[candidate])
        .map(|candidate| candidate as u64)
        .collect()
}

fn error_with_detail(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn call(value: Value) -> BuiltinResult<Tensor> {
        match block_on(primes_builtin(value, Vec::new()))? {
            Value::Tensor(tensor) => Ok(tensor),
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn primes_returns_row_vector_of_double_primes() {
        let out = call(Value::Num(25.0)).expect("primes");
        assert_eq!(out.shape, vec![1, 9]);
        assert_eq!(out.dtype, NumericDType::F64);
        assert_eq!(
            out.data,
            vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]
        );
    }

    #[test]
    fn primes_returns_empty_row_for_values_below_two() {
        for value in [Value::Num(1.0), Value::Num(0.0), Value::Num(-8.0)] {
            let out = call(value).expect("primes");
            assert_eq!(out.shape, vec![1, 0]);
            assert!(out.data.is_empty());
        }
    }

    #[test]
    fn primes_preserves_floating_and_exact_integer_output_classes() {
        let single = Tensor::new_with_dtype(vec![12.0], vec![1, 1], NumericDType::F32).unwrap();
        let out = call(Value::Tensor(single)).expect("single primes");
        assert_eq!(out.dtype, NumericDType::F32);
        assert_eq!(out.data, vec![2.0, 3.0, 5.0, 7.0, 11.0]);

        let expected_primes = vec![2, 3, 5, 7, 11];
        let cases = [
            (
                IntValue::I8(12),
                IntegerStorage::I8(expected_primes.iter().map(|&v| v as i8).collect()),
            ),
            (
                IntValue::I16(12),
                IntegerStorage::I16(expected_primes.iter().map(|&v| v as i16).collect()),
            ),
            (
                IntValue::I32(12),
                IntegerStorage::I32(expected_primes.iter().map(|&v| v as i32).collect()),
            ),
            (
                IntValue::I64(12),
                IntegerStorage::I64(expected_primes.iter().map(|&v| v as i64).collect()),
            ),
            (
                IntValue::U8(12),
                IntegerStorage::U8(expected_primes.iter().map(|&v| v as u8).collect()),
            ),
            (
                IntValue::U16(12),
                IntegerStorage::U16(expected_primes.iter().map(|&v| v as u16).collect()),
            ),
            (
                IntValue::U32(12),
                IntegerStorage::U32(expected_primes.iter().map(|&v| v as u32).collect()),
            ),
            (IntValue::U64(12), IntegerStorage::U64(expected_primes)),
        ];

        for (input, expected) in cases {
            let out = call(Value::Int(input)).expect("integer primes");
            assert_eq!(out.shape, vec![1, 5]);
            assert_eq!(out.integer_storage(), Some(&expected));
        }

        let input = Tensor::new_integer(IntegerStorage::I64(vec![12]), vec![1, 1])
            .expect("exact int64 tensor");
        let out = call(Value::Tensor(input)).expect("integer tensor primes");
        assert_eq!(
            out.integer_storage(),
            Some(&IntegerStorage::I64(vec![2, 3, 5, 7, 11]))
        );

        let negative = call(Value::Int(IntValue::I16(-4))).expect("negative integer primes");
        assert_eq!(negative.shape, vec![1, 0]);
        assert_eq!(
            negative.integer_storage(),
            Some(&IntegerStorage::I16(Vec::new()))
        );
    }

    #[test]
    fn primes_rejects_non_scalar_gpu_handle_before_gather() {
        let handle = GpuTensorHandle {
            shape: vec![1, 2],
            device_id: 0,
            buffer_id: 999,
        };
        let err = block_on(primes_builtin(Value::GpuTensor(handle), Vec::new()))
            .expect_err("non-scalar gpu handle should fail before gather");
        assert!(err.to_string().contains("n must be a scalar value"));
    }

    #[test]
    fn primes_rejects_non_scalar_non_integer_and_nonfinite_inputs() {
        let matrix = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        assert!(block_on(primes_builtin(Value::Tensor(matrix), Vec::new())).is_err());
        assert!(block_on(primes_builtin(Value::Num(10.5), Vec::new())).is_err());
        assert!(block_on(primes_builtin(Value::Num(f64::INFINITY), Vec::new())).is_err());
        assert!(block_on(primes_builtin(Value::Bool(true), Vec::new())).is_err());
    }

    #[test]
    fn primes_rejects_huge_requests_before_allocation() {
        let err = block_on(primes_builtin(
            Value::Num((MAX_PRIMES_LIMIT + 1) as f64),
            Vec::new(),
        ))
        .expect_err("limit should fail");
        assert!(err.to_string().contains("requested limit is too large"));
    }
}
