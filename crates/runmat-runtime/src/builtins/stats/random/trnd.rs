//! Student's t random variates.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "trnd";

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from the Student's t distribution.",
}];

const INPUT_NU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Degrees of freedom parameter.",
};

const INPUT_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output size arguments.",
};

const INPUTS_NU: [BuiltinParamDescriptor; 1] = [INPUT_NU];
const INPUTS_NU_SZ: [BuiltinParamDescriptor; 2] = [INPUT_NU, INPUT_SZ];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = trnd(nu)",
        inputs: &INPUTS_NU,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = trnd(nu, sz)",
        inputs: &INPUTS_NU_SZ,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = trnd(nu, sz1, sz2, ...)",
        inputs: &INPUTS_NU_SZ,
        outputs: &OUTPUT_R,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:trnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing, malformed, or incompatible.",
    message: "trnd: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRND.INTERNAL",
    identifier: Some("RunMat:trnd:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "trnd: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const TRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn trnd_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn trnd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args {
        [_] => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runtime_builtin(
    name = "trnd",
    category = "stats/random",
    summary = "Generate Student's t random numbers.",
    keywords = "trnd,student t,random,statistics,distribution",
    type_resolver(trnd_type),
    descriptor(crate::builtins::stats::random::trnd::TRND_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::random::trnd"
)]
pub(crate) async fn trnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (nu, shape) = parse_args(args).await?;
    let len = tensor::element_count(&shape);
    let data = random::generate_student_t(&nu.data, len, BUILTIN_NAME)?;
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| trnd_error(&ERROR_INTERNAL, format!("trnd: {err}")))
}

async fn parse_args(args: Vec<Value>) -> BuiltinResult<(Tensor, Vec<usize>)> {
    if args.is_empty() {
        return Err(trnd_error(
            &ERROR_INVALID_ARGUMENT,
            "trnd: nu argument is required",
        ));
    }
    let nu_value = gather_if_needed_async(&args[0])
        .await
        .map_err(|err| trnd_error(&ERROR_INVALID_ARGUMENT, format!("trnd: {err}")))?;
    let nu = tensor::value_into_tensor_for(BUILTIN_NAME, nu_value)
        .map_err(|err| trnd_error(&ERROR_INVALID_ARGUMENT, format!("trnd: {err}")))?;
    let nu = tensor::integer_tensor_to_f64(nu)
        .map_err(|err| trnd_error(&ERROR_INVALID_ARGUMENT, format!("trnd: {err}")))?;
    if nu.data.iter().any(|value| value.is_nan() || *value <= 0.0) {
        return Err(trnd_error(
            &ERROR_INVALID_ARGUMENT,
            "trnd: nu must contain positive degrees of freedom",
        ));
    }

    let shape = if args.len() == 1 {
        normalize_shape(nu.shape.clone())
    } else {
        parse_shape_args(&args[1..]).await?
    };
    if nu.data.len() != 1 && normalize_shape(nu.shape.clone()) != shape {
        return Err(trnd_error(
            &ERROR_INVALID_ARGUMENT,
            "trnd: requested size must match non-scalar nu",
        ));
    }
    Ok((nu, shape))
}

async fn parse_shape_args(rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    let mut dims = Vec::new();
    for arg in rest {
        match extract_dims(arg, BUILTIN_NAME).await {
            Ok(Some(values)) => dims.extend(values),
            Ok(None) => {
                return Err(trnd_error(
                    &ERROR_INVALID_ARGUMENT,
                    format!("trnd: invalid size argument: {arg:?}"),
                ));
            }
            Err(err) => return Err(trnd_error(&ERROR_INVALID_ARGUMENT, err)),
        }
    }
    Ok(normalize_dims(dims))
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape = vec![1, 1];
    } else if shape.len() == 1 {
        shape.push(1);
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        normalize_shape(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn reset() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        let mut tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        tensor.data.fill(f64::NAN);
        tensor
    }

    #[test]
    fn trnd_scalar_is_deterministic_and_finite() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let result = block_on(trnd_builtin(vec![Value::Num(10.0)])).expect("trnd");
        match result {
            Value::Num(value) => assert!(value.is_finite()),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn trnd_accepts_size_forms() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let out = block_on(trnd_builtin(vec![
            Value::Num(5.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ]))
        .expect("trnd");
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let size = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let out = block_on(trnd_builtin(vec![Value::Num(5.0), Value::Tensor(size)]))
            .expect("trnd size vector");
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn trnd_non_scalar_nu_shape_must_match_requested_shape() {
        let nu = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let out = block_on(trnd_builtin(vec![Value::Tensor(nu.clone())])).expect("trnd");
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![3, 1]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let err = block_on(trnd_builtin(vec![
            Value::Tensor(nu),
            Value::Num(1.0),
            Value::Num(3.0),
        ]))
        .expect_err("mismatched shape should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn trnd_reads_typed_integer_nu_and_size_exactly() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let nu = poisoned_int_tensor(IntegerStorage::U16(vec![5, 6, 7]), vec![3, 1]);
        let out = block_on(trnd_builtin(vec![Value::Tensor(nu)])).expect("trnd");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 1]);
                assert!(tensor.data.iter().all(|value| value.is_finite()));
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let nu = poisoned_int_tensor(IntegerStorage::I16(vec![5]), vec![1, 1]);
        let size = poisoned_int_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
        let out =
            block_on(trnd_builtin(vec![Value::Tensor(nu), Value::Tensor(size)])).expect("trnd");
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn trnd_rejects_nonpositive_degrees_of_freedom() {
        let err =
            block_on(trnd_builtin(vec![Value::Num(0.0)])).expect_err("nonpositive nu should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn trnd_distribution_has_heavier_tails_than_normal() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let n = 20_000;
        let out = block_on(trnd_builtin(vec![
            Value::Num(3.0),
            Value::Num(n as f64),
            Value::Num(1.0),
        ]))
        .expect("trnd");
        let data = match out {
            Value::Tensor(tensor) => tensor.data,
            other => panic!("expected tensor, got {other:?}"),
        };
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance =
            data.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / data.len() as f64;
        assert!(mean.abs() < 0.1, "sample mean {mean}");
        assert!((variance - 3.0).abs() < 0.35, "sample variance {variance}");
    }
}
