use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::random;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;
use runmat_builtins::ResolveContext;
use runmat_builtins::Type;

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("unifrnd").build()
}

fn unifrnd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() <= 2 {
        Type::Num
    } else {
        Type::Unknown
    }
}

#[runtime_builtin(
    name = "unifrnd",
    category = "stats/random",
    summary = "Uniformly-distributed random numbers on the interval [a, b).",
    keywords = "unifrnd,uniform,random,distribution,statistics",
    type_resolver(unifrnd_type),
    builtin_path = "crate::builtins::stats::random::unifrnd"
)]
async fn unifrnd_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (a, b, shape) = parse_args(args).await?;
    if a >= b {
        return Err(builtin_error("unifrnd: a must be less than b"));
    }
    if let Some(value) = try_gpu_unifrnd(a, b, &shape)? {
        return Ok(value);
    }
    let len = tensor::element_count(&shape);
    let data = random::generate_uniform_scaled(a, b, len, "unifrnd")?;
    let t = Tensor::new(data, shape).map_err(|e| builtin_error(format!("unifrnd: {e}")))?;
    Ok(tensor::tensor_into_value(t))
}

async fn parse_args(args: Vec<Value>) -> crate::BuiltinResult<(f64, f64, Vec<usize>)> {
    if args.len() < 2 {
        return Err(builtin_error(
            "unifrnd: requires at least two arguments (a, b)",
        ));
    }
    let a = scalar_f64(&args[0])?;
    let b = scalar_f64(&args[1])?;
    let shape = parse_shape_args(&args[2..]).await?;
    Ok((a, b, shape))
}

fn scalar_f64(value: &Value) -> crate::BuiltinResult<f64> {
    match value {
        Value::Num(v) => Ok(*v),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(builtin_error(format!(
            "unifrnd: expected scalar parameter, got {other:?}"
        ))),
    }
}

async fn parse_shape_args(rest: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims: Vec<usize> = Vec::new();
    for arg in rest {
        match extract_dims(arg, "unifrnd").await? {
            Some(d) => dims.extend(d),
            None => {
                return Err(builtin_error(format!(
                    "unifrnd: invalid size argument: {arg:?}"
                )))
            }
        }
    }
    Ok(normalize_dims(dims))
}

fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        dims
    }
}

fn try_gpu_unifrnd(a: f64, b: f64, shape: &[usize]) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
        return Ok(None);
    }
    match provider.random_unifrnd(a, b, shape) {
        Ok(handle) => {
            let len = tensor::element_count(shape);
            random::skip_uniform(len, "unifrnd")?;
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;
    use futures::executor::block_on;

    fn reset() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    #[test]
    fn unifrnd_scalar_deterministic() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let result =
            block_on(unifrnd_builtin(vec![Value::Num(2.0), Value::Num(5.0)])).expect("unifrnd");
        let expected = random::expected_uniform_scaled_sequence(2.0, 5.0, 1)[0];
        match result {
            Value::Num(v) => {
                assert!((2.0..5.0).contains(&v));
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn unifrnd_matrix_dims() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let args = vec![
            Value::Num(0.0),
            Value::Num(10.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let result = block_on(unifrnd_builtin(args)).expect("unifrnd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 4]);
                assert!(t.data.iter().all(|&v| (0.0..10.0).contains(&v)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unifrnd_size_vec() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let size = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let args = vec![Value::Num(0.0), Value::Num(1.0), Value::Tensor(size)];
        let result = block_on(unifrnd_builtin(args)).expect("unifrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unifrnd_rejects_a_ge_b() {
        let args = vec![Value::Num(5.0), Value::Num(2.0)];
        assert!(block_on(unifrnd_builtin(args)).is_err());
    }

    #[test]
    fn unifrnd_rejects_a_eq_b() {
        let args = vec![Value::Num(3.0), Value::Num(3.0)];
        assert!(block_on(unifrnd_builtin(args)).is_err());
    }

    #[test]
    fn unifrnd_distribution_bounds() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let a = 2.0_f64;
        let b = 7.0_f64;
        let n = 50_000_usize;
        let args = vec![
            Value::Num(a),
            Value::Num(b),
            Value::Num(n as f64),
            Value::Num(1.0),
        ];
        let result = block_on(unifrnd_builtin(args)).expect("unifrnd");
        let data = match result {
            Value::Tensor(t) => t.data,
            other => panic!("expected tensor, got {other:?}"),
        };
        assert!(
            data.iter().all(|&v| v >= a && v < b),
            "some values outside [{a}, {b})"
        );
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let expected_mean = (a + b) / 2.0;
        assert!(
            (mean - expected_mean).abs() / (b - a) < 0.05,
            "sample mean {mean:.4} not within 5% of expected {expected_mean:.4}"
        );
    }
}
