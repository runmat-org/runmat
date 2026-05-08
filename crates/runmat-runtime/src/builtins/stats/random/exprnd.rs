use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::random;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;
use runmat_builtins::ResolveContext;
use runmat_builtins::Type;

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("exprnd").build()
}

fn exprnd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() <= 1 {
        Type::Num
    } else {
        Type::Unknown
    }
}

#[runtime_builtin(
    name = "exprnd",
    category = "stats/random",
    summary = "Exponentially-distributed random numbers with mean mu.",
    keywords = "exprnd,exponential,random,distribution,statistics",
    type_resolver(exprnd_type),
    builtin_path = "crate::builtins::stats::random::exprnd"
)]
async fn exprnd_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (mu, shape) = parse_args(args).await?;
    if mu <= 0.0 {
        return Err(builtin_error("exprnd: mu must be greater than zero"));
    }
    if let Some(value) = try_gpu_exponential(mu, &shape)? {
        return Ok(value);
    }
    let len = tensor::element_count(&shape);
    let data = random::generate_exponential(mu, len, "exprnd")?;
    let t = Tensor::new(data, shape).map_err(|e| builtin_error(format!("exprnd: {e}")))?;
    Ok(tensor::tensor_into_value(t))
}

async fn parse_args(args: Vec<Value>) -> crate::BuiltinResult<(f64, Vec<usize>)> {
    if args.is_empty() {
        return Err(builtin_error("exprnd: requires at least one argument (mu)"));
    }
    let mu = scalar_f64(&args[0])?;
    let shape = parse_shape_args(&args[1..]).await?;
    Ok((mu, shape))
}

fn scalar_f64(value: &Value) -> crate::BuiltinResult<f64> {
    match value {
        Value::Num(v) => Ok(*v),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(builtin_error(format!(
            "exprnd: expected scalar parameter, got {other:?}"
        ))),
    }
}

async fn parse_shape_args(rest: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims: Vec<usize> = Vec::new();
    for arg in rest {
        match extract_dims(arg, "exprnd").await? {
            Some(d) => dims.extend(d),
            None => {
                return Err(builtin_error(format!(
                    "exprnd: invalid size argument: {arg:?}"
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

fn try_gpu_exponential(mu: f64, shape: &[usize]) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
        return Ok(None);
    }
    match provider.random_exponential(mu, shape) {
        Ok(handle) => {
            let len = tensor::element_count(shape);
            random::skip_uniform(len, "exprnd")?;
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
    fn exprnd_scalar_deterministic() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let result = block_on(exprnd_builtin(vec![Value::Num(2.0)])).expect("exprnd");
        let expected = random::expected_exponential_sequence(2.0, 1)[0];
        match result {
            Value::Num(v) => {
                assert!(v > 0.0);
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn exprnd_matrix_dims() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let args = vec![Value::Num(1.0), Value::Num(3.0), Value::Num(4.0)];
        let result = block_on(exprnd_builtin(args)).expect("exprnd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 4]);
                assert!(t.data.iter().all(|&v| v > 0.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn exprnd_size_vec() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let size = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let args = vec![Value::Num(1.0), Value::Tensor(size)];
        let result = block_on(exprnd_builtin(args)).expect("exprnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn exprnd_rejects_negative_mu() {
        let args = vec![Value::Num(-1.0)];
        assert!(block_on(exprnd_builtin(args)).is_err());
    }

    #[test]
    fn exprnd_rejects_zero_mu() {
        let args = vec![Value::Num(0.0)];
        assert!(block_on(exprnd_builtin(args)).is_err());
    }

    #[test]
    fn exprnd_distribution_mean() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let mu = 3.0_f64;
        let n = 50_000_usize;
        let args = vec![Value::Num(mu), Value::Num(n as f64), Value::Num(1.0)];
        let result = block_on(exprnd_builtin(args)).expect("exprnd");
        let data = match result {
            Value::Tensor(t) => t.data,
            other => panic!("expected tensor, got {other:?}"),
        };
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        assert!(
            (mean - mu).abs() / mu < 0.05,
            "sample mean {mean:.4} not within 5% of mu={mu}"
        );
    }
}
