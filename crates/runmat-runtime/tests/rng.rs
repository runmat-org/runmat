use runmat_builtins::{CharArray, NumericDType, Value};

fn mean_variance(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples
        .iter()
        .map(|x| {
            let delta = x - mean;
            delta * delta
        })
        .sum::<f64>()
        / n;
    (mean, variance)
}

#[test]
fn randn_single_stats_are_unit() {
    let count = 50_000.0;
    let tensor_value = runmat_runtime::call_builtin(
        "randn",
        &[
            Value::Num(count),
            Value::Num(1.0),
            Value::CharArray(CharArray::new_row("single")),
        ],
    )
    .expect("randn single");

    let Value::Tensor(tensor) = tensor_value else {
        panic!("expected tensor result from randn single");
    };
    assert_eq!(tensor.dtype, NumericDType::F32);

    let (mean, variance) = mean_variance(&tensor.data);
    assert!(mean.abs() < 0.02, "mean drift too high for single: {mean}");
    assert!(
        (variance - 1.0).abs() < 0.05,
        "variance drift too high for single: {variance}"
    );
}

#[test]
fn randn_double_stats_are_unit() {
    let count = 50_000.0;
    let tensor_value = runmat_runtime::call_builtin("randn", &[Value::Num(count), Value::Num(1.0)])
        .expect("randn double");

    let Value::Tensor(tensor) = tensor_value else {
        panic!("expected tensor result from randn double");
    };
    assert_eq!(tensor.dtype, NumericDType::F64);

    let (mean, variance) = mean_variance(&tensor.data);
    assert!(mean.abs() < 0.01, "mean drift too high for double: {mean}");
    assert!(
        (variance - 1.0).abs() < 0.02,
        "variance drift too high for double: {variance}"
    );
}
