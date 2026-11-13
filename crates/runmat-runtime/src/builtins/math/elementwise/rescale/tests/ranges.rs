use runmat_value::Value;

use super::super::rescale_builtin;
use super::support::{assert_close, tensor, values};

#[tokio::test]
async fn rescales_default_and_custom_intervals() {
    let source = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]);
    let (unit, shape, _) = values(rescale_builtin(source.clone(), vec![]).await.expect("unit"));
    assert_eq!(shape, vec![1, 5]);
    assert_close(&unit, &[0.0, 0.25, 0.5, 0.75, 1.0]);

    let (custom, _, _) = values(
        rescale_builtin(source, vec![Value::Num(-1.0), Value::Num(1.0)])
            .await
            .expect("custom"),
    );
    assert_close(&custom, &[-1.0, -0.5, 0.0, 0.5, 1.0]);
}

#[tokio::test]
async fn clips_and_broadcasts_named_input_ranges() {
    let result = rescale_builtin(
        tensor(vec![-30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 70.0], vec![1, 7]),
        vec![
            Value::from("InputMin"),
            Value::Num(1.0),
            Value::from("InputMax"),
            Value::Num(5.0),
        ],
    )
    .await
    .expect("clipped");
    assert_close(&values(result).0, &[0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
}

#[tokio::test]
async fn scales_columns_with_vector_ranges() {
    let result = rescale_builtin(
        tensor(vec![0.4, 0.5, 0.9, 0.2, -4.0, -5.0, 9.0, 1.0], vec![4, 2]),
        vec![
            Value::from("InputMin"),
            tensor(vec![0.2, -5.0], vec![1, 2]),
            Value::from("InputMax"),
            tensor(vec![0.9, 9.0], vec![1, 2]),
        ],
    )
    .await
    .expect("columns");
    let (data, shape, _) = values(result);
    assert_eq!(shape, vec![4, 2]);
    assert_close(
        &data,
        &[
            2.0 / 7.0,
            3.0 / 7.0,
            1.0,
            0.0,
            1.0 / 14.0,
            0.0,
            1.0,
            3.0 / 7.0,
        ],
    );
}

#[tokio::test]
async fn constant_and_nan_ranges_follow_contract() {
    let constant = rescale_builtin(
        tensor(vec![7.0, 7.0], vec![1, 2]),
        vec![Value::Num(-2.0), Value::Num(2.0)],
    )
    .await
    .expect("constant");
    assert_close(&values(constant).0, &[-2.0, -2.0]);

    let mixed = rescale_builtin(tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]), vec![])
        .await
        .expect("mixed NaN");
    assert_close(&values(mixed).0, &[0.0, f64::NAN, 1.0]);

    let all_nan = rescale_builtin(tensor(vec![f64::NAN, f64::NAN], vec![1, 2]), vec![])
        .await
        .expect("all NaN");
    assert!(values(all_nan).0.iter().all(|value| value.is_nan()));

    let explicit_nan = rescale_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        vec![Value::from("InputMin"), Value::Num(f64::NAN)],
    )
    .await
    .expect("explicit NaN");
    assert!(values(explicit_nan).0.iter().all(|value| value.is_nan()));

    let infinite = rescale_builtin(
        tensor(vec![7.0, 7.0], vec![1, 2]),
        vec![Value::Num(0.0), Value::Num(f64::INFINITY)],
    )
    .await
    .expect("infinite");
    assert!(values(infinite).0.iter().all(|value| value.is_nan()));
}
