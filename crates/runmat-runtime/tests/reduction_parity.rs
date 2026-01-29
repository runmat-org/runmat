#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use futures::executor::block_on;
use runmat_builtins::{NumericDType, Tensor, Value};
use runmat_runtime as rt;

fn tensor_from_f32(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data.to_vec(), shape.to_vec()).expect("tensor")
}

fn expect_tensor(value: Value) -> Tensor {
    if let Value::Tensor(t) = value {
        t
    } else {
        panic!("expected tensor result, got {:?}", value);
    }
}

fn column_sum_of_products(a: &[f32], b: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; cols];
    for (col, out_value) in out.iter_mut().enumerate().take(cols) {
        let mut acc = 0.0;
        for row in 0..rows {
            let idx = row + col * rows;
            acc += (a[idx] as f64) * (b[idx] as f64);
        }
        *out_value = acc;
    }
    out
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (a, e) in actual.iter().zip(expected.iter()) {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "value mismatch: got {a} expected {e} (diff {diff})"
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mean_all_native_preserves_single_dtype_and_value() {
    let rows = 2;
    let cols = 3;
    let data: Vec<f32> = vec![0.1, -0.25, 0.5, 1.0, -0.75, 0.33];
    let tensor = tensor_from_f32(&data, &[rows, cols]);
    let args = [
        Value::Tensor(tensor),
        Value::from("all"),
        Value::from("native"),
    ];
    let result = block_on(rt::call_builtin_async("mean", &args)).expect("mean");
    let scalar = expect_tensor(result);
    assert_eq!(
        scalar.shape,
        vec![1, 1],
        "mean('all') should produce scalar"
    );
    assert_eq!(
        scalar.dtype,
        NumericDType::F32,
        "mean(...,'native') should retain single dtype"
    );
    let expected = data.iter().map(|&v| v as f64).sum::<f64>() / (rows * cols) as f64;
    assert_close(&scalar.data, &[expected], 1e-7);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nlms_style_column_reductions_match_reference() {
    let rows = 4;
    let cols = 3;
    // Column-major layout: each group of `rows` entries is one column.
    let x_vals: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.4, // col 0
        -0.5, -0.4, -0.3, -0.2, // col 1
        0.9, 0.7, 0.5, 0.3, // col 2
    ];
    let w_vals: Vec<f32> = vec![
        0.05, 0.1, 0.15, 0.2, // col 0
        0.8, 0.6, 0.4, 0.2, // col 1
        -0.3, -0.2, -0.1, 0.0, // col 2
    ];
    let x_value = Value::Tensor(tensor_from_f32(&x_vals, &[rows, cols]));
    let w_value = Value::Tensor(tensor_from_f32(&w_vals, &[rows, cols]));

    // sum(x .* x, 1, 'native')
    let xx = block_on(rt::call_builtin_async(
        "times",
        &[x_value.clone(), x_value.clone()],
    ))
    .expect("times");
    let sum_xx_args = [xx, Value::Num(1.0), Value::from("native")];
    let sum_xx = block_on(rt::call_builtin_async("sum", &sum_xx_args)).expect("sum");
    let sum_xx_tensor = expect_tensor(sum_xx);
    assert_eq!(
        sum_xx_tensor.shape,
        vec![1, cols],
        "sum along dim=1 should preserve column count"
    );
    assert_eq!(
        sum_xx_tensor.dtype,
        NumericDType::F32,
        "sum(...,'native') should retain single dtype"
    );
    let expected_xx = column_sum_of_products(&x_vals, &x_vals, rows, cols);
    assert_close(&sum_xx_tensor.data, &expected_xx, 1e-6);

    // sum(x .* W, 1, 'native')
    let xw = block_on(rt::call_builtin_async(
        "times",
        &[x_value.clone(), w_value.clone()],
    ))
    .expect("times");
    let sum_xw_args = [xw, Value::Num(1.0), Value::from("native")];
    let sum_xw = block_on(rt::call_builtin_async("sum", &sum_xw_args)).expect("sum");
    let sum_xw_tensor = expect_tensor(sum_xw);
    assert_eq!(sum_xw_tensor.shape, vec![1, cols]);
    assert_eq!(sum_xw_tensor.dtype, NumericDType::F32);
    let expected_xw = column_sum_of_products(&x_vals, &w_vals, rows, cols);
    assert_close(&sum_xw_tensor.data, &expected_xw, 1e-6);
}
