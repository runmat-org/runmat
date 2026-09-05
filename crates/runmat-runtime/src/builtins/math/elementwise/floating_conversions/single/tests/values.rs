use super::*;

#[test]
fn single_large_ones_preserve_values() {
    // Create a large ones tensor and ensure single() preserves ones exactly.
    let m = 200_000usize;
    let tensor = Tensor::ones(vec![m, 1]);
    let result = single_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, tensor.shape);
            // Sum should equal m exactly.
            let sum: f64 = t.materialize_f64().iter().copied().sum();
            assert!(
                (sum - (m as f64)).abs() < 1e-9,
                "sum expected {m}, got {sum}"
            );
            // All entries must be exactly 1.0
            assert!(t.materialize_f64().iter().all(|&v| v == 1.0));
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}
