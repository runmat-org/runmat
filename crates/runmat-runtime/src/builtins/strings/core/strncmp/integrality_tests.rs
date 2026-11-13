use super::*;
use runmat_value::Tensor;

fn compare(case_insensitive: bool, limit: Value) -> BuiltinResult<Value> {
    let left = Value::String("ab".into());
    let right = Value::String(if case_insensitive { "AB" } else { "ab" }.into());
    futures::executor::block_on(async {
        if case_insensitive {
            crate::builtins::strings::core::strncmpi::strncmpi_builtin(
                None, None, left, right, limit,
            )
            .await
        } else {
            strncmp_builtin(None, None, left, right, limit).await
        }
    })
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn source_known_float_boundary_matches_both_runtime_prefix_routes() {
    for case_insensitive in [false, true] {
        assert_eq!(
            compare(case_insensitive, Value::Num(1.0)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            compare(case_insensitive, Value::Num(-1.0)).unwrap(),
            Value::Bool(true)
        );
        for bad in [1.0 + f64::EPSILON, -0.5, f64::NAN, f64::INFINITY, 1e100] {
            let error = compare(case_insensitive, Value::Num(bad)).expect_err("invalid N");
            if case_insensitive {
                assert!(error.to_string().contains("integer scalar count"));
            } else {
                assert_eq!(
                    error.identifier(),
                    STRNCMP_ERROR_INVALID_PREFIX_LENGTH.identifier
                );
            }
        }
        let single_fraction = Tensor::from_f32(vec![1.0000001], vec![1, 1]).unwrap();
        assert!(compare(case_insensitive, Value::Tensor(single_fraction)).is_err());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn direct_numeric_text_has_priority_over_invalid_count_in_both_routes() {
    futures::executor::block_on(async {
        let a = Value::Num(3.0);
        let b = Value::String("ab".into());
        let n = Value::Num(f64::NAN);
        assert_eq!(
            strncmp_builtin(None, None, a.clone(), b.clone(), n.clone())
                .await
                .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            crate::builtins::strings::core::strncmpi::strncmpi_builtin(None, None, a, b, n)
                .await
                .unwrap(),
            Value::Bool(false)
        );
    });
}
