#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use runmat_builtins::Value;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn zeros_ones_variants() {
    let z = runmat_runtime::call_builtin("zeros", &[Value::Num(3.0)]).unwrap();
    let z2 = runmat_runtime::call_builtin("zeros", &[Value::Num(2.0), Value::Num(3.0)]).unwrap();
    let o = runmat_runtime::call_builtin("ones", &[Value::Num(4.0)]).unwrap();
    let o2 = runmat_runtime::call_builtin("ones", &[Value::Num(2.0), Value::Num(2.0)]).unwrap();
    match (z, z2, o, o2) {
        (Value::Tensor(_), Value::Tensor(_), Value::Tensor(_), Value::Tensor(_)) => {}
        _ => panic!(),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sum_prod_mean_any_all_variants() {
    let a = runmat_builtins::Tensor::new(vec![1.0, 0.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let v = Value::Tensor(a);
    for name in ["sum", "prod", "mean", "any", "all"] {
        let _ = runmat_runtime::call_builtin(name, std::slice::from_ref(&v)).unwrap();
        let _ = runmat_runtime::call_builtin(name, &[v.clone(), Value::Num(1.0)]).unwrap();
        let _ = runmat_runtime::call_builtin(name, &[v.clone(), Value::Num(2.0)]).unwrap();
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_min_variants() {
    let a = runmat_builtins::Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let v = Value::Tensor(a);
    let _ = runmat_runtime::call_builtin("max", &[Value::Num(3.0), Value::Num(5.0)]).unwrap();
    let _ = runmat_runtime::call_builtin("min", &[Value::Num(3.0), Value::Num(5.0)]).unwrap();
    let _ = runmat_runtime::call_builtin("max", std::slice::from_ref(&v)).unwrap();
    let _ = runmat_runtime::call_builtin("min", std::slice::from_ref(&v)).unwrap();
    let _ = runmat_runtime::call_builtin("max", &[v.clone(), Value::Num(1.0)]).unwrap();
    let _ = runmat_runtime::call_builtin("min", &[v.clone(), Value::Num(2.0)]).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn find_variants() {
    let a = runmat_builtins::Tensor::new(vec![0.0, 2.0, 0.0, 4.0], vec![2, 2]).unwrap();
    let v = Value::Tensor(a);
    let _ = runmat_runtime::call_builtin("find", std::slice::from_ref(&v)).unwrap();
    let _ = runmat_runtime::call_builtin("find", &[v.clone(), Value::Num(1.0)]).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn degree_radian_conversion_builtins_dispatch() {
    let radians = runmat_runtime::call_builtin("deg2rad", &[Value::Num(90.0)]).unwrap();
    match radians {
        Value::Num(value) => assert!((value - std::f64::consts::FRAC_PI_2).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }

    let degrees =
        runmat_runtime::call_builtin("rad2deg", &[Value::Num(std::f64::consts::PI)]).unwrap();
    match degrees {
        Value::Num(value) => assert!((value - 180.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn degree_trig_builtins_return_matlab_exact_values() {
    let sin_zero = runmat_runtime::call_builtin("sind", &[Value::Num(180.0)]).unwrap();
    match sin_zero {
        Value::Num(value) => assert_eq!(value, 0.0),
        other => panic!("expected scalar result, got {other:?}"),
    }

    let sin_half = runmat_runtime::call_builtin("sind", &[Value::Num(30.0)]).unwrap();
    match sin_half {
        Value::Num(value) => assert_eq!(value, 0.5),
        other => panic!("expected scalar result, got {other:?}"),
    }

    let cos_zero = runmat_runtime::call_builtin("cosd", &[Value::Num(90.0)]).unwrap();
    match cos_zero {
        Value::Num(value) => assert_eq!(value, 0.0),
        other => panic!("expected scalar result, got {other:?}"),
    }

    let tan_one = runmat_runtime::call_builtin("tand", &[Value::Num(45.0)]).unwrap();
    match tan_one {
        Value::Num(value) => assert_eq!(value, 1.0),
        other => panic!("expected scalar result, got {other:?}"),
    }

    let tan_pos_inf = runmat_runtime::call_builtin("tand", &[Value::Num(90.0)]).unwrap();
    match tan_pos_inf {
        Value::Num(value) => assert!(value.is_infinite() && value.is_sign_positive()),
        other => panic!("expected scalar result, got {other:?}"),
    }

    let tan_neg_inf = runmat_runtime::call_builtin("tand", &[Value::Num(-90.0)]).unwrap();
    match tan_neg_inf {
        Value::Num(value) => assert!(value.is_infinite() && value.is_sign_negative()),
        other => panic!("expected scalar result, got {other:?}"),
    }
}
