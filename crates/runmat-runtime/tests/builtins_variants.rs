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
