use runmat_accelerate::simple_provider::register_inprocess_provider;
use runmat_builtins::{builtin_functions, Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_runtime::call_builtin;

type BuiltinResult<T> = runmat_runtime::BuiltinResult<T>;

#[runtime_builtin(name = "double", builtin_path = "tests::double_fn")]
fn double_fn(x: i32) -> Result<i32, String> {
    Ok(x * 2)
}

#[runtime_builtin(name = "host_only_trace", builtin_path = "tests::host_only_trace")]
fn host_only_trace(value: Value) -> Result<Value, String> {
    match value {
        Value::Tensor(t) => {
            let sum: f64 = t.data.iter().copied().sum();
            Ok(Value::Num(sum))
        }
        other => Err(format!("host_only_trace: unsupported input {other:?}")),
    }
}

#[runtime_builtin(
    name = "host_only_add_tensors",
    builtin_path = "tests::host_only_add_tensors"
)]
fn host_only_add_tensors(a: Value, b: Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                return Err("host_only_add_tensors: shape mismatch".to_string());
            }
            let data: Vec<f64> = ta
                .data
                .iter()
                .zip(tb.data.iter())
                .map(|(x, y)| x + y)
                .collect();
            let tensor = Tensor::new(data, ta.shape.clone()).map_err(|e| e.to_string())?;
            Ok(Value::Tensor(tensor))
        }
        (lhs, rhs) => Err(format!(
            "host_only_add_tensors: unsupported inputs {lhs:?} and {rhs:?}"
        )),
    }
}

#[test]
fn call_registered_builtin() {
    let result = call_builtin("double", &[Value::Int(runmat_builtins::IntValue::I32(4))]).unwrap();
    if let Value::Int(n) = result {
        assert_eq!(n.to_i64(), 8);
    } else {
        panic!();
    }
    let names: Vec<&str> = builtin_functions().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"double"));
}

#[test]
fn dispatcher_gathers_gpu_argument_for_host_builtin() {
    register_inprocess_provider();

    let cpu_tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let gpu_value = call_builtin("gpuArray", &[Value::Tensor(cpu_tensor.clone())]).unwrap();
    let result = call_builtin("host_only_trace", &[gpu_value]).unwrap();

    match result {
        Value::Num(sum) => assert!((sum - 10.0).abs() < 1e-9),
        other => panic!("expected numeric result, got {other:?}"),
    }
}

#[test]
fn dispatcher_gathers_multiple_gpu_arguments() {
    register_inprocess_provider();

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let ga = call_builtin("gpuArray", &[Value::Tensor(a.clone())]).unwrap();
    let gb = call_builtin("gpuArray", &[Value::Tensor(b.clone())]).unwrap();

    let result = call_builtin("host_only_add_tensors", &[ga, gb]).unwrap();

    match result {
        Value::Tensor(t) => assert_eq!(t.data, vec![6.0, 8.0, 10.0, 12.0]),
        other => panic!("expected tensor result, got {other:?}"),
    }
}
