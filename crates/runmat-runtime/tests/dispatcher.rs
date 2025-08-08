use runmat_builtins::{builtin_functions, Value};
use runmat_macros::runtime_builtin;
use runmat_runtime::call_builtin;

#[runtime_builtin(name = "double")]
fn double_fn(x: i32) -> Result<i32, String> {
    Ok(x * 2)
}

#[test]
fn call_registered_builtin() {
    let result = call_builtin("double", &[Value::Int(4)]).unwrap();
    if let Value::Int(n) = result {
        assert_eq!(n, 8);
    } else {
        panic!();
    }
    let names: Vec<&str> = builtin_functions().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"double"));
}
