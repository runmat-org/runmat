use rustmat_builtins::{builtin_functions, Matrix, Value};
use rustmat_macros::runtime_builtin;

#[runtime_builtin(name = "add")]
fn add(x: i32, y: i32) -> Result<i32, String> {
    Ok(x + y)
}

#[runtime_builtin(name = "sub")]
fn sub(x: i32, y: i32) -> Result<i32, String> {
    Ok(x - y)
}

#[runtime_builtin(name = "matrix_sum")]
fn matrix_sum(m: Matrix) -> Result<f64, String> {
    Ok(m.data.iter().sum())
}

#[runtime_builtin(name = "str_length")]
fn str_length(s: String) -> Result<i32, String> {
    Ok(s.len() as i32)
}

#[test]
fn contains_registered_functions() {
    let names: Vec<&str> = builtin_functions().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"add"));
    assert!(names.contains(&"sub"));
    assert!(names.contains(&"matrix_sum"));
    assert!(names.contains(&"str_length"));
}

#[test]
fn test_value_conversions() {
    // Test basic types
    let int_val = Value::Int(42);
    let num_val = Value::Num(3.15);
    let bool_val = Value::Bool(true);
    let str_val = Value::String("hello".to_string());

    // Test From implementations
    assert_eq!(Value::from(42), int_val);
    assert_eq!(Value::from(3.15), num_val);
    assert_eq!(Value::from(true), bool_val);
    assert_eq!(Value::from("hello"), str_val);

    // Test TryFrom implementations
    use std::convert::TryInto;
    assert_eq!((&int_val).try_into(), Ok(42i32));
    assert_eq!((&num_val).try_into(), Ok(3.15f64));
    assert_eq!((&bool_val).try_into(), Ok(true));
    assert_eq!((&str_val).try_into(), Ok("hello".to_string()));
}

#[test]
fn test_matrix_operations() {
    let mut matrix = Matrix::zeros(2, 3);
    assert_eq!(matrix.rows, 2);
    assert_eq!(matrix.cols, 3);
    assert_eq!(matrix.data.len(), 6);

    // Test setting and getting values
    matrix.set(1, 2, 5.0).unwrap();
    assert_eq!(matrix.get(1, 2).unwrap(), 5.0);

    // Test bounds checking
    assert!(matrix.get(2, 0).is_err());
    assert!(matrix.set(0, 3, 1.0).is_err());

    // Test matrix creation
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let matrix2 = Matrix::new(data, 2, 2).unwrap();
    assert_eq!(matrix2.get(0, 1).unwrap(), 2.0);
    assert_eq!(matrix2.get(1, 1).unwrap(), 4.0);

    // Test invalid dimensions
    assert!(Matrix::new(vec![1.0, 2.0], 2, 2).is_err());
}

#[test]
fn test_cell_arrays() {
    let cell = Value::Cell(vec![
        Value::Int(1),
        Value::String("test".to_string()),
        Value::Bool(false),
    ]);

    if let Value::Cell(contents) = cell {
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0], Value::Int(1));
        assert_eq!(contents[1], Value::String("test".to_string()));
        assert_eq!(contents[2], Value::Bool(false));
    } else {
        panic!("Expected Cell value");
    }
}
