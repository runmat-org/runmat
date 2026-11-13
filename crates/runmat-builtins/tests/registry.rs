#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use runmat_value::{CellArray, Tensor, Value};

#[cfg(not(target_arch = "wasm32"))]
#[path = "registry/native_registration.rs"]
mod native_registration;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn test_value_conversions() {
    // Test basic types
    let int_val = Value::Int(runmat_value::IntValue::I32(42));
    let num_val = Value::Num(3.15);
    let bool_val = Value::Bool(true);
    let str_val = Value::string("hello".to_string());

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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn test_matrix_operations() {
    let mut matrix = Tensor::zeros2(2, 3);
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    assert_eq!(matrix.len(), 6);

    // Test setting and getting values (0-based helpers)
    matrix.set2(1, 2, 5.0).unwrap();
    assert_eq!(matrix.get2(1, 2).unwrap(), 5.0);

    // Test bounds checking
    assert!(matrix.get2(2, 0).is_err());
    assert!(matrix.set2(0, 3, 1.0).is_err());

    // Test matrix creation
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let matrix2 = Tensor::new_2d(data, 2, 2).unwrap();
    // Column-major: data = [1,2,3,4] laid out as [[1,3];[2,4]] for 2x2
    assert_eq!(matrix2.get2(0, 1).unwrap(), 3.0);
    assert_eq!(matrix2.get2(1, 1).unwrap(), 4.0);

    // Test invalid dimensions
    assert!(Tensor::new_2d(vec![1.0, 2.0], 2, 2).is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn test_cell_arrays() {
    let cell = Value::Cell(
        CellArray::new(
            vec![
                Value::Int(runmat_value::IntValue::I32(1)),
                Value::string("test".to_string()),
                Value::Bool(false),
            ],
            1,
            3,
        )
        .unwrap(),
    );

    if let Value::Cell(contents) = cell {
        assert_eq!(contents.data().len(), 3);
        assert_eq!(
            &contents.data()[0],
            &Value::Int(runmat_value::IntValue::I32(1))
        );
        assert_eq!(&contents.data()[1], &Value::string("test".to_string()));
        assert_eq!(&contents.data()[2], &Value::Bool(false));
    } else {
        panic!("Expected Cell value");
    }
}
