use futures::executor::block_on;
use runmat_value::{CharArray, Tensor, Value};

#[test]
fn character_operands_preserve_column_major_order_and_complete_shape() {
    let _mode = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for (shape, data) in [
        (vec![2, 3], runmat_value::encode_utf16_code_units("ADBECF")),
        (vec![1, 1, 3], runmat_value::encode_utf16_code_units("ABC")),
        (vec![2, 0, 3], vec![]),
    ] {
        let chars = CharArray::from_column_major(data.clone(), shape.clone()).unwrap();
        let numeric = Tensor::new(
            data.iter().map(|c| f64::from(c.raw())).collect(),
            shape.clone(),
        )
        .unwrap();
        for name in ["plus", "minus", "times", "ldivide", "rdivide", "power"] {
            for char_on_left in [false, true] {
                let character = Value::CharArray(chars.clone());
                let reference = Value::Tensor(numeric.clone());
                let scalar = Value::Num(2.0);
                let (actual_args, expected_args) = if char_on_left {
                    (vec![character, scalar.clone()], vec![reference, scalar])
                } else {
                    (vec![scalar.clone(), character], vec![scalar, reference])
                };
                let actual =
                    block_on(runmat_runtime::call_builtin_async(name, &actual_args)).unwrap();
                let expected =
                    block_on(runmat_runtime::call_builtin_async(name, &expected_args)).unwrap();
                assert_eq!(
                    actual, expected,
                    "{name}, {shape:?}, char_on_left={char_on_left}"
                );
            }
        }
        for name in ["conj", "real", "imag"] {
            let actual = block_on(runmat_runtime::call_builtin_async(
                name,
                &[Value::CharArray(chars.clone())],
            ))
            .unwrap();
            let expected = block_on(runmat_runtime::call_builtin_async(
                name,
                &[Value::Tensor(numeric.clone())],
            ))
            .unwrap();
            assert_eq!(actual, expected, "{name}, {shape:?}");
        }
    }
}
