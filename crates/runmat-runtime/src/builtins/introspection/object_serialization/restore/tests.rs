use super::*;
use futures::executor::block_on;
use runmat_value::StructArray;

fn nested_structs(depth: usize) -> Value {
    let mut value = Value::Int(runmat_value::IntValue::U64(u64::MAX));
    for _ in 0..depth {
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("next", value);
        value = Value::Struct(structure.finish());
    }
    value
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn maximal_valid_restoration_depth_preserves_exact_integer_without_recursive_polling() {
    let restored = block_on(restore_value_after_load(nested_structs(
        MAX_SERIALIZATION_DEPTH,
    )))
    .expect("maximum valid nesting");
    let mut observed = restored.value();
    for _ in 0..MAX_SERIALIZATION_DEPTH {
        let Value::Struct(structure) = observed else {
            panic!("expected nested structure");
        };
        observed = structure.fields.get("next").expect("next field");
    }
    assert_eq!(observed, &Value::Int(runmat_value::IntValue::U64(u64::MAX)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn overdepth_restoration_reports_canonical_recursion_error_on_default_stack() {
    let error = block_on(restore_value_after_load(nested_structs(
        MAX_SERIALIZATION_DEPTH + 1,
    )))
    .expect_err("over-depth restoration");
    assert_eq!(error.identifier(), SERIALIZATION_ERROR_RECURSION.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn structure_array_restoration_visits_field_major_values() {
    let array = StructArray::with_fields(
        vec!["first".into(), "second".into()],
        vec![
            {
                let mut element = runmat_value::UnboundedStructValueBuilder::new();
                element.insert("first", Value::Num(1.0));
                element.insert("second", Value::Num(2.0));
                element.finish()
            },
            {
                let mut element = runmat_value::UnboundedStructValueBuilder::new();
                element.insert("first", nested_structs(MAX_SERIALIZATION_DEPTH + 1));
                element.insert("second", Value::Num(2.0));
                element.finish()
            },
        ],
        vec![1, 2],
    )
    .expect("non-scalar structure array");
    let error = block_on(restore_value_after_load(Value::StructArray(array)))
        .expect_err("first column's second element must exceed depth");
    assert_eq!(error.identifier(), SERIALIZATION_ERROR_RECURSION.identifier);
}
