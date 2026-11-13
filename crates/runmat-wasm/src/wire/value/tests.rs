use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn string_wire_distinguishes_missing_from_every_literal_spelling() {
    let elements = vec![
        "<missing>".into(),
        "<MISSING>".into(),
        "".into(),
        runmat_value::StringElement::Missing,
    ];
    for element in &elements {
        let json = value_to_json(&Value::String(element.clone()), 0);
        assert_eq!(json["value"], serde_json::to_value(element).unwrap());
        assert_eq!(json["shape"], json!([1, 1]));
    }
    let array = runmat_value::StringArray::from_elements_unbounded(elements, vec![1, 4]).unwrap();
    let json = value_to_json(&Value::StringArray(array), 0);
    assert_eq!(json["preview"], json!(["<missing>", "<MISSING>", "", null]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn transient_output_sequence_is_not_projected_as_a_portable_value() {
    let json = value_to_json(&Value::output_list(vec![Value::Num(1.0)]), 0);
    assert_eq!(json["kind"], "unavailable");
    assert_eq!(json["identifier"], "TransientSequenceNotPortable");
    assert!(json.get("items").is_none());
}
use runmat_value::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, ObjectArray, SparseTensor,
    StructArray, Tensor,
};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn character_wire_preserves_utf16_geometry_and_unpaired_units() {
    let value = runmat_value::CharArray::from_code_units(
        vec![
            runmat_value::Utf16CodeUnit::from_raw(0xD83D),
            runmat_value::Utf16CodeUnit::from_raw(0xDE42),
            runmat_value::Utf16CodeUnit::from_raw(0xD800),
        ],
        1,
        3,
    )
    .expect("raw UTF-16 character row");
    let json = value_to_json(&Value::CharArray(value), 0);
    assert_eq!(json["encoding"], "utf16-code-units-v1");
    assert_eq!(json["shape"], json!([1, 3]));
    assert_eq!(json["codeUnits"], json!([0xD83D, 0xDE42, 0xD800]));
    assert!(json["text"].is_null());
}

#[wasm_bindgen_test]
fn integer_json_preserves_exact_64_bit_values_and_dtype() {
    let scalar = value_to_json(&Value::Int(IntValue::U64(u64::MAX)), 0);
    assert_eq!(scalar["value"], "18446744073709551615");

    let tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![42, u64::MAX]), vec![1, 2]).expect("tensor");
    let json = value_to_json(&Value::Tensor(tensor), 0);
    assert_eq!(json["dtype"], "uint64");
    assert_eq!(json["preview"], json!([42, "18446744073709551615"]));
    assert_eq!(json["length"], 2);
}

#[wasm_bindgen_test]
fn native_single_tensor_json_preserves_dtype_and_values() {
    let tensor = Tensor::from_f32(vec![1.25, -3.5], vec![1, 2]).expect("single tensor");
    let json = value_to_json(&Value::Tensor(tensor), 0);

    assert_eq!(json["dtype"], "single");
    assert_eq!(json["preview"], json!([1.25, -3.5]));

    let length = MAX_DATA_PREVIEW + 1;
    for (tensor, class) in [
        (
            Tensor::from_f32(vec![1.25; length], vec![1, length]).unwrap(),
            "single",
        ),
        (
            Tensor::new(vec![1.25; length], vec![1, length]).unwrap(),
            "double",
        ),
    ] {
        let preview = value_to_json(&Value::Tensor(tensor), 0);
        assert_eq!(preview["dtype"], class);
        assert_eq!(preview["length"], length);
        assert_eq!(
            preview["preview"].as_array().unwrap().len(),
            MAX_DATA_PREVIEW
        );
        assert_eq!(preview["preview"][MAX_DATA_PREVIEW - 1], 1.25);
        assert_eq!(preview["truncated"], true);
    }
    assert_eq!(json["length"], 2);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn structure_array_uses_the_v1_display_fallback() {
    let mut element = runmat_value::UnboundedStructValueBuilder::with_capacity(1);
    element.insert("name", Value::from("entry"));
    let element = element.finish();
    let array = StructArray::new(vec![element.clone(), element], vec![1, 2]).unwrap();
    let json = value_to_json(&Value::StructArray(array), 0);

    assert_eq!(json["kind"], "display");
    assert_eq!(json["className"], "struct");
    assert_eq!(json["shape"], json!([1, 2]));
    assert_eq!(json["value"], "1x2 struct array with fields:\n    name");
    assert!(json.get("fields").is_none());
    assert!(json.get("items").is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn floating_complex_json_uses_the_canonical_element_layout() {
    let double = ComplexTensor::new(vec![(1.25, -2.5)], vec![1, 1]).expect("double complex");
    let double_json = value_to_json(&Value::ComplexTensor(double), 0);
    assert_eq!(double_json["dtype"], "double");
    assert_eq!(
        double_json["preview"],
        json!([{"real": 1.25, "imag": -2.5}])
    );

    let single = ComplexTensor::from_f32(vec![(3.5, -4.25)], vec![1, 1]).expect("single complex");
    let single_json = value_to_json(&Value::ComplexTensor(single), 0);
    assert_eq!(single_json["dtype"], "single");
    assert_eq!(
        single_json["preview"],
        json!([{"real": 3.5, "imag": -4.25}])
    );
}

#[wasm_bindgen_test]
fn sparse_tensor_json_uses_bounded_storage_previews() {
    let rows = MAX_DATA_PREVIEW + 2;
    let cols = 1;
    let col_ptrs = vec![0, rows];
    let row_indices = (0..rows).collect::<Vec<_>>();
    let values = (0..rows).map(|idx| idx as f64).collect::<Vec<_>>();
    let sparse = SparseTensor::new(rows, cols, col_ptrs, row_indices, values).unwrap();

    let json = value_to_json(&Value::SparseTensor(sparse), 0);

    assert!(json.get("colPtrs").is_none());
    assert!(json.get("rowIndices").is_none());
    assert!(json.get("values").is_none());
    assert_eq!(json["rowIndicesLength"], rows);
    assert_eq!(json["valuesLength"], rows);
    assert_eq!(
        json["rowIndicesPreview"].as_array().unwrap().len(),
        MAX_DATA_PREVIEW
    );
    assert_eq!(
        json["valuesPreview"].as_array().unwrap().len(),
        MAX_DATA_PREVIEW
    );
    assert_eq!(json["rowIndicesTruncated"], true);
    assert_eq!(json["valuesTruncated"], true);
    assert_eq!(json["truncated"], true);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn object_array_json_preserves_class_shape_and_items() {
    let mut first = runmat_value::UnboundedObjectBuilder::new("matlab.unittest.TestResult");
    first.insert("Passed".into(), Value::Bool(true));
    let mut second = runmat_value::UnboundedObjectBuilder::new("matlab.unittest.TestResult");
    second.insert("Passed".into(), Value::Bool(false));
    let array = ObjectArray::row(
        "matlab.unittest.TestResult",
        vec![
            Value::Object(first.finish()),
            Value::Object(second.finish()),
        ],
    )
    .unwrap();

    let json = value_to_json(&Value::ObjectArray(array), 0);

    assert_eq!(json["kind"], "object-array");
    assert_eq!(json["className"], "matlab.unittest.TestResult");
    assert_eq!(json["shape"], json!([1, 2]));
    assert_eq!(json["items"].as_array().unwrap().len(), 2);
    for (ordinal, displayed) in [(0, "1"), (1, "0")] {
        assert_eq!(
            json["items"][ordinal]["properties"]["Passed"],
            json!({
                "kind": "display", "className": "logical", "shape": [1, 1], "value": displayed
            })
        );
    }
    assert_eq!(json["truncated"], false);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn foreign_json_is_opaque_and_does_not_expose_runtime_handle_identity() {
    let value = Value::Foreign(runmat_value::ForeignRef::detached(
        runmat_value::ForeignResourceKey {
            host_identity: "private-host".into(),
            handle: 42,
            generation: 7,
        },
        runmat_value::ForeignTypeIdentity {
            family: "java".into(),
            name: "java.lang.StringBuilder".into(),
            version: 1,
        },
        runmat_value::ForeignOwnership::Shared,
        runmat_value::ForeignAffinity::OriginProcess,
        runmat_value::ForeignLifetime::Session,
    ));

    let json = value_to_json(&value, 0);
    assert_eq!(json["kind"], "foreign");
    assert_eq!(json["family"], "java");
    assert_eq!(json["typeName"], "java.lang.StringBuilder");
    assert_eq!(json["opaque"], true);
    assert!(json.get("hostIdentity").is_none());
    assert!(json.get("handle").is_none());
    assert!(json.get("generation").is_none());
}

#[wasm_bindgen_test]
fn native_single_sparse_json_preserves_dtype_and_values() {
    let sparse = SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![1, 0], vec![1.25, 3.5])
        .expect("single sparse");
    let json = value_to_json(&Value::SparseTensor(sparse), 0);
    assert_eq!(json["dtype"], "single");
    assert_eq!(json["valuesPreview"], json!([1.25, 3.5]));
    assert_eq!(
        json["preview"],
        json!([
            {"row": 2, "col": 1, "value": 1.25},
            {"row": 1, "col": 2, "value": 3.5}
        ])
    );
}

#[wasm_bindgen_test]
fn typed_complex_and_sparse_json_preserve_exact_64_bit_values() {
    let complex = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX]),
            IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
        )
        .expect("matching components"),
        vec![1, 1],
    )
    .expect("complex tensor");
    let complex_json = value_to_json(&Value::ComplexTensor(complex), 0);
    assert_eq!(complex_json["dtype"], "uint64");
    assert_eq!(
        complex_json["preview"],
        json!([{"real": "18446744073709551615", "imag": "9007199254740993"}])
    );

    let sparse = SparseTensor::new_integer(
        1,
        1,
        vec![0, 1],
        vec![0],
        IntegerStorage::U64(vec![u64::MAX]),
    )
    .expect("sparse tensor");
    let sparse_json = value_to_json(&Value::SparseTensor(sparse), 0);
    assert_eq!(
        sparse_json["valuesPreview"],
        json!(["18446744073709551615"])
    );
    assert_eq!(
        sparse_json["preview"],
        json!([{"row": 1, "col": 1, "value": "18446744073709551615"}])
    );
    assert_eq!(sparse_json["valuesLength"], 1);
}
