use super::*;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{CharArray, IntValue};

#[test]
fn non_text_and_all_integer_scalar_classes_reject() {
    let values = [
        IntValue::I8(1),
        IntValue::I16(1),
        IntValue::I32(1),
        IntValue::I64(1),
        IntValue::U8(1),
        IntValue::U16(1),
        IntValue::U32(1),
        IntValue::U64(1),
    ];
    for value in values {
        let error = call(Value::Int(value)).expect_err("integer input");
        assert_eq!(error.identifier(), Some("RunMat:cellstr:InvalidInput"));
    }
}

#[test]
fn cell_contents_must_be_character_vectors_or_string_scalars() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    for value in [
        Value::Num(1.0),
        Value::CharArray(CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap()),
    ] {
        let input = Value::Cell(CellArray::new(vec![value], 1, 1).unwrap());
        let error = call(input).expect_err("invalid cell content");
        assert_eq!(error.identifier(), Some("RunMat:cellstr:InvalidContents"));
    }
}

#[test]
fn resident_values_reject_without_provider_lookup() {
    let resident = Value::GpuTensor(GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    });
    let top_level = call(resident.clone()).expect_err("resident input");
    assert_eq!(top_level.identifier(), Some("RunMat:cellstr:InvalidInput"));

    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let nested = Value::Cell(CellArray::new(vec![resident], 1, 1).unwrap());
    let nested = call(nested).expect_err("nested resident input");
    assert_eq!(nested.identifier(), Some("RunMat:cellstr:InvalidContents"));
    assert!(!nested.message().contains("provider"));
}
