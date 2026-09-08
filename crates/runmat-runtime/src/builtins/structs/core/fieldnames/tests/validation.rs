use super::run;
use runmat_value::{CellArray, IntValue, Value};

#[test]
fn rejects_non_struct_input() {
    let error = run(Value::Num(1.0)).expect_err("invalid target");
    assert_eq!(
        error.identifier(),
        runmat_builtins::FIELDNAMES_ERROR_INVALID_TARGET.identifier
    );
}

#[test]
fn rejects_non_struct_represented_array_contents() {
    let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell");
    let error = run(Value::Cell(cell)).expect_err("invalid array contents");
    assert_eq!(
        error.identifier(),
        runmat_builtins::FIELDNAMES_ERROR_STRUCT_ARRAY_CONTENTS.identifier
    );
}

#[test]
fn rejects_numeric_and_resident_targets_without_provider_access() {
    assert!(run(Value::Int(IntValue::U64(u64::MAX))).is_err());
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    });
    assert!(run(resident).is_err());
}

#[test]
fn gates_object_extension_before_introspection() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let object = runmat_value::ObjectInstance::new("runmat.unittest.FieldnamesGated");
    let error = run(Value::Object(object)).expect_err("object-family extension");
    assert_eq!(
        error.identifier(),
        runmat_builtins::FIELDNAMES_OBJECT_FAMILY_EXTENSION.error_identifier
    );
}
