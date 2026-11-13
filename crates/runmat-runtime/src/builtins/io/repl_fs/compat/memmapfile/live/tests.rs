use super::*;
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use runmat_filesystem::{self as vfs, MemoryFsProvider};
use runmat_value::{IntValue, IntegerStorage, Tensor};
use std::sync::Arc;

#[path = "tests/repeat_effects.rs"]
mod repeat_effects;

fn property(name: &str) -> Value {
    ObjectSubscriptPath::single(ObjectSubscript::member(name))
        .to_standard_substruct_value()
        .expect("typed property path")
}

fn indexed_data(index: usize) -> Value {
    ObjectSubscriptPath::new(vec![
        ObjectSubscript::member("Data"),
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                ObjectIndexComponent::Value(Value::Num(index as f64)),
            ]),
        }),
    ])
    .unwrap()
    .to_standard_substruct_value()
    .unwrap()
}

fn indexed_field(record: usize, field: &str, index: usize) -> Value {
    ObjectSubscriptPath::new(vec![
        ObjectSubscript::member("Data"),
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                ObjectIndexComponent::Value(Value::Num(record as f64)),
            ]),
        }),
        ObjectSubscript::member(field),
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                ObjectIndexComponent::Value(Value::Num(index as f64)),
            ]),
        }),
    ])
    .unwrap()
    .to_standard_substruct_value()
    .unwrap()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn method_output_bounds_precede_path_parse_and_mapped_effects() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let invalid_path = Value::Num(42.0);
    let reference =
        futures::executor::block_on(subsref(None, Value::Num(0.0), invalid_path.clone()))
            .unwrap_err();
    assert_eq!(
        reference.identifier(),
        Some("RunMat:memmapfile:TooManyOutputs")
    );
    let assignment = futures::executor::block_on(subsasgn(
        None,
        Value::Num(0.0),
        invalid_path,
        Value::Num(1.0),
    ))
    .unwrap_err();
    assert_eq!(
        assignment.identifier(),
        Some("RunMat:memmapfile:TooManyOutputs")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn explicit_gpu_method_controls_are_gated_before_path_or_receiver_work() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap()
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let reference = futures::executor::block_on(subsref(
        None,
        Value::Num(0.0),
        Value::GpuTensor(handle.clone()),
    ))
    .unwrap_err();
    assert_eq!(
        reference.identifier(),
        MEMMAPFILE_EXPLICIT_GPU_EXTENSION.error_identifier
    );
    let assignment = futures::executor::block_on(subsasgn(
        None,
        Value::Num(0.0),
        property("Data"),
        Value::GpuTensor(handle),
    ))
    .unwrap_err();
    assert_eq!(
        assignment.identifier(),
        MEMMAPFILE_EXPLICIT_GPU_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mapped_data_is_live_and_indexed_write_stays_on_original_provider() {
    let _lock = vfs::provider_override_lock();
    let root = std::path::PathBuf::from("/memmap-live");
    let selected = Arc::new(MemoryFsProvider::with_current_dir(root.clone()));
    selected
        .write_project_path(root.join("data.bin"), b"\x01\x02\x03\x04")
        .unwrap();
    let _selected = vfs::replace_provider(selected.clone());
    vfs::set_current_dir(&root).unwrap();
    let mapped = futures::executor::block_on(super::super::super::memmapfile_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("data.bin".into()),
            Value::String("Writable".into()),
            Value::Bool(true),
        ]),
    ))
    .expect("live memmap receiver");
    let Value::Object(object) = &mapped else {
        panic!("object receiver")
    };
    assert!(object.memmap_receiver().is_some());
    assert!(!object.properties().contains_key("Data"));
    let alias =
        crate::execution::RootedValue::new(mapped.clone()).expect("root mapped receiver alias");
    let replacement = Arc::new(MemoryFsProvider::with_current_dir(root.join("replacement")));
    replacement
        .write_project_path(root.join("replacement/data.bin"), b"xxxx")
        .unwrap();
    let _replacement = vfs::replace_provider(replacement.clone());
    vfs::set_current_dir(root.join("replacement")).unwrap();

    let updated = futures::executor::block_on(subsasgn(
        None,
        mapped,
        indexed_data(2),
        Value::Int(IntValue::U8(9)),
    ))
    .expect("indexed mapped write");
    assert_eq!(
        selected.read_project_path(root.join("data.bin")).unwrap(),
        b"\x01\x09\x03\x04"
    );
    assert_eq!(
        replacement
            .read_project_path(root.join("replacement/data.bin"))
            .unwrap(),
        b"xxxx"
    );
    for receiver in [alias, updated] {
        let data = receiver
            .with_owned(|receiver| {
                futures::executor::block_on(subsref(None, receiver, property("Data")))
            })
            .expect("live read through original owner");
        let Value::Tensor(data) = data.value() else {
            panic!("numeric Data")
        };
        assert_eq!(
            data.integer_storage(),
            Some(&IntegerStorage::U8(vec![1, 9, 3, 4]))
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mapped_receiver_controls_clone_independently_without_changing_backing() {
    let _lock = vfs::provider_override_lock();
    let root = std::path::PathBuf::from("/memmap-clones");
    let provider = Arc::new(MemoryFsProvider::with_current_dir(root.clone()));
    provider
        .write_project_path(root.join("data.bin"), b"a")
        .unwrap();
    let _selected = vfs::replace_provider(provider);
    vfs::set_current_dir(&root).unwrap();
    let original = futures::executor::block_on(super::super::super::memmapfile_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("data.bin".into()),
            Value::String("Writable".into()),
            Value::Bool(true),
            Value::String("Repeat".into()),
            Value::Num(f64::INFINITY),
        ]),
    ))
    .unwrap();
    assert_eq!(
        futures::executor::block_on(subsref(None, original.clone(), property("Repeat"))).unwrap(),
        Value::Num(f64::INFINITY)
    );
    let finite = futures::executor::block_on(subsasgn(
        None,
        original.clone(),
        property("Repeat"),
        Value::Num(1.0),
    ))
    .unwrap();
    assert_eq!(
        futures::executor::block_on(subsref(None, finite.value().clone(), property("Repeat")))
            .unwrap(),
        Value::Num(1.0)
    );
    let unlimited = futures::executor::block_on(subsasgn(
        None,
        finite.value().clone(),
        property("Repeat"),
        Value::Num(f64::INFINITY),
    ))
    .unwrap();
    assert_eq!(
        unlimited
            .with_owned(|receiver| {
                futures::executor::block_on(subsref(None, receiver, property("Repeat")))
            })
            .unwrap(),
        Value::Num(f64::INFINITY)
    );
    let read_only_clone = futures::executor::block_on(subsasgn(
        None,
        original.clone(),
        property("Writable"),
        Value::Bool(false),
    ))
    .unwrap();
    assert_eq!(
        futures::executor::block_on(subsref(None, original, property("Writable"))).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        futures::executor::block_on(subsref(
            None,
            read_only_clone.value().clone(),
            property("Writable")
        ))
        .unwrap(),
        Value::Bool(false)
    );
    let rhs = Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![7]), vec![1, 1]).unwrap());
    let error = read_only_clone
        .with_owned(|receiver| {
            futures::executor::block_on(subsasgn(None, receiver, property("Data"), rhs))
        })
        .unwrap_err();
    assert!(error.message().contains("read-only"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn indexed_read_observes_only_current_selected_byte_and_forged_class_has_no_authority() {
    let _lock = vfs::provider_override_lock();
    let root = std::path::PathBuf::from("/memmap-selected-read");
    let provider = Arc::new(MemoryFsProvider::with_current_dir(root.clone()));
    provider
        .write_project_path(root.join("data.bin"), b"\x01\x02\x03\x04")
        .unwrap();
    let _selected = vfs::replace_provider(provider.clone());
    vfs::set_current_dir(&root).unwrap();
    let mapped = futures::executor::block_on(super::super::super::memmapfile_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("data.bin".into()),
            Value::String("Writable".into()),
            Value::Bool(true),
        ]),
    ))
    .unwrap();
    let alias = mapped.clone();
    futures::executor::block_on(subsasgn(
        None,
        alias,
        indexed_data(2),
        Value::Int(IntValue::U8(9)),
    ))
    .unwrap();
    let selected = futures::executor::block_on(subsref(None, mapped, indexed_data(2))).unwrap();
    let Value::Tensor(selected) = selected.value() else {
        panic!("selected tensor")
    };
    assert_eq!(
        selected.integer_storage(),
        Some(&IntegerStorage::U8(vec![9]))
    );

    let forged = Value::Object(runmat_value::ObjectInstance::new(MEMMAPFILE_CLASS.owned()));
    let error = futures::executor::block_on(subsref(None, forged, property("Data"))).unwrap_err();
    assert!(error.message().contains("sealed state"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn indexed_named_field_write_preserves_other_fields_and_elements() {
    let _lock = vfs::provider_override_lock();
    let root = std::path::PathBuf::from("/memmap-field-range");
    let provider = Arc::new(MemoryFsProvider::with_current_dir(root.clone()));
    provider
        .write_project_path(root.join("record.bin"), b"\x01\x02\x03\x04")
        .unwrap();
    let _selected = vfs::replace_provider(provider.clone());
    vfs::set_current_dir(&root).unwrap();
    let format = super::super::MemmapFormat {
        dtype: super::super::MemmapDataType::Numeric(runmat_value::NumericDType::U8),
        shape: vec![1, 2],
        field: Some("first".into()),
        additional_fields: vec![super::super::MemmapField {
            dtype: super::super::MemmapDataType::Numeric(runmat_value::NumericDType::U8),
            shape: vec![1, 2],
            name: "second".into(),
        }],
        presentation: super::super::MemmapFormatPresentation::Cell { columns: 3 },
    }
    .to_value()
    .unwrap();
    let mapped = futures::executor::block_on(super::super::super::memmapfile_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("record.bin".into()),
            Value::String("Format".into()),
            format,
            Value::String("Writable".into()),
            Value::Bool(true),
        ]),
    ))
    .unwrap();
    let selected = indexed_field(1, "second", 1);
    for invalid in [indexed_field(1, "first", 3), indexed_field(2, "first", 1)] {
        let error = futures::executor::block_on(subsasgn(
            None,
            mapped.clone(),
            invalid,
            Value::Int(IntValue::U8(99)),
        ))
        .expect_err("fixed mapped field and record cannot grow");
        assert!(error.message().contains("index") || error.message().contains("Index"));
        assert_eq!(
            provider.read_project_path(root.join("record.bin")).unwrap(),
            b"\x01\x02\x03\x04",
            "invalid mapped selection must not change adjacent bytes"
        );
    }
    let updated = futures::executor::block_on(subsasgn(
        None,
        mapped,
        selected.clone(),
        Value::Int(IntValue::U8(9)),
    ))
    .unwrap();
    assert_eq!(
        provider.read_project_path(root.join("record.bin")).unwrap(),
        b"\x01\x02\x09\x04"
    );
    let field = updated
        .with_owned(|receiver| futures::executor::block_on(subsref(None, receiver, selected)))
        .unwrap();
    let Value::Tensor(field) = field.value() else {
        panic!("selected field tensor")
    };
    assert_eq!(field.integer_storage(), Some(&IntegerStorage::U8(vec![9])));
}
