use std::ptr::NonNull;

use runmat_types::ClassIdentity;

use super::*;

fn test_handle() -> (*mut u8, runmat_gc_api::GcHandle) {
    let raw = Box::into_raw(Box::new(1_u8));
    let handle = unsafe {
        runmat_gc_api::GcHandle::from_ptr_unchecked(NonNull::new(raw.cast()).expect("test pointer"))
    };
    (raw, handle)
}

#[test]
fn value_tracing_reaches_deep_container_handle_without_recursive_calls() {
    let (raw, handle) = test_handle();
    let mut value = Value::HandleObject(HandleRef {
        class_name: ClassIdentity::from("DeepTrace"),
        target: handle,
        valid: true,
    });
    for _ in 0..1024 {
        let mut structure = StructValue::new();
        structure.insert("next", value);
        value = Value::Struct(structure);
    }
    let mut tracer = RecordingTracer(Vec::new());
    value.trace(&mut tracer);
    assert_eq!(tracer.0, vec![handle]);
    loop {
        value = match value {
            Value::Struct(structure) => structure.fields.into_iter().next().unwrap().1,
            other => {
                assert!(matches!(other, Value::HandleObject(_)));
                break;
            }
        };
    }
    unsafe {
        drop(Box::from_raw(raw));
    }
}

#[test]
fn value_tracing_includes_object_native_property_and_metadata_edges() {
    let (native_raw, native) = test_handle();
    let (property_raw, property) = test_handle();
    let (metadata_raw, metadata) = test_handle();
    let handle_value = |target| {
        Value::HandleObject(HandleRef {
            class_name: ClassIdentity::from("TraceEdge"),
            target,
            valid: true,
        })
    };
    let mut storage = ContainersMapStorage::new(
        ContainersMapKeyType::Char,
        ContainersMapValueType::Any,
        false,
    );
    storage.insert_unique(ContainersMapEntry {
        normalized: ContainersMapKey::String("key".into()),
        key_value: Value::String("key".into()),
        value: handle_value(native),
    });
    let mut object = ObjectInstance::new("TraceObject").with_containers_map_payload(storage);
    object
        .properties
        .insert("property".into(), handle_value(property));
    let mut definition = DynamicPropertyDef::new("metadata".into(), "TraceObject".into());
    definition.metadata_handle = Some(metadata);
    object.insert_dynamic_property("metadata".into(), definition);
    let mut tracer = RecordingTracer(Vec::new());
    Value::Object(object).trace(&mut tracer);
    assert_eq!(tracer.0.len(), 3);
    for expected in [native, property, metadata] {
        assert!(tracer.0.contains(&expected));
    }
    unsafe {
        drop(Box::from_raw(native_raw));
        drop(Box::from_raw(property_raw));
        drop(Box::from_raw(metadata_raw));
    }
}

#[test]
fn value_tracing_covers_all_nested_container_edges() {
    let allocated = (0..7).map(|_| test_handle()).collect::<Vec<_>>();
    let handles = allocated
        .iter()
        .map(|(_, handle)| *handle)
        .collect::<Vec<_>>();
    let child = |target| {
        Value::HandleObject(HandleRef {
            class_name: ClassIdentity::from("TraceEdge"),
            target,
            valid: true,
        })
    };
    let cell = Value::Cell(CellArray::new_with_shape(vec![child(handles[0])], vec![1, 1]).unwrap());
    let mut first = StructValue::new();
    first.insert("item", child(handles[1]));
    let mut second = StructValue::new();
    second.insert("item", child(handles[2]));
    let structures = Value::StructArray(
        StructArray::with_fields(vec!["item".into()], vec![first, second], vec![1, 2]).unwrap(),
    );
    let objects = Value::ObjectArray(
        ObjectArray::new(
            "TraceEdge",
            vec![child(handles[3]), child(handles[4])],
            vec![1, 2],
        )
        .unwrap(),
    );
    let closure = Value::Closure(Closure::from_data(crate::ClosureData {
        lexical_environment: None,
        function_name: "trace".into(),
        binding: None,
        source_role: runmat_types::FunctionSourceRole::Unknown,
        source_file: None,
        captures: vec![Value::Listener(Listener {
            id: 1,
            target: handles[5],
            target_class_name: "TraceEdge".into(),
            event_name: "Changed".into(),
            callback: handles[6],
            enabled: true,
            valid: true,
        })],
    }));
    let value = Value::OutputList(vec![cell, structures, objects, closure]);
    let mut tracer = RecordingTracer(Vec::new());
    value.trace(&mut tracer);
    assert_eq!(tracer.0.len(), handles.len());
    for handle in &handles {
        assert!(tracer.0.contains(handle));
    }
    drop(value);
    for (raw, _) in allocated {
        unsafe {
            drop(Box::from_raw(raw));
        }
    }
}

#[test]
fn nested_closure_traces_each_live_ancestor_activation() {
    let (parent_raw, parent_handle) = test_handle();
    let (child_raw, child_handle) = test_handle();
    let parent = LexicalEnvironment::new(parent_handle, None);
    let child = LexicalEnvironment::new(child_handle, Some(parent));
    let closure = Value::Closure(Closure::from_data(crate::ClosureData {
        lexical_environment: Some(child),
        function_name: "nested".into(),
        binding: None,
        source_role: runmat_types::FunctionSourceRole::Nested,
        source_file: None,
        captures: Vec::new(),
    }));
    let mut tracer = RecordingTracer(Vec::new());
    closure.trace(&mut tracer);
    assert_eq!(tracer.0.len(), 2);
    assert!(tracer.0.contains(&parent_handle));
    assert!(tracer.0.contains(&child_handle));
    drop(closure);
    unsafe {
        drop(Box::from_raw(parent_raw));
        drop(Box::from_raw(child_raw));
    }
}

struct RecordingTracer(Vec<runmat_gc_api::GcHandle>);

impl Tracer for RecordingTracer {
    fn mark(&mut self, handle: runmat_gc_api::GcHandle) {
        self.0.push(handle);
    }
}

#[test]
fn sequence_traces_handles_reachable_from_every_output() {
    let first_raw = Box::into_raw(Box::new(1_u8));
    let second_raw = Box::into_raw(Box::new(2_u8));
    let first = unsafe {
        runmat_gc_api::GcHandle::from_ptr_unchecked(
            NonNull::new(first_raw.cast()).expect("test pointer"),
        )
    };
    let second = unsafe {
        runmat_gc_api::GcHandle::from_ptr_unchecked(
            NonNull::new(second_raw.cast()).expect("test pointer"),
        )
    };
    let make_handle = |target| {
        Value::HandleObject(HandleRef {
            class_name: ClassIdentity::from("SequenceTestHandle"),
            target,
            valid: true,
        })
    };
    let sequence =
        ValueSequence::comma_separated(vec![make_handle(first), make_handle(second)]).unwrap();

    let mut tracer = RecordingTracer(Vec::new());
    sequence.trace(&mut tracer);
    assert_eq!(tracer.0, vec![first, second]);

    unsafe {
        drop(Box::from_raw(first_raw));
        drop(Box::from_raw(second_raw));
    }
}
