use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::HostTensorView;
use runmat_types::MemberAccess;
use runmat_value::{CharArray, HandleRef, ObjectInstance, StringArray, Tensor};
use std::collections::HashMap;

#[test]
fn inferred_storage_matches_runtime_cell_storage() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
        StorageFact,
    };
    let (_, _, child) = register_hierarchy("StorageOracle");
    for name in [child, "uint64".to_string()] {
        let input = Value::String(name.clone().into());
        let request = CallRequest {
            arguments: vec![crate::value_fact::value_fact(&input)],
            literals: LiteralContext::new(vec![LiteralValue::String(name)]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let entry = runmat_builtins::builtin_catalog_entry_by_name("superclasses").unwrap();
        let inferred = runmat_builtins::infer_catalog_call(entry, &request);
        let actual = crate::value_fact::value_fact(&superclasses_builtin(input).unwrap());
        assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
        assert_eq!(inferred.outputs[0].storage, actual.storage);
    }
}

fn method(name: &str) -> crate::class_registry::RuntimeMethod {
    crate::class_registry::RuntimeMethod {
        name: name.into(),
        is_static: false,
        is_abstract: false,
        is_sealed: false,
        access: MemberAccess::Public,
        function_name: format!("{name}_impl"),
        implicit_class_argument: None,
        role: runmat_types::ClassMethodRole::Ordinary,
    }
}

fn unique_class_name(label: &str) -> String {
    format!(
        "Superclasses{}{}",
        label,
        std::thread::current().name().unwrap_or("thread")
    )
}

fn register_hierarchy(label: &str) -> (String, String, String) {
    let grand = unique_class_name(&format!("{label}Grand"));
    let parent = unique_class_name(&format!("{label}Parent"));
    let child = unique_class_name(&format!("{label}Child"));

    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: grand.clone().into(),
        parent: None,
        properties: HashMap::new(),
        methods: HashMap::from([("root".into(), method("root"))]),
    });
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: parent.clone().into(),
        parent: Some(grand.clone().into()),
        properties: HashMap::new(),
        methods: HashMap::from([("mid".into(), method("mid"))]),
    });
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: child.clone().into(),
        parent: Some(parent.clone().into()),
        properties: HashMap::new(),
        methods: HashMap::from([("leaf".into(), method("leaf"))]),
    });

    (grand, parent, child)
}

fn cell_strings(value: Value) -> Vec<String> {
    let Value::Cell(cell) = value else {
        panic!("superclasses should return cell");
    };
    cell.data()
        .iter()
        .map(|value| match value {
            Value::CharArray(chars) => {
                runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture")
            }
            other => panic!("expected char superclass name, got {other:?}"),
        })
        .collect()
}

fn call(value: Value) -> Vec<String> {
    cell_strings(superclasses_builtin(value).expect("superclasses"))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn superclasses_returns_parent_chain_for_class_name_text() {
    let (grand, parent, child) = register_hierarchy("ClassName");

    assert_eq!(
        call(Value::String(child.clone().into())),
        vec![parent.clone(), grand.clone()]
    );
    assert_eq!(
        call(Value::CharArray(CharArray::new_row(&child))),
        vec![parent.clone(), grand.clone()]
    );
    assert_eq!(
        call(Value::StringArray(
            StringArray::new(vec![child], vec![1, 1]).expect("string scalar")
        )),
        vec![parent, grand]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn superclasses_returns_parent_chain_for_objects_and_class_refs() {
    let (grand, parent, child) = register_hierarchy("Object");
    let target = runmat_gc::gc_allocate(Value::Object(ObjectInstance::new(child.clone())))
        .expect("gc allocation");

    assert_eq!(
        call(Value::Object(ObjectInstance::new(child.clone()))),
        vec![parent.clone(), grand.clone()]
    );
    assert_eq!(
        call(Value::HandleObject(HandleRef {
            class_name: child.clone().into(),
            target,
            valid: true,
        })),
        vec![parent.clone(), grand.clone()]
    );
    assert_eq!(call(Value::ClassRef(child.into())), vec![parent, grand]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn superclasses_returns_empty_cell_for_leaf_builtin_classes() {
    assert!(call(Value::from("char")).is_empty());
    assert!(call(Value::Object(ObjectInstance::new("double".to_string()))).is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn superclasses_handles_known_runtime_leaf_classes_without_registry_metadata() {
    assert!(call(Value::GpuTensor(test_support::with_test_provider(
        |provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            provider.upload(&view).expect("upload")
        }
    )))
    .is_empty());
    assert!(call(Value::HandleObject(HandleRef {
        class_name: runmat_types::standard::HANDLE.into(),
        target: runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation"),
        valid: true,
    }))
    .is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn superclasses_errors_for_invalid_or_unknown_class_inputs() {
    let invalid = superclasses_builtin(Value::Num(1.0)).expect_err("numeric is invalid");
    assert_eq!(
        invalid.identifier.as_deref(),
        Some("RunMat:SuperclassesClassInvalid")
    );

    let invalid_string_array = superclasses_builtin(Value::StringArray(
        StringArray::new(
            vec!["ClassA".to_string(), "ClassB".to_string()],
            vec![1, 1, 2],
        )
        .expect("string array"),
    ))
    .expect_err("nonscalar string array is invalid");
    assert_eq!(
        invalid_string_array.identifier.as_deref(),
        Some("RunMat:SuperclassesClassInvalid")
    );

    let unknown =
        superclasses_builtin(Value::from("MissingRunMatClass")).expect_err("missing class");
    assert_eq!(
        unknown.identifier.as_deref(),
        Some("RunMat:SuperclassesClassUnknown")
    );
}
