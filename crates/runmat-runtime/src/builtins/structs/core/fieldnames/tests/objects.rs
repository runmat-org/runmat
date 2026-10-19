use super::{run, strings, support::register_class};
use runmat_value::{HandleRef, Listener, ObjectInstance, Value};

#[test]
fn object_includes_inherited_and_dynamic_instance_properties() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let parent = "runmat.unittest.FieldnamesObjectParent";
    let child = "runmat.unittest.FieldnamesObjectChild";
    register_class(parent, None, &["ParentValue"], &[]);
    register_class(child, Some(parent), &["ChildValue"], &["Version"]);
    let mut object = ObjectInstance::new(child);
    object.properties.insert("Step".into(), Value::Num(2.0));
    let (names, _) = strings(run(Value::Object(object)).expect("fieldnames object"));
    assert_eq!(names, ["ChildValue", "ParentValue", "Step"]);
}

#[test]
fn handle_merges_inherited_class_and_target_names() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let parent = "runmat.unittest.FieldnamesHandleParent";
    let child = "runmat.unittest.FieldnamesHandleChild";
    register_class(parent, None, &["ParentEnabled"], &[]);
    register_class(child, Some(parent), &["ChildEnabled"], &[]);
    let mut payload = ObjectInstance::new(child);
    payload
        .properties
        .insert("Status".into(), Value::from("ready"));
    let target = runmat_gc::gc_allocate(Value::Object(payload)).expect("target");
    let handle = HandleRef {
        class_name: child.into(),
        target,
        valid: true,
    };
    let (names, _) = strings(run(Value::HandleObject(handle)).expect("fieldnames handle"));
    assert_eq!(names, ["ChildEnabled", "ParentEnabled", "Status"]);
}

#[test]
fn listener_exposes_stable_public_metadata_names() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let listener = Listener {
        id: 1,
        target: runmat_gc::gc_allocate(Value::Num(1.0)).expect("target"),
        target_class_name: "runmat.unittest.EventSource".into(),
        event_name: "event".into(),
        callback: runmat_gc::gc_allocate(Value::Num(2.0)).expect("callback"),
        enabled: true,
        valid: true,
    };
    let (names, _) = strings(run(Value::Listener(listener)).expect("fieldnames listener"));
    assert_eq!(
        names,
        ["callback", "enabled", "event_name", "id", "target", "valid"]
    );
}
