#[path = "support/mod.rs"]
mod support;

use runmat_runtime::class_registry::{register_class, RuntimeClass, RuntimeProperty};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_types::MemberAccess;
use runmat_value::{HandleRef, ObjectInstance, Value};
use std::rc::Rc;

#[test]
fn strict_dot_store_updates_existing_handle_without_setfield_extension() {
    runmat_gc::gc_test_context(|| {
        let bytecode = support::compile_source("obj.Value = 9; result = obj.Value;").unwrap();
        let slot = |name: &str| {
            bytecode
                .var_names
                .iter()
                .find_map(|(slot, candidate)| (candidate == name).then_some(*slot))
                .unwrap()
        };
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let handle = futures::executor::block_on(runtime.scope(async {
            runmat_runtime::compatibility::set_runmat_extensions_enabled(false);
            let property = RuntimeProperty {
                name: "Value".into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            };
            register_class(RuntimeClass {
                events: Default::default(),
                name: "StrictDotHandle".into(),
                parent: None,
                properties: [(property.name.clone(), property)].into(),
                methods: Default::default(),
            });
            let mut object = ObjectInstance::new("StrictDotHandle");
            object.properties.insert("Value".into(), Value::Num(1.0));
            object
                .properties
                .insert("UnrelatedNaN".into(), Value::Num(f64::NAN));
            HandleRef {
                class_name: "StrictDotHandle".into(),
                target: runmat_gc::gc_allocate(Value::Object(object)).unwrap(),
                valid: true,
            }
        }));
        let mut variables = vec![
            Value::Num(0.0);
            bytecode
                .var_names
                .keys()
                .map(|slot| *slot + 1)
                .max()
                .unwrap()
        ];
        variables[slot("obj")] = Value::HandleObject(handle.clone());
        let result = futures::executor::block_on(runmat_vm::interpret_with_vars_in_context(
            &bytecode,
            &mut variables,
            None,
            runtime,
        ))
        .unwrap();
        let runmat_vm::InterpreterOutcome::Completed(completion) = result;
        assert_eq!(completion.values[slot("result")], Value::Num(9.0));
        let Value::HandleObject(updated) = &completion.values[slot("obj")] else {
            panic!("handle result")
        };
        assert_eq!(updated.target, handle.target);
        runmat_gc::gc_with_value(&handle.target, |value| {
            let Value::Object(object) = value else {
                panic!("object target")
            };
            assert_eq!(object.properties["Value"], Value::Num(9.0));
            assert!(
                matches!(object.properties["UnrelatedNaN"], Value::Num(value) if value.is_nan())
            );
        })
        .unwrap();
    });
}
