use super::*;

#[test]
fn indexing_context_enumerations_are_registered_in_each_runtime_session() {
    for _ in 0..2 {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _activation = runtime.enter();
        for class in [INDEXING_CONTEXT_CLASS, LEGACY_INDEXING_CONTEXT_CLASS] {
            let reference = block_on(crate::builtins::introspection::classref::classref_builtin(
                class.to_string(),
            ))
            .expect("class reference");
            let Value::ClassRef(identity) = reference else {
                panic!("classref must return a class identity");
            };
            assert_eq!(identity.as_identity(), &class);
            let member = crate::object::resolve::load_static_member(
                identity.as_identity(),
                INDEXING_CONTEXT_STATEMENT,
                None,
            )
            .expect("session-local enum member");
            let Value::Object(value) = member else {
                panic!("enum member must retain its class identity");
            };
            assert_eq!(value.class_name(), &identity);
            assert_eq!(
                value.property(ENUM_MEMBER_PROPERTY),
                Some(&Value::String(INDEXING_CONTEXT_STATEMENT.to_owned().into()))
            );
        }
    }
}
