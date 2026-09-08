use runmat_types::MemberAccess;
use std::collections::HashMap;

pub(super) fn register_class(
    name: &str,
    parent: Option<&str>,
    instance_properties: &[&str],
    static_properties: &[&str],
) {
    let mut class = crate::class_registry::RuntimeClass {
        name: name.into(),
        parent: parent.map(Into::into),
        properties: HashMap::new(),
        methods: HashMap::new(),
    };
    for property in instance_properties {
        class
            .properties
            .insert((*property).into(), runtime_property(property, false));
    }
    for property in static_properties {
        class
            .properties
            .insert((*property).into(), runtime_property(property, true));
    }
    crate::class_registry::register_class(class);
}

fn runtime_property(name: &str, is_static: bool) -> crate::class_registry::RuntimeProperty {
    crate::class_registry::RuntimeProperty {
        name: name.into(),
        is_static,
        is_constant: false,
        is_dependent: false,
        get_access: MemberAccess::Public,
        set_access: MemberAccess::Public,
        default_value: None,
    }
}
