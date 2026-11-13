//! Builders for host-defined value classes (FEA, geometry, meshing objects).
//! Every such class has public read-only properties and named methods bound to
//! builtin functions; these helpers keep that shape in one place.

use std::collections::HashMap;

use runmat_types::{ClassIdentity, ClassMethodRole, MemberAccess, MemberName, MethodName};

use super::{register_class, RuntimeClass, RuntimeMethod, RuntimeProperty};

/// Public, read-only, non-dependent instance properties.
pub fn immutable_properties(
    fields: impl IntoIterator<Item = impl AsRef<str>>,
) -> HashMap<MemberName, RuntimeProperty> {
    fields
        .into_iter()
        .map(|field| immutable_property(field.as_ref(), MemberAccess::Public))
        .collect()
}

/// One read-only, non-dependent instance property readable with `get_access`.
pub fn immutable_property(name: &str, get_access: MemberAccess) -> (MemberName, RuntimeProperty) {
    (
        name.into(),
        RuntimeProperty {
            name: name.into(),
            is_static: false,
            is_constant: true,
            is_dependent: false,
            get_access,
            set_access: MemberAccess::Private,
            default_value: None,
        },
    )
}

/// One public ordinary method bound to `function_name`.
pub fn method(name: &str, function_name: &str, is_static: bool) -> (MethodName, RuntimeMethod) {
    (
        name.into(),
        RuntimeMethod {
            name: name.into(),
            role: ClassMethodRole::Ordinary,
            is_static,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: function_name.to_string(),
            implicit_class_argument: None,
        },
    )
}

/// Public static methods `(method name, builtin function name)`.
pub fn static_methods<'a>(
    entries: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> HashMap<MethodName, RuntimeMethod> {
    entries
        .into_iter()
        .map(|(name, function_name)| method(name, function_name, true))
        .collect()
}

/// Public instance methods `(method name, builtin function name)`.
pub fn instance_methods<'a>(
    entries: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> HashMap<MethodName, RuntimeMethod> {
    entries
        .into_iter()
        .map(|(name, function_name)| method(name, function_name, false))
        .collect()
}

/// Register a parentless host value class with no events.
pub fn register_value_class(
    name: impl Into<ClassIdentity>,
    properties: HashMap<MemberName, RuntimeProperty>,
    methods: HashMap<MethodName, RuntimeMethod>,
) {
    register_class(RuntimeClass {
        name: name.into(),
        parent: None,
        properties,
        methods,
        events: HashMap::new(),
    });
}
