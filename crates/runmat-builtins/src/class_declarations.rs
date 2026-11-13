use runmat_types::{
    standard, BuiltinId, CallableIdentity, ClassIdentity, ClassKind, ExternalClassDeclaration,
    ExternalMethodDeclaration, ExternalPropertyDeclaration, MemberAccess, MethodAttributes,
    PropertyAttributes, QualifiedName, StaticClassIdentity, StaticMethodName, SymbolName,
};

mod arrays;
mod equation_problem;
mod layers;
mod optim_warm_start;
mod optimization_exitflag;
mod optimization_model;
mod optimization_problem;
mod optimization_variable;
pub use arrays::{
    CALENDAR_DAYS_PROPERTY, CALENDAR_MONTHS_PROPERTY, DATETIME_SERIAL_PROPERTY,
    DLARRAY_DATA_PROPERTY, DURATION_DAYS_PROPERTY,
};
pub use layers::*;

/// Borrow the immutable shape protocol without rebuilding class/member
/// declarations on each runtime structural query.
pub fn standard_class_array_shape(
    identity: &ClassIdentity,
) -> Option<&'static runmat_types::ClassArrayShape> {
    arrays::shape(identity)
}

/// Return immutable standard-library class metadata used during composition.
/// Mutable runtime registrations and static property values are intentionally
/// not visible through this interface.
pub fn standard_class_declaration(identity: &ClassIdentity) -> Option<ExternalClassDeclaration> {
    if let Some(declaration) = optim_warm_start::declaration(identity) {
        return Some(declaration);
    }
    if let Some(declaration) = optimization_exitflag::declaration(identity) {
        return Some(declaration);
    }
    if let Some(declaration) = equation_problem::declaration(identity) {
        return Some(declaration);
    }
    if let Some(declaration) = optimization_problem::declaration(identity) {
        return Some(declaration);
    }
    if let Some(declaration) = optimization_model::declaration(identity) {
        return Some(declaration);
    }
    if let Some(declaration) = optimization_variable::declaration(identity) {
        return Some(declaration);
    }
    if let Some(declaration) = arrays::declaration(identity) {
        return Some(declaration);
    }
    if identity.is(standard::STRING) {
        return Some(ExternalClassDeclaration {
            name: identity.qualified_name(),
            parent: None,
            kind: ClassKind::Value,
            is_sealed: false,
            is_abstract: false,
            properties: Vec::new(),
            methods: vec![ExternalMethodDeclaration {
                name: StaticMethodName::new("empty").owned(),
                attributes: MethodAttributes {
                    access: MemberAccess::Public,
                    ..MethodAttributes::default()
                },
                is_static: true,
                callable: CallableIdentity::Builtin(BuiltinId("string.empty".into())),
                implicit_class_argument: None,
            }],
            array_shape: None,
        });
    }
    if identity.is(standard::GPU_ARRAY) {
        return Some(ExternalClassDeclaration {
            name: identity.qualified_name(),
            parent: None,
            kind: ClassKind::Value,
            is_sealed: false,
            is_abstract: false,
            properties: Vec::new(),
            methods: vec![
                ExternalMethodDeclaration {
                    name: StaticMethodName::new("zeros").owned(),
                    attributes: MethodAttributes {
                        access: MemberAccess::Public,
                        ..MethodAttributes::default()
                    },
                    is_static: true,
                    callable: CallableIdentity::Builtin(BuiltinId("zeros".into())),
                    implicit_class_argument: Some(standard::GPU_ARRAY.owned()),
                },
                public_instance_method(standard::GPU_ARRAY, StaticMethodName::new("arrayfun")),
                public_instance_method(standard::GPU_ARRAY, StaticMethodName::new("existsOnGPU")),
                public_instance_method(standard::GPU_ARRAY, StaticMethodName::new("gather")),
                public_instance_method(standard::GPU_ARRAY, StaticMethodName::new("isgpuarray")),
                public_instance_method(
                    standard::GPU_ARRAY,
                    StaticMethodName::new("isUnderlyingType"),
                ),
                public_instance_method(standard::GPU_ARRAY, StaticMethodName::new("ndims")),
                public_instance_method(standard::GPU_ARRAY, StaticMethodName::new("pagefun")),
                public_instance_method(standard::GPU_ARRAY, StaticMethodName::new("size")),
                public_instance_method(
                    standard::GPU_ARRAY,
                    StaticMethodName::new("underlyingType"),
                ),
            ],
            array_shape: None,
        });
    }
    let primitives = [
        standard::DOUBLE,
        standard::SINGLE,
        standard::LOGICAL,
        standard::INT8,
        standard::INT16,
        standard::INT32,
        standard::INT64,
        standard::UINT8,
        standard::UINT16,
        standard::UINT32,
        standard::UINT64,
    ];
    if let Some(primitive) = primitives
        .into_iter()
        .find(|candidate| identity.is(*candidate))
    {
        let mut methods = vec![ExternalMethodDeclaration {
            name: StaticMethodName::new("zeros").owned(),
            attributes: MethodAttributes {
                access: MemberAccess::Public,
                ..MethodAttributes::default()
            },
            is_static: true,
            callable: CallableIdentity::Builtin(BuiltinId("zeros".into())),
            implicit_class_argument: Some(primitive.owned()),
        }];
        if [
            standard::INT8,
            standard::INT16,
            standard::INT32,
            standard::INT64,
            standard::UINT8,
            standard::UINT16,
            standard::UINT32,
            standard::UINT64,
        ]
        .into_iter()
        .any(|class| identity.is(class))
        {
            methods.push(ExternalMethodDeclaration {
                name: StaticMethodName::new("empty").owned(),
                attributes: MethodAttributes {
                    access: MemberAccess::Public,
                    ..MethodAttributes::default()
                },
                is_static: true,
                callable: CallableIdentity::Builtin(BuiltinId(format!(
                    "{}.empty",
                    primitive.display_name()
                ))),
                implicit_class_argument: None,
            });
        }
        return Some(ExternalClassDeclaration {
            name: identity.qualified_name(),
            parent: None,
            kind: ClassKind::Value,
            is_sealed: false,
            is_abstract: false,
            properties: Vec::new(),
            methods,
            array_shape: None,
        });
    }
    let (parent, kind) = if identity.is(standard::HANDLE) {
        (None, ClassKind::Handle)
    } else if identity.is(standard::DYNAMIC_PROPERTIES)
        || identity.is(standard::EVENT_DATA)
        || identity.is(standard::METADATA_DYNAMIC_PROPERTY)
        || identity.is(standard::UNIT_TEST_CASE)
    {
        (Some(standard::HANDLE), ClassKind::Handle)
    } else if identity.is(standard::METADATA_PROPERTY) {
        (None, ClassKind::Value)
    } else {
        return None;
    };
    let methods = if identity.is(standard::METADATA_DYNAMIC_PROPERTY) {
        vec![public_instance_method(
            standard::METADATA_DYNAMIC_PROPERTY,
            StaticMethodName::new("delete"),
        )]
    } else {
        Vec::new()
    };
    let properties = if identity.is(standard::EVENT_DATA) {
        ["Source", "EventName"]
            .into_iter()
            .map(|name| ExternalPropertyDeclaration {
                name: name.into(),
                attributes: PropertyAttributes {
                    set_access: MemberAccess::Private,
                    ..PropertyAttributes::default()
                },
            })
            .collect()
    } else {
        Vec::new()
    };
    Some(ExternalClassDeclaration {
        name: identity.qualified_name(),
        parent: parent.map(qualified_static),
        kind,
        is_sealed: false,
        is_abstract: false,
        properties,
        methods,
        array_shape: None,
    })
}

pub fn standard_class_is_subclass(
    class_name: &ClassIdentity,
    ancestor_name: &ClassIdentity,
) -> bool {
    let mut current = Some(class_name.clone());
    let mut visited = std::collections::BTreeSet::new();
    while let Some(name) = current {
        if !visited.insert(name.clone()) {
            return false;
        }
        if &name == ancestor_name {
            return true;
        }
        current = standard_class_declaration(&name)
            .and_then(|declaration| declaration.parent)
            .and_then(|name| ClassIdentity::from_qualified_name(&name).ok());
    }
    false
}

fn qualified_static(name: StaticClassIdentity) -> QualifiedName {
    name.owned().qualified_name()
}

fn public_instance_method(
    owner: StaticClassIdentity,
    name: StaticMethodName,
) -> ExternalMethodDeclaration {
    let mut callable = owner.owned().qualified_name();
    callable.0.push(SymbolName(name.display_name().to_owned()));
    ExternalMethodDeclaration {
        name: name.owned(),
        attributes: MethodAttributes::default(),
        is_static: false,
        callable: CallableIdentity::ExternalName(callable),
        implicit_class_argument: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_and_handle_metadata_is_deterministic() {
        let double = standard_class_declaration(&standard::DOUBLE.owned()).unwrap();
        assert_eq!(double.methods[0].name.0, "zeros");
        assert!(standard_class_is_subclass(
            &standard::DYNAMIC_PROPERTIES.owned(),
            &standard::HANDLE.owned()
        ));
        assert!(!standard_class_is_subclass(
            &standard::DOUBLE.owned(),
            &standard::HANDLE.owned()
        ));
    }

    #[test]
    fn event_data_is_a_handle_standard_class_with_private_set_metadata() {
        let class = standard_class_declaration(&standard::EVENT_DATA.owned()).unwrap();
        assert_eq!(
            class.parent,
            Some(standard::HANDLE.owned().qualified_name())
        );
        assert_eq!(class.kind, ClassKind::Handle);
        assert_eq!(class.properties.len(), 2);
        for property in class.properties {
            assert_eq!(property.attributes.get_access, MemberAccess::Public);
            assert_eq!(property.attributes.set_access, MemberAccess::Private);
        }
    }

    #[test]
    fn gpu_array_methods_carry_complete_typed_declarations() {
        let declaration = standard_class_declaration(&standard::GPU_ARRAY.owned()).unwrap();
        assert_eq!(declaration.methods.len(), 10);
        for method in declaration.methods {
            if method.is_static {
                assert_eq!(method.name.0, "zeros");
                assert_eq!(
                    method.callable,
                    CallableIdentity::Builtin(BuiltinId("zeros".into()))
                );
                assert_eq!(
                    method.implicit_class_argument,
                    Some(standard::GPU_ARRAY.owned())
                );
                continue;
            }
            let CallableIdentity::ExternalName(callable) = method.callable else {
                panic!("gpuArray method must carry a qualified callable identity");
            };
            let mut expected = standard::GPU_ARRAY.owned().qualified_name();
            expected.0.push(SymbolName(method.name.0.clone()));
            assert_eq!(callable, expected);
            assert_eq!(method.attributes.access, MemberAccess::Public);
            assert!(!method.is_static);
        }
    }
}
