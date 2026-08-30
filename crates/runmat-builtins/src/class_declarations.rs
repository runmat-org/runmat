use runmat_types::{
    standard, BuiltinId, CallableIdentity, ClassIdentity, ClassKind, ExternalClassDeclaration,
    ExternalMethodDeclaration, MemberAccess, MethodAttributes, MethodName, QualifiedName,
    StaticClassIdentity, StaticMethodName, SymbolName,
};

const GPU_ARRAY_METHODS: &[StaticMethodName] = &[
    StaticMethodName::new("arrayfun"),
    StaticMethodName::new("existsOnGPU"),
    StaticMethodName::new("gather"),
    StaticMethodName::new("isgpuarray"),
    StaticMethodName::new("isUnderlyingType"),
    StaticMethodName::new("ndims"),
    StaticMethodName::new("pagefun"),
    StaticMethodName::new("size"),
    StaticMethodName::new("underlyingType"),
];

/// Return immutable standard-library class metadata used during composition.
/// Mutable runtime registrations and static property values are intentionally
/// not visible through this interface.
pub fn standard_class_declaration(identity: &ClassIdentity) -> Option<ExternalClassDeclaration> {
    if identity.is(standard::GPU_ARRAY) {
        return Some(ExternalClassDeclaration {
            name: identity.qualified_name(),
            parent: None,
            kind: ClassKind::Value,
            is_sealed: false,
            is_abstract: false,
            properties: Vec::new(),
            methods: GPU_ARRAY_METHODS
                .iter()
                .map(|method| ExternalMethodDeclaration {
                    name: method.owned(),
                    attributes: MethodAttributes::default(),
                    is_static: false,
                    callable: CallableIdentity::ExternalName(qualified(&format!(
                        "{}.{}",
                        standard::GPU_ARRAY,
                        method.display_name()
                    ))),
                    implicit_class_argument: None,
                })
                .collect(),
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
        return Some(ExternalClassDeclaration {
            name: identity.qualified_name(),
            parent: None,
            kind: ClassKind::Value,
            is_sealed: false,
            is_abstract: false,
            properties: Vec::new(),
            methods: vec![ExternalMethodDeclaration {
                name: MethodName("zeros".into()),
                attributes: MethodAttributes {
                    access: MemberAccess::Public,
                    ..MethodAttributes::default()
                },
                is_static: true,
                callable: CallableIdentity::Builtin(BuiltinId("zeros".into())),
                implicit_class_argument: Some(primitive.display_name().to_owned()),
            }],
        });
    }
    let (parent, kind) = if identity.is(standard::HANDLE) {
        (None, ClassKind::Handle)
    } else if identity.is(standard::DYNAMIC_PROPERTIES)
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
        vec![ExternalMethodDeclaration {
            name: MethodName("delete".into()),
            attributes: MethodAttributes::default(),
            is_static: false,
            callable: CallableIdentity::ExternalName(qualified(
                "matlab.metadata.DynamicProperty.delete",
            )),
            implicit_class_argument: None,
        }]
    } else {
        Vec::new()
    };
    Some(ExternalClassDeclaration {
        name: identity.qualified_name(),
        parent: parent.map(qualified_static),
        kind,
        is_sealed: false,
        is_abstract: false,
        properties: Vec::new(),
        methods,
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

fn qualified(name: &str) -> QualifiedName {
    QualifiedName(
        name.split('.')
            .map(|segment| SymbolName(segment.to_owned()))
            .collect(),
    )
}

fn qualified_static(name: StaticClassIdentity) -> QualifiedName {
    name.owned().qualified_name()
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
}
