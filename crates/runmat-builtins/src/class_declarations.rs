use runmat_types::{
    BuiltinId, CallableIdentity, ClassKind, ExternalClassDeclaration, ExternalMethodDeclaration,
    MemberAccess, MethodAttributes, MethodName, QualifiedName, SymbolName,
};

/// Return immutable standard-library class metadata used during composition.
/// Mutable runtime registrations and static property values are intentionally
/// not visible through this interface.
pub fn standard_class_declaration(name: &str) -> Option<ExternalClassDeclaration> {
    let primitive = [
        "double", "single", "logical", "int8", "int16", "int32", "int64", "uint8", "uint16",
        "uint32", "uint64",
    ];
    if primitive.contains(&name) {
        return Some(ExternalClassDeclaration {
            name: qualified(name),
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
                implicit_class_argument: Some(name.to_owned()),
            }],
        });
    }
    let (parent, kind) = match name {
        "handle" => (None, ClassKind::Handle),
        "dynamicprops" => (Some("handle"), ClassKind::Handle),
        "matlab.metadata.Property" => (None, ClassKind::Value),
        "matlab.metadata.DynamicProperty" => (Some("handle"), ClassKind::Handle),
        "matlab.unittest.TestCase" => (Some("handle"), ClassKind::Handle),
        _ => return None,
    };
    let methods = if name == "matlab.metadata.DynamicProperty" {
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
        name: qualified(name),
        parent: parent.map(qualified),
        kind,
        is_sealed: false,
        is_abstract: false,
        properties: Vec::new(),
        methods,
    })
}

pub fn standard_class_is_subclass(class_name: &str, ancestor_name: &str) -> bool {
    let mut current = Some(class_name.to_owned());
    let mut visited = std::collections::BTreeSet::new();
    while let Some(name) = current {
        if !visited.insert(name.clone()) {
            return false;
        }
        if name == ancestor_name {
            return true;
        }
        current = standard_class_declaration(&name)
            .and_then(|declaration| declaration.parent)
            .and_then(|name| name.display_name());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_and_handle_metadata_is_deterministic() {
        let double = standard_class_declaration("double").unwrap();
        assert_eq!(double.methods[0].name.0, "zeros");
        assert!(standard_class_is_subclass("dynamicprops", "handle"));
        assert!(!standard_class_is_subclass("double", "handle"));
    }
}
