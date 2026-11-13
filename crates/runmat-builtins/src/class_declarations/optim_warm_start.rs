//! Warm properties and methods use one canonical class contract.
use super::*;
pub(super) fn declaration(identity: &ClassIdentity) -> Option<ExternalClassDeclaration> {
    let methods = if identity.is(crate::LSQLIN_WARM_START_CLASS) {
        [
            &crate::LSQLIN_WARM_SUBSREF_CATALOG_ENTRY,
            &crate::LSQLIN_WARM_SUBSASGN_CATALOG_ENTRY,
        ]
    } else if identity.is(crate::QUADPROG_WARM_START_CLASS) {
        [
            &crate::QUADPROG_WARM_SUBSREF_CATALOG_ENTRY,
            &crate::QUADPROG_WARM_SUBSASGN_CATALOG_ENTRY,
        ]
    } else {
        return None;
    };
    Some(ExternalClassDeclaration {
        name: identity.qualified_name(),
        parent: None,
        kind: ClassKind::Value,
        is_sealed: true,
        is_abstract: false,
        array_shape: None,
        properties: crate::OptimWarmStartProperty::ALL
            .into_iter()
            .map(|property| ExternalPropertyDeclaration {
                name: property.name().into(),
                attributes: PropertyAttributes {
                    is_dependent: true,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Private,
                    ..Default::default()
                },
            })
            .collect(),
        methods: [
            StaticMethodName::new("subsref"),
            StaticMethodName::new("subsasgn"),
        ]
        .into_iter()
        .zip(methods)
        .map(|(name, entry)| ExternalMethodDeclaration {
            name: name.owned(),
            attributes: MethodAttributes {
                access: MemberAccess::Public,
                ..Default::default()
            },
            is_static: false,
            callable: CallableIdentity::Builtin(BuiltinId(entry.identity.name.into())),
            implicit_class_argument: None,
        })
        .collect(),
    })
}
