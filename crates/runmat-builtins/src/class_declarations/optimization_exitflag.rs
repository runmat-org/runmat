//! Canonical method roles for the existing enumeration value representation.
use super::*;
pub(super) fn declaration(identity: &ClassIdentity) -> Option<ExternalClassDeclaration> {
    if !identity.is(standard::OPTIMIZATION_EXITFLAG) {
        return None;
    }
    Some(ExternalClassDeclaration {
        name: identity.qualified_name(),
        parent: None,
        kind: ClassKind::Value,
        is_sealed: false,
        is_abstract: false,
        array_shape: None,
        properties: Vec::new(),
        methods: [
            (
                StaticMethodName::new("double"),
                &crate::OPTIMEXITFLAG_DOUBLE_CATALOG_ENTRY,
            ),
            (
                StaticMethodName::new("string"),
                &crate::OPTIMEXITFLAG_STRING_CATALOG_ENTRY,
            ),
            (
                StaticMethodName::new("eq"),
                &crate::OPTIMEXITFLAG_EQ_CATALOG_ENTRY,
            ),
            (
                StaticMethodName::new("ne"),
                &crate::OPTIMEXITFLAG_NE_CATALOG_ENTRY,
            ),
        ]
        .into_iter()
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
