//! Problem properties retain one native value generation and actual graph roots.
use super::*;
use crate::OptimizationProblemProperty as Property;
pub(super) fn declaration(identity: &ClassIdentity) -> Option<ExternalClassDeclaration> {
    if !identity.is(standard::OPTIMIZATION_PROBLEM) {
        return None;
    }
    Some(ExternalClassDeclaration {
        name: identity.qualified_name(),
        parent: None,
        kind: ClassKind::Value,
        is_sealed: false,
        is_abstract: false,
        array_shape: None,
        properties: Property::ALL
            .into_iter()
            .map(|property| ExternalPropertyDeclaration {
                name: property.name().into(),
                attributes: PropertyAttributes {
                    is_dependent: true,
                    get_access: MemberAccess::Public,
                    set_access: if property == Property::Variables {
                        MemberAccess::Private
                    } else {
                        MemberAccess::Public
                    },
                    ..Default::default()
                },
            })
            .collect(),
        methods: [
            (
                StaticMethodName::new("solve"),
                &crate::OPTIMPROBLEM_SOLVE_METHOD_CATALOG_ENTRY,
            ),
            (
                StaticMethodName::new("subsref"),
                &crate::OPTIMPROBLEM_SUBSREF_CATALOG_ENTRY,
            ),
            (
                StaticMethodName::new("subsasgn"),
                &crate::OPTIMPROBLEM_SUBSASGN_CATALOG_ENTRY,
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
