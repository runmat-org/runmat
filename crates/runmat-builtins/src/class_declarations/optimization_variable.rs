//! Public variable properties and real native method bindings. Reference-level
//! setter restrictions are enforced by the reference role, not a new class.
use super::*;
use crate::OptimizationVariableProperty as Property;
use runmat_types::StaticMethodName;
pub(super) fn declaration(identity: &ClassIdentity) -> Option<ExternalClassDeclaration> {
    if !identity.is(standard::OPTIMIZATION_VARIABLE) {
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
                    set_access: if property == Property::Name {
                        MemberAccess::Private
                    } else {
                        MemberAccess::Public
                    },
                    ..PropertyAttributes::default()
                },
            })
            .collect(),
        methods: [
            (
                StaticMethodName::new("subsref"),
                "optim.problemdef.OptimizationVariable.subsref",
            ),
            (
                StaticMethodName::new("subsasgn"),
                "optim.problemdef.OptimizationVariable.subsasgn",
            ),
        ]
        .into_iter()
        .map(|(name, builtin)| ExternalMethodDeclaration {
            name: name.owned(),
            attributes: MethodAttributes {
                access: MemberAccess::Public,
                ..MethodAttributes::default()
            },
            is_static: false,
            callable: CallableIdentity::Builtin(BuiltinId(builtin.into())),
            implicit_class_argument: None,
        })
        .chain(super::optimization_model::arithmetic_methods())
        .collect(),
    })
}
