//! Standard problem-model class roles. Arithmetic methods share the actual
//! expression producer; variable subscripting remains its original-definition
//! protocol rather than being replaced with expression indexing.
use super::*;
use runmat_types::OperatorKind as Op;

pub(super) const OPERATORS: &[Op] = &[
    Op::UnaryPlus,
    Op::UnaryMinus,
    Op::Transpose,
    Op::ConjugateTranspose,
    Op::Add,
    Op::Subtract,
    Op::MatrixMultiply,
    Op::ElementwiseMultiply,
    Op::MatrixPower,
    Op::ElementwisePower,
    Op::Mldivide,
    Op::Mrdivide,
    Op::ElementwiseDivide,
    Op::ElementwiseLeftDivide,
    Op::LessEqual,
    Op::GreaterEqual,
    Op::Equal,
];
fn method(name: &str, builtin: String) -> ExternalMethodDeclaration {
    ExternalMethodDeclaration {
        name: name.into(),
        attributes: MethodAttributes {
            access: MemberAccess::Public,
            ..Default::default()
        },
        is_static: false,
        callable: CallableIdentity::Builtin(BuiltinId(builtin)),
        implicit_class_argument: None,
    }
}
pub(super) fn arithmetic_methods() -> Vec<ExternalMethodDeclaration> {
    OPERATORS
        .iter()
        .filter_map(|operator| operator.overload_name())
        .map(|name| {
            method(
                name.display_name(),
                format!("optim.problemdef.OptimizationExpression.{name}"),
            )
        })
        .chain(
            crate::OPTIMMODEL_NUMERICAL_METHODS
                .iter()
                .map(|(name, entry)| method(name.display_name(), entry.identity.name.to_owned())),
        )
        .chain(std::iter::once(zeros_like_method()))
        .collect()
}
fn zeros_like_method() -> ExternalMethodDeclaration {
    let mut declaration = method(
        crate::OPTIMEXPR_ZEROS_LIKE_METHOD.display_name(),
        crate::OPTIMEXPR_ZEROS_LIKE_CATALOG_ENTRY
            .identity
            .name
            .to_owned(),
    );
    declaration.attributes.is_hidden = true;
    declaration
}
pub(super) fn declaration(identity: &ClassIdentity) -> Option<ExternalClassDeclaration> {
    let expression = identity.is(standard::OPTIMIZATION_EXPRESSION);
    let constraint = identity.is(standard::OPTIMIZATION_CONSTRAINT);
    let equality = identity.is(standard::OPTIMIZATION_EQUALITY);
    let inequality = identity.is(standard::OPTIMIZATION_INEQUALITY);
    if !(expression || constraint || equality || inequality) {
        return None;
    }
    let properties = vec![("IndexNames", true), ("Variables", false)];
    let mut methods = if expression {
        arithmetic_methods()
    } else {
        Vec::new()
    };
    if expression {
        let mut creation = method(
            crate::OPTIMEXPR_ZEROS_METHOD.display_name(),
            crate::OPTIMEXPR_ZEROS_CATALOG_ENTRY
                .identity
                .name
                .to_owned(),
        );
        creation.is_static = true;
        methods.push(creation);
    }
    for name in ["subsref", "subsasgn", "evaluate"] {
        let owner = if expression {
            standard::OPTIMIZATION_EXPRESSION
        } else {
            standard::OPTIMIZATION_CONSTRAINT
        };
        methods.push(method(name, format!("{}.{name}", owner.display_name())));
    }
    Some(ExternalClassDeclaration {
        name: identity.qualified_name(),
        parent: (equality || inequality)
            .then(|| standard::OPTIMIZATION_CONSTRAINT.owned().qualified_name()),
        kind: ClassKind::Value,
        is_sealed: false,
        is_abstract: false,
        array_shape: None,
        properties: properties
            .into_iter()
            .map(|(name, writable)| ExternalPropertyDeclaration {
                name: name.into(),
                attributes: PropertyAttributes {
                    is_dependent: true,
                    get_access: MemberAccess::Public,
                    set_access: if writable {
                        MemberAccess::Public
                    } else {
                        MemberAccess::Private
                    },
                    ..Default::default()
                },
            })
            .collect(),
        methods,
    })
}
