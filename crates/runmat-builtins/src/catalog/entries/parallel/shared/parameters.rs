use crate::{BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType};

pub(in crate::catalog::entries::parallel) const ANY_REQUIRED: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value operated on by the parallel runtime.",
    };
pub(in crate::catalog::entries::parallel) const ANY_OPTIONAL: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional value supplied by the designated lab.",
    };
pub(in crate::catalog::entries::parallel) const LAB_REQUIRED: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "lab",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based lab index.",
    };
pub(in crate::catalog::entries::parallel) const LAB_OPTIONAL: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "lab",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional one-based source lab index.",
    };
pub(in crate::catalog::entries::parallel) const TAG_OPTIONAL: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "tag",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional message tag.",
    };
pub(in crate::catalog::entries::parallel) const DIMENSION_OPTIONAL: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "dimension",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("2"),
        description: "One-based concatenation dimension.",
    };
pub(in crate::catalog::entries::parallel) const REDUCER_REQUIRED: BuiltinParamDescriptor =
    BuiltinParamDescriptor {
        name: "reducer",
        ty: BuiltinParamType::Callable,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Associative binary reduction function.",
    };
pub(in crate::catalog::entries::parallel) const ANY_OUTPUT: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "result",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Result produced by the parallel operation.",
    }];
