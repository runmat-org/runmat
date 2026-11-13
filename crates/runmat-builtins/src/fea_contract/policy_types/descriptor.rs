//! Signature and error facts shared by typed FEA policy constructors.

use super::super::descriptors::{ERROR_INPUT, ERROR_INTERNAL, ERROR_OPERATION};
use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

/// Typed policy constructors report invalid arguments as `RunMat:fea:InvalidInput`
/// and conversion failures as `RunMat:fea:Internal`; the operation error is part of
/// the shared FEA builtin error set the runtime raises through the same helper.
pub const POLICY_ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INPUT, ERROR_OPERATION, ERROR_INTERNAL];

pub const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

/// A trailing argument the caller may omit; `default` is the documented value used then.
pub const fn optional(
    name: &'static str,
    ty: BuiltinParamType,
    default: &'static str,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Optional,
        default: Some(default),
        description,
    }
}

pub const fn output(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    required(name, BuiltinParamType::Any, description)
}

pub const fn fixed_descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &POLICY_ERRORS,
    }
}
