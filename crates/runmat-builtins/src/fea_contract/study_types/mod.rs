//! Shared catalog facts for the typed FEA study-contract constructors:
//! observation requests, targets, sampling, storage, monitors, independent
//! domains, state, restart, and analysis sequences. Each class family owns its
//! signatures under `catalog/entries/fea/study_types/<class>/descriptor.rs`;
//! this module owns only the layout every family shares.

use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

pub mod classes;
pub mod descriptors;
pub mod fixtures;
pub mod inference;
pub mod integer;
#[cfg(test)]
pub mod test_support;

pub use super::descriptors::{ERROR_INPUT, ERROR_INTERNAL};

/// A typed study constructor returns exactly one immutable value.
pub const FEA_STUDY_TYPE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.STUDY_TYPE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:fea:TooManyOutputs"),
    when: "More than one output is requested from a typed study constructor.",
    message: "fea: too many output arguments",
};

/// Typed study constructors validate their inputs (`ERROR_INPUT`), convert
/// the immutable analysis value into a RunMat object (`ERROR_INTERNAL`), and
/// return one output. They never run a validation, plan, or solve operation.
pub const FEA_STUDY_TYPE_ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INPUT, ERROR_INTERNAL, FEA_STUDY_TYPE_ERROR_OUTPUTS];

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

pub const fn signature(
    label: &'static str,
    inputs: &'static [BuiltinParamDescriptor],
    outputs: &'static [BuiltinParamDescriptor],
) -> BuiltinSignatureDescriptor {
    BuiltinSignatureDescriptor {
        label,
        inputs,
        outputs,
    }
}

/// Fixed one-output public descriptor over a family's signature and error set.
pub const fn descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
    errors: &'static [BuiltinErrorDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors,
    }
}

pub const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
