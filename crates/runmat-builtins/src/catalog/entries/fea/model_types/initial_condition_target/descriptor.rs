//! Signatures, parameters, and integer facts for `fea.InitialConditionTarget` constructors.
//! The runtime binding re-exports these facts; it does not redefine them.

use crate::fea_contract::descriptors::ERRORS;
use crate::*;

const fn required(
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

pub const FEA_INITIAL_CONDITION_TARGET_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "The whole-model target constructor has no numeric role.",
    };

const WHOLE_MODEL_INPUTS: [BuiltinParamDescriptor; 0] = [];
const WHOLE_MODEL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "target",
    BuiltinParamType::Any,
    "fea.InitialConditionTarget object.",
)];
const WHOLE_MODEL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "target = fea.InitialConditionTarget.wholeModel()",
    inputs: &WHOLE_MODEL_INPUTS,
    outputs: &WHOLE_MODEL_OUTPUTS,
}];
pub const FEA_INITIAL_CONDITION_TARGET_WHOLE_MODEL_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &WHOLE_MODEL_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
