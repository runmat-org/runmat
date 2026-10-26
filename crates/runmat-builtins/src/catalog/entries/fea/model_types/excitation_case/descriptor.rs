//! Signatures, parameters, and integer facts for `fea.ExcitationCase` constructors.
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

pub const FEA_EXCITATION_CASE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed excitation constructors accept identifiers, geometry.Region values, and typed signal objects; physical numeric conversion belongs to the signal constructor.",
};

const ALWAYS_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Excitation-case identifier (nonempty text).",
    ),
    required(
        "actions",
        BuiltinParamType::Any,
        "One typed excitation action or a cell array of actions.",
    ),
];
const ALWAYS_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "load_case",
    BuiltinParamType::Any,
    "fea.ExcitationCase object.",
)];
const ALWAYS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "load_case = fea.ExcitationCase.always(id, actions)",
    inputs: &ALWAYS_INPUTS,
    outputs: &ALWAYS_OUTPUTS,
}];
pub const FEA_EXCITATION_CASE_ALWAYS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ALWAYS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const DURING_STEPS_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Excitation-case identifier (nonempty text).",
    ),
    required(
        "actions",
        BuiltinParamType::Any,
        "One typed excitation action or a cell array of actions.",
    ),
    required(
        "stepIds",
        BuiltinParamType::Any,
        "Step id text, string array, or cell array of step ids.",
    ),
];
const DURING_STEPS_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "load_case",
    BuiltinParamType::Any,
    "fea.ExcitationCase object.",
)];
const DURING_STEPS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "load_case = fea.ExcitationCase.duringSteps(id, actions, stepIds)",
    inputs: &DURING_STEPS_INPUTS,
    outputs: &DURING_STEPS_OUTPUTS,
}];
pub const FEA_EXCITATION_CASE_DURING_STEPS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURING_STEPS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const MUTUALLY_EXCLUSIVE_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Excitation-case identifier (nonempty text).",
    ),
    required(
        "actions",
        BuiltinParamType::Any,
        "One typed excitation action or a cell array of actions.",
    ),
    required(
        "groupId",
        BuiltinParamType::StringScalar,
        "Combination-group identifier (nonempty text).",
    ),
];
const MUTUALLY_EXCLUSIVE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "load_case",
    BuiltinParamType::Any,
    "fea.ExcitationCase object.",
)];
const MUTUALLY_EXCLUSIVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "load_case = fea.ExcitationCase.mutuallyExclusive(id, actions, groupId)",
        inputs: &MUTUALLY_EXCLUSIVE_INPUTS,
        outputs: &MUTUALLY_EXCLUSIVE_OUTPUTS,
    }];
pub const FEA_EXCITATION_CASE_MUTUALLY_EXCLUSIVE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &MUTUALLY_EXCLUSIVE_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
