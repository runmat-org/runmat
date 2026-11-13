use super::super::shared;
use crate::*;

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "c = spring()",
        inputs: &shared::INPUTS_NONE,
        outputs: &shared::OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "c = spring(m)",
        inputs: &shared::INPUTS_LENGTH,
        outputs: &shared::OUTPUTS,
    },
];

pub(super) const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &shared::ERRORS,
};

pub(super) const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "c = spring(integer_m)",
        inputs: &shared::LENGTH_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "An exact host integer row count is range-checked before RGB generation; output remains host double.",
    }];
