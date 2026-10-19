//! Signatures, parameters, and integer facts for `fea.AcousticInitialCondition` constructors.
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

const PRESSURE_INPUTS: [BuiltinParamDescriptor; 6] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Initial-condition identifier (nonempty text).",
    ),
    required(
        "target",
        BuiltinParamType::Any,
        "geometry.Region or fea.InitialConditionTarget.wholeModel().",
    ),
    required(
        "signal",
        BuiltinParamType::Any,
        "fea.PressureSignal that supplies the initial value.",
    ),
    required(
        "abscissa",
        BuiltinParamType::NumericScalar,
        "Finite signal abscissa (time, frequency, or parameter) at which the signal is sampled.",
    ),
    required(
        "sampleApproach",
        BuiltinParamType::Any,
        "fea.SignalSampleApproach policy: at, leftLimit, or rightLimit.",
    ),
    required(
        "beforeStepId",
        BuiltinParamType::StringScalar,
        "Id of the procedure step that starts from this state.",
    ),
];
const PRESSURE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "condition",
    BuiltinParamType::Any,
    "fea.AcousticInitialCondition object.",
)];
const PRESSURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "condition = fea.AcousticInitialCondition.pressure(id, target, signal, abscissa, sampleApproach, beforeStepId)",
    inputs: &PRESSURE_INPUTS,
    outputs: &PRESSURE_OUTPUTS,
}];
pub const FEA_ACOUSTIC_INITIAL_CONDITION_PRESSURE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &PRESSURE_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
