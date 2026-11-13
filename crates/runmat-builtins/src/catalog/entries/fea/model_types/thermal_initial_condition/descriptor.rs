//! Signatures, parameters, and integer facts for `fea.ThermalInitialCondition` constructors.
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

const TEMPERATURE_INPUTS: [BuiltinParamDescriptor; 6] = [
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
        "fea.TemperatureSignal that supplies the initial value.",
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
const TEMPERATURE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "condition",
    BuiltinParamType::Any,
    "fea.ThermalInitialCondition object.",
)];
const TEMPERATURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "condition = fea.ThermalInitialCondition.temperature(id, target, signal, abscissa, sampleApproach, beforeStepId)",
    inputs: &TEMPERATURE_INPUTS,
    outputs: &TEMPERATURE_OUTPUTS,
}];
pub const FEA_THERMAL_INITIAL_CONDITION_TEMPERATURE_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &TEMPERATURE_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
