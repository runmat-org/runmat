//! Signatures, parameters, and integer facts for `fea.MagneticInitialCondition` constructors.
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

const VECTOR_POTENTIAL_INPUTS: [BuiltinParamDescriptor; 6] = [
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
        "fea.MagneticVectorPotentialSignal that supplies the initial value.",
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
const VECTOR_POTENTIAL_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "condition",
    BuiltinParamType::Any,
    "fea.MagneticInitialCondition object.",
)];
const VECTOR_POTENTIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "condition = fea.MagneticInitialCondition.vectorPotential(id, target, signal, abscissa, sampleApproach, beforeStepId)",
    inputs: &VECTOR_POTENTIAL_INPUTS,
    outputs: &VECTOR_POTENTIAL_OUTPUTS,
}];
pub const FEA_MAGNETIC_INITIAL_CONDITION_VECTOR_POTENTIAL_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &VECTOR_POTENTIAL_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
