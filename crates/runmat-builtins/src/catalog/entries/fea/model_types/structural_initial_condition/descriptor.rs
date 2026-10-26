//! Signatures, parameters, and integer facts for `fea.StructuralInitialCondition` constructors.
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

const DISPLACEMENT_INPUTS: [BuiltinParamDescriptor; 6] = [
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
        "fea.DisplacementSignal that supplies the initial value.",
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
const DISPLACEMENT_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "condition",
    BuiltinParamType::Any,
    "fea.StructuralInitialCondition object.",
)];
const DISPLACEMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "condition = fea.StructuralInitialCondition.displacement(id, target, signal, abscissa, sampleApproach, beforeStepId)",
    inputs: &DISPLACEMENT_INPUTS,
    outputs: &DISPLACEMENT_OUTPUTS,
}];
pub const FEA_STRUCTURAL_INITIAL_CONDITION_DISPLACEMENT_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &DISPLACEMENT_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const VELOCITY_INPUTS: [BuiltinParamDescriptor; 6] = [
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
        "fea.VelocitySignal that supplies the initial value.",
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
const VELOCITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "condition",
    BuiltinParamType::Any,
    "fea.StructuralInitialCondition object.",
)];
const VELOCITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "condition = fea.StructuralInitialCondition.velocity(id, target, signal, abscissa, sampleApproach, beforeStepId)",
    inputs: &VELOCITY_INPUTS,
    outputs: &VELOCITY_OUTPUTS,
}];
pub const FEA_STRUCTURAL_INITIAL_CONDITION_VELOCITY_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &VELOCITY_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };

const ANGULAR_VELOCITY_INPUTS: [BuiltinParamDescriptor; 6] = [
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
        "fea.AngularVelocitySignal that supplies the initial value.",
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
const ANGULAR_VELOCITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "condition",
    BuiltinParamType::Any,
    "fea.StructuralInitialCondition object.",
)];
const ANGULAR_VELOCITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "condition = fea.StructuralInitialCondition.angularVelocity(id, target, signal, abscissa, sampleApproach, beforeStepId)",
    inputs: &ANGULAR_VELOCITY_INPUTS,
    outputs: &ANGULAR_VELOCITY_OUTPUTS,
}];
pub const FEA_STRUCTURAL_INITIAL_CONDITION_ANGULAR_VELOCITY_DESCRIPTOR: BuiltinDescriptor =
    BuiltinDescriptor {
        signatures: &ANGULAR_VELOCITY_SIGNATURES,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    };
