//! Signature for `fea.ProcedureStep.configure`.

use crate::fea_contract::study_types::{descriptor, required, signature, FEA_STUDY_TYPE_ERRORS};
use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [required(
    "step",
    BuiltinParamType::Any,
    "The configured family step, with the input step's class.",
)];

const CONFIGURE_INPUTS: [BuiltinParamDescriptor; 8] = [
    required(
        "step",
        BuiltinParamType::Any,
        "Family step such as fea.StructuralStep.",
    ),
    required(
        "independentDomain",
        BuiltinParamType::Any,
        "Typed fea.IndependentDomain.",
    ),
    required(
        "activations",
        BuiltinParamType::Any,
        "fea.StepObjectActivation or cell array.",
    ),
    required(
        "stateInputs",
        BuiltinParamType::Any,
        "fea.StateInput or cell array.",
    ),
    required(
        "stateOutputs",
        BuiltinParamType::Any,
        "fea.StateOutput or cell array.",
    ),
    required(
        "observations",
        BuiltinParamType::Any,
        "fea.Observation or cell array.",
    ),
    required("accuracy", BuiltinParamType::Any, "Typed fea.AccuracyPlan."),
    required("restart", BuiltinParamType::Any, "Typed fea.RestartPolicy."),
];
const CONFIGURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [signature("step = fea.ProcedureStep.configure(step, independentDomain, activations, stateInputs, stateOutputs, observations, accuracy, restart)", &CONFIGURE_INPUTS, &OUTPUT)];
pub const FEA_PROCEDURE_STEP_CONFIGURE_DESCRIPTOR: BuiltinDescriptor =
    descriptor(&CONFIGURE_SIGNATURES, &FEA_STUDY_TYPE_ERRORS);
