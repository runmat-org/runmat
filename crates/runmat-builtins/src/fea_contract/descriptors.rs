//! Signature and error descriptors for the FEA operation surface. The runtime
//! re-exports these values; it does not declare its own copies.

use super::parameters::{
    COMPARE_INPUTS, FIELD_INPUTS, FIELD_PAGE_INPUTS, IN_INPUT, IN_STUDY_ARGS, IN_VARIADIC_ARGS,
    MATERIAL_ASSIGNMENT_INPUTS, MATERIAL_INPUTS, MESH_CONTEXT_INPUTS, MODEL_INPUTS, OUT_ANY,
    PLOT_CONTEXT_INPUTS, RESULTS_INPUTS, STEP_RESULT_INPUTS, SWEEP_INPUTS, TRENDS_INPUTS,
};
use crate::BuiltinCompletionPolicy;
use crate::BuiltinDescriptor;
use crate::BuiltinErrorDescriptor;
use crate::BuiltinOutputMode;
use crate::BuiltinSignatureDescriptor;

pub const STUDY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "study = fea.study(id, geometry, model, Name, Value, ...)",
    inputs: &IN_STUDY_ARGS,
    outputs: &OUT_ANY,
}];
pub const VALIDATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "result = fea.validate(studyOrSweep)",
    inputs: &IN_INPUT,
    outputs: &OUT_ANY,
}];
pub const PLAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "plan = fea.plan(study)",
    inputs: &IN_INPUT,
    outputs: &OUT_ANY,
}];
pub const RUN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "run = fea.run(studyOrSweep)",
    inputs: &IN_INPUT,
    outputs: &OUT_ANY,
}];
pub const SWEEP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "sweep = fea.sweep(id, studies, Name, Value, ...)",
    inputs: &SWEEP_INPUTS,
    outputs: &OUT_ANY,
}];
pub const MODEL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "model = fea.model(id, geometry, Name, Value, ...)",
    inputs: &MODEL_INPUTS,
    outputs: &OUT_ANY,
}];
pub const MATERIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "material = fea.material(id, Name, Value, ...)",
    inputs: &MATERIAL_INPUTS,
    outputs: &OUT_ANY,
}];
pub const MATERIAL_ASSIGNMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "assignment = fea.materialAssignment(target, material, Name, Value, ...)",
        inputs: &MATERIAL_ASSIGNMENT_INPUTS,
        outputs: &OUT_ANY,
    }];
pub const COMPONENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "component = fea.component(args, ...)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];
pub const RESULTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "results = fea.results(runOrRunId, Name, Value, ...)",
    inputs: &RESULTS_INPUTS,
    outputs: &OUT_ANY,
}];
pub const FIELD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.field(resultsOrRun, fieldSelector, sample)",
    inputs: &FIELD_INPUTS,
    outputs: &OUT_ANY,
}];
pub const FIELD_PAGE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "page = fea.fieldPage(field, start, count)",
    inputs: &FIELD_PAGE_INPUTS,
    outputs: &OUT_ANY,
}];
pub const STEP_RESULT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "stepRun = fea.stepResult(run, stepId)",
    inputs: &STEP_RESULT_INPUTS,
    outputs: &OUT_ANY,
}];
pub const PLOT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "figure = plot(field, Name, Value, ...)",
    inputs: &PLOT_CONTEXT_INPUTS,
    outputs: &OUT_ANY,
}];
pub const MESH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "figure = mesh(runOrResultsOrField, Name, Value, ...)",
    inputs: &MESH_CONTEXT_INPUTS,
    outputs: &OUT_ANY,
}];
pub const COMPARE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "comparison = fea.compare(baselineRunId, candidateRunId)",
    inputs: &COMPARE_INPUTS,
    outputs: &OUT_ANY,
}];
pub const TRENDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "trends = fea.trends(Name, Value, ...)",
    inputs: &TRENDS_INPUTS,
    outputs: &OUT_ANY,
}];

pub const ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.INVALID_INPUT",
    identifier: Some("RunMat:fea:InvalidInput"),
    when: "A builtin receives an unsupported argument pattern or object type.",
    message: "fea: invalid input",
};
pub const ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.OPERATION_FAILED",
    identifier: Some("RunMat:fea:OperationFailed"),
    when: "A validation, planning, run, result-query, comparison, or trend operation fails.",
    message: "fea: operation failed",
};
pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.INTERNAL",
    identifier: Some("RunMat:fea:Internal"),
    when: "An FEA object or operation result cannot be converted to a RunMat value.",
    message: "fea: internal error",
};
pub const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INPUT, ERROR_OPERATION, ERROR_INTERNAL];

const fn public(signatures: &'static [BuiltinSignatureDescriptor]) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    }
}

const fn method_only(signatures: &'static [BuiltinSignatureDescriptor]) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::MethodOnly,
        errors: &ERRORS,
    }
}

pub const FEA_STUDY_DESCRIPTOR: BuiltinDescriptor = public(&STUDY_SIGNATURES);
pub const FEA_VALIDATE_DESCRIPTOR: BuiltinDescriptor = public(&VALIDATE_SIGNATURES);
pub const FEA_PLAN_DESCRIPTOR: BuiltinDescriptor = public(&PLAN_SIGNATURES);
pub const FEA_RUN_DESCRIPTOR: BuiltinDescriptor = public(&RUN_SIGNATURES);
pub const FEA_SWEEP_DESCRIPTOR: BuiltinDescriptor = public(&SWEEP_SIGNATURES);
pub const FEA_MODEL_DESCRIPTOR: BuiltinDescriptor = public(&MODEL_SIGNATURES);
pub const FEA_MATERIAL_DESCRIPTOR: BuiltinDescriptor = public(&MATERIAL_SIGNATURES);
pub const FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR: BuiltinDescriptor =
    public(&MATERIAL_ASSIGNMENT_SIGNATURES);
/// Shared descriptor for generic typed FEA component constructors.
pub const FEA_COMPONENT_DESCRIPTOR: BuiltinDescriptor = public(&COMPONENT_SIGNATURES);
pub const FEA_RESULTS_DESCRIPTOR: BuiltinDescriptor = public(&RESULTS_SIGNATURES);
pub const FEA_FIELD_DESCRIPTOR: BuiltinDescriptor = public(&FIELD_SIGNATURES);
pub const FEA_FIELD_PAGE_DESCRIPTOR: BuiltinDescriptor = public(&FIELD_PAGE_SIGNATURES);
pub const FEA_STEP_RESULT_DESCRIPTOR: BuiltinDescriptor = public(&STEP_RESULT_SIGNATURES);
pub const FEA_PLOT_DESCRIPTOR: BuiltinDescriptor = method_only(&PLOT_SIGNATURES);
pub const FEA_MESH_DESCRIPTOR: BuiltinDescriptor = method_only(&MESH_SIGNATURES);
pub const FEA_COMPARE_DESCRIPTOR: BuiltinDescriptor = public(&COMPARE_SIGNATURES);
pub const FEA_TRENDS_DESCRIPTOR: BuiltinDescriptor = public(&TRENDS_SIGNATURES);
