//! Shared descriptors of the typed FEA public and method surface.

use super::{errors::ERRORS, parameters::*};
use crate::*;

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
    label: "plan = fea.plan(studyOrSweep)",
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

pub const RESULTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "results = fea.results(runOrRunId, Name, Value, ...)",
    inputs: &RESULTS_INPUTS,
    outputs: &OUT_ANY,
}];

pub const FIELD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.field(resultsOrRun, fieldSelector[, sample])",
    inputs: &FIELD_INPUTS,
    outputs: &OUT_ANY,
}];

pub const FIELD_PAGE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "page = fea.fieldPage(field, start, count)",
    inputs: &FIELD_PAGE_INPUTS,
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

pub const FEA_STUDY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STUDY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_VALIDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VALIDATE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_PLAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PLAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_RUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_SWEEP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SWEEP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_MODEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MODEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_MATERIAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATERIAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_RESULTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RESULTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_FIELD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIELD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_FIELD_PAGE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIELD_PAGE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_PLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &ERRORS,
};

pub const FEA_MESH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MESH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &ERRORS,
};

pub const FEA_COMPARE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPARE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FEA_TRENDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRENDS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const MATERIAL_ASSIGNMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "assignment = fea.materialAssignment(target, material, Name, Value, ...)",
        inputs: &MATERIAL_ASSIGNMENT_INPUTS,
        outputs: &OUT_ANY,
    }];

pub const FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATERIAL_ASSIGNMENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
