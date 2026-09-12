use crate::builtins::fea::contracts::parameters::{
    COMPARE_INPUTS, DOMAIN_INPUTS, FIELD_INPUTS, INTERFACE_INPUTS, IN_AUTHOR_STUDY_ARGS, IN_INPUT,
    IN_PATH, IN_STUDY_ARGS, IN_VARIADIC_ARGS, LOAD_CASE_INPUTS, MATERIAL_ASSIGNMENT_INPUTS,
    MATERIAL_INPUTS, MODEL_INPUTS, OUT_ANY, PLOT_CONTEXT_INPUTS, PLOT_STUDY_INPUTS, RESULTS_INPUTS,
    RUN_OPTIONS_INPUTS, STEP_INPUTS, SWEEP_INPUTS, TRENDS_INPUTS,
};
use runmat_builtins::BuiltinCompletionPolicy;
use runmat_builtins::BuiltinDescriptor;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::BuiltinOutputMode;
use runmat_builtins::BuiltinParamArity;
use runmat_builtins::BuiltinParamDescriptor;
use runmat_builtins::BuiltinParamType;
use runmat_builtins::BuiltinSignatureDescriptor;

pub(in crate::builtins::fea) const LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "doc = fea.load(path)",
        inputs: &IN_PATH,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const STUDY_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "study = fea.study(path)",
        inputs: &IN_PATH,
        outputs: &OUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "study = fea.study(id, geometry, Name, Value, ...)",
        inputs: &IN_STUDY_ARGS,
        outputs: &OUT_ANY,
    },
];
pub(in crate::builtins::fea) const AUTHOR_STUDY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "study = fea.authorStudy(id, geometry, meshAuthoringSummary, Name, Value, ...)",
        inputs: &IN_AUTHOR_STUDY_ARGS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const VALIDATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "result = fea.validate(studyOrSweepOrPath)",
        inputs: &IN_INPUT,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const PLAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "plan = fea.plan(study)",
        inputs: &IN_INPUT,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const RUN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "run = fea.run(studyOrSweepOrPath)",
        inputs: &IN_INPUT,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const SWEEP_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "sweep = fea.sweep(id, studies, Name, Value, ...)",
        inputs: &SWEEP_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const MODEL_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "model = fea.model(id, geometry, Name, Value, ...)",
        inputs: &MODEL_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const MATERIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "material = fea.material(id, Name, Value, ...)",
        inputs: &MATERIAL_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const MATERIAL_ASSIGNMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "assignment = fea.materialAssignment(region, material, Name, Value, ...)",
        inputs: &MATERIAL_ASSIGNMENT_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const LOAD_CASE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "load = fea.loadCase(id, region, kind, Name, Value, ...)",
        inputs: &LOAD_CASE_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const DOMAIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "domain = fea.domain(kind, Name, Value, ...)",
        inputs: &DOMAIN_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const INTERFACE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "interface = fea.interface(id, primaryRegion, secondaryRegion, Name, Value, ...)",
        inputs: &INTERFACE_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const COMPONENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "component = fea.component(args, ...)",
        inputs: &IN_VARIADIC_ARGS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const STEP_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "step = fea.step(id, kind)",
        inputs: &STEP_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const RUN_OPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "options = fea.runOptions(solver, Name, Value, ...)",
        inputs: &RUN_OPTIONS_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const BOUNDARY_CONDITION_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Boundary-condition id.",
    },
    BuiltinParamDescriptor {
        name: "region",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target region id.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Boundary-condition kind.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Numeric fields required by the selected kind.",
    },
];
pub(in crate::builtins::fea) const BOUNDARY_CONDITION_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "bc = fea.boundaryCondition(id, region, kind, Name, Value, ...)",
        inputs: &BOUNDARY_CONDITION_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const RESULTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "results = fea.results(runOrRunId, Name, Value, ...)",
        inputs: &RESULTS_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const FIELD_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.field(resultsOrRun, fieldId)",
        inputs: &FIELD_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const PLOT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "figure = fea.plot(runOrResultsOrField, fieldId)",
        inputs: &PLOT_CONTEXT_INPUTS,
        outputs: &OUT_ANY,
    },
    BuiltinSignatureDescriptor {
        label: "figure = fea.plot(study, runId, fieldId)",
        inputs: &PLOT_STUDY_INPUTS,
        outputs: &OUT_ANY,
    },
];
pub(in crate::builtins::fea) const COMPARE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "comparison = fea.compare(baselineRunId, candidateRunId)",
        inputs: &COMPARE_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(in crate::builtins::fea) const TRENDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "trends = fea.trends(Name, Value, ...)",
        inputs: &TRENDS_INPUTS,
        outputs: &OUT_ANY,
    }];

pub(in crate::builtins::fea) const ERROR_LOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.LOAD_FAILED",
    identifier: Some("RunMat:fea:LoadFailed"),
    when: "A .fea document cannot be read, parsed, or resolved.",
    message: "fea: failed to load FEA document",
};
pub(in crate::builtins::fea) const ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.INVALID_INPUT",
    identifier: Some("RunMat:fea:InvalidInput"),
    when: "A builtin receives an unsupported argument pattern or object type.",
    message: "fea: invalid input",
};
pub(in crate::builtins::fea) const ERROR_OPERATION: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.FEA.BUILTIN.OPERATION_FAILED",
        identifier: Some("RunMat:fea:OperationFailed"),
        when: "A validation, planning, run, result-query, comparison, or trend operation fails.",
        message: "fea: operation failed",
    };
pub(in crate::builtins::fea) const ERROR_INTERNAL: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.FEA.BUILTIN.INTERNAL",
        identifier: Some("RunMat:fea:Internal"),
        when: "An FEA object or operation result cannot be converted to a RunMat value.",
        message: "fea: internal error",
    };
pub(in crate::builtins::fea) const ERRORS: [BuiltinErrorDescriptor; 4] =
    [ERROR_LOAD, ERROR_INPUT, ERROR_OPERATION, ERROR_INTERNAL];

pub const FEA_LOAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_STUDY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STUDY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_AUTHOR_STUDY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AUTHOR_STUDY_SIGNATURES,
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
pub const FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATERIAL_ASSIGNMENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_LOAD_CASE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOAD_CASE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_DOMAIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOMAIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_INTERFACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTERFACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_COMPONENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPONENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_STEP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STEP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const FEA_RUN_OPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RUN_OPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
