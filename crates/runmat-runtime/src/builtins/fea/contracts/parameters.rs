use runmat_builtins::BuiltinParamArity;
use runmat_builtins::BuiltinParamDescriptor;
use runmat_builtins::BuiltinParamType;

pub(in crate::builtins::fea) const OUT_ANY: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "result",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "FEA object or operation result.",
    }];
pub(in crate::builtins::fea) const IN_PATH: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to a .fea file.",
    }];
pub(in crate::builtins::fea) const IN_INPUT: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "study",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A .fea path, fea.Study object, or fea.Sweep object.",
    }];
pub(in crate::builtins::fea) const IN_STUDY_ARGS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Study id.",
    },
    BuiltinParamDescriptor {
        name: "geometry",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "geometry.Asset returned by geometry.load.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Required Profile plus Backend, ModelId, and model setup options.",
    },
];
pub(in crate::builtins::fea) const IN_AUTHOR_STUDY_ARGS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Study id.",
    },
    BuiltinParamDescriptor {
        name: "geometry",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "geometry.Asset returned by geometry.load.",
    },
    BuiltinParamDescriptor {
        name: "meshAuthoringSummary",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Compact mesh authoring evidence summary.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Required Profile plus Backend, boundary/driving region selectors, structural force vector, and analysis mesh artifact paths.",
    },
];
pub(in crate::builtins::fea) const IN_VARIADIC_ARGS: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Constructor or query arguments.",
    }];

pub(in crate::builtins::fea) const MODEL_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Model id.",
    },
    BuiltinParamDescriptor {
        name: "geometry",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing geometry.Asset object.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Profile, frame, defaults, and typed model-component options.",
    },
];
pub(in crate::builtins::fea) const MATERIAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Material id.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Mechanical, thermal, acoustic, electrical, and plastic material fields.",
    },
];
pub(in crate::builtins::fea) const MATERIAL_ASSIGNMENT_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "region",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Geometry region selector.",
    },
    BuiltinParamDescriptor {
        name: "material",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned material id.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Expected-material and confidence options.",
    },
];
pub(in crate::builtins::fea) const LOAD_CASE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Load-case id.",
    },
    BuiltinParamDescriptor {
        name: "region",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Geometry region selector.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Load kind.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Numeric fields required by the selected load kind.",
    },
];
pub(in crate::builtins::fea) const DOMAIN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Physics-domain kind.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Typed fields for the selected domain kind.",
    },
];
pub(in crate::builtins::fea) const INTERFACE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Interface id.",
    },
    BuiltinParamDescriptor {
        name: "primaryRegion",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Primary geometry region selector.",
    },
    BuiltinParamDescriptor {
        name: "secondaryRegion",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Secondary geometry region selector.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Interface kind and numeric fields.",
    },
];
pub(in crate::builtins::fea) const STEP_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analysis-step id.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Static, modal, transient, thermal, nonlinear, electromagnetic, or CFD step kind.",
    },
];
pub(in crate::builtins::fea) const RUN_OPTIONS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "solver",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "FEA solver family.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Family-specific structural, tolerance, timing, precision, and quality options.",
    },
];
pub(in crate::builtins::fea) const SWEEP_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sweep id.",
    },
    BuiltinParamDescriptor {
        name: "studies",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A fea.Study or cell array of fea.Study objects.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional logical FailFast control.",
    },
];
pub(in crate::builtins::fea) const RESULTS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "runOrRunId",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Persisted run id, fea.RunResult, or fea.Results object.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Field, diagnostic, one-based mode/snapshot selector, and inclusion options.",
    },
];
pub(in crate::builtins::fea) const TRENDS_INPUTS: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional positive integer WindowSize.",
    }];
pub(in crate::builtins::fea) const FIELD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "resultsOrRun",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "fea.Results, fea.RunResult, or persisted run id.",
    },
    BuiltinParamDescriptor {
        name: "fieldId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exact field id or unique dotted suffix.",
    },
];
pub(in crate::builtins::fea) const PLOT_CONTEXT_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "context",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "FEA run, results, field, or study context.",
    },
    BuiltinParamDescriptor {
        name: "fieldId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional field id.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional Field or FieldId selector.",
    },
];
pub(in crate::builtins::fea) const PLOT_STUDY_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "study",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "FEA study carrying geometry context.",
    },
    BuiltinParamDescriptor {
        name: "runId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Persisted run id.",
    },
    BuiltinParamDescriptor {
        name: "fieldId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional field id.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional Field or FieldId selector.",
    },
];
pub(in crate::builtins::fea) const COMPARE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "baselineRunId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Baseline persisted run id.",
    },
    BuiltinParamDescriptor {
        name: "candidateRunId",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Candidate persisted run id.",
    },
];
