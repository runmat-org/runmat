use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, AnalysisInterface, AnalysisInterfaceKind, AnalysisModel,
    AnalysisModelId, AnalysisStep, AnalysisStepKind, BoundaryCondition, BoundaryConditionKind,
    CfdDomain, ElectroThermalDomain, ElectromagneticDomain, EvidenceConfidence, LoadCase, LoadKind,
    MaterialAcousticModel, MaterialAssignment, MaterialElectricalModel, MaterialMechanicalModel,
    MaterialModel, MaterialPlasticModel, MaterialThermalModel, ReferenceFrame,
    ThermoMechanicalDomain,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ClassDef, IntValue, IntegerStorage, MethodDef, NumericScalar,
    ObjectInstance, Tensor, Value,
};
use runmat_geometry_core::GeometryAsset;
use runmat_macros::runtime_builtin;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::analysis::{
    analysis_create_model_op, analysis_plan_study_op, analysis_plan_study_sweep_op,
    analysis_results_by_run_id_op, analysis_results_compare_op, analysis_run_study_op,
    analysis_run_study_sweep_op, analysis_trends_op, analysis_validate_study_op,
    analysis_validate_study_sweep_op, load_fea_document_from_path_async,
    AnalysisAcousticRunOptions, AnalysisCfdRunOptions, AnalysisChtRunOptions,
    AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile, AnalysisElectromagneticRunOptions,
    AnalysisFieldDescriptor, AnalysisFsiRunOptions, AnalysisModalRunOptions,
    AnalysisNonlinearRunOptions, AnalysisResultsCompareQuery, AnalysisResultsQuery,
    AnalysisRunKind, AnalysisRunOptions, AnalysisStudySpec, AnalysisStudySweepData,
    AnalysisStudySweepFailureEntry, AnalysisStudySweepPlanData, AnalysisStudySweepSpec,
    AnalysisThermalRunOptions, AnalysisTransientRunOptions, AnalysisTrendsQuery,
    FeaResolvedDocument,
};
use crate::builtins::common::{json::int_value_to_json, tensor as tensor_utils};
use crate::builtins::geometry::{GEOMETRY_ASSET_CLASS, GEOMETRY_ASSET_JSON_PROPERTY};
use crate::builtins::io::json::jsondecode::value_from_json;
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod author_study;

const FEA_STUDY_CLASS: &str = "fea.Study";
const FEA_SWEEP_CLASS: &str = "fea.Sweep";
const FEA_VALIDATION_CLASS: &str = "fea.Validation";
const FEA_PLAN_CLASS: &str = "fea.Plan";
const FEA_RUN_RESULT_CLASS: &str = "fea.RunResult";
const FEA_MODEL_CLASS: &str = "fea.Model";
const FEA_MATERIAL_CLASS: &str = "fea.Material";
const FEA_MATERIAL_ASSIGNMENT_CLASS: &str = "fea.MaterialAssignment";
const FEA_BOUNDARY_CONDITION_CLASS: &str = "fea.BoundaryCondition";
const FEA_LOAD_CASE_CLASS: &str = "fea.LoadCase";
const FEA_STEP_CLASS: &str = "fea.Step";
const FEA_DOMAIN_CLASS: &str = "fea.Domain";
const FEA_INTERFACE_CLASS: &str = "fea.Interface";
const FEA_RUN_OPTIONS_CLASS: &str = "fea.RunOptions";
const FEA_RESULTS_CLASS: &str = "fea.Results";
const FEA_FIELD_CLASS: &str = "fea.Field";
const FEA_COMPARE_CLASS: &str = "fea.Compare";
const FEA_TRENDS_CLASS: &str = "fea.Trends";
const FEA_STUDY_SPEC_JSON_PROPERTY: &str = "__runmat_fea_study_spec_json";
const FEA_SWEEP_SPEC_JSON_PROPERTY: &str = "__runmat_fea_sweep_spec_json";
const FEA_PAYLOAD_JSON_PROPERTY: &str = "__runmat_fea_payload_json";
const FEA_STUDY_CONTEXT_JSON_PROPERTY: &str = "__runmat_fea_study_context_json";
const FEA_RUN_ID_CONTEXT_PROPERTY: &str = "__runmat_fea_run_id";

const LOAD_NAME: &str = "fea.load";
const STUDY_NAME: &str = "fea.study";
const AUTHOR_STUDY_NAME: &str = "fea.authorStudy";
const SWEEP_NAME: &str = "fea.sweep";
const MODEL_NAME: &str = "fea.model";
const MATERIAL_NAME: &str = "fea.material";
const MATERIAL_ASSIGNMENT_NAME: &str = "fea.materialAssignment";
const BOUNDARY_CONDITION_NAME: &str = "fea.boundaryCondition";
const LOAD_CASE_NAME: &str = "fea.loadCase";
const STEP_NAME: &str = "fea.step";
const DOMAIN_NAME: &str = "fea.domain";
const INTERFACE_NAME: &str = "fea.interface";
const RUN_OPTIONS_NAME: &str = "fea.runOptions";
const VALIDATE_NAME: &str = "fea.validate";
const PLAN_NAME: &str = "fea.plan";
const RUN_NAME: &str = "fea.run";
const RESULTS_NAME: &str = "fea.results";
const FIELD_NAME: &str = "fea.field";
const PLOT_NAME: &str = "fea.plot";
const COMPARE_NAME: &str = "fea.compare";
const TRENDS_NAME: &str = "fea.trends";

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "FEA object or operation result.",
}];
const IN_PATH: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to a .fea file.",
}];
const IN_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "study",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A .fea path, fea.Study object, or fea.Sweep object.",
}];
const IN_STUDY_ARGS: [BuiltinParamDescriptor; 3] = [
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
const IN_AUTHOR_STUDY_ARGS: [BuiltinParamDescriptor; 4] = [
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
const IN_VARIADIC_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Constructor or query arguments.",
}];

const MODEL_INPUTS: [BuiltinParamDescriptor; 3] = [
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
const MATERIAL_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const MATERIAL_ASSIGNMENT_INPUTS: [BuiltinParamDescriptor; 3] = [
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
const LOAD_CASE_INPUTS: [BuiltinParamDescriptor; 4] = [
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
const DOMAIN_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const INTERFACE_INPUTS: [BuiltinParamDescriptor; 4] = [
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
const STEP_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const RUN_OPTIONS_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const SWEEP_INPUTS: [BuiltinParamDescriptor; 3] = [
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
const RESULTS_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const TRENDS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Name, Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional positive integer WindowSize.",
}];
const FIELD_INPUTS: [BuiltinParamDescriptor; 2] = [
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
const PLOT_CONTEXT_INPUTS: [BuiltinParamDescriptor; 3] = [
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
const PLOT_STUDY_INPUTS: [BuiltinParamDescriptor; 4] = [
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
const COMPARE_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "doc = fea.load(path)",
    inputs: &IN_PATH,
    outputs: &OUT_ANY,
}];
const STUDY_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
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
const AUTHOR_STUDY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "study = fea.authorStudy(id, geometry, meshAuthoringSummary, Name, Value, ...)",
    inputs: &IN_AUTHOR_STUDY_ARGS,
    outputs: &OUT_ANY,
}];
const VALIDATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "result = fea.validate(studyOrSweepOrPath)",
    inputs: &IN_INPUT,
    outputs: &OUT_ANY,
}];
const PLAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "plan = fea.plan(study)",
    inputs: &IN_INPUT,
    outputs: &OUT_ANY,
}];
const RUN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "run = fea.run(studyOrSweepOrPath)",
    inputs: &IN_INPUT,
    outputs: &OUT_ANY,
}];
const SWEEP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "sweep = fea.sweep(id, studies, Name, Value, ...)",
    inputs: &SWEEP_INPUTS,
    outputs: &OUT_ANY,
}];
const MODEL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "model = fea.model(id, geometry, Name, Value, ...)",
    inputs: &MODEL_INPUTS,
    outputs: &OUT_ANY,
}];
const MATERIAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "material = fea.material(id, Name, Value, ...)",
    inputs: &MATERIAL_INPUTS,
    outputs: &OUT_ANY,
}];
const MATERIAL_ASSIGNMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "assignment = fea.materialAssignment(region, material, Name, Value, ...)",
        inputs: &MATERIAL_ASSIGNMENT_INPUTS,
        outputs: &OUT_ANY,
    }];
const LOAD_CASE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "load = fea.loadCase(id, region, kind, Name, Value, ...)",
    inputs: &LOAD_CASE_INPUTS,
    outputs: &OUT_ANY,
}];
const DOMAIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "domain = fea.domain(kind, Name, Value, ...)",
    inputs: &DOMAIN_INPUTS,
    outputs: &OUT_ANY,
}];
const INTERFACE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "interface = fea.interface(id, primaryRegion, secondaryRegion, Name, Value, ...)",
    inputs: &INTERFACE_INPUTS,
    outputs: &OUT_ANY,
}];
const COMPONENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "component = fea.component(args, ...)",
    inputs: &IN_VARIADIC_ARGS,
    outputs: &OUT_ANY,
}];
const STEP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "step = fea.step(id, kind)",
    inputs: &STEP_INPUTS,
    outputs: &OUT_ANY,
}];
const RUN_OPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "options = fea.runOptions(solver, Name, Value, ...)",
    inputs: &RUN_OPTIONS_INPUTS,
    outputs: &OUT_ANY,
}];
const BOUNDARY_CONDITION_INPUTS: [BuiltinParamDescriptor; 4] = [
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
const BOUNDARY_CONDITION_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "bc = fea.boundaryCondition(id, region, kind, Name, Value, ...)",
        inputs: &BOUNDARY_CONDITION_INPUTS,
        outputs: &OUT_ANY,
    }];
const RESULTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "results = fea.results(runOrRunId, Name, Value, ...)",
    inputs: &RESULTS_INPUTS,
    outputs: &OUT_ANY,
}];
const FIELD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.field(resultsOrRun, fieldId)",
    inputs: &FIELD_INPUTS,
    outputs: &OUT_ANY,
}];
const PLOT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
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
const COMPARE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "comparison = fea.compare(baselineRunId, candidateRunId)",
    inputs: &COMPARE_INPUTS,
    outputs: &OUT_ANY,
}];
const TRENDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "trends = fea.trends(Name, Value, ...)",
    inputs: &TRENDS_INPUTS,
    outputs: &OUT_ANY,
}];

const ERROR_LOAD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.LOAD_FAILED",
    identifier: Some("RunMat:fea:LoadFailed"),
    when: "A .fea document cannot be read, parsed, or resolved.",
    message: "fea: failed to load FEA document",
};
const ERROR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.INVALID_INPUT",
    identifier: Some("RunMat:fea:InvalidInput"),
    when: "A builtin receives an unsupported argument pattern or object type.",
    message: "fea: invalid input",
};
const ERROR_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.OPERATION_FAILED",
    identifier: Some("RunMat:fea:OperationFailed"),
    when: "A validation, planning, run, result-query, comparison, or trend operation fails.",
    message: "fea: operation failed",
};
const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEA.BUILTIN.INTERNAL",
    identifier: Some("RunMat:fea:Internal"),
    when: "An FEA object or operation result cannot be converted to a RunMat value.",
    message: "fea: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] =
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

const fn fea_floating_input(
    name: &'static str,
    scalar_double: BuiltinIntegerScalarDoubleRule,
) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double,
        notes: "Host integers cross once into a finite binary64 physics field; provider-resident values are rejected.",
    }
}

const fn fea_floating_capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
    overload: BuiltinIntegerOverloadKind,
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload,
        notes: "The RunMat-native constructor validates its host value and performs one explicit IEEE-754 binary64 model-storage conversion; wide integers can round.",
    }
}

const MATERIAL_MECHANICAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "mechanical numeric fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];
const MATERIAL_THERMAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "thermal numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
const MATERIAL_ACOUSTIC_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "acoustic numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
const MATERIAL_ELECTRICAL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [fea_floating_input(
        "electrical numeric fields",
        BuiltinIntegerScalarDoubleRule::Allowed,
    )];
const MATERIAL_RESPONSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "conductivity response numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
const MATERIAL_PLASTIC_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "plastic numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
pub const FEA_MATERIAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 6] = [
    fea_floating_capability(
        "mechanical material fields",
        &MATERIAL_MECHANICAL_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "thermal material fields",
        &MATERIAL_THERMAL_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "acoustic material fields",
        &MATERIAL_ACOUSTIC_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "electrical material fields",
        &MATERIAL_ELECTRICAL_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "electrical frequency-response fields",
        &MATERIAL_RESPONSE_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "plastic material fields",
        &MATERIAL_PLASTIC_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
];

const LOAD_VECTOR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "three-element vector",
    BuiltinIntegerScalarDoubleRule::NotApplicable,
)];
const LOAD_SCALAR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "scalar physics fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
const LOAD_CURRENT_DENSITY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    fea_floating_input(
        "three-element vector",
        BuiltinIntegerScalarDoubleRule::NotApplicable,
    ),
    fea_floating_input(
        "phase and amplitude",
        BuiltinIntegerScalarDoubleRule::Allowed,
    ),
];
pub const FEA_LOAD_CASE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 7] = [
    fea_floating_capability(
        "force vector",
        &LOAD_VECTOR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::FunctionSpecific,
    ),
    fea_floating_capability(
        "moment or torque vector",
        &LOAD_VECTOR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::FunctionSpecific,
    ),
    fea_floating_capability(
        "pressure magnitude",
        &LOAD_SCALAR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
    fea_floating_capability(
        "body-force vector",
        &LOAD_VECTOR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::FunctionSpecific,
    ),
    fea_floating_capability(
        "current-density fields",
        &LOAD_CURRENT_DENSITY_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "coil-current fields",
        &LOAD_SCALAR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "volumetric heat source",
        &LOAD_SCALAR_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::ScalarOnly,
    ),
];

const DOMAIN_FLOATING_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "domain numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
const DOMAIN_REVISION_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "field_source.revision",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes: "The structural revision is decoded exactly as u32 and rejects negative or out-of-range integer values.",
}];
pub const FEA_DOMAIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    fea_floating_capability("thermo-mechanical physics fields", &DOMAIN_FLOATING_INTEGER_INPUTS, BuiltinIntegerOverloadKind::Multiple),
    BuiltinIntegerCapabilityDescriptor { form: "thermo field-source revision", inputs: &DOMAIN_REVISION_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The RunMat-native constructor preserves the exact u32 revision in both typed model storage and its public object representation." },
    fea_floating_capability("electro-thermal physics fields", &DOMAIN_FLOATING_INTEGER_INPUTS, BuiltinIntegerOverloadKind::Multiple),
    fea_floating_capability("electromagnetic physics fields", &DOMAIN_FLOATING_INTEGER_INPUTS, BuiltinIntegerOverloadKind::Multiple),
    fea_floating_capability("CFD physics fields", &DOMAIN_FLOATING_INTEGER_INPUTS, BuiltinIntegerOverloadKind::Multiple),
];

const INTERFACE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "interface numeric fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];
pub const FEA_INTERFACE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    fea_floating_capability(
        "contact interface fields",
        &INTERFACE_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "fluid-structure interface fields",
        &INTERFACE_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
    fea_floating_capability(
        "conjugate-heat-transfer interface fields",
        &INTERFACE_INTEGER_INPUTS,
        BuiltinIntegerOverloadKind::Multiple,
    ),
];

pub const FEA_STRUCTURAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This RunMat-native FEA API accepts object, text, or enum inputs rather than numeric data; nested typed objects retain their already-defined numeric contracts and no provider gather occurs.",
};

const RUN_OPTIONS_EXACT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "structural iteration, step, retry, mode, and refresh counts",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Host integer scalars and integral scalar doubles are range-checked and decoded exactly as usize; provider-resident values are rejected.",
    }];
const RUN_OPTIONS_FLOATING_INPUTS: [BuiltinIntegerInputCapability; 1] = [fea_floating_input(
    "tolerance, timing, residual, convergence, and frequency fields",
    BuiltinIntegerScalarDoubleRule::Allowed,
)];

const fn run_options_exact_capability(form: &'static str) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &RUN_OPTIONS_EXACT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact count is preserved in the typed run-options payload and public object representation.",
    }
}

const fn run_options_floating_capability(form: &'static str) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &RUN_OPTIONS_FLOATING_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Floating solver controls use finite IEEE-754 binary64 storage; wide integer inputs can round while structural counts remain exact.",
    }
}

pub const FEA_RUN_OPTIONS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 18] = [
    run_options_exact_capability("modal structural controls"),
    run_options_floating_capability("modal floating controls"),
    run_options_exact_capability("acoustic structural controls"),
    run_options_floating_capability("acoustic floating controls"),
    run_options_exact_capability("thermal structural controls"),
    run_options_floating_capability("thermal floating controls"),
    run_options_exact_capability("transient structural controls"),
    run_options_floating_capability("transient floating controls"),
    run_options_exact_capability("CFD structural controls"),
    run_options_floating_capability("CFD floating controls"),
    run_options_exact_capability("CHT structural controls"),
    run_options_floating_capability("CHT floating controls"),
    run_options_exact_capability("FSI structural controls"),
    run_options_floating_capability("FSI floating controls"),
    run_options_exact_capability("nonlinear structural controls"),
    run_options_floating_capability("nonlinear floating controls"),
    run_options_exact_capability("electromagnetic structural controls"),
    run_options_floating_capability("electromagnetic floating controls"),
];

const RESULTS_SELECTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "one-based result indices",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Host numeric scalars or vectors are decoded exactly, require positive one-based indices, preserve order and duplicates, and reject matrix and provider-resident inputs.",
    }];
const RESULTS_FLAG_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric inclusion predicate",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Host scalar logical values and exact numeric zero or one are accepted; every other numeric value and provider-resident input is rejected.",
    }];
pub const FEA_RESULTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "ModeIndices", inputs: &RESULTS_SELECTOR_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The one-based public selector is translated once to the operation layer's zero-based index and public structural result fields remain exact." },
    BuiltinIntegerCapabilityDescriptor { form: "TransientSnapshotIndices", inputs: &RESULTS_SELECTOR_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The one-based public selector is translated once to the operation layer's zero-based index and public structural result fields remain exact." },
    BuiltinIntegerCapabilityDescriptor { form: "numeric inclusion predicates", inputs: &RESULTS_FLAG_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Logical and numeric zero/one select query projections without converting provider data." },
];

const TRENDS_WINDOW_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "WindowSize",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A positive host integer scalar or integral scalar double is decoded exactly as usize; provider-resident values are rejected.",
    }];
pub const FEA_TRENDS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "WindowSize", inputs: &TRENDS_WINDOW_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The positive window size and structural trend counts remain exact in the public result object; time and rate fields remain binary64." }];
pub const FEA_BOUNDARY_CONDITION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BOUNDARY_CONDITION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const fn boundary_integer_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "An exact scalar is converted once to the model's binary64 storage field using Rust's IEEE-754 integer-to-f64 conversion.",
    }
}

const BOUNDARY_ROTATION_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    boundary_integer_input("rx"),
    boundary_integer_input("ry"),
    boundary_integer_input("rz"),
];
const BOUNDARY_IMPEDANCE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [boundary_integer_input("specificImpedancePaSPerM")];
const BOUNDARY_TEMPERATURE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [boundary_integer_input("temperatureK")];
const BOUNDARY_HEAT_FLUX_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [boundary_integer_input("heatFluxWPerM2")];
const BOUNDARY_CONVECTION_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    boundary_integer_input("ambientTemperatureK"),
    boundary_integer_input("coefficientWPerM2K"),
];
const BOUNDARY_INLET_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [boundary_integer_input("velocityMPerS")];
const BOUNDARY_OUTLET_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [boundary_integer_input("pressurePa")];

const fn boundary_integer_capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The constructor validates scalar shape and finiteness, then performs one explicit IEEE-754 binary64 model-storage conversion; wide integers can therefore round.",
    }
}

pub const FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 7] = [
    boundary_integer_capability(
        "prescribedRotation integer fields",
        &BOUNDARY_ROTATION_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "acousticImpedance integer field",
        &BOUNDARY_IMPEDANCE_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "thermalPrescribedTemperature integer field",
        &BOUNDARY_TEMPERATURE_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "thermalHeatFlux integer field",
        &BOUNDARY_HEAT_FLUX_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "thermalConvection integer fields",
        &BOUNDARY_CONVECTION_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "cfdInletVelocity integer field",
        &BOUNDARY_INLET_INTEGER_INPUTS,
    ),
    boundary_integer_capability(
        "cfdOutletPressure integer field",
        &BOUNDARY_OUTLET_INTEGER_INPUTS,
    ),
];
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
pub const FEA_PLOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PLOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
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

#[runtime_builtin(
    name = "fea.load",
    category = "fea",
    summary = "Load a .fea study or sweep document.",
    keywords = "fea,study,sweep,load,yaml",
    descriptor(crate::builtins::fea::FEA_LOAD_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_load_builtin(path: String) -> BuiltinResult<Value> {
    load_document_object(PathBuf::from(path)).await
}

#[runtime_builtin(
    name = "fea.study",
    category = "fea",
    summary = "Create a typed FEA study from geometry, model data, and run settings.",
    keywords = "fea,study,geometry,run",
    descriptor(crate::builtins::fea::FEA_STUDY_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_study_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() == 1 {
        let path = scalar_string(&args[0], STUDY_NAME, &ERROR_INPUT)?;
        return load_document_object(PathBuf::from(path)).await;
    }
    create_study_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.authorStudy",
    category = "fea",
    summary = "Author a typed FEA study from compact mesh authoring evidence.",
    keywords = "fea,study,author,mesh,evidence,agent",
    descriptor(crate::builtins::fea::FEA_AUTHOR_STUDY_DESCRIPTOR),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_author_study_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    author_study::create_author_study_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.sweep",
    category = "fea",
    summary = "Create a FEA study sweep from study objects.",
    keywords = "fea,sweep,study,run",
    descriptor(crate::builtins::fea::FEA_SWEEP_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_sweep_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_sweep_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.model",
    category = "fea",
    summary = "Create a typed FEA model object from geometry and model components.",
    keywords = "fea,model,materials,boundary,loads,domains",
    descriptor(crate::builtins::fea::FEA_MODEL_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_model_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_model_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.material",
    category = "fea",
    summary = "Create a typed FEA material object.",
    keywords = "fea,material,mechanical,thermal,electrical,plastic",
    descriptor(crate::builtins::fea::FEA_MATERIAL_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_MATERIAL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_material_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_material_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.materialAssignment",
    category = "fea",
    summary = "Create a typed FEA material assignment.",
    keywords = "fea,material,assignment,region",
    descriptor(crate::builtins::fea::FEA_MATERIAL_ASSIGNMENT_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_material_assignment_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_material_assignment_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.boundaryCondition",
    category = "fea",
    summary = "Create a typed FEA boundary condition.",
    keywords = "fea,boundary,condition,region,prescribed,rotation",
    descriptor(crate::builtins::fea::FEA_BOUNDARY_CONDITION_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_boundary_condition_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_boundary_condition_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.loadCase",
    category = "fea",
    summary = "Create a typed FEA load case.",
    keywords = "fea,load,force,moment,torque,pressure,current",
    descriptor(crate::builtins::fea::FEA_LOAD_CASE_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_LOAD_CASE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_load_case_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_load_case_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.step",
    category = "fea",
    summary = "Create a typed FEA analysis step.",
    keywords = "fea,step,static,modal,transient",
    descriptor(crate::builtins::fea::FEA_STEP_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_step_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_step_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.domain",
    category = "fea",
    summary = "Create a typed FEA physics domain object.",
    keywords = "fea,domain,thermal,electromagnetic,cfd",
    descriptor(crate::builtins::fea::FEA_DOMAIN_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_DOMAIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_domain_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_domain_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.interface",
    category = "fea",
    summary = "Create a typed FEA interface object.",
    keywords = "fea,interface,contact,region",
    descriptor(crate::builtins::fea::FEA_INTERFACE_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_INTERFACE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_interface_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_interface_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.runOptions",
    category = "fea",
    summary = "Create typed FEA run options for a solver.",
    keywords = "fea,run,options,solver,quality",
    descriptor(crate::builtins::fea::FEA_RUN_OPTIONS_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_RUN_OPTIONS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_run_options_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_run_options_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.validate",
    category = "fea",
    summary = "Validate a FEA study or sweep without planning or solving.",
    keywords = "fea,validate,study,sweep",
    descriptor(crate::builtins::fea::FEA_VALIDATE_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_validate_builtin(input: Value) -> BuiltinResult<Value> {
    match resolve_document_input(input, VALIDATE_NAME).await? {
        FeaResolvedDocument::Study(spec) => operation_result_to_object(
            VALIDATE_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_VALIDATION_CLASS,
            analysis_validate_study_op(&spec, OperationContext::new(None, None)),
            None,
        ),
        FeaResolvedDocument::Sweep(spec) => operation_result_to_object(
            VALIDATE_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_VALIDATION_CLASS,
            analysis_validate_study_sweep_op(&spec, OperationContext::new(None, None)),
            None,
        ),
    }
}

#[runtime_builtin(
    name = "fea.plan",
    category = "fea",
    summary = "Plan a FEA study or sweep without solving it.",
    keywords = "fea,plan,study,sweep",
    descriptor(crate::builtins::fea::FEA_PLAN_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_plan_builtin(input: Value) -> BuiltinResult<Value> {
    match resolve_document_input(input, PLAN_NAME).await? {
        FeaResolvedDocument::Study(spec) => operation_result_to_object(
            PLAN_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_PLAN_CLASS,
            analysis_plan_study_op(&spec, OperationContext::new(None, None)),
            None,
        ),
        FeaResolvedDocument::Sweep(spec) => sweep_plan_result_to_object(
            analysis_plan_study_sweep_op(&spec, OperationContext::new(None, None)),
        ),
    }
}

#[runtime_builtin(
    name = "fea.run",
    category = "fea",
    summary = "Run a FEA study or sweep.",
    keywords = "fea,run,study,sweep,solve",
    descriptor(crate::builtins::fea::FEA_RUN_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_run_builtin(input: Value) -> BuiltinResult<Value> {
    match resolve_document_input(input, RUN_NAME).await? {
        FeaResolvedDocument::Study(spec) => run_study_result_to_object(&spec),
        FeaResolvedDocument::Sweep(spec) => sweep_run_result_to_object(
            analysis_run_study_sweep_op(&spec, OperationContext::new(None, None)),
        ),
    }
}

#[runtime_builtin(
    name = "fea.results",
    category = "fea",
    summary = "Load or project FEA run results for post-processing.",
    keywords = "fea,results,run_id,fields,diagnostics",
    descriptor(crate::builtins::fea::FEA_RESULTS_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_RESULTS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_results_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_results_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.field",
    category = "fea",
    summary = "Extract a field from FEA results or a run result.",
    keywords = "fea,field,displacement,von_mises,post",
    descriptor(crate::builtins::fea::FEA_FIELD_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_field_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_field_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.plot",
    category = "fea",
    summary = "Create a RunMat figure for an FEA result field on its geometry mesh.",
    keywords = "fea,plot,visualize,mesh,von_mises,stress,field",
    descriptor(crate::builtins::fea::FEA_PLOT_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_plot_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_plot_from_args(args)
}

#[runtime_builtin(
    name = "fea.compare",
    category = "fea",
    summary = "Compare two persisted FEA runs by run id.",
    keywords = "fea,compare,run_id,quality",
    descriptor(crate::builtins::fea::FEA_COMPARE_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_compare_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_compare_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.trends",
    category = "fea",
    summary = "Summarize recent persisted FEA run trends.",
    keywords = "fea,trends,history,quality",
    descriptor(crate::builtins::fea::FEA_TRENDS_DESCRIPTOR),
    integer_capabilities(crate::builtins::fea::FEA_TRENDS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_trends_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_trends_object_from_args(args)
}

async fn load_document_object(path: PathBuf) -> BuiltinResult<Value> {
    let document = load_fea_document_from_path_async(&path)
        .await
        .map_err(|err| builtin_error(LOAD_NAME, &ERROR_LOAD, err))?;
    resolved_document_to_object(document)
}

async fn resolve_document_input(
    input: Value,
    builtin: &'static str,
) -> BuiltinResult<FeaResolvedDocument> {
    match input {
        Value::Object(object) if object.class_name == FEA_STUDY_CLASS => {
            let spec: AnalysisStudySpec =
                object_json_property(builtin, &object, FEA_STUDY_SPEC_JSON_PROPERTY, &ERROR_INPUT)?;
            Ok(FeaResolvedDocument::Study(Box::new(spec)))
        }
        Value::Object(object) if object.class_name == FEA_SWEEP_CLASS => {
            let spec: AnalysisStudySweepSpec =
                object_json_property(builtin, &object, FEA_SWEEP_SPEC_JSON_PROPERTY, &ERROR_INPUT)?;
            Ok(FeaResolvedDocument::Sweep(spec))
        }
        Value::String(path) => load_fea_document_from_path_async(&PathBuf::from(path))
            .await
            .map_err(|err| builtin_error(builtin, &ERROR_LOAD, err)),
        Value::CharArray(chars) if chars.rows == 1 => {
            let path: String = chars.data.iter().collect();
            load_fea_document_from_path_async(&PathBuf::from(path))
                .await
                .map_err(|err| builtin_error(builtin, &ERROR_LOAD, err))
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected .fea path, {FEA_STUDY_CLASS}, or {FEA_SWEEP_CLASS}; got {other:?}"),
        )),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunOptionsPayload {
    run_kind: AnalysisRunKind,
    options: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DomainPayload {
    kind: String,
    data: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelDefaultsMode {
    ProfileScaffold,
    None,
}

impl Default for ModelDefaultsMode {
    fn default() -> Self {
        Self::ProfileScaffold
    }
}

#[derive(Debug, Default)]
struct StudyConstructorOptions {
    run_kind: Option<AnalysisRunKind>,
    profile: Option<AnalysisCreateModelProfile>,
    backend: Option<ComputeBackend>,
    model_id: Option<String>,
    model: Option<AnalysisModel>,
    frame: Option<ReferenceFrame>,
    model_defaults: ModelDefaultsMode,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<MaterialAssignment>,
    boundary_conditions: Vec<BoundaryCondition>,
    loads: Vec<LoadCase>,
    steps: Vec<AnalysisStep>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<AnalysisInterface>,
    run_options: Option<RunOptionsPayload>,
}

fn create_study_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            "fea.study requires id and geometry arguments",
        ));
    }
    let study_id = scalar_string(&args[0], STUDY_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(STUDY_NAME, &args[1])?;
    let options = StudyConstructorOptions::parse(&args[2..])?;
    let (profile, run_kind) = resolve_study_profile_and_run_kind(&options)?;
    let model_id = options.model_id.clone().unwrap_or_else(|| {
        options
            .model
            .as_ref()
            .map(|model| model.model_id.0.clone())
            .unwrap_or_else(|| format!("{}_model", sanitize_id(&study_id)))
    });
    let model = match options.model {
        Some(model) => Some(model),
        None if options.has_model_components() => Some(build_model_from_parts(
            STUDY_NAME,
            &geometry,
            model_id.clone(),
            profile,
            options.model_defaults,
            options.frame,
            options.materials,
            options.material_assignments,
            options.boundary_conditions,
            options.loads,
            options.steps,
            options.domains,
            options.interfaces,
        )?),
        None => None,
    };
    let run_options = options
        .run_options
        .map(|payload| resolved_run_options_from_payload(STUDY_NAME, payload, run_kind))
        .transpose()?
        .unwrap_or_default();
    let spec = AnalysisStudySpec {
        study_id,
        geometry,
        create_model_intent: AnalysisCreateModelIntentSpec {
            model_id,
            profile,
            prep_context: None,
        },
        model,
        run_kind,
        backend: options.backend.unwrap_or(ComputeBackend::Cpu),
        mesh_options: None,
        outputs: Vec::new(),
        analysis_mesh_artifact_path: None,
        analysis_mesh_evidence_artifact_path: None,
        linear_static_run_options: run_options.linear_static,
        modal_run_options: run_options.modal,
        acoustic_run_options: run_options.acoustic,
        thermal_run_options: run_options.thermal,
        transient_run_options: run_options.transient,
        cfd_run_options: run_options.cfd,
        cht_run_options: run_options.cht,
        fsi_run_options: run_options.fsi,
        nonlinear_run_options: run_options.nonlinear,
        electromagnetic_run_options: run_options.electromagnetic,
    };
    study_to_object(spec)
}

impl StudyConstructorOptions {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(builtin_error(
                STUDY_NAME,
                &ERROR_INPUT,
                "fea.study options must be Name, Value pairs",
            ));
        }
        let mut options = Self::default();
        let mut seen = HashSet::new();
        for pair in args.chunks(2) {
            let key = option_key(&pair[0], STUDY_NAME)?;
            let canonical = match key.as_str() {
                "runkind" | "kind" => "runkind",
                "materialassignments" | "assignments" => "materialassignments",
                "boundaryconditions" | "bcs" => "boundaryconditions",
                "loads" | "loadcases" => "loads",
                "runoptions" | "options" => "runoptions",
                other => other,
            };
            if !seen.insert(canonical.to_string()) {
                return Err(builtin_error(
                    STUDY_NAME,
                    &ERROR_INPUT,
                    format!("duplicate fea.study option `{canonical}`"),
                ));
            }
            match key.as_str() {
                "runkind" | "kind" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.run_kind = Some(parse_scalar_enum(&text, "RunKind")?);
                }
                "profile" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.profile = Some(parse_scalar_enum(&text, "Profile")?);
                }
                "backend" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.backend = Some(parse_scalar_enum(&text, "Backend")?);
                }
                "modelid" => {
                    options.model_id = Some(scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?);
                }
                "model" => {
                    options.model = Some(model_from_value(STUDY_NAME, &pair[1])?);
                }
                "frame" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.frame = Some(parse_scalar_enum(&text, "Frame")?);
                }
                "defaults" => {
                    options.model_defaults = parse_model_defaults_mode(&scalar_string(
                        &pair[1],
                        STUDY_NAME,
                        &ERROR_INPUT,
                    )?)?;
                }
                "materials" => options.materials = material_vec_from_value(STUDY_NAME, &pair[1])?,
                "materialassignments" | "assignments" => {
                    options.material_assignments =
                        material_assignment_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "boundaryconditions" | "bcs" => {
                    options.boundary_conditions =
                        boundary_condition_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "loads" | "loadcases" => {
                    options.loads = load_case_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "steps" => options.steps = step_vec_from_value(STUDY_NAME, &pair[1])?,
                "domains" => options.domains = domain_vec_from_value(STUDY_NAME, &pair[1])?,
                "interfaces" => {
                    options.interfaces = interface_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "runoptions" | "options" => {
                    options.run_options =
                        Some(run_options_payload_from_value(STUDY_NAME, &pair[1])?);
                }
                other => {
                    return Err(builtin_error(
                        STUDY_NAME,
                        &ERROR_INPUT,
                        format!("unsupported fea.study option `{other}`"),
                    ));
                }
            }
        }
        Ok(options)
    }

    fn has_model_components(&self) -> bool {
        self.frame.is_some()
            || !self.materials.is_empty()
            || !self.material_assignments.is_empty()
            || !self.boundary_conditions.is_empty()
            || !self.loads.is_empty()
            || !self.steps.is_empty()
            || !self.domains.is_empty()
            || !self.interfaces.is_empty()
    }
}

#[derive(Debug, Default)]
struct ModelConstructorOptions {
    profile: Option<AnalysisCreateModelProfile>,
    frame: Option<ReferenceFrame>,
    defaults: ModelDefaultsMode,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<MaterialAssignment>,
    boundary_conditions: Vec<BoundaryCondition>,
    loads: Vec<LoadCase>,
    steps: Vec<AnalysisStep>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<AnalysisInterface>,
}

#[derive(Debug, Default)]
struct ResolvedRunOptions {
    linear_static: Option<AnalysisRunOptions>,
    modal: Option<AnalysisModalRunOptions>,
    acoustic: Option<AnalysisAcousticRunOptions>,
    thermal: Option<AnalysisThermalRunOptions>,
    transient: Option<AnalysisTransientRunOptions>,
    cfd: Option<AnalysisCfdRunOptions>,
    cht: Option<AnalysisChtRunOptions>,
    fsi: Option<AnalysisFsiRunOptions>,
    nonlinear: Option<AnalysisNonlinearRunOptions>,
    electromagnetic: Option<AnalysisElectromagneticRunOptions>,
}

fn create_sweep_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            SWEEP_NAME,
            &ERROR_INPUT,
            "fea.sweep requires id and studies arguments",
        ));
    }
    let sweep_id = scalar_string(&args[0], SWEEP_NAME, &ERROR_INPUT)?;
    let studies = study_vec_from_value(SWEEP_NAME, &args[1])?;
    let mut fail_fast = true;
    let mut fail_fast_seen = false;
    for pair in expect_name_value_tail(SWEEP_NAME, &args[2..])? {
        match pair.key.as_str() {
            "failfast" => {
                if fail_fast_seen {
                    return Err(builtin_error(
                        SWEEP_NAME,
                        &ERROR_INPUT,
                        "duplicate fea.sweep option `failfast`",
                    ));
                }
                fail_fast_seen = true;
                fail_fast = logical_from_value(SWEEP_NAME, pair.value)?;
            }
            other => {
                return Err(builtin_error(
                    SWEEP_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.sweep option `{other}`"),
                ));
            }
        }
    }
    sweep_to_object(AnalysisStudySweepSpec {
        sweep_id,
        studies,
        fail_fast,
    })
}

fn create_model_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            "fea.model requires id and geometry arguments",
        ));
    }
    let model_id = scalar_string(&args[0], MODEL_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(MODEL_NAME, &args[1])?;
    let options = parse_model_constructor_options(MODEL_NAME, &args[2..])?;
    let profile = options.profile.ok_or_else(|| {
        builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            "fea.model requires Profile; choose a physics profile from fea.capabilities().physicsProfiles",
        )
    })?;
    let model = build_model_from_parts(
        MODEL_NAME,
        &geometry,
        model_id,
        profile,
        options.defaults,
        options.frame,
        options.materials,
        options.material_assignments,
        options.boundary_conditions,
        options.loads,
        options.steps,
        options.domains,
        options.interfaces,
    )?;
    serializable_to_object_preserving_integers(
        MODEL_NAME,
        &ERROR_INTERNAL,
        FEA_MODEL_CLASS,
        &model,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[],
        &["geometry_revision", "revision"],
    )
}

fn parse_model_constructor_options(
    builtin: &'static str,
    args: &[Value],
) -> BuiltinResult<ModelConstructorOptions> {
    let mut options = ModelConstructorOptions::default();
    for pair in expect_name_value_tail(builtin, args)? {
        match pair.key.as_str() {
            "profile" => {
                let text = scalar_string(pair.value, builtin, &ERROR_INPUT)?;
                options.profile = Some(parse_scalar_enum(&text, "Profile")?);
            }
            "frame" => {
                let text = scalar_string(pair.value, builtin, &ERROR_INPUT)?;
                options.frame = Some(parse_scalar_enum(&text, "Frame")?);
            }
            "defaults" => {
                options.defaults =
                    parse_model_defaults_mode(&scalar_string(pair.value, builtin, &ERROR_INPUT)?)?;
            }
            "materials" => options.materials = material_vec_from_value(builtin, pair.value)?,
            "materialassignments" | "assignments" => {
                options.material_assignments =
                    material_assignment_vec_from_value(builtin, pair.value)?;
            }
            "boundaryconditions" | "bcs" => {
                options.boundary_conditions =
                    boundary_condition_vec_from_value(builtin, pair.value)?;
            }
            "loads" | "loadcases" => options.loads = load_case_vec_from_value(builtin, pair.value)?,
            "steps" => options.steps = step_vec_from_value(builtin, pair.value)?,
            "domains" => options.domains = domain_vec_from_value(builtin, pair.value)?,
            "interfaces" => options.interfaces = interface_vec_from_value(builtin, pair.value)?,
            other => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported {builtin} option `{other}`"),
                ));
            }
        }
    }
    Ok(options)
}

fn create_material_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            MATERIAL_NAME,
            &ERROR_INPUT,
            "fea.material requires a material id",
        ));
    }
    let material_id = scalar_string(&args[0], MATERIAL_NAME, &ERROR_INPUT)?;
    let mut fields = json_fields_from_name_values(MATERIAL_NAME, &args[1..])?;
    let name = fields
        .remove("name")
        .map(json_to_string)
        .transpose()?
        .unwrap_or_else(|| material_id.clone());
    let mechanical = if let Some(value) = fields.remove("mechanical") {
        json_deserialize(MATERIAL_NAME, value, "mechanical material model")?
    } else {
        let youngs = remove_required_f64(&mut fields, MATERIAL_NAME, "youngs_modulus_pa")?;
        let poisson = remove_required_f64(&mut fields, MATERIAL_NAME, "poisson_ratio")?;
        let density =
            remove_optional_f64(&mut fields, MATERIAL_NAME, "density_kg_per_m3")?.unwrap_or(7850.0);
        MaterialMechanicalModel {
            youngs_modulus_pa: youngs,
            poisson_ratio: poisson,
            density_kg_per_m3: density,
        }
    };
    let thermal = if let Some(value) = fields.remove("thermal") {
        json_deserialize(MATERIAL_NAME, value, "thermal material model")?
    } else {
        let mut thermal = serde_json::to_value(MaterialThermalModel::default())
            .map_err(|err| builtin_error(MATERIAL_NAME, &ERROR_INTERNAL, err.to_string()))?;
        move_known_fields(
            &mut fields,
            thermal.as_object_mut().expect("thermal model is object"),
            &[
                "reference_temperature_k",
                "modulus_temp_coeff_per_k",
                "conductivity_w_per_mk",
                "specific_heat_j_per_kgk",
                "expansion_coefficient_per_k",
            ],
        );
        json_deserialize(MATERIAL_NAME, thermal, "thermal material model")?
    };
    let electrical = if let Some(value) = fields.remove("electrical") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "electrical material model",
        )?)
    } else {
        let mut electrical = serde_json::to_value(MaterialElectricalModel::default())
            .map_err(|err| builtin_error(MATERIAL_NAME, &ERROR_INTERNAL, err.to_string()))?;
        let moved = move_known_fields(
            &mut fields,
            electrical
                .as_object_mut()
                .expect("electrical material model is object"),
            &[
                "reference_temperature_k",
                "conductivity_s_per_m",
                "resistive_heating_coefficient",
                "relative_permittivity",
                "relative_permeability",
                "conductivity_frequency_response",
            ],
        );
        if moved {
            Some(json_deserialize(
                MATERIAL_NAME,
                electrical,
                "electrical material model",
            )?)
        } else {
            None
        }
    };
    let acoustic = if let Some(value) = fields.remove("acoustic") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "acoustic material model",
        )?)
    } else {
        let mut acoustic = serde_json::to_value(MaterialAcousticModel::default())
            .map_err(|err| builtin_error(MATERIAL_NAME, &ERROR_INTERNAL, err.to_string()))?;
        let moved = move_known_fields(
            &mut fields,
            acoustic
                .as_object_mut()
                .expect("acoustic material model is object"),
            &[
                "density_kg_per_m3",
                "speed_of_sound_m_per_s",
                "damping_ratio",
            ],
        );
        if moved {
            Some(json_deserialize(
                MATERIAL_NAME,
                acoustic,
                "acoustic material model",
            )?)
        } else {
            None
        }
    };
    let plastic = if let Some(value) = fields.remove("plastic") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "plastic material model",
        )?)
    } else if fields.contains_key("yield_strain")
        || fields.contains_key("hardening_modulus_ratio")
        || fields.contains_key("saturation_exponent")
    {
        Some(MaterialPlasticModel {
            yield_strain: remove_required_f64(&mut fields, MATERIAL_NAME, "yield_strain")?,
            hardening_modulus_ratio: remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "hardening_modulus_ratio",
            )?,
            saturation_exponent: remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "saturation_exponent",
            )?,
        })
    } else {
        None
    };
    reject_unknown_fields(MATERIAL_NAME, fields)?;
    material_to_object(MaterialModel {
        material_id,
        name,
        mechanical,
        thermal,
        acoustic,
        electrical,
        plastic,
    })
}

fn create_material_assignment_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            MATERIAL_ASSIGNMENT_NAME,
            &ERROR_INPUT,
            "fea.materialAssignment requires region and material arguments",
        ));
    }
    let region_id = scalar_string(&args[0], MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
    let assigned_material_id = scalar_string(&args[1], MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
    let mut expected_material_id = assigned_material_id.clone();
    let mut confidence = EvidenceConfidence::Verified;
    for pair in expect_name_value_tail(MATERIAL_ASSIGNMENT_NAME, &args[2..])? {
        match pair.key.as_str() {
            "expectedmaterial" | "expectedmaterialid" => {
                expected_material_id =
                    scalar_string(pair.value, MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
            }
            "confidence" => {
                let text = scalar_string(pair.value, MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
                confidence = parse_scalar_enum(&text, "Confidence")?;
            }
            other => {
                return Err(builtin_error(
                    MATERIAL_ASSIGNMENT_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.materialAssignment option `{other}`"),
                ));
            }
        }
    }
    material_assignment_to_object(MaterialAssignment {
        region_id,
        expected_material_id,
        assigned_material_id,
        confidence,
    })
}

fn create_boundary_condition_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            BOUNDARY_CONDITION_NAME,
            &ERROR_INPUT,
            "fea.boundaryCondition requires id, region, and kind arguments",
        ));
    }
    let bc_id = scalar_string(&args[0], BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
    let region_id = scalar_string(&args[1], BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
    let kind_text = scalar_string(&args[2], BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
    let mut fields = boundary_fields_from_name_values(&args[3..])?;
    let kind = match normalize_token(&kind_text).as_str() {
        "prescribedrotation" => BoundaryConditionKind::PrescribedRotation {
            rx: remove_required_boundary_f64(&mut fields, "rx")?,
            ry: remove_required_boundary_f64(&mut fields, "ry")?,
            rz: remove_required_boundary_f64(&mut fields, "rz")?,
        },
        "acousticimpedance" => BoundaryConditionKind::AcousticImpedance {
            specific_impedance_pa_s_per_m: remove_required_boundary_f64(
                &mut fields,
                "specific_impedance_pa_s_per_m",
            )?,
        },
        "thermalprescribedtemperature" => BoundaryConditionKind::ThermalPrescribedTemperature {
            temperature_k: remove_required_boundary_f64(&mut fields, "temperature_k")?,
        },
        "thermalheatflux" => BoundaryConditionKind::ThermalHeatFlux {
            heat_flux_w_per_m2: remove_required_boundary_f64(&mut fields, "heat_flux_w_per_m2")?,
        },
        "thermalconvection" => BoundaryConditionKind::ThermalConvection {
            ambient_temperature_k: remove_required_boundary_f64(
                &mut fields,
                "ambient_temperature_k",
            )?,
            coefficient_w_per_m2k: remove_required_boundary_f64(
                &mut fields,
                "coefficient_w_per_m2k",
            )?,
        },
        "cfdinletvelocity" => BoundaryConditionKind::CfdInletVelocity {
            velocity_m_per_s: remove_required_boundary_f64(&mut fields, "velocity_m_per_s")?,
        },
        "cfdoutletpressure" => BoundaryConditionKind::CfdOutletPressure {
            pressure_pa: remove_required_boundary_f64(&mut fields, "pressure_pa")?,
        },
        _ => parse_scalar_enum_for_builtin::<BoundaryConditionKind>(
            BOUNDARY_CONDITION_NAME,
            &kind_text,
            "BoundaryConditionKind",
        )?,
    };
    reject_unknown_boundary_fields(fields)?;
    boundary_condition_to_object(BoundaryCondition {
        bc_id,
        region_id,
        kind,
    })
}

fn boundary_fields_from_name_values(args: &[Value]) -> BuiltinResult<HashMap<String, &Value>> {
    let mut fields = HashMap::new();
    for pair in expect_name_value_tail(BOUNDARY_CONDITION_NAME, args)? {
        let raw = scalar_string(pair.name, BOUNDARY_CONDITION_NAME, &ERROR_INPUT)?;
        let key = canonical_field_name(&raw);
        if fields.insert(key.clone(), pair.value).is_some() {
            return Err(builtin_error(
                BOUNDARY_CONDITION_NAME,
                &ERROR_INPUT,
                format!("duplicate fea.boundaryCondition option `{key}`"),
            ));
        }
    }
    Ok(fields)
}

fn remove_required_boundary_f64(
    fields: &mut HashMap<String, &Value>,
    key: &str,
) -> BuiltinResult<f64> {
    let Some(value) = fields.remove(key) else {
        return Err(builtin_error(
            BOUNDARY_CONDITION_NAME,
            &ERROR_INPUT,
            format!("missing required option `{key}`"),
        ));
    };
    boundary_numeric_scalar_f64(value, key)
}

fn boundary_numeric_scalar_f64(value: &Value, key: &str) -> BuiltinResult<f64> {
    let converted = match value {
        Value::Num(value) => *value,
        Value::Int(value) => boundary_integer_to_f64(value),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            boundary_numeric_storage_scalar_to_f64(
                tensor
                    .numeric_value_at(0)
                    .expect("validated scalar tensor storage"),
            )
        }
        _ => {
            return Err(builtin_error(
                BOUNDARY_CONDITION_NAME,
                &ERROR_INPUT,
                format!("numeric option `{key}` must be a real numeric scalar"),
            ))
        }
    };
    if !converted.is_finite() {
        return Err(builtin_error(
            BOUNDARY_CONDITION_NAME,
            &ERROR_INPUT,
            format!("numeric option `{key}` must be finite"),
        ));
    }
    Ok(converted)
}

fn boundary_integer_to_f64(value: &IntValue) -> f64 {
    match value {
        IntValue::I8(value) => f64::from(*value),
        IntValue::I16(value) => f64::from(*value),
        IntValue::I32(value) => f64::from(*value),
        IntValue::I64(value) => *value as f64,
        IntValue::U8(value) => f64::from(*value),
        IntValue::U16(value) => f64::from(*value),
        IntValue::U32(value) => f64::from(*value),
        IntValue::U64(value) => *value as f64,
    }
}

fn boundary_numeric_storage_scalar_to_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        NumericScalar::I8(value) => f64::from(value),
        NumericScalar::I16(value) => f64::from(value),
        NumericScalar::I32(value) => f64::from(value),
        NumericScalar::I64(value) => value as f64,
        NumericScalar::U8(value) => f64::from(value),
        NumericScalar::U16(value) => f64::from(value),
        NumericScalar::U32(value) => f64::from(value),
        NumericScalar::U64(value) => value as f64,
    }
}

fn reject_unknown_boundary_fields(fields: HashMap<String, &Value>) -> BuiltinResult<()> {
    if let Some(key) = fields.keys().next() {
        return Err(builtin_error(
            BOUNDARY_CONDITION_NAME,
            &ERROR_INPUT,
            format!("unsupported fea.boundaryCondition option `{key}`"),
        ));
    }
    Ok(())
}

fn create_load_case_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            LOAD_CASE_NAME,
            &ERROR_INPUT,
            "fea.loadCase requires id, region, and kind arguments",
        ));
    }
    let load_id = scalar_string(&args[0], LOAD_CASE_NAME, &ERROR_INPUT)?;
    let region_id = scalar_string(&args[1], LOAD_CASE_NAME, &ERROR_INPUT)?;
    let kind_text = scalar_string(&args[2], LOAD_CASE_NAME, &ERROR_INPUT)?;
    let mut fields = json_fields_from_name_values(LOAD_CASE_NAME, &args[3..])?;
    let kind = match normalize_token(&kind_text).as_str() {
        "force" => {
            let [fx, fy, fz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::Force { fx, fy, fz }
        }
        "moment" | "torque" => {
            let [mx, my, mz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::Moment { mx, my, mz }
        }
        "pressure" => LoadKind::Pressure {
            magnitude_pa: remove_required_f64(&mut fields, LOAD_CASE_NAME, "magnitude_pa")?,
        },
        "bodyforce" => {
            let [gx, gy, gz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::BodyForce { gx, gy, gz }
        }
        "currentdensity" => {
            let [jx, jy, jz] = remove_required_vector3(&mut fields, LOAD_CASE_NAME, "vector")?;
            LoadKind::CurrentDensity {
                jx,
                jy,
                jz,
                phase_rad: remove_optional_f64(&mut fields, LOAD_CASE_NAME, "phase_rad")?
                    .unwrap_or_default(),
                amplitude_scale: remove_optional_f64(
                    &mut fields,
                    LOAD_CASE_NAME,
                    "amplitude_scale",
                )?
                .unwrap_or(1.0),
            }
        }
        "coilcurrent" => LoadKind::CoilCurrent {
            current_a: remove_required_f64(&mut fields, LOAD_CASE_NAME, "current_a")?,
            phase_rad: remove_optional_f64(&mut fields, LOAD_CASE_NAME, "phase_rad")?
                .unwrap_or_default(),
            amplitude_scale: remove_optional_f64(&mut fields, LOAD_CASE_NAME, "amplitude_scale")?
                .unwrap_or(1.0),
        },
        "heatsource" => LoadKind::HeatSource {
            volumetric_w_per_m3: remove_required_f64(
                &mut fields,
                LOAD_CASE_NAME,
                "volumetric_w_per_m3",
            )?,
        },
        other => {
            return Err(builtin_error(
                LOAD_CASE_NAME,
                &ERROR_INPUT,
                format!("unsupported load kind `{other}`"),
            ));
        }
    };
    reject_unknown_fields(LOAD_CASE_NAME, fields)?;
    load_case_to_object(LoadCase {
        load_id,
        region_id,
        kind,
    })
}

fn create_step_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 2 {
        return Err(builtin_error(
            STEP_NAME,
            &ERROR_INPUT,
            "fea.step requires exactly id and kind arguments",
        ));
    }
    let step_id = scalar_string(&args[0], STEP_NAME, &ERROR_INPUT)?;
    let kind_text = scalar_string(&args[1], STEP_NAME, &ERROR_INPUT)?;
    let kind = parse_scalar_enum::<AnalysisStepKind>(&kind_text, "AnalysisStepKind")?;
    step_to_object(AnalysisStep { step_id, kind })
}

fn create_domain_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            DOMAIN_NAME,
            &ERROR_INPUT,
            "fea.domain requires a domain kind",
        ));
    }
    let kind_text = scalar_string(&args[0], DOMAIN_NAME, &ERROR_INPUT)?;
    let kind = normalize_token(&kind_text);
    let fields = json_fields_from_name_values(DOMAIN_NAME, &args[1..])?;
    let payload = match kind.as_str() {
        "thermomechanical" => DomainPayload {
            kind: "thermo_mechanical".to_string(),
            data: typed_domain_data::<ThermoMechanicalDomain>(
                DOMAIN_NAME,
                "thermo_mechanical domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "reference_temperature_k": 293.15,
                        "applied_temperature_delta_k": 0.0,
                        "field_artifact_id": null,
                        "field_source": null,
                        "region_temperature_deltas": [],
                        "time_profile": []
                    }),
                    fields,
                    "thermo_mechanical domain",
                )?,
            )?,
        },
        "electrothermal" => DomainPayload {
            kind: "electro_thermal".to_string(),
            data: typed_domain_data::<ElectroThermalDomain>(
                DOMAIN_NAME,
                "electro_thermal domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "reference_temperature_k": 293.15,
                        "applied_voltage_v": 0.0,
                        "region_conductivity_scales": [],
                        "time_profile": []
                    }),
                    fields,
                    "electro_thermal domain",
                )?,
            )?,
        },
        "electromagnetic" => DomainPayload {
            kind: "electromagnetic".to_string(),
            data: typed_domain_data::<ElectromagneticDomain>(
                DOMAIN_NAME,
                "electromagnetic domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "reference_frequency_hz": 0.0,
                        "applied_current_a": 0.0
                    }),
                    fields,
                    "electromagnetic domain",
                )?,
            )?,
        },
        "cfd" => DomainPayload {
            kind: "cfd".to_string(),
            data: typed_domain_data::<CfdDomain>(
                DOMAIN_NAME,
                "cfd domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "solve_family": "steady_state",
                        "reference_density_kg_per_m3": 1.225,
                        "dynamic_viscosity_pa_s": 1.8e-5,
                        "inlet_velocity_m_per_s": 0.0,
                        "turbulence_intensity": 0.0,
                        "time_profile": []
                    }),
                    fields,
                    "cfd domain",
                )?,
            )?,
        },
        other => {
            return Err(builtin_error(
                DOMAIN_NAME,
                &ERROR_INPUT,
                format!("unsupported FEA domain kind `{other}`"),
            ));
        }
    };
    domain_to_object(payload)
}

fn create_interface_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            INTERFACE_NAME,
            &ERROR_INPUT,
            "fea.interface requires id, primary region, and secondary region arguments",
        ));
    }
    let interface_id = scalar_string(&args[0], INTERFACE_NAME, &ERROR_INPUT)?;
    let primary_region_id = scalar_string(&args[1], INTERFACE_NAME, &ERROR_INPUT)?;
    let secondary_region_id = scalar_string(&args[2], INTERFACE_NAME, &ERROR_INPUT)?;
    let mut kind = "contact".to_string();
    let mut kind_seen = false;
    let mut fields = serde_json::Map::new();
    for pair in expect_name_value_tail(INTERFACE_NAME, &args[3..])? {
        if pair.key == "kind" {
            if kind_seen {
                return Err(builtin_error(
                    INTERFACE_NAME,
                    &ERROR_INPUT,
                    "duplicate fea.interface option `kind`",
                ));
            }
            kind_seen = true;
            kind = scalar_string(pair.value, INTERFACE_NAME, &ERROR_INPUT)?;
        } else {
            let key =
                canonical_field_name(&scalar_string(pair.name, INTERFACE_NAME, &ERROR_INPUT)?);
            if fields
                .insert(key.clone(), value_to_json(INTERFACE_NAME, pair.value)?)
                .is_some()
            {
                return Err(builtin_error(
                    INTERFACE_NAME,
                    &ERROR_INPUT,
                    format!("duplicate fea.interface option `{key}`"),
                ));
            }
        }
    }
    let kind = match normalize_token(&kind).as_str() {
        "contact" => AnalysisInterfaceKind::Contact(json_deserialize(
            INTERFACE_NAME,
            json_with_overrides(
                INTERFACE_NAME,
                serde_json::json!({
                    "penalty_stiffness_scale": 1.0,
                    "max_penetration_ratio": 0.0,
                    "friction_coefficient": 0.0
                }),
                fields,
                "contact interface",
            )?,
            "contact interface",
        )?),
        "fluid_structure" | "fluidstructure" | "fsi" => {
            AnalysisInterfaceKind::FluidStructure(json_deserialize(
                INTERFACE_NAME,
                json_with_overrides(
                    INTERFACE_NAME,
                    serde_json::json!({
                        "normal_stiffness_pa_per_m": 1.0e9,
                        "damping_ratio": 0.0,
                        "relaxation_factor": 0.5
                    }),
                    fields,
                    "fluid-structure interface",
                )?,
                "fluid-structure interface",
            )?)
        }
        "conjugate_heat_transfer" | "conjugateheattransfer" | "cht" => {
            AnalysisInterfaceKind::ConjugateHeatTransfer(json_deserialize(
                INTERFACE_NAME,
                json_with_overrides(
                    INTERFACE_NAME,
                    serde_json::json!({
                        "thermal_conductance_w_per_m2k": 500.0,
                        "contact_resistance_m2k_per_w": 0.0,
                        "relaxation_factor": 0.5
                    }),
                    fields,
                    "conjugate heat-transfer interface",
                )?,
                "conjugate heat-transfer interface",
            )?)
        }
        other => {
            return Err(builtin_error(
                INTERFACE_NAME,
                &ERROR_INPUT,
                format!("unsupported interface kind `{other}`"),
            ));
        }
    };
    interface_to_object(AnalysisInterface {
        interface_id,
        primary_region_id,
        secondary_region_id,
        kind,
    })
}

fn create_run_options_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            RUN_OPTIONS_NAME,
            &ERROR_INPUT,
            "fea.runOptions requires a solver",
        ));
    }
    let kind_text = scalar_string(&args[0], RUN_OPTIONS_NAME, &ERROR_INPUT)?;
    let run_kind = parse_scalar_enum::<AnalysisRunKind>(&kind_text, "solver")?;
    let fields = run_options_fields_from_name_values(&args[1..])?;
    let data = run_options_json_for_kind(RUN_OPTIONS_NAME, run_kind, fields)?;
    run_options_to_object(RunOptionsPayload {
        run_kind,
        options: data,
    })
}

fn run_options_fields_from_name_values(
    args: &[Value],
) -> BuiltinResult<serde_json::Map<String, serde_json::Value>> {
    const EXACT_FIELDS: &[&str] = &[
        "mode_count",
        "step_count",
        "max_linear_iters",
        "max_step_retries",
        "increment_count",
        "max_newton_iters",
        "max_line_search_backtracks",
        "tangent_refresh_interval",
        "harmonic_max_iterations",
    ];
    let mut fields = serde_json::Map::new();
    for pair in expect_name_value_tail(RUN_OPTIONS_NAME, args)? {
        let raw = scalar_string(pair.name, RUN_OPTIONS_NAME, &ERROR_INPUT)?;
        let key = canonical_field_name(&raw);
        if key == "prep_context" {
            return Err(builtin_error(
                RUN_OPTIONS_NAME,
                &ERROR_INPUT,
                "fea.runOptions does not expose the internal PrepContext; use PrepArtifactId or PrepCalibrationProfile",
            ));
        }
        let value = if EXACT_FIELDS.contains(&key.as_str()) {
            serde_json::Value::from(usize_from_value(RUN_OPTIONS_NAME, pair.value)? as u64)
        } else {
            value_to_json(RUN_OPTIONS_NAME, pair.value)?
        };
        if fields.insert(key.clone(), value).is_some() {
            return Err(builtin_error(
                RUN_OPTIONS_NAME,
                &ERROR_INPUT,
                format!("duplicate fea.runOptions option `{key}`"),
            ));
        }
    }
    Ok(fields)
}

fn create_results_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            RESULTS_NAME,
            &ERROR_INPUT,
            "fea.results requires a run id or fea.RunResult",
        ));
    }
    if let Value::Object(object) = &args[0] {
        if object.class_name == FEA_RESULTS_CLASS && args.len() == 1 {
            return Ok(args[0].clone());
        }
    }
    let run_id = run_id_from_value(RESULTS_NAME, &args[0])?;
    let query = results_query_from_args(&args[1..])?;
    let envelope = analysis_results_by_run_id_op(&run_id, query, OperationContext::new(None, None))
        .map_err(|err| operation_error(RESULTS_NAME, &ERROR_OPERATION, err))?;
    let mut public_data = envelope.data;
    for index in &mut public_data.summary.available_mode_indices {
        *index = index.checked_add(1).ok_or_else(|| {
            builtin_error(
                RESULTS_NAME,
                &ERROR_INTERNAL,
                "available mode index cannot be represented at the one-based public boundary",
            )
        })?;
    }
    let value = serializable_to_object_preserving_integers(
        RESULTS_NAME,
        &ERROR_INTERNAL,
        FEA_RESULTS_CLASS,
        &public_data,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[],
        &[
            "shape",
            "element_count",
            "component_count",
            "size_bytes",
            "solver_host_sync_count",
            "field_count",
            "total_elements",
            "mode_count",
            "available_mode_indices",
            "snapshot_count",
            "increment_count",
            "failed_increment_count",
            "max_nonlinear_iteration_count",
            "nonlinear_line_search_backtracks",
            "nonlinear_max_backtracks_per_increment",
            "nonlinear_tangent_rebuild_count",
            "nonlinear_iteration_spike_count",
            "nonlinear_convergence_stall_count",
            "nonlinear_backtrack_burst_count",
            "prep_calibration_fingerprint",
            "prep_acceptance_fingerprint",
            "thermo_coupling_fingerprint",
            "electro_thermal_coupling_fingerprint",
            "iteration_counts",
            "failed_increments",
            "line_search_backtracks",
            "max_line_search_backtracks_per_increment",
            "tangent_rebuild_count",
            "iteration_spike_count",
            "convergence_stall_count",
            "backtrack_burst_count",
        ],
    )?;
    let Value::Object(mut object) = value else {
        unreachable!("integer-preserving FEA result serialization returns an object")
    };
    object
        .properties
        .insert("run_id".to_string(), Value::String(run_id.clone()));
    object.properties.insert(
        FEA_RUN_ID_CONTEXT_PROPERTY.to_string(),
        Value::String(run_id),
    );
    copy_study_context_property(&args[0], &mut object);
    Ok(Value::Object(object))
}

fn create_field_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 2 {
        return Err(builtin_error(
            FIELD_NAME,
            &ERROR_INPUT,
            "fea.field requires results/run input and field id",
        ));
    }
    let field_id = scalar_string(&args[1], FIELD_NAME, &ERROR_INPUT)?;
    let results = results_data_from_value(FIELD_NAME, &args[0])?;
    let field = find_field(results.fields.into_iter(), &field_id).ok_or_else(|| {
        builtin_error(
            FIELD_NAME,
            &ERROR_INPUT,
            format!("FEA field `{field_id}` was not found in results"),
        )
    })?;
    let descriptor = find_descriptor(results.field_descriptors.iter(), &field_id)
        .cloned()
        .unwrap_or_else(|| AnalysisFieldDescriptor::from_field(&field));
    let mut object = field_to_object(&field, &descriptor)?;
    copy_study_context_property(&args[0], &mut object);
    copy_run_id_context_property(&args[0], &mut object);
    Ok(Value::Object(object))
}

fn create_plot_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    #[cfg(feature = "plot-core")]
    {
        let request = plot_request_from_args(&args)?;
        reject_requested_device_field(&request)?;
        let mut figures = generate_plot_figures(&request.study, &request.run_id, &request.options)?;
        let figure = select_generated_figure(&mut figures, request.field_id.as_deref())?;
        let handle = import_generated_figure(figure)?;
        Ok(Value::Num(f64::from(handle)))
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = args;
        Err(builtin_error(
            PLOT_NAME,
            &ERROR_OPERATION,
            "fea.plot requires the plot-core runtime feature",
        ))
    }
}

#[cfg(feature = "plot-core")]
fn reject_requested_device_field(request: &FeaPlotRequest) -> BuiltinResult<()> {
    let Some(field_id) = request.field_id.as_deref() else {
        return Ok(());
    };
    let results = analysis_results_by_run_id_op(
        &request.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .map(|envelope| envelope.data)
    .map_err(|err| operation_error(PLOT_NAME, &ERROR_OPERATION, err))?;
    if find_field(results.fields, field_id)
        .is_some_and(|field| matches!(field.values, AnalysisFieldValues::DeviceRef(_)))
    {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            format!(
                "FEA field `{field_id}` is device-backed and cannot be plotted without explicit host materialization"
            ),
        ));
    }
    Ok(())
}

fn create_compare_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 2 {
        return Err(builtin_error(
            COMPARE_NAME,
            &ERROR_INPUT,
            "fea.compare requires baseline and candidate run ids",
        ));
    }
    let baseline_run_id = scalar_string(&args[0], COMPARE_NAME, &ERROR_INPUT)?;
    let candidate_run_id = scalar_string(&args[1], COMPARE_NAME, &ERROR_INPUT)?;
    operation_result_to_object_preserving_integers(
        COMPARE_NAME,
        &ERROR_OPERATION,
        &ERROR_INTERNAL,
        FEA_COMPARE_CLASS,
        analysis_results_compare_op(
            AnalysisResultsCompareQuery {
                baseline_run_id,
                candidate_run_id,
            },
            OperationContext::new(None, None),
        ),
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[
            "quality_reason_count_delta",
            "failed_increment_delta",
            "max_iteration_delta",
            "nonlinear_spike_count_delta",
            "nonlinear_stall_count_delta",
        ],
        &[],
    )
}

fn create_trends_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    let mut window_size = AnalysisTrendsQuery::default().window_size;
    let mut window_size_seen = false;
    for pair in expect_name_value_tail(TRENDS_NAME, args.as_slice())? {
        match pair.key.as_str() {
            "windowsize" => {
                if window_size_seen {
                    return Err(builtin_error(
                        TRENDS_NAME,
                        &ERROR_INPUT,
                        "duplicate fea.trends option `windowsize`",
                    ));
                }
                window_size_seen = true;
                window_size = usize_from_value(TRENDS_NAME, pair.value)?;
                if window_size == 0 {
                    return Err(builtin_error(
                        TRENDS_NAME,
                        &ERROR_INPUT,
                        "fea.trends WindowSize must be positive",
                    ));
                }
            }
            other => {
                return Err(builtin_error(
                    TRENDS_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.trends option `{other}`"),
                ));
            }
        }
    }
    operation_result_to_object(
        TRENDS_NAME,
        &ERROR_OPERATION,
        &ERROR_INTERNAL,
        FEA_TRENDS_CLASS,
        analysis_trends_op(
            AnalysisTrendsQuery { window_size },
            OperationContext::new(None, None),
        ),
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn build_model_from_parts(
    builtin: &'static str,
    geometry: &GeometryAsset,
    model_id: String,
    profile: AnalysisCreateModelProfile,
    defaults: ModelDefaultsMode,
    frame: Option<ReferenceFrame>,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<MaterialAssignment>,
    boundary_conditions: Vec<BoundaryCondition>,
    loads: Vec<LoadCase>,
    steps: Vec<AnalysisStep>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<AnalysisInterface>,
) -> BuiltinResult<AnalysisModel> {
    let mut model = match defaults {
        ModelDefaultsMode::ProfileScaffold => analysis_create_model_op(
            geometry,
            AnalysisCreateModelIntentSpec {
                model_id: model_id.clone(),
                profile,
                prep_context: None,
            },
            OperationContext::new(None, None),
        )
        .map(|envelope| envelope.data)
        .map_err(|err| operation_error(builtin, &ERROR_OPERATION, err))?,
        ModelDefaultsMode::None => empty_model(model_id, geometry),
    };

    if let Some(frame) = frame {
        model.frame = frame;
    }
    if !materials.is_empty() {
        model.materials = materials;
    }
    if !material_assignments.is_empty() {
        model.material_assignments = material_assignments
            .into_iter()
            .map(|mut assignment| {
                assignment.region_id =
                    resolve_region_selector(builtin, &assignment.region_id, geometry)?;
                Ok(assignment)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !boundary_conditions.is_empty() {
        model.boundary_conditions = boundary_conditions
            .into_iter()
            .map(|mut bc| {
                bc.region_id = resolve_region_selector(builtin, &bc.region_id, geometry)?;
                Ok(bc)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !loads.is_empty() {
        model.loads = loads
            .into_iter()
            .map(|mut load| {
                load.region_id = resolve_region_selector(builtin, &load.region_id, geometry)?;
                Ok(load)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !steps.is_empty() {
        model.steps = steps;
    }
    for domain in domains {
        match domain.kind.as_str() {
            "thermo_mechanical" => {
                let mut domain: ThermoMechanicalDomain =
                    json_deserialize(builtin, domain.data, "thermo_mechanical domain")?;
                for entry in &mut domain.region_temperature_deltas {
                    entry.region_id = resolve_region_selector(builtin, &entry.region_id, geometry)?;
                }
                if let Some(source) = &mut domain.field_source {
                    for region_id in &mut source.expected_region_ids {
                        *region_id = resolve_region_selector(builtin, region_id, geometry)?;
                    }
                }
                model.thermo_mechanical = Some(domain);
            }
            "electro_thermal" => {
                let mut domain: ElectroThermalDomain =
                    json_deserialize(builtin, domain.data, "electro_thermal domain")?;
                for entry in &mut domain.region_conductivity_scales {
                    entry.region_id = resolve_region_selector(builtin, &entry.region_id, geometry)?;
                }
                model.electro_thermal = Some(domain);
            }
            "electromagnetic" => {
                model.electromagnetic = Some(json_deserialize(
                    builtin,
                    domain.data,
                    "electromagnetic domain",
                )?);
            }
            "cfd" => {
                model.cfd = Some(json_deserialize(builtin, domain.data, "cfd domain")?);
            }
            other => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported domain payload `{other}`"),
                ));
            }
        }
    }
    if !interfaces.is_empty() {
        model.interfaces = interfaces
            .into_iter()
            .map(|mut interface| {
                interface.primary_region_id =
                    resolve_region_selector(builtin, &interface.primary_region_id, geometry)?;
                interface.secondary_region_id =
                    resolve_region_selector(builtin, &interface.secondary_region_id, geometry)?;
                Ok(interface)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    Ok(model)
}

fn empty_model(model_id: String, geometry: &GeometryAsset) -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId(model_id),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: ReferenceFrame::Global,
        materials: Vec::new(),
        material_assignments: Vec::new(),
        structural: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        boundary_conditions: Vec::new(),
        loads: Vec::new(),
        steps: Vec::new(),
    }
}

fn resolve_region_selector(
    builtin: &'static str,
    selector: &str,
    geometry: &GeometryAsset,
) -> BuiltinResult<String> {
    if let Some(id) = selector
        .strip_prefix("id:")
        .or_else(|| selector.strip_prefix("region:"))
    {
        return require_region_id(builtin, id, geometry);
    }
    if let Some(tag) = selector.strip_prefix("tag:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.tag.as_deref() == Some(tag))
            .map(|region| region.region_id.clone())
            .ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("region tag `{tag}` was not found in geometry"),
                )
            });
    }
    if let Some(name) = selector.strip_prefix("name:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.name == name)
            .map(|region| region.region_id.clone())
            .ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("region name `{name}` was not found in geometry"),
                )
            });
    }
    require_region_id(builtin, selector, geometry)
}

fn require_region_id(
    builtin: &'static str,
    region_id: &str,
    geometry: &GeometryAsset,
) -> BuiltinResult<String> {
    geometry
        .regions
        .iter()
        .find(|region| region.region_id == region_id)
        .map(|region| region.region_id.clone())
        .ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("region id `{region_id}` was not found in geometry"),
            )
        })
}

fn material_to_object(material: MaterialModel) -> BuiltinResult<Value> {
    serializable_to_object(
        MATERIAL_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_CLASS,
        &material,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn material_assignment_to_object(assignment: MaterialAssignment) -> BuiltinResult<Value> {
    serializable_to_object(
        MATERIAL_ASSIGNMENT_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_ASSIGNMENT_CLASS,
        &assignment,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn boundary_condition_to_object(bc: BoundaryCondition) -> BuiltinResult<Value> {
    serializable_to_object(
        BOUNDARY_CONDITION_NAME,
        &ERROR_INTERNAL,
        FEA_BOUNDARY_CONDITION_CLASS,
        &bc,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn load_case_to_object(load: LoadCase) -> BuiltinResult<Value> {
    serializable_to_object(
        LOAD_CASE_NAME,
        &ERROR_INTERNAL,
        FEA_LOAD_CASE_CLASS,
        &load,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn step_to_object(step: AnalysisStep) -> BuiltinResult<Value> {
    serializable_to_object(
        STEP_NAME,
        &ERROR_INTERNAL,
        FEA_STEP_CLASS,
        &step,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn domain_to_object(domain: DomainPayload) -> BuiltinResult<Value> {
    serializable_to_object_preserving_integers(
        DOMAIN_NAME,
        &ERROR_INTERNAL,
        FEA_DOMAIN_CLASS,
        &domain,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[],
        &["revision"],
    )
}

fn interface_to_object(interface: AnalysisInterface) -> BuiltinResult<Value> {
    serializable_to_object(
        INTERFACE_NAME,
        &ERROR_INTERNAL,
        FEA_INTERFACE_CLASS,
        &interface,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn run_options_to_object(payload: RunOptionsPayload) -> BuiltinResult<Value> {
    serializable_to_object(
        RUN_OPTIONS_NAME,
        &ERROR_INTERNAL,
        FEA_RUN_OPTIONS_CLASS,
        &payload,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

fn model_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<AnalysisModel> {
    object_payload(builtin, value, FEA_MODEL_CLASS)
}

fn study_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<AnalysisStudySpec>> {
    object_vec_from_value_with_property(
        builtin,
        value,
        FEA_STUDY_CLASS,
        FEA_STUDY_SPEC_JSON_PROPERTY,
    )
}

fn material_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<MaterialModel>> {
    object_vec_from_value(builtin, value, FEA_MATERIAL_CLASS)
}

fn material_assignment_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<MaterialAssignment>> {
    object_vec_from_value(builtin, value, FEA_MATERIAL_ASSIGNMENT_CLASS)
}

fn boundary_condition_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<BoundaryCondition>> {
    object_vec_from_value(builtin, value, FEA_BOUNDARY_CONDITION_CLASS)
}

fn load_case_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<LoadCase>> {
    object_vec_from_value(builtin, value, FEA_LOAD_CASE_CLASS)
}

fn step_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<AnalysisStep>> {
    object_vec_from_value(builtin, value, FEA_STEP_CLASS)
}

fn domain_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<DomainPayload>> {
    object_vec_from_value(builtin, value, FEA_DOMAIN_CLASS)
}

fn interface_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<AnalysisInterface>> {
    object_vec_from_value(builtin, value, FEA_INTERFACE_CLASS)
}

fn object_vec_from_value<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
) -> BuiltinResult<Vec<T>> {
    object_vec_from_value_with_property(builtin, value, expected_class, FEA_PAYLOAD_JSON_PROPERTY)
}

fn object_vec_from_value_with_property<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
    payload_property: &'static str,
) -> BuiltinResult<Vec<T>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| {
                object_payload_with_property(builtin, item, expected_class, payload_property)
            })
            .collect(),
        Value::Object(_) => Ok(vec![object_payload_with_property(
            builtin,
            value,
            expected_class,
            payload_property,
        )?]),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class} object or cell array; got {other:?}"),
        )),
    }
}

fn object_payload<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
) -> BuiltinResult<T> {
    object_payload_with_property(builtin, value, expected_class, FEA_PAYLOAD_JSON_PROPERTY)
}

fn object_payload_with_property<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
    payload_property: &'static str,
) -> BuiltinResult<T> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class} object"),
        ));
    };
    if object.class_name != expected_class {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class}, got {}", object.class_name),
        ));
    }
    object_json_property(builtin, object, payload_property, &ERROR_INPUT)
}

fn run_options_payload_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<RunOptionsPayload> {
    object_payload(builtin, value, FEA_RUN_OPTIONS_CLASS)
}

fn resolved_run_options_from_payload(
    builtin: &'static str,
    payload: RunOptionsPayload,
    expected_kind: AnalysisRunKind,
) -> BuiltinResult<ResolvedRunOptions> {
    if payload.run_kind != expected_kind {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "run options kind {:?} does not match selected study solver {:?}",
                payload.run_kind, expected_kind
            ),
        ));
    }
    let mut resolved = ResolvedRunOptions::default();
    match payload.run_kind {
        AnalysisRunKind::LinearStatic => {
            resolved.linear_static = Some(json_deserialize(
                builtin,
                payload.options,
                "linear_static run options",
            )?);
        }
        AnalysisRunKind::Modal => {
            resolved.modal = Some(json_deserialize(
                builtin,
                payload.options,
                "modal run options",
            )?);
        }
        AnalysisRunKind::Acoustic => {
            resolved.acoustic = Some(json_deserialize(
                builtin,
                payload.options,
                "acoustic run options",
            )?);
        }
        AnalysisRunKind::Thermal => {
            resolved.thermal = Some(json_deserialize(
                builtin,
                payload.options,
                "thermal run options",
            )?);
        }
        AnalysisRunKind::Transient => {
            resolved.transient = Some(json_deserialize(
                builtin,
                payload.options,
                "transient run options",
            )?);
        }
        AnalysisRunKind::Cfd => {
            resolved.cfd = Some(json_deserialize(
                builtin,
                payload.options,
                "cfd run options",
            )?);
        }
        AnalysisRunKind::Cht => {
            resolved.cht = Some(json_deserialize(
                builtin,
                payload.options,
                "cht run options",
            )?);
        }
        AnalysisRunKind::Fsi => {
            resolved.fsi = Some(json_deserialize(
                builtin,
                payload.options,
                "fsi run options",
            )?);
        }
        AnalysisRunKind::Nonlinear => {
            resolved.nonlinear = Some(json_deserialize(
                builtin,
                payload.options,
                "nonlinear run options",
            )?);
        }
        AnalysisRunKind::Electromagnetic => {
            resolved.electromagnetic = Some(json_deserialize(
                builtin,
                payload.options,
                "electromagnetic run options",
            )?);
        }
    }
    Ok(resolved)
}

fn run_options_json_for_kind(
    builtin: &'static str,
    run_kind: AnalysisRunKind,
    fields: serde_json::Map<String, serde_json::Value>,
) -> BuiltinResult<serde_json::Value> {
    match run_kind {
        AnalysisRunKind::LinearStatic => typed_json_with_overrides::<AnalysisRunOptions>(
            builtin,
            AnalysisRunOptions::default(),
            fields,
            "linear_static run options",
        ),
        AnalysisRunKind::Modal => typed_json_with_overrides::<AnalysisModalRunOptions>(
            builtin,
            AnalysisModalRunOptions::default(),
            fields,
            "modal run options",
        ),
        AnalysisRunKind::Acoustic => typed_json_with_overrides::<AnalysisAcousticRunOptions>(
            builtin,
            AnalysisAcousticRunOptions::default(),
            fields,
            "acoustic run options",
        ),
        AnalysisRunKind::Thermal => typed_json_with_overrides::<AnalysisThermalRunOptions>(
            builtin,
            AnalysisThermalRunOptions::default(),
            fields,
            "thermal run options",
        ),
        AnalysisRunKind::Transient => typed_json_with_overrides::<AnalysisTransientRunOptions>(
            builtin,
            AnalysisTransientRunOptions::default(),
            fields,
            "transient run options",
        ),
        AnalysisRunKind::Cfd => typed_json_with_overrides::<AnalysisCfdRunOptions>(
            builtin,
            AnalysisCfdRunOptions::default(),
            fields,
            "cfd run options",
        ),
        AnalysisRunKind::Cht => typed_json_with_overrides::<AnalysisChtRunOptions>(
            builtin,
            AnalysisChtRunOptions::default(),
            fields,
            "cht run options",
        ),
        AnalysisRunKind::Fsi => typed_json_with_overrides::<AnalysisFsiRunOptions>(
            builtin,
            AnalysisFsiRunOptions::default(),
            fields,
            "fsi run options",
        ),
        AnalysisRunKind::Nonlinear => typed_json_with_overrides::<AnalysisNonlinearRunOptions>(
            builtin,
            AnalysisNonlinearRunOptions::default(),
            fields,
            "nonlinear run options",
        ),
        AnalysisRunKind::Electromagnetic => {
            typed_json_with_overrides::<AnalysisElectromagneticRunOptions>(
                builtin,
                AnalysisElectromagneticRunOptions::default(),
                fields,
                "electromagnetic run options",
            )
        }
    }
}

fn results_query_from_args(args: &[Value]) -> BuiltinResult<AnalysisResultsQuery> {
    let mut query = AnalysisResultsQuery::default();
    let mut seen = HashSet::new();
    for pair in expect_name_value_tail(RESULTS_NAME, args)? {
        let canonical = match pair.key.as_str() {
            "includefields" | "fields" => "includefields",
            "includefieldvalues" | "fieldvalues" => "includefieldvalues",
            other => other,
        };
        if !seen.insert(canonical.to_string()) {
            return Err(builtin_error(
                RESULTS_NAME,
                &ERROR_INPUT,
                format!("duplicate fea.results option `{canonical}`"),
            ));
        }
        match pair.key.as_str() {
            "includefields" | "fields" => {
                query.include_fields = string_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includefieldvalues" | "fieldvalues" => {
                query.include_field_values = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "includediagnostics" => {
                query.include_diagnostics = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "diagnosticcodes" => {
                query.diagnostic_codes = string_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includemodalresults" => {
                query.include_modal_results = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "modeindices" => {
                query.mode_indices = one_based_usize_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includetransientresults" => {
                query.include_transient_results = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "transientsnapshotindices" => {
                query.transient_snapshot_indices =
                    one_based_usize_vec_from_value(RESULTS_NAME, pair.value)?;
            }
            "includenonlinearresults" => {
                query.include_nonlinear_results = exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            "includeelectromagneticresults" => {
                query.include_electromagnetic_results =
                    exact_bool_from_value(RESULTS_NAME, pair.value)?;
            }
            other => {
                return Err(builtin_error(
                    RESULTS_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.results option `{other}`"),
                ));
            }
        }
    }
    Ok(query)
}

fn run_id_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::Object(object) if object.class_name == FEA_RUN_RESULT_CLASS => {
            run_id_from_object(object).ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "fea.RunResult does not contain a run_id; sweep results expose run_entries",
                )
            })
        }
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            scalar_string(value, builtin, &ERROR_INPUT)
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected run id string or fea.RunResult; got {other:?}"),
        )),
    }
}

fn run_id_from_object(object: &ObjectInstance) -> Option<String> {
    object
        .properties
        .get(FEA_RUN_ID_CONTEXT_PROPERTY)
        .or_else(|| object.properties.get("run_id"))
        .or_else(|| object.properties.get("runId"))
        .and_then(|value| match value {
            Value::String(run_id) => Some(run_id.clone()),
            _ => None,
        })
}

fn results_data_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<crate::analysis::AnalysisResultsData> {
    match value {
        Value::Object(object) if object.class_name == FEA_RESULTS_CLASS => {
            object_json_property(builtin, object, FEA_PAYLOAD_JSON_PROPERTY, &ERROR_INPUT)
        }
        _ => {
            let run_id = run_id_from_value(builtin, value)?;
            analysis_results_by_run_id_op(
                &run_id,
                AnalysisResultsQuery::default(),
                OperationContext::new(None, None),
            )
            .map(|envelope| envelope.data)
            .map_err(|err| operation_error(builtin, &ERROR_OPERATION, err))
        }
    }
}

fn run_study_result_to_object(spec: &AnalysisStudySpec) -> BuiltinResult<Value> {
    let envelope = analysis_run_study_op(spec, OperationContext::new(None, None))
        .map_err(|err| operation_error(RUN_NAME, &ERROR_OPERATION, err))?;
    let mut object = serializable_to_object_value(
        RUN_NAME,
        &ERROR_INTERNAL,
        FEA_RUN_RESULT_CLASS,
        &envelope.data,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )?;
    object.properties.insert(
        FEA_RUN_ID_CONTEXT_PROPERTY.to_string(),
        Value::String(envelope.data.run_id.clone()),
    );
    object.properties.insert(
        "run_id".to_string(),
        Value::String(envelope.data.run_id.clone()),
    );
    object.properties.insert(
        "runId".to_string(),
        Value::String(envelope.data.run_id.clone()),
    );
    insert_study_context(&mut object, spec)?;
    Ok(Value::Object(object))
}

fn insert_study_context(
    object: &mut ObjectInstance,
    spec: &AnalysisStudySpec,
) -> BuiltinResult<()> {
    let json = serde_json::to_string(spec).map_err(|err| {
        builtin_error_with_source(RUN_NAME, &ERROR_INTERNAL, err.to_string(), err)
    })?;
    object.properties.insert(
        FEA_STUDY_CONTEXT_JSON_PROPERTY.to_string(),
        Value::String(json),
    );
    Ok(())
}

fn copy_study_context_property(source: &Value, target: &mut ObjectInstance) {
    if let Some(json) = study_context_json_from_value(source) {
        target.properties.insert(
            FEA_STUDY_CONTEXT_JSON_PROPERTY.to_string(),
            Value::String(json),
        );
    }
}

fn copy_run_id_context_property(source: &Value, target: &mut ObjectInstance) {
    if let Some(run_id) = run_id_context_from_value(source) {
        target.properties.insert(
            FEA_RUN_ID_CONTEXT_PROPERTY.to_string(),
            Value::String(run_id.clone()),
        );
        target
            .properties
            .entry("run_id".to_string())
            .or_insert(Value::String(run_id.clone()));
        target
            .properties
            .entry("runId".to_string())
            .or_insert(Value::String(run_id));
    }
}

fn study_context_json_from_value(value: &Value) -> Option<String> {
    let Value::Object(object) = value else {
        return None;
    };
    if object.class_name == FEA_STUDY_CLASS {
        if let Some(Value::String(json)) = object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY) {
            return Some(json.clone());
        }
    }
    object
        .properties
        .get(FEA_STUDY_CONTEXT_JSON_PROPERTY)
        .and_then(|value| match value {
            Value::String(json) => Some(json.clone()),
            _ => None,
        })
}

fn study_context_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<AnalysisStudySpec> {
    let Some(json) = study_context_json_from_value(value) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin}: FEA plot requires study geometry context; pass a fea.RunResult from fea.run(study), a derived fea.Results/fea.Field, or call fea.plot(study, runId, fieldId)"),
        ));
    };
    serde_json::from_str(&json)
        .map_err(|err| builtin_error_with_source(builtin, &ERROR_INPUT, err.to_string(), err))
}

fn run_id_context_from_value(value: &Value) -> Option<String> {
    let Value::Object(object) = value else {
        return None;
    };
    run_id_from_object(object)
}

fn field_to_object(
    field: &AnalysisField,
    descriptor: &AnalysisFieldDescriptor,
) -> BuiltinResult<ObjectInstance> {
    ensure_fea_classes_registered();
    let mut object = ObjectInstance::new(FEA_FIELD_CLASS.to_string());
    object.properties.insert(
        "field_id".to_string(),
        Value::String(field.field_id.clone()),
    );
    object
        .properties
        .insert("id".to_string(), Value::String(field.field_id.clone()));
    object.properties.insert(
        "shape".to_string(),
        usize_slice_tensor(&field.shape, 1, field.shape.len())?,
    );
    object
        .properties
        .insert("values".to_string(), field_values_value(field)?);
    object.properties.insert(
        "unit".to_string(),
        Value::String(descriptor.unit.clone().unwrap_or_default()),
    );
    object.properties.insert(
        "location".to_string(),
        Value::String(format!("{:?}", descriptor.location).to_ascii_lowercase()),
    );
    object.properties.insert(
        "kind".to_string(),
        Value::String(format!("{:?}", descriptor.kind).to_ascii_lowercase()),
    );
    object.properties.insert(
        "family".to_string(),
        Value::String(descriptor.family.clone()),
    );
    object.properties.insert(
        "quantity".to_string(),
        Value::String(descriptor.quantity.clone()),
    );
    object.properties.insert(
        "topology_id".to_string(),
        descriptor
            .topology_id
            .as_ref()
            .map(|value| Value::String(value.clone()))
            .unwrap_or_else(empty_double_value),
    );
    object.properties.insert(
        "element_kind".to_string(),
        descriptor
            .element_kind
            .as_ref()
            .map(|value| Value::String(value.clone()))
            .unwrap_or_else(empty_double_value),
    );
    object.properties.insert(
        "component_count".to_string(),
        descriptor
            .component_count
            .map(|value| Value::Int(IntValue::U64(value as u64)))
            .unwrap_or_else(empty_double_value),
    );
    object.properties.insert(
        "element_count".to_string(),
        Value::Int(IntValue::U64(descriptor.element_count as u64)),
    );
    object.properties.insert(
        "entity_count".to_string(),
        Value::Int(IntValue::U64(descriptor.entity_count as u64)),
    );
    object.properties.insert(
        "value_count".to_string(),
        Value::Int(IntValue::U64(descriptor.value_count as u64)),
    );
    object.properties.insert(
        "storage".to_string(),
        Value::String(format!("{:?}", descriptor.storage).to_ascii_lowercase()),
    );
    object.properties.insert(
        "descriptor".to_string(),
        serializable_to_value_preserving_integers(
            FIELD_NAME,
            &ERROR_INTERNAL,
            descriptor,
            &[],
            &["shape", "element_count", "component_count", "size_bytes"],
        )?,
    );
    let json = serde_json::to_string(field).map_err(|err| {
        builtin_error_with_source(FIELD_NAME, &ERROR_INTERNAL, err.to_string(), err)
    })?;
    object
        .properties
        .insert(FEA_PAYLOAD_JSON_PROPERTY.to_string(), Value::String(json));
    Ok(object)
}

fn field_values_value(field: &AnalysisField) -> BuiltinResult<Value> {
    match &field.values {
        AnalysisFieldValues::HostF64(values) => Tensor::new(values.clone(), field.shape.clone())
            .map(Value::Tensor)
            .map_err(|err| {
                builtin_error(
                    FIELD_NAME,
                    &ERROR_INTERNAL,
                    format!("fea.field: failed to build values tensor: {err}"),
                )
            }),
        AnalysisFieldValues::DeviceRef(device) => serializable_to_value_preserving_integers(
            FIELD_NAME,
            &ERROR_INTERNAL,
            device,
            &[],
            &["element_count"],
        ),
    }
}

fn usize_slice_tensor(values: &[usize], rows: usize, cols: usize) -> BuiltinResult<Value> {
    let values = values
        .iter()
        .map(|value| u64::try_from(*value))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            builtin_error(
                FIELD_NAME,
                &ERROR_INTERNAL,
                "FEA field shape exceeds uint64",
            )
        })?;
    Tensor::new_integer(IntegerStorage::U64(values), vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| {
            builtin_error(
                FIELD_NAME,
                &ERROR_INTERNAL,
                format!("fea.field: failed to build metadata tensor: {err}"),
            )
        })
}

fn empty_double_value() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor shape is valid"))
}

fn find_field<I>(fields: I, requested: &str) -> Option<AnalysisField>
where
    I: IntoIterator<Item = AnalysisField>,
{
    let mut suffix_matches = Vec::new();
    for field in fields {
        if field.field_id == requested {
            return Some(field);
        }
        if field_id_matches(&field.field_id, requested) {
            suffix_matches.push(field);
        }
    }
    if suffix_matches.len() == 1 {
        suffix_matches.pop()
    } else {
        None
    }
}

fn find_descriptor<'a, I>(descriptors: I, requested: &str) -> Option<&'a AnalysisFieldDescriptor>
where
    I: IntoIterator<Item = &'a AnalysisFieldDescriptor>,
{
    let mut suffix_matches = Vec::new();
    for descriptor in descriptors {
        if descriptor.field_id == requested {
            return Some(descriptor);
        }
        if field_id_matches(&descriptor.field_id, requested) {
            suffix_matches.push(descriptor);
        }
    }
    if suffix_matches.len() == 1 {
        suffix_matches.pop()
    } else {
        None
    }
}

fn field_id_matches(candidate: &str, requested: &str) -> bool {
    candidate == requested
        || candidate
            .strip_suffix(requested)
            .is_some_and(|prefix| prefix.ends_with('.'))
        || candidate
            .rsplit_once('.')
            .is_some_and(|(_, tail)| tail == requested)
}

struct FeaPlotRequest {
    study: AnalysisStudySpec,
    run_id: String,
    field_id: Option<String>,
    options: FeaPlotOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FeaPlotOptions {
    field_id: Option<String>,
    mesh_source: crate::analysis::AnalysisFigureMeshSource,
    show_solver_mesh_edges: bool,
    apply_deformation_overlay: bool,
}

impl Default for FeaPlotOptions {
    fn default() -> Self {
        Self {
            field_id: None,
            mesh_source: crate::analysis::AnalysisFigureMeshSource::Auto,
            show_solver_mesh_edges: false,
            apply_deformation_overlay: true,
        }
    }
}

fn plot_request_from_args(args: &[Value]) -> BuiltinResult<FeaPlotRequest> {
    if args.is_empty() {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            "fea.plot requires a run, results, field, or study/run pair",
        ));
    }

    let (core, options) = split_plot_options(args)?;
    match core {
        [single] => plot_request_from_context_value(single, options),
        [first, second] if is_fea_study(first) => {
            let study = study_context_from_value(PLOT_NAME, first)?;
            let run_id = run_id_from_value(PLOT_NAME, second)?;
            Ok(FeaPlotRequest {
                study,
                run_id,
                field_id: options.field_id.clone(),
                options,
            })
        }
        [first, second] => {
            let mut request = plot_request_from_context_value(first, options)?;
            request.field_id = Some(scalar_string(second, PLOT_NAME, &ERROR_INPUT)?);
            if request.options.field_id.is_some() {
                request.field_id = request.options.field_id.clone();
            }
            Ok(request)
        }
        [first, second, third] if is_fea_study(first) => {
            let study = study_context_from_value(PLOT_NAME, first)?;
            let run_id = run_id_from_value(PLOT_NAME, second)?;
            let field_id = match options.field_id.clone() {
                Some(field_id) => Some(field_id),
                None => Some(scalar_string(third, PLOT_NAME, &ERROR_INPUT)?),
            };
            Ok(FeaPlotRequest {
                study,
                run_id,
                field_id,
                options,
            })
        }
        _ => Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            "fea.plot supports plot(run, field), plot(results, field), plot(field), or plot(study, runId, field)",
        )),
    }
}

fn split_plot_options(args: &[Value]) -> BuiltinResult<(&[Value], FeaPlotOptions)> {
    let mut options = FeaPlotOptions::default();
    let mut end = args.len();
    while end >= 2 && is_plot_option_name(&args[end - 2]) {
        let key = scalar_string(&args[end - 2], PLOT_NAME, &ERROR_INPUT)?.to_ascii_lowercase();
        match key.as_str() {
            "field" | "fieldid" | "field_id" => {
                options.field_id = Some(scalar_string(&args[end - 1], PLOT_NAME, &ERROR_INPUT)?);
            }
            "mesh" => {
                options.show_solver_mesh_edges =
                    plot_mesh_option_shows_solver_edges(&args[end - 1])?;
            }
            "overlay" => {
                options.mesh_source = plot_overlay_option_mesh_source(&args[end - 1])?;
            }
            "deformed" => {
                options.apply_deformation_overlay = bool_from_value(PLOT_NAME, &args[end - 1])?;
            }
            _ => unreachable!("is_plot_option_name only accepts supported plot option names"),
        }
        end -= 2;
    }
    Ok((&args[..end], options))
}

fn is_plot_option_name(value: &Value) -> bool {
    scalar_string(value, PLOT_NAME, &ERROR_INPUT)
        .map(|name| {
            matches!(
                name.to_ascii_lowercase().as_str(),
                "field" | "fieldid" | "field_id" | "mesh" | "overlay" | "deformed"
            )
        })
        .unwrap_or(false)
}

fn plot_mesh_option_shows_solver_edges(value: &Value) -> BuiltinResult<bool> {
    let mesh = scalar_string(value, PLOT_NAME, &ERROR_INPUT)?;
    match mesh.to_ascii_lowercase().as_str() {
        "solver" | "solver_edges" | "solveredges" | "edges" => Ok(true),
        "cad" | "geometry" | "surface" | "none" => Ok(false),
        other => Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            format!(
                "unsupported fea.plot mesh option `{other}`; expected solver, solver_edges, cad, geometry, surface, or none"
            ),
        )),
    }
}

fn plot_overlay_option_mesh_source(
    value: &Value,
) -> BuiltinResult<crate::analysis::AnalysisFigureMeshSource> {
    let overlay = scalar_string(value, PLOT_NAME, &ERROR_INPUT)?;
    match overlay.to_ascii_lowercase().as_str() {
        "auto" => Ok(crate::analysis::AnalysisFigureMeshSource::Auto),
        "solver" | "mesh" | "boundary" | "solver_boundary" | "solverboundary" => {
            Ok(crate::analysis::AnalysisFigureMeshSource::Solver)
        }
        "cad" | "cad_reference" | "reference" | "geometry" | "surface" => {
            Ok(crate::analysis::AnalysisFigureMeshSource::CadReference)
        }
        other => Err(builtin_error(
            PLOT_NAME,
            &ERROR_INPUT,
            format!("unsupported fea.plot overlay option `{other}`; expected auto, solver, or cad"),
        )),
    }
}

fn is_fea_study(value: &Value) -> bool {
    matches!(value, Value::Object(object) if object.class_name == FEA_STUDY_CLASS)
}

fn plot_request_from_context_value(
    value: &Value,
    options: FeaPlotOptions,
) -> BuiltinResult<FeaPlotRequest> {
    let study = study_context_from_value(PLOT_NAME, value)?;
    let run_id = run_id_context_from_value(value)
        .or_else(|| run_id_from_value(PLOT_NAME, value).ok())
        .ok_or_else(|| {
            builtin_error(
                PLOT_NAME,
                &ERROR_INPUT,
                "fea.plot requires a run_id; use a fea.RunResult from fea.run or pass fea.plot(study, runId, field)",
            )
        })?;
    let field_id = options.field_id.clone().or_else(|| match value {
        Value::Object(object) if object.class_name == FEA_FIELD_CLASS => object
            .properties
            .get("field_id")
            .and_then(|value| match value {
                Value::String(field_id) => Some(field_id.clone()),
                _ => None,
            }),
        _ => None,
    });
    Ok(FeaPlotRequest {
        study,
        run_id,
        field_id,
        options,
    })
}

#[cfg(feature = "plot-core")]
fn generate_plot_figures(
    study: &AnalysisStudySpec,
    run_id: &str,
    options: &FeaPlotOptions,
) -> BuiltinResult<Vec<crate::analysis::AnalysisGeneratedFigure>> {
    crate::analysis::analysis_generate_study_run_figures(
        study,
        run_id,
        crate::analysis::AnalysisFigureGenerationOptions {
            include_comparison: false,
            include_trends: false,
            max_mesh_result_figures: 8,
            mesh_source: options.mesh_source,
            show_solver_mesh_edges: options.show_solver_mesh_edges,
            apply_deformation_overlay: options.apply_deformation_overlay,
            ..crate::analysis::AnalysisFigureGenerationOptions::default()
        },
    )
    .map_err(|err| builtin_error(PLOT_NAME, &ERROR_OPERATION, err))
}

#[cfg(feature = "plot-core")]
fn select_generated_figure(
    figures: &mut Vec<crate::analysis::AnalysisGeneratedFigure>,
    field_id: Option<&str>,
) -> BuiltinResult<crate::analysis::AnalysisGeneratedFigure> {
    if figures.is_empty() {
        return Err(builtin_error(
            PLOT_NAME,
            &ERROR_OPERATION,
            "fea.plot could not generate a renderable FEA figure for this run",
        ));
    }
    let Some(field_id) = field_id else {
        if let Some(index) = default_generated_figure_index(figures) {
            return Ok(figures.remove(index));
        }
        return Ok(figures.remove(0));
    };
    if let Some(index) = figures.iter().position(|figure| {
        figure
            .field_ids
            .iter()
            .any(|candidate| field_id_matches(candidate, field_id))
    }) {
        return Ok(figures.remove(index));
    }
    let available = figures
        .iter()
        .flat_map(|figure| figure.field_ids.iter())
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    Err(builtin_error(
        PLOT_NAME,
        &ERROR_INPUT,
        format!("FEA field `{field_id}` did not produce a mesh figure; available figure fields: {available}"),
    ))
}

#[cfg(feature = "plot-core")]
fn default_generated_figure_index(
    figures: &[crate::analysis::AnalysisGeneratedFigure],
) -> Option<usize> {
    let mut best: Option<(usize, u8)> = None;
    for (index, figure) in figures.iter().enumerate() {
        let score = default_generated_figure_score(figure);
        if best
            .map(|(_, best_score)| score > best_score)
            .unwrap_or(true)
        {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| index)
}

#[cfg(feature = "plot-core")]
fn default_generated_figure_score(figure: &crate::analysis::AnalysisGeneratedFigure) -> u8 {
    let kind_score = match figure.kind {
        crate::analysis::AnalysisGeneratedFigureKind::MeshResult => 40,
        crate::analysis::AnalysisGeneratedFigureKind::Modal
        | crate::analysis::AnalysisGeneratedFigureKind::Electromagnetic => 35,
        crate::analysis::AnalysisGeneratedFigureKind::Summary
        | crate::analysis::AnalysisGeneratedFigureKind::Convergence => 20,
        crate::analysis::AnalysisGeneratedFigureKind::Comparison
        | crate::analysis::AnalysisGeneratedFigureKind::Trend => 15,
    };
    figure
        .field_ids
        .iter()
        .map(|field_id| default_field_figure_score(field_id))
        .max()
        .unwrap_or(kind_score)
        .max(kind_score)
}

#[cfg(feature = "plot-core")]
fn default_field_figure_score(field_id: &str) -> u8 {
    let normalized = field_id.to_ascii_lowercase();
    if normalized.contains("residual")
        || normalized.contains("iteration")
        || normalized.contains("orthogonality")
        || normalized.contains("condition")
    {
        return 25;
    }
    if normalized.contains("von_mises") || normalized.contains("stress") {
        return 95;
    }
    if normalized.contains("temperature")
        || normalized.contains("heat_flux")
        || normalized.contains("velocity")
        || normalized.contains("pressure")
        || normalized.contains("magnetic_flux_density")
        || normalized.contains("electric_field")
        || normalized.contains("sound_pressure")
        || normalized.contains("coupling")
    {
        return 90;
    }
    if normalized.contains("mode_shape") || normalized.contains("displacement") {
        return 85;
    }
    if normalized.starts_with("structural.")
        || normalized.starts_with("modal.")
        || normalized.starts_with("thermal.")
        || normalized.starts_with("transient.")
        || normalized.starts_with("nonlinear.")
        || normalized.starts_with("em.")
        || normalized.starts_with("electro_thermal.")
        || normalized.starts_with("thermo_mechanical.")
        || normalized.starts_with("acoustic.")
        || normalized.starts_with("cfd.")
        || normalized.starts_with("fluid.")
        || normalized.starts_with("cht.")
        || normalized.starts_with("fsi.")
    {
        return 70;
    }
    40
}

#[cfg(feature = "plot-core")]
fn import_generated_figure(figure: crate::analysis::AnalysisGeneratedFigure) -> BuiltinResult<u32> {
    Ok(crate::builtins::plotting::import_runtime_figure(
        figure.figure,
    ))
}

struct NameValuePair<'a> {
    name: &'a Value,
    key: String,
    value: &'a Value,
}

fn expect_name_value_tail<'a>(
    builtin: &'static str,
    args: &'a [Value],
) -> BuiltinResult<Vec<NameValuePair<'a>>> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin} options must be Name, Value pairs"),
        ));
    }
    args.chunks(2)
        .map(|pair| {
            let key = option_key(&pair[0], builtin)?;
            Ok(NameValuePair {
                name: &pair[0],
                key,
                value: &pair[1],
            })
        })
        .collect()
}

fn json_fields_from_name_values(
    builtin: &'static str,
    args: &[Value],
) -> BuiltinResult<serde_json::Map<String, serde_json::Value>> {
    let mut fields = serde_json::Map::new();
    for pair in expect_name_value_tail(builtin, args)? {
        let raw = scalar_string(pair.name, builtin, &ERROR_INPUT)?;
        let key = canonical_field_name(&raw);
        if fields
            .insert(key.clone(), value_to_json(builtin, pair.value)?)
            .is_some()
        {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("duplicate {builtin} option `{key}`"),
            ));
        }
    }
    Ok(fields)
}

fn option_key(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    Ok(normalize_token(&scalar_string(
        value,
        builtin,
        &ERROR_INPUT,
    )?))
}

fn normalize_token(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn canonical_field_name(text: &str) -> String {
    let mut out = String::new();
    let mut previous_lower_or_digit = false;
    for ch in text.chars() {
        if ch == '-' || ch == ' ' {
            if !out.ends_with('_') && !out.is_empty() {
                out.push('_');
            }
            previous_lower_or_digit = false;
            continue;
        }
        if ch == '_' {
            if !out.ends_with('_') && !out.is_empty() {
                out.push('_');
            }
            previous_lower_or_digit = false;
            continue;
        }
        if ch.is_ascii_uppercase() {
            if previous_lower_or_digit && !out.ends_with('_') {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            previous_lower_or_digit = false;
        } else if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            previous_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    match normalize_token(&out).as_str() {
        "youngsmoduluspa" => "youngs_modulus_pa".to_string(),
        "poissonratio" => "poisson_ratio".to_string(),
        "density" | "densitykgperm3" => "density_kg_per_m3".to_string(),
        "magnitude" | "magnitudepa" => "magnitude_pa".to_string(),
        "current" | "currenta" => "current_a".to_string(),
        "phase" | "phaserad" => "phase_rad".to_string(),
        "specificimpedancepasperm" => "specific_impedance_pa_s_per_m".to_string(),
        "temperaturek" => "temperature_k".to_string(),
        "heatfluxwperm2" => "heat_flux_w_per_m2".to_string(),
        "ambienttemperaturek" => "ambient_temperature_k".to_string(),
        "coefficientwperm2k" => "coefficient_w_per_m2k".to_string(),
        "velocitympers" => "velocity_m_per_s".to_string(),
        "pressurepa" => "pressure_pa".to_string(),
        "amplitudescale" => "amplitude_scale".to_string(),
        "conductivitywpermk" => "conductivity_w_per_mk".to_string(),
        "specificheatjperkgk" => "specific_heat_j_per_kgk".to_string(),
        "conductivitysperm" => "conductivity_s_per_m".to_string(),
        "speedofsoundmpers" => "speed_of_sound_m_per_s".to_string(),
        "volumetricwperm3" => "volumetric_w_per_m3".to_string(),
        "inletvelocitympers" => "inlet_velocity_m_per_s".to_string(),
        "thermalconductancewperm2k" => "thermal_conductance_w_per_m2k".to_string(),
        "contactresistancem2kperw" => "contact_resistance_m2k_per_w".to_string(),
        "deterministicmode" => "deterministic_mode".to_string(),
        "precisionmode" => "precision_mode".to_string(),
        "preconditionermode" => "preconditioner_mode".to_string(),
        "qualitypolicy" => "quality_policy".to_string(),
        "prepcalibrationprofile" => "prep_calibration_profile".to_string(),
        "prepartifactid" => "prep_artifact_id".to_string(),
        "sweepfrequencyhz" => "sweep_frequency_hz".to_string(),
        "sweepenabled" => "sweep_enabled".to_string(),
        _ => out.trim_matches('_').to_string(),
    }
}

fn value_to_json(builtin: &'static str, value: &Value) -> BuiltinResult<serde_json::Value> {
    match value {
        Value::Num(n) => json_number(builtin, *n),
        Value::Int(i) => Ok(int_value_to_json(i)),
        Value::Bool(b) => Ok(serde_json::Value::Bool(*b)),
        Value::String(s) => Ok(serde_json::Value::String(s.clone())),
        Value::CharArray(chars) if chars.rows == 1 => {
            Ok(serde_json::Value::String(chars.data.iter().collect()))
        }
        Value::StringArray(array) if array.data.len() == 1 => {
            Ok(serde_json::Value::String(array.data[0].clone()))
        }
        Value::StringArray(array) => Ok(serde_json::Value::Array(
            array
                .data
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        )),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => numeric_scalar_to_json(
            builtin,
            tensor
                .numeric_value_at(0)
                .expect("validated scalar tensor storage"),
        ),
        Value::Tensor(tensor) => Ok(serde_json::Value::Array(
            (0..tensor.len())
                .map(|index| {
                    numeric_scalar_to_json(
                        builtin,
                        tensor
                            .numeric_value_at(index)
                            .expect("validated tensor storage"),
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Value::Cell(cell) => Ok(serde_json::Value::Array(
            cell.data
                .iter()
                .map(|item| value_to_json(builtin, item))
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Value::Struct(fields) => {
            let mut object = serde_json::Map::new();
            for (key, value) in &fields.fields {
                object.insert(canonical_field_name(key), value_to_json(builtin, value)?);
            }
            Ok(serde_json::Value::Object(object))
        }
        Value::Object(object) => {
            if let Some(Value::String(json)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) {
                serde_json::from_str(json).map_err(|err| {
                    builtin_error_with_source(builtin, &ERROR_INPUT, err.to_string(), err)
                })
            } else {
                let mut object_json = serde_json::Map::new();
                for (key, value) in &object.properties {
                    if key.starts_with("__runmat_") {
                        continue;
                    }
                    object_json.insert(canonical_field_name(key), value_to_json(builtin, value)?);
                }
                Ok(serde_json::Value::Object(object_json))
            }
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("cannot convert value to FEA JSON payload: {other:?}"),
        )),
    }
}

fn numeric_scalar_to_json(
    builtin: &'static str,
    value: NumericScalar,
) -> BuiltinResult<serde_json::Value> {
    match value {
        NumericScalar::F64(value) => json_number(builtin, value),
        NumericScalar::F32(value) => json_number(builtin, f64::from(value)),
        value => Ok(int_value_to_json(
            &value
                .into_int_value()
                .expect("non-floating numeric scalar is integer"),
        )),
    }
}

fn json_number(builtin: &'static str, value: f64) -> BuiltinResult<serde_json::Value> {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                "FEA numeric option values must be finite JSON numbers",
            )
        })
}

fn typed_json_with_overrides<T: Serialize + DeserializeOwned>(
    builtin: &'static str,
    default: T,
    fields: serde_json::Map<String, serde_json::Value>,
    label: &str,
) -> BuiltinResult<serde_json::Value> {
    let base = serde_json::to_value(default)
        .map_err(|err| builtin_error(builtin, &ERROR_INTERNAL, err.to_string()))?;
    let merged = json_with_overrides(builtin, base, fields, label)?;
    let typed: T = json_deserialize(builtin, merged, label)?;
    serde_json::to_value(typed)
        .map_err(|err| builtin_error_with_source(builtin, &ERROR_INTERNAL, err.to_string(), err))
}

fn json_with_overrides(
    builtin: &'static str,
    mut base: serde_json::Value,
    fields: serde_json::Map<String, serde_json::Value>,
    label: &str,
) -> BuiltinResult<serde_json::Value> {
    let Some(object) = base.as_object_mut() else {
        return Err(builtin_error(
            builtin,
            &ERROR_INTERNAL,
            format!("{label} default payload is not an object"),
        ));
    };
    for (key, value) in fields {
        if !object.contains_key(&key) {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("unsupported {label} option `{key}`"),
            ));
        }
        object.insert(key, value);
    }
    Ok(base)
}

fn json_deserialize<T: DeserializeOwned>(
    builtin: &'static str,
    value: serde_json::Value,
    label: &str,
) -> BuiltinResult<T> {
    serde_json::from_value(value)
        .map_err(|err| builtin_error(builtin, &ERROR_INPUT, format!("invalid {label}: {err}")))
}

fn typed_domain_data<T: DeserializeOwned + Serialize>(
    builtin: &'static str,
    label: &str,
    value: serde_json::Value,
) -> BuiltinResult<serde_json::Value> {
    let typed: T = json_deserialize(builtin, value, label)?;
    serde_json::to_value(typed)
        .map_err(|err| builtin_error_with_source(builtin, &ERROR_INTERNAL, err.to_string(), err))
}

fn json_to_string(value: serde_json::Value) -> BuiltinResult<String> {
    serde_json::from_value(value).map_err(|err| {
        builtin_error(
            MATERIAL_NAME,
            &ERROR_INPUT,
            format!("invalid string option: {err}"),
        )
    })
}

fn remove_required_f64(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    builtin: &'static str,
    key: &str,
) -> BuiltinResult<f64> {
    let Some(value) = fields.remove(key) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("missing required option `{key}`"),
        ));
    };
    serde_json::from_value(value).map_err(|err| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("invalid numeric option `{key}`: {err}"),
        )
    })
}

fn remove_optional_f64(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    builtin: &'static str,
    key: &str,
) -> BuiltinResult<Option<f64>> {
    fields
        .remove(key)
        .map(|value| {
            serde_json::from_value(value).map_err(|err| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("invalid numeric option `{key}`: {err}"),
                )
            })
        })
        .transpose()
}

fn remove_required_vector3(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    builtin: &'static str,
    key: &str,
) -> BuiltinResult<[f64; 3]> {
    let Some(value) = fields.remove(key) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("missing required vector option `{key}`"),
        ));
    };
    let values: Vec<f64> = serde_json::from_value(value).map_err(|err| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("invalid vector option `{key}`: {err}"),
        )
    })?;
    if values.len() != 3 {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("vector option `{key}` must contain exactly 3 values"),
        ));
    }
    Ok([values[0], values[1], values[2]])
}

fn move_known_fields(
    source: &mut serde_json::Map<String, serde_json::Value>,
    target: &mut serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> bool {
    let mut moved = false;
    for key in keys {
        if let Some(value) = source.remove(*key) {
            target.insert((*key).to_string(), value);
            moved = true;
        }
    }
    moved
}

fn reject_unknown_fields(
    builtin: &'static str,
    fields: serde_json::Map<String, serde_json::Value>,
) -> BuiltinResult<()> {
    if fields.is_empty() {
        return Ok(());
    }
    let keys = fields.keys().cloned().collect::<Vec<_>>().join(", ");
    Err(builtin_error(
        builtin,
        &ERROR_INPUT,
        format!("unsupported option field(s): {keys}"),
    ))
}

fn logical_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected logical scalar; got {other:?}"),
        )),
    }
}

fn exact_bool_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<bool> {
    if let Ok(value) = logical_from_value(builtin, value) {
        return Ok(value);
    }
    if let Some(integer) = tensor_utils::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "numeric logical option must be exactly zero or one",
            )),
        };
    }
    match ordinary_double_scalar(value) {
        Some(0.0) => Ok(false),
        Some(1.0) => Ok(true),
        _ => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "logical option must be a logical scalar or exact numeric zero or one",
        )),
    }
}

fn bool_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<bool> {
    bool::try_from(value).map_err(|err| builtin_error(builtin, &ERROR_INPUT, err))
}

fn usize_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<usize> {
    if let Some(int) = tensor_utils::scalar_integer_value(value) {
        return int.try_to_usize().ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                "expected non-negative integer value outside the platform range",
            )
        });
    }
    match ordinary_double_scalar(value) {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => {
            if n > usize::MAX as f64 || (usize::BITS == 64 && n == usize::MAX as f64) {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "expected non-negative integer value outside the platform range",
                ));
            }
            Ok(n as usize)
        }
        _ => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected non-negative integer value; got {value:?}"),
        )),
    }
}

fn ordinary_double_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Tensor(tensor)
            if tensor.len() == 1
                && tensor.numeric_dtype() == runmat_builtins::NumericDType::F64 =>
        {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn string_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| scalar_string(item, builtin, &ERROR_INPUT))
            .collect(),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::String(_) | Value::CharArray(_) => {
            Ok(vec![scalar_string(value, builtin, &ERROR_INPUT)?])
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected string, string array, or cell array of strings; got {other:?}"),
        )),
    }
}

fn usize_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor
                .shape
                .iter()
                .filter(|&&dimension| dimension > 1)
                .count()
                > 1
            {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "expected a numeric scalar or vector of indices, not a matrix",
                ));
            }
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .exact_values()
                    .into_iter()
                    .map(|value| usize_from_value(builtin, &Value::Int(value)))
                    .collect();
            }
            if tensor.numeric_dtype() != runmat_builtins::NumericDType::F64 {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "floating index selectors must use ordinary double storage",
                ));
            }
            tensor_utils::tensor_values_f64(tensor)
                .into_iter()
                .map(|value| usize_from_value(builtin, &Value::Num(value)))
                .collect()
        }
        Value::Int(_) | Value::Num(_) => Ok(vec![usize_from_value(builtin, value)?]),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected numeric scalar or vector of indices; got {other:?}"),
        )),
    }
}

fn one_based_usize_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<usize>> {
    usize_vec_from_value(builtin, value)?
        .into_iter()
        .map(|value| {
            value.checked_sub(1).ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "result indices are one-based and must be positive",
                )
            })
        })
        .collect()
}

fn parse_model_defaults_mode(text: &str) -> BuiltinResult<ModelDefaultsMode> {
    match normalize_token(text).as_str() {
        "profilescaffold" | "scaffold" | "profile" => Ok(ModelDefaultsMode::ProfileScaffold),
        "none" | "empty" => Ok(ModelDefaultsMode::None),
        other => Err(builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            format!("unsupported model defaults mode `{other}`"),
        )),
    }
}

fn resolved_document_to_object(document: FeaResolvedDocument) -> BuiltinResult<Value> {
    match document {
        FeaResolvedDocument::Study(spec) => study_to_object(*spec),
        FeaResolvedDocument::Sweep(spec) => sweep_to_object(spec),
    }
}

fn study_to_object(spec: AnalysisStudySpec) -> BuiltinResult<Value> {
    let mut object = serializable_to_object(
        STUDY_NAME,
        &ERROR_INTERNAL,
        FEA_STUDY_CLASS,
        &spec,
        Some(FEA_STUDY_SPEC_JSON_PROPERTY),
    )?;
    if let Value::Object(ref mut object) = object {
        object
            .properties
            .insert("id".to_string(), Value::String(spec.study_id));
    }
    Ok(object)
}

fn sweep_to_object(spec: AnalysisStudySweepSpec) -> BuiltinResult<Value> {
    let mut object = serializable_to_object(
        SWEEP_NAME,
        &ERROR_INTERNAL,
        FEA_SWEEP_CLASS,
        &spec,
        Some(FEA_SWEEP_SPEC_JSON_PROPERTY),
    )?;
    if let Value::Object(ref mut object) = object {
        object
            .properties
            .insert("id".to_string(), Value::String(spec.sweep_id));
    }
    Ok(object)
}

fn operation_result_to_object<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<Value> {
    let envelope =
        result.map_err(|err| operation_error(builtin, operation_error_descriptor, err))?;
    serializable_to_object(
        builtin,
        internal_error_descriptor,
        class_name,
        &envelope.data,
        hidden_json_property,
    )
}

fn operation_result_to_object_preserving_integers<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
    hidden_json_property: Option<&'static str>,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<Value> {
    let envelope =
        result.map_err(|err| operation_error(builtin, operation_error_descriptor, err))?;
    serializable_to_object_preserving_integers(
        builtin,
        internal_error_descriptor,
        class_name,
        &envelope.data,
        hidden_json_property,
        signed_fields,
        unsigned_fields,
    )
}

fn sweep_plan_result_to_object(
    result: Result<OperationEnvelope<AnalysisStudySweepPlanData>, OperationErrorEnvelope>,
) -> BuiltinResult<Value> {
    let mut envelope = result
        .map_err(|error| operation_error(PLAN_NAME, &ERROR_OPERATION, public_sweep_error(error)))?;
    one_base_failure_entries(PLAN_NAME, &mut envelope.data.failure_entries)?;
    serializable_to_object_preserving_integers(
        PLAN_NAME,
        &ERROR_INTERNAL,
        FEA_PLAN_CLASS,
        &envelope.data,
        None,
        &[],
        &[
            "study_count",
            "planned_count",
            "failed_count",
            "study_index",
        ],
    )
}

fn sweep_run_result_to_object(
    result: Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope>,
) -> BuiltinResult<Value> {
    let mut envelope = result
        .map_err(|error| operation_error(RUN_NAME, &ERROR_OPERATION, public_sweep_error(error)))?;
    one_base_failure_entries(RUN_NAME, &mut envelope.data.failure_entries)?;
    serializable_to_object_preserving_integers(
        RUN_NAME,
        &ERROR_INTERNAL,
        FEA_RUN_RESULT_CLASS,
        &envelope.data,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[],
        &[
            "study_count",
            "success_count",
            "failed_count",
            "study_index",
        ],
    )
}

fn one_base_failure_entries(
    builtin: &'static str,
    entries: &mut [AnalysisStudySweepFailureEntry],
) -> BuiltinResult<()> {
    for entry in entries {
        entry.study_index = entry.study_index.checked_add(1).ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INTERNAL,
                "study index cannot be represented at the one-based public boundary",
            )
        })?;
    }
    Ok(())
}

fn public_sweep_error(mut error: OperationErrorEnvelope) -> OperationErrorEnvelope {
    let Some(index) = error
        .context
        .get("study_index")
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return error;
    };
    let Some(public_index) = index.checked_add(1) else {
        return error;
    };
    error
        .context
        .insert("study_index".to_string(), public_index.to_string());
    error.message = error.message.replacen(
        &format!("at index {index} "),
        &format!("at index {public_index} "),
        1,
    );
    error
}

fn serializable_to_object<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    value: &T,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<Value> {
    serializable_to_object_value(builtin, error, class_name, value, hidden_json_property)
        .map(Value::Object)
}

fn serializable_to_object_preserving_integers<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    value: &T,
    hidden_json_property: Option<&'static str>,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<Value> {
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let object =
        serializable_to_object_value(builtin, error, class_name, value, hidden_json_property)?;
    let mut wrapped = Value::Object(object);
    promote_named_integer_fields(
        builtin,
        error,
        &mut wrapped,
        &json,
        signed_fields,
        unsigned_fields,
    )?;
    Ok(wrapped)
}

fn promote_named_integer_fields(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &mut Value,
    json: &serde_json::Value,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<()> {
    match (value, json) {
        (Value::Object(object), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                let Some(target) = object.properties.get_mut(name) else {
                    continue;
                };
                if signed_fields.contains(&name.as_str()) {
                    if let Some(exact) = exact_integer_json_value(builtin, error, child, true)? {
                        *target = exact;
                    }
                } else if unsigned_fields.contains(&name.as_str()) {
                    if let Some(exact) = exact_integer_json_value(builtin, error, child, false)? {
                        *target = exact;
                    }
                } else {
                    promote_named_integer_fields(
                        builtin,
                        error,
                        target,
                        child,
                        signed_fields,
                        unsigned_fields,
                    )?;
                }
            }
        }
        (Value::Struct(value), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                let Some(target) = value.fields.get_mut(name) else {
                    continue;
                };
                if signed_fields.contains(&name.as_str()) {
                    if let Some(exact) = exact_integer_json_value(builtin, error, child, true)? {
                        *target = exact;
                    }
                } else if unsigned_fields.contains(&name.as_str()) {
                    if let Some(exact) = exact_integer_json_value(builtin, error, child, false)? {
                        *target = exact;
                    }
                } else {
                    promote_named_integer_fields(
                        builtin,
                        error,
                        target,
                        child,
                        signed_fields,
                        unsigned_fields,
                    )?;
                }
            }
        }
        (Value::Cell(cell), serde_json::Value::Array(items)) => {
            for (target, child) in cell.data.iter_mut().zip(items) {
                promote_named_integer_fields(
                    builtin,
                    error,
                    target,
                    child,
                    signed_fields,
                    unsigned_fields,
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn exact_integer_json_value(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    json: &serde_json::Value,
    signed: bool,
) -> BuiltinResult<Option<Value>> {
    match json {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(number) if signed => number
            .as_i64()
            .map(|value| Some(Value::Int(IntValue::I64(value))))
            .ok_or_else(|| {
                builtin_error(builtin, error, "signed structural integer is out of range")
            }),
        serde_json::Value::Number(number) => number
            .as_u64()
            .map(|value| Some(Value::Int(IntValue::U64(value))))
            .ok_or_else(|| {
                builtin_error(
                    builtin,
                    error,
                    "unsigned structural integer is out of range",
                )
            }),
        serde_json::Value::Array(items) => {
            if signed {
                let values = items
                    .iter()
                    .map(|item| {
                        item.as_i64().ok_or_else(|| {
                            builtin_error(
                                builtin,
                                error,
                                "signed structural integer array is out of range",
                            )
                        })
                    })
                    .collect::<BuiltinResult<Vec<_>>>()?;
                Tensor::new_integer(IntegerStorage::I64(values), vec![1, items.len()])
                    .map(Value::Tensor)
                    .map(Some)
                    .map_err(|message| builtin_error(builtin, error, message))
            } else {
                let values = items
                    .iter()
                    .map(|item| {
                        item.as_u64().ok_or_else(|| {
                            builtin_error(
                                builtin,
                                error,
                                "unsigned structural integer array is out of range",
                            )
                        })
                    })
                    .collect::<BuiltinResult<Vec<_>>>()?;
                Tensor::new_integer(IntegerStorage::U64(values), vec![1, items.len()])
                    .map(Value::Tensor)
                    .map(Some)
                    .map_err(|message| builtin_error(builtin, error, message))
            }
        }
        _ => Err(builtin_error(
            builtin,
            error,
            "structural integer field has a noninteger representation",
        )),
    }
}

fn serializable_to_value_preserving_integers<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &T,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<Value> {
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let mut converted = value_from_json_preserving_integer_kinds(builtin, error, &json)?;
    promote_named_integer_fields(
        builtin,
        error,
        &mut converted,
        &json,
        signed_fields,
        unsigned_fields,
    )?;
    Ok(converted)
}

fn value_from_json_preserving_integer_kinds(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    json: &serde_json::Value,
) -> BuiltinResult<Value> {
    let mut converted = value_from_json(json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.message().to_string(), err))?;
    promote_json_integer_kinds(builtin, error, &mut converted, json)?;
    Ok(converted)
}

fn promote_json_integer_kinds(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &mut Value,
    json: &serde_json::Value,
) -> BuiltinResult<()> {
    match (value, json) {
        (target, serde_json::Value::Number(number)) if number.is_u64() => {
            *target = Value::Int(IntValue::U64(
                number.as_u64().expect("checked unsigned number"),
            ));
        }
        (target, serde_json::Value::Number(number)) if number.is_i64() => {
            *target = Value::Int(IntValue::I64(
                number.as_i64().expect("checked signed number"),
            ));
        }
        (Value::Tensor(tensor), serde_json::Value::Array(_)) => {
            if let Some(storage) = exact_json_integer_array(json, &tensor.shape) {
                *tensor = Tensor::new_integer(storage, tensor.shape.clone())
                    .map_err(|message| builtin_error(builtin, error, message))?;
            }
        }
        (Value::Struct(structure), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                if let Some(target) = structure.fields.get_mut(name) {
                    promote_json_integer_kinds(builtin, error, target, child)?;
                }
            }
        }
        (Value::Object(object), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                if let Some(target) = object.properties.get_mut(name) {
                    promote_json_integer_kinds(builtin, error, target, child)?;
                }
            }
        }
        (Value::Cell(cell), serde_json::Value::Array(items)) => {
            for (target, child) in cell.data.iter_mut().zip(items) {
                promote_json_integer_kinds(builtin, error, target, child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn exact_json_integer_array(json: &serde_json::Value, shape: &[usize]) -> Option<IntegerStorage> {
    fn collect<'a>(
        json: &'a serde_json::Value,
        numbers: &mut Vec<&'a serde_json::Number>,
    ) -> Option<()> {
        match json {
            serde_json::Value::Number(number) if number.is_i64() || number.is_u64() => {
                numbers.push(number);
                Some(())
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    collect(item, numbers)?;
                }
                Some(())
            }
            _ => None,
        }
    }

    let mut numbers = Vec::new();
    collect(json, &mut numbers)?;
    if numbers.is_empty() || numbers.len() != shape.iter().product::<usize>() {
        return None;
    }
    let row_major_index = |column_major_index: usize| {
        let mut row_major_index = 0;
        let mut column_stride = 1;
        for (dimension_index, &dimension) in shape.iter().enumerate() {
            let coordinate = (column_major_index / column_stride) % dimension;
            let row_stride = shape[dimension_index + 1..].iter().product::<usize>();
            row_major_index += coordinate * row_stride;
            column_stride *= dimension;
        }
        row_major_index
    };
    if numbers.iter().all(|number| number.is_u64()) {
        return Some(IntegerStorage::U64(
            (0..numbers.len())
                .map(|index| {
                    numbers[row_major_index(index)]
                        .as_u64()
                        .expect("checked unsigned number")
                })
                .collect(),
        ));
    }
    if numbers.iter().all(|number| number.is_i64()) {
        return Some(IntegerStorage::I64(
            (0..numbers.len())
                .map(|index| {
                    numbers[row_major_index(index)]
                        .as_i64()
                        .expect("checked signed number")
                })
                .collect(),
        ));
    }
    None
}

fn serializable_to_object_value<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    value: &T,
    hidden_json_property: Option<&'static str>,
) -> BuiltinResult<ObjectInstance> {
    ensure_fea_classes_registered();
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let converted = value_from_json_preserving_integer_kinds(builtin, error, &json)?;
    let mut object = ObjectInstance::new(class_name.to_string());
    if let Value::Struct(fields) = converted {
        object.properties = fields.fields.into_iter().collect();
    } else {
        object.properties.insert("value".to_string(), converted);
    }
    if let Some(property) = hidden_json_property {
        object
            .properties
            .insert(property.to_string(), Value::String(json.to_string()));
    }
    Ok(object)
}

fn geometry_asset_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<GeometryAsset> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin} geometry must be {GEOMETRY_ASSET_CLASS}"),
        ));
    };
    if object.class_name != GEOMETRY_ASSET_CLASS {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "{builtin} geometry must be {GEOMETRY_ASSET_CLASS}, got {}",
                object.class_name
            ),
        ));
    }
    object_json_property(builtin, object, GEOMETRY_ASSET_JSON_PROPERTY, &ERROR_INPUT)
}

fn geometry_asset_from_value_with_builtin(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<GeometryAsset> {
    geometry_asset_from_value(builtin, value)
}

fn object_json_property<T: DeserializeOwned>(
    builtin: &'static str,
    object: &ObjectInstance,
    property: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    let Some(Value::String(json)) = object.properties.get(property) else {
        return Err(builtin_error(
            builtin,
            error,
            format!(
                "{} is missing required runtime payload property `{property}`",
                object.class_name
            ),
        ));
    };
    serde_json::from_str(json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))
}

fn scalar_string(
    value: &Value,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        _ => Err(builtin_error(
            builtin,
            error,
            format!("{builtin} expected a text scalar"),
        )),
    }
}

fn parse_scalar_enum<T: DeserializeOwned>(text: &str, label: &str) -> BuiltinResult<T> {
    parse_scalar_enum_for_builtin(STUDY_NAME, text, label)
}

fn parse_scalar_enum_for_builtin<T: DeserializeOwned>(
    builtin: &'static str,
    text: &str,
    label: &str,
) -> BuiltinResult<T> {
    serde_yaml::from_str::<T>(&text.to_ascii_lowercase()).map_err(|err| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("invalid {label} value `{text}`: {err}"),
        )
    })
}

fn resolve_study_profile_and_run_kind(
    options: &StudyConstructorOptions,
) -> BuiltinResult<(AnalysisCreateModelProfile, AnalysisRunKind)> {
    let profile = options.profile.ok_or_else(|| {
        builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            "fea.study requires Profile; choose a physics profile from fea.capabilities().physicsProfiles",
        )
    })?;
    let run_kind = profile.derived_run_kind();
    if let Some(explicit_run_kind) = options.run_kind {
        if explicit_run_kind != run_kind {
            return Err(builtin_error(
                STUDY_NAME,
                &ERROR_INPUT,
                format!(
                    "explicit solver {} does not match Profile {}; omit RunKind or choose a matching Profile",
                    explicit_run_kind.as_snake_case(),
                    profile.as_snake_case()
                ),
            ));
        }
    }
    Ok((profile, run_kind))
}

fn ensure_fea_classes_registered() {
    static REGISTER: OnceLock<()> = OnceLock::new();
    REGISTER.get_or_init(|| {
        let workflow_methods = workflow_methods();
        for class_name in [FEA_STUDY_CLASS, FEA_SWEEP_CLASS] {
            runmat_builtins::register_class(ClassDef {
                name: class_name.to_string(),
                parent: None,
                properties: HashMap::new(),
                methods: workflow_methods.clone(),
            });
        }
        runmat_builtins::register_class(ClassDef {
            name: FEA_RUN_RESULT_CLASS.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: run_result_methods(),
        });
        runmat_builtins::register_class(ClassDef {
            name: FEA_RESULTS_CLASS.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: results_methods(),
        });
        for class_name in [FEA_VALIDATION_CLASS, FEA_PLAN_CLASS, FEA_RUN_RESULT_CLASS] {
            if class_name == FEA_RUN_RESULT_CLASS {
                continue;
            }
            runmat_builtins::register_class(ClassDef {
                name: class_name.to_string(),
                parent: None,
                properties: HashMap::new(),
                methods: HashMap::new(),
            });
        }
        for class_name in [
            FEA_MODEL_CLASS,
            FEA_MATERIAL_CLASS,
            FEA_MATERIAL_ASSIGNMENT_CLASS,
            FEA_BOUNDARY_CONDITION_CLASS,
            FEA_LOAD_CASE_CLASS,
            FEA_STEP_CLASS,
            FEA_DOMAIN_CLASS,
            FEA_INTERFACE_CLASS,
            FEA_RUN_OPTIONS_CLASS,
            FEA_FIELD_CLASS,
            FEA_COMPARE_CLASS,
            FEA_TRENDS_CLASS,
        ] {
            runmat_builtins::register_class(ClassDef {
                name: class_name.to_string(),
                parent: None,
                properties: HashMap::new(),
                methods: if class_name == FEA_FIELD_CLASS {
                    field_methods()
                } else {
                    HashMap::new()
                },
            });
        }
    });
}

fn workflow_methods() -> HashMap<String, MethodDef> {
    [
        ("validate", VALIDATE_NAME),
        ("plan", PLAN_NAME),
        ("run", RUN_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.to_string(),
            MethodDef {
                name: name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: function_name.to_string(),
                implicit_class_argument: None,
            },
        )
    })
    .collect()
}

fn run_result_methods() -> HashMap<String, MethodDef> {
    [
        ("results", RESULTS_NAME),
        ("field", FIELD_NAME),
        ("plot", PLOT_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.to_string(),
            MethodDef {
                name: name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: function_name.to_string(),
                implicit_class_argument: None,
            },
        )
    })
    .collect()
}

fn results_methods() -> HashMap<String, MethodDef> {
    [("field", FIELD_NAME), ("plot", PLOT_NAME)]
        .into_iter()
        .map(|(name, function_name)| {
            (
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: function_name.to_string(),
                    implicit_class_argument: None,
                },
            )
        })
        .collect()
}

fn field_methods() -> HashMap<String, MethodDef> {
    [("plot", PLOT_NAME)]
        .into_iter()
        .map(|(name, function_name)| {
            (
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: function_name.to_string(),
                    implicit_class_argument: None,
                },
            )
        })
        .collect()
}

fn operation_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    source: OperationErrorEnvelope,
) -> RuntimeError {
    let message = format!(
        "{}: {}: {}",
        error.message, source.error_code, source.message
    );
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(
            error
                .identifier
                .unwrap_or(ERROR_OPERATION.identifier.expect("descriptor identifier")),
        )
        .build()
}

fn builtin_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_identifier(
            error
                .identifier
                .unwrap_or(ERROR_INTERNAL.identifier.expect("descriptor identifier")),
        )
        .build()
}

fn builtin_error_with_source<E>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    build_runtime_error(format!("{}: {}", error.message, message.into()))
        .with_builtin(builtin)
        .with_identifier(
            error
                .identifier
                .unwrap_or(ERROR_INTERNAL.identifier.expect("descriptor identifier")),
        )
        .with_source(source)
        .build()
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CellArray, StructValue};

    const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
    const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

    fn cell(values: Vec<Value>) -> Value {
        let cols = values.len().max(1);
        Value::Cell(CellArray::new(values, 1, cols).expect("cell should build"))
    }

    fn force_vector() -> Value {
        Value::Tensor(Tensor::new_2d(vec![0.0, -1000.0, 0.0], 1, 3).expect("tensor should build"))
    }

    fn moment_vector() -> Value {
        Value::Tensor(Tensor::new_2d(vec![10.0, 20.0, 30.0], 1, 3).expect("tensor should build"))
    }

    fn boundary_payload(value: Value) -> BoundaryCondition {
        let Value::Object(object) = value else {
            panic!("expected boundary condition object");
        };
        let Some(Value::String(payload)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) else {
            panic!("expected boundary condition JSON payload");
        };
        serde_json::from_str(payload).expect("boundary condition payload should decode")
    }

    fn object_payload<T: DeserializeOwned>(value: &Value) -> T {
        let Value::Object(object) = value else {
            panic!("expected FEA object");
        };
        let Some(Value::String(payload)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) else {
            panic!("expected FEA JSON payload");
        };
        serde_json::from_str(payload).expect("FEA payload should decode")
    }

    fn boundary_args(kind: &str, fields: Vec<(&str, Value)>) -> Vec<Value> {
        let mut args = vec![
            Value::String("bc".into()),
            Value::String("region".into()),
            Value::String(kind.into()),
        ];
        for (name, value) in fields {
            args.push(Value::String(name.into()));
            args.push(value);
        }
        args
    }

    #[test]
    fn fea_usize_parsers_preserve_typed_bounds_and_reject_invalid_values() {
        use runmat_builtins::{IntValue, IntegerStorage};

        assert_eq!(
            usize_from_value(INTERFACE_NAME, &Value::Int(IntValue::U16(7))).unwrap(),
            7
        );
        assert!(usize_from_value(INTERFACE_NAME, &Value::Int(IntValue::I8(-1))).is_err());
        assert!(usize_from_value(INTERFACE_NAME, &Value::Num(1.5)).is_err());
        assert!(usize_vec_from_value(
            INTERFACE_NAME,
            &Value::Tensor(Tensor::new_2d(vec![1.0, -1.0], 1, 2).unwrap())
        )
        .is_err());
        let typed_indices =
            Tensor::new_integer(IntegerStorage::U16(vec![2, 4]), vec![1, 2]).unwrap();
        assert_eq!(
            usize_vec_from_value(INTERFACE_NAME, &Value::Tensor(typed_indices)).unwrap(),
            vec![2, 4]
        );

        let maximum = usize_from_value(INTERFACE_NAME, &Value::Int(IntValue::U64(u64::MAX)));
        if usize::BITS == 64 {
            assert_eq!(maximum.unwrap(), usize::MAX);
        } else {
            assert!(maximum.is_err());
        }
    }

    #[test]
    fn fea_public_object_mirror_preserves_recursive_integer_kinds() {
        #[derive(Serialize)]
        struct IntegerMirror {
            signed: i64,
            unsigned: u64,
            matrix: Vec<Vec<u64>>,
            floating: f64,
            floating_vector: Vec<f64>,
        }

        let object = serializable_to_object_value(
            RUN_OPTIONS_NAME,
            &ERROR_INTERNAL,
            FEA_RUN_OPTIONS_CLASS,
            &IntegerMirror {
                signed: -9,
                unsigned: u64::MAX,
                matrix: vec![vec![1, 2], vec![3, 4]],
                floating: 1.0,
                floating_vector: vec![2.0, 3.0],
            },
            None,
        )
        .expect("integer-preserving public mirror");
        assert!(matches!(
            object.properties.get("signed"),
            Some(Value::Int(IntValue::I64(-9)))
        ));
        assert!(matches!(
            object.properties.get("unsigned"),
            Some(Value::Int(IntValue::U64(u64::MAX)))
        ));
        let Some(Value::Tensor(matrix)) = object.properties.get("matrix") else {
            panic!("exact integer matrix");
        };
        assert_eq!(matrix.shape, vec![2, 2]);
        assert_eq!(
            matrix
                .integer_storage()
                .expect("integer storage")
                .exact_values(),
            vec![
                IntValue::U64(1),
                IntValue::U64(3),
                IntValue::U64(2),
                IntValue::U64(4),
            ]
        );
        assert!(matches!(
            object.properties.get("floating"),
            Some(Value::Num(1.0))
        ));
        assert!(matches!(
            object.properties.get("floating_vector"),
            Some(Value::Tensor(values)) if values.integer_storage().is_none()
        ));

        #[derive(Serialize)]
        struct EmptyIntegerMirror {
            available_mode_indices: Vec<usize>,
            iteration_counts: Vec<usize>,
        }
        let empty = serializable_to_object_preserving_integers(
            RESULTS_NAME,
            &ERROR_INTERNAL,
            FEA_RESULTS_CLASS,
            &EmptyIntegerMirror {
                available_mode_indices: Vec::new(),
                iteration_counts: Vec::new(),
            },
            None,
            &[],
            &["available_mode_indices", "iteration_counts"],
        )
        .expect("schema-aware empty integer vectors");
        let Value::Object(empty) = empty else {
            panic!("results object");
        };
        for name in ["available_mode_indices", "iteration_counts"] {
            let Some(Value::Tensor(values)) = empty.properties.get(name) else {
                panic!("empty exact vector {name}");
            };
            assert!(values.integer_storage().is_some());
            assert!(values.is_empty());
        }
    }

    #[test]
    fn fea_execution_controls_enforce_exact_structural_boundaries() {
        let options = block_on(fea_run_options_builtin(vec![
            Value::String("modal".into()),
            Value::String("ModeCount".into()),
            Value::Num(3.0),
            Value::String("ResidualWarnThreshold".into()),
            Value::Int(IntValue::U64(1)),
        ]))
        .expect("ordinary integral double count and typed floating control");
        let Value::Object(options) = options else {
            panic!("run options object");
        };
        let Some(Value::Struct(payload)) = options.properties.get("options") else {
            panic!("run options payload");
        };
        assert!(matches!(
            payload.fields.get("mode_count"),
            Some(Value::Int(IntValue::U64(3)))
        ));
        assert!(matches!(
            payload.fields.get("residual_warn_threshold"),
            Some(Value::Num(1.0))
        ));

        for (solver, exact_field, floating_field) in [
            ("modal", "ModeCount", "ResidualWarnThreshold"),
            ("acoustic", "ModeCount", "ResidualWarnThreshold"),
            ("thermal", "StepCount", "TimeStepS"),
            ("transient", "MaxStepRetries", "Tolerance"),
            ("cfd", "MaxLinearIters", "ResidualWarnThreshold"),
            ("cht", "StepCount", "ResidualWarnThreshold"),
            ("fsi", "MaxLinearIters", "Tolerance"),
            ("nonlinear", "TangentRefreshInterval", "Tolerance"),
            (
                "electromagnetic",
                "HarmonicMaxIterations",
                "HarmonicTolerance",
            ),
        ] {
            let value = block_on(fea_run_options_builtin(vec![
                Value::String(solver.into()),
                Value::String(exact_field.into()),
                Value::Int(IntValue::U32(3)),
                Value::String(floating_field.into()),
                Value::Int(IntValue::I16(1)),
            ]))
            .unwrap_or_else(|error| panic!("{solver} typed run options: {error}"));
            let Value::Object(object) = value else {
                panic!("{solver} run options object");
            };
            let Some(Value::Struct(payload)) = object.properties.get("options") else {
                panic!("{solver} run options payload");
            };
            let exact_field = canonical_field_name(exact_field);
            let floating_field = canonical_field_name(floating_field);
            assert!(matches!(
                payload.fields.get(&exact_field),
                Some(Value::Int(IntValue::U64(3)))
            ));
            assert!(matches!(
                payload.fields.get(&floating_field),
                Some(Value::Num(1.0))
            ));
        }

        let prep_context = block_on(fea_run_options_builtin(vec![
            Value::String("modal".into()),
            Value::String("PrepContext".into()),
            Value::String("internal".into()),
        ]))
        .expect_err("internal prep context must not be public");
        assert_eq!(prep_context.identifier(), Some("RunMat:fea:InvalidInput"));

        let extra_step = block_on(fea_step_builtin(vec![
            Value::String("step".into()),
            Value::String("modal".into()),
            Value::Num(1.0),
        ]))
        .expect_err("step has exact arity");
        assert_eq!(extra_step.identifier(), Some("RunMat:fea:InvalidInput"));

        let zero_window = block_on(fea_trends_builtin(vec![
            Value::String("WindowSize".into()),
            Value::Int(IntValue::U8(0)),
        ]))
        .expect_err("trend window must be positive");
        assert_eq!(zero_window.identifier(), Some("RunMat:fea:InvalidInput"));

        let scalar_double = Tensor::new(vec![4.0], vec![1, 1]).expect("double scalar tensor");
        assert_eq!(
            usize_from_value(RUN_OPTIONS_NAME, &Value::Tensor(scalar_double.clone())).unwrap(),
            4
        );
        assert!(exact_bool_from_value(
            RESULTS_NAME,
            &Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("double flag"))
        )
        .unwrap());

        let scalar_single =
            Tensor::new_with_dtype(vec![4.0], vec![1, 1], runmat_builtins::NumericDType::F32)
                .expect("single scalar tensor");
        assert!(usize_from_value(RUN_OPTIONS_NAME, &Value::Tensor(scalar_single)).is_err());
    }

    #[test]
    fn fea_result_selectors_are_exact_one_based_vectors_and_flags_are_zero_one() {
        let wide = 9_007_199_254_740_993_u64;
        let selectors = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("wide selector vector");
        let decoded = one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(selectors));
        if usize::BITS == 64 {
            assert_eq!(decoded.unwrap(), vec![(wide - 1) as usize]);
        } else {
            assert_eq!(
                decoded.unwrap_err().identifier(),
                Some("RunMat:fea:InvalidInput")
            );
        }

        let selectors = Tensor::new_integer(IntegerStorage::U32(vec![3, 1, 3]), vec![3, 1])
            .expect("selector vector");
        assert_eq!(
            one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(selectors)).unwrap(),
            vec![2, 0, 2]
        );
        let matrix = Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 3, 4]), vec![2, 2])
            .expect("selector matrix");
        assert!(one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(matrix)).is_err());
        assert!(!exact_bool_from_value(RESULTS_NAME, &Value::Int(IntValue::I8(0))).unwrap());
        assert!(exact_bool_from_value(RESULTS_NAME, &Value::Int(IntValue::U64(1))).unwrap());
        assert!(exact_bool_from_value(RESULTS_NAME, &Value::Int(IntValue::I16(2))).is_err());

        let single_selectors = Tensor::new_with_dtype(
            vec![1.0, 2.0],
            vec![1, 2],
            runmat_builtins::NumericDType::F32,
        )
        .expect("single selectors");
        assert!(
            one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(single_selectors)).is_err()
        );
    }

    #[test]
    fn fea_sweep_failures_cross_to_one_based_public_indices() {
        let mut entries = vec![AnalysisStudySweepFailureEntry {
            study_id: "bad-study".into(),
            study_index: 0,
            error_code: "RM.TEST".into(),
            message: "failed".into(),
        }];
        one_base_failure_entries(RUN_NAME, &mut entries).expect("public index translation");
        assert_eq!(entries[0].study_index, 1);

        let mut context = std::collections::BTreeMap::new();
        context.insert("study_index".into(), "0".into());
        let error = public_sweep_error(OperationErrorEnvelope {
            error_code: "RM.TEST".into(),
            error_type: crate::operations::OperationErrorType::Validation,
            message: "study sweep failed at index 0 for study_id bad-study".into(),
            operation: "test".into(),
            op_version: "1".into(),
            retryable: false,
            severity: crate::operations::OperationErrorSeverity::Error,
            context,
            trace_id: None,
            request_id: None,
            timestamp: "test".into(),
        });
        assert_eq!(
            error.context.get("study_index").map(String::as_str),
            Some("1")
        );
        assert!(error.message.contains("at index 1 "));
    }

    #[test]
    fn fea_json_preserves_native_integer_scalars_and_tensors() {
        let maximum = runmat_builtins::IntValue::U64(u64::MAX);
        assert_eq!(
            value_to_json(INTERFACE_NAME, &Value::Int(maximum.clone()))
                .expect("scalar json")
                .to_string(),
            maximum.decimal_string()
        );

        let scalar = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX]),
            vec![1, 1],
        )
        .expect("scalar tensor");
        assert_eq!(
            value_to_json(INTERFACE_NAME, &Value::Tensor(scalar))
                .expect("scalar tensor json")
                .to_string(),
            u64::MAX.to_string()
        );

        let tensor = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![42, u64::MAX]),
            vec![1, 2],
        )
        .expect("tensor");
        assert_eq!(
            value_to_json(INTERFACE_NAME, &Value::Tensor(tensor))
                .expect("tensor json")
                .to_string(),
            "[42,18446744073709551615]"
        );
    }

    #[test]
    fn fea_numeric_constructors_cross_all_integer_classes_once_into_binary64() {
        for integer in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ] {
            let expected = boundary_integer_to_f64(&integer);
            let domain = block_on(fea_domain_builtin(vec![
                Value::String("electromagnetic".into()),
                Value::String("AppliedCurrentA".into()),
                Value::Int(integer.clone()),
            ]))
            .expect("domain integer field");
            let domain: DomainPayload = object_payload(&domain);
            let domain: runmat_analysis_core::ElectromagneticDomain =
                json_deserialize(DOMAIN_NAME, domain.data, "electromagnetic domain")
                    .expect("typed domain storage boundary");
            assert_eq!(domain.applied_current_a, expected);

            let interface = block_on(fea_interface_builtin(vec![
                Value::String("contact".into()),
                Value::String("left".into()),
                Value::String("right".into()),
                Value::String("FrictionCoefficient".into()),
                Value::Int(integer.clone()),
            ]))
            .expect("interface integer field");
            let interface: AnalysisInterface = object_payload(&interface);
            let AnalysisInterfaceKind::Contact(contact) = interface.kind else {
                panic!("expected contact interface");
            };
            assert_eq!(contact.friction_coefficient, expected);

            let load = block_on(fea_load_case_builtin(vec![
                Value::String("pressure".into()),
                Value::String("face".into()),
                Value::String("pressure".into()),
                Value::String("MagnitudePa".into()),
                Value::Int(integer.clone()),
            ]))
            .expect("load integer field");
            let load: LoadCase = object_payload(&load);
            assert!(
                matches!(load.kind, LoadKind::Pressure { magnitude_pa } if magnitude_pa == expected)
            );

            let material = block_on(fea_material_builtin(vec![
                Value::String("material".into()),
                Value::String("YoungsModulusPa".into()),
                Value::Int(integer),
                Value::String("PoissonRatio".into()),
                Value::Int(IntValue::U8(0)),
            ]))
            .expect("material integer field");
            let material: MaterialModel = object_payload(&material);
            assert_eq!(material.mechanical.youngs_modulus_pa, expected);
        }
    }

    #[test]
    fn fea_domain_revision_and_field_metadata_remain_exact_in_public_objects() {
        let mut field_source = StructValue::new();
        field_source.insert("source_id", Value::String("temperature".into()));
        field_source.insert("revision", Value::Int(IntValue::U32(u32::MAX)));
        let domain = block_on(fea_domain_builtin(vec![
            Value::String("thermoMechanical".into()),
            Value::String("FieldSource".into()),
            Value::Struct(field_source),
        ]))
        .expect("domain source revision");
        let Value::Object(domain) = domain else {
            panic!("expected domain object");
        };
        let Value::Struct(data) = domain.properties.get("data").expect("domain data") else {
            panic!("expected domain data struct");
        };
        let Value::Struct(source) = data.fields.get("field_source").expect("field source") else {
            panic!("expected field source struct");
        };
        assert_eq!(
            source.fields.get("revision"),
            Some(&Value::Int(IntValue::U64(u64::from(u32::MAX))))
        );

        for revision in [
            Value::Int(IntValue::I8(-1)),
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Num(1.0),
        ] {
            let mut field_source = StructValue::new();
            field_source.insert("source_id", Value::String("temperature".into()));
            field_source.insert("revision", revision);
            let error = block_on(fea_domain_builtin(vec![
                Value::String("thermoMechanical".into()),
                Value::String("FieldSource".into()),
                Value::Struct(field_source),
            ]))
            .expect_err("invalid structural revision must reject");
            assert_eq!(error.identifier(), Some("RunMat:fea:InvalidInput"));
            assert_eq!(error.context.builtin.as_deref(), Some(DOMAIN_NAME));
        }

        let field = AnalysisField {
            field_id: "stress".into(),
            shape: vec![u32::MAX as usize, 0],
            values: AnalysisFieldValues::HostF64(Vec::new()),
        };
        let descriptor = AnalysisFieldDescriptor::from_field(&field);
        let object = field_to_object(&field, &descriptor).expect("field object");
        let Value::Tensor(shape) = object.properties.get("shape").expect("shape") else {
            panic!("expected integer shape tensor");
        };
        assert_eq!(shape.numeric_dtype(), runmat_builtins::NumericDType::U64);
        assert_eq!(
            shape.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::from(u32::MAX), 0]))
        );
        assert_eq!(
            object.properties.get("element_count"),
            Some(&Value::Int(IntValue::U64(0)))
        );

        let device_field = AnalysisField {
            field_id: "device".into(),
            shape: vec![u32::MAX as usize],
            values: AnalysisFieldValues::DeviceRef(runmat_analysis_core::DeviceFieldRef {
                backend: "wgpu".into(),
                token: "buffer".into(),
                element_count: u32::MAX as usize,
            }),
        };
        let Value::Struct(device) = field_values_value(&device_field).expect("device metadata")
        else {
            panic!("expected device field metadata");
        };
        assert_eq!(
            device.fields.get("element_count"),
            Some(&Value::Int(IntValue::U64(u64::from(u32::MAX))))
        );
    }

    #[test]
    fn fea_numeric_constructors_reject_resident_fields_without_provider_access() {
        let resident = || {
            Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: u32::MAX,
                buffer_id: u64::MAX - 1,
            })
        };
        let cases = [
            block_on(fea_domain_builtin(vec![
                Value::String("electromagnetic".into()),
                Value::String("AppliedCurrentA".into()),
                resident(),
            ])),
            block_on(fea_interface_builtin(vec![
                Value::String("contact".into()),
                Value::String("left".into()),
                Value::String("right".into()),
                Value::String("FrictionCoefficient".into()),
                resident(),
            ])),
            block_on(fea_load_case_builtin(vec![
                Value::String("pressure".into()),
                Value::String("face".into()),
                Value::String("pressure".into()),
                Value::String("MagnitudePa".into()),
                resident(),
            ])),
            block_on(fea_material_builtin(vec![
                Value::String("material".into()),
                Value::String("YoungsModulusPa".into()),
                resident(),
                Value::String("PoissonRatio".into()),
                Value::Num(0.3),
            ])),
        ];
        for result in cases {
            let error = result.expect_err("resident FEA constructor field must reject");
            assert_eq!(error.identifier(), Some("RunMat:fea:InvalidInput"));
            assert!(error.message().contains("cannot convert value"));
        }
    }

    #[test]
    fn fea_structural_serializer_preserves_plan_counts_and_compare_deltas() {
        #[derive(Serialize)]
        struct FailureEntry {
            study_index: usize,
        }
        #[derive(Serialize)]
        struct StructuralPayload {
            study_count: usize,
            failure_entries: Vec<FailureEntry>,
            quality_reason_count_delta: i64,
            optional_delta: Option<i64>,
        }
        let value = serializable_to_object_preserving_integers(
            PLAN_NAME,
            &ERROR_INTERNAL,
            FEA_PLAN_CLASS,
            &StructuralPayload {
                study_count: 3,
                failure_entries: vec![FailureEntry { study_index: 2 }],
                quality_reason_count_delta: -4,
                optional_delta: None,
            },
            None,
            &["quality_reason_count_delta", "optional_delta"],
            &["study_count", "study_index"],
        )
        .expect("structural serializer");
        let Value::Object(object) = value else {
            panic!("expected structural object");
        };
        assert_eq!(
            object.properties.get("study_count"),
            Some(&Value::Int(IntValue::U64(3)))
        );
        assert_eq!(
            object.properties.get("quality_reason_count_delta"),
            Some(&Value::Int(IntValue::I64(-4)))
        );
        assert!(matches!(
            object.properties.get("optional_delta"),
            Some(Value::Tensor(tensor)) if tensor.is_empty()
        ));
        let Some(Value::Cell(entries)) = object.properties.get("failure_entries") else {
            panic!("expected failure entry cell");
        };
        let Some(Value::Struct(entry)) = entries.data.first() else {
            panic!("expected failure entry struct");
        };
        assert_eq!(
            entry.fields.get("study_index"),
            Some(&Value::Int(IntValue::U64(2)))
        );
    }

    #[test]
    fn fea_constructor_aliases_arity_and_error_attribution_are_stable() {
        for (alias, expected) in [
            ("ConductivityWPerMk", "conductivity_w_per_mk"),
            ("SpecificHeatJPerKgK", "specific_heat_j_per_kgk"),
            ("ConductivitySPerM", "conductivity_s_per_m"),
            ("SpeedOfSoundMPerS", "speed_of_sound_m_per_s"),
            ("VolumetricWPerM3", "volumetric_w_per_m3"),
            ("InletVelocityMPerS", "inlet_velocity_m_per_s"),
            ("ThermalConductanceWPerM2K", "thermal_conductance_w_per_m2k"),
            ("ContactResistanceM2KPerW", "contact_resistance_m2k_per_w"),
        ] {
            assert_eq!(canonical_field_name(alias), expected, "{alias}");
        }

        let field_error = create_field_object_from_args(vec![
            Value::Num(1.0),
            Value::String("stress".into()),
            Value::Num(2.0),
        ])
        .expect_err("surplus fea.field argument");
        assert_eq!(field_error.context.builtin.as_deref(), Some(FIELD_NAME));

        let compare_error = create_compare_object_from_args(vec![
            Value::String("base".into()),
            Value::String("candidate".into()),
            Value::Num(2.0),
        ])
        .expect_err("surplus fea.compare argument");
        assert_eq!(compare_error.context.builtin.as_deref(), Some(COMPARE_NAME));

        let model_error = block_on(fea_model_builtin(vec![
            Value::String("model".into()),
            Value::Num(1.0),
        ]))
        .expect_err("invalid model geometry");
        assert_eq!(model_error.context.builtin.as_deref(), Some(MODEL_NAME));

        let duplicate_error = block_on(fea_material_builtin(vec![
            Value::String("material".into()),
            Value::String("YoungsModulusPa".into()),
            Value::Num(1.0),
            Value::String("youngs_modulus_pa".into()),
            Value::Num(2.0),
            Value::String("PoissonRatio".into()),
            Value::Num(0.3),
        ]))
        .expect_err("duplicate normalized field");
        assert_eq!(
            duplicate_error.context.builtin.as_deref(),
            Some(MATERIAL_NAME)
        );
        assert!(duplicate_error.message().contains("duplicate"));
    }

    #[test]
    fn fea_study_requires_geometry_asset() {
        let err = block_on(fea_study_builtin(vec![
            Value::String("demo".to_string()),
            Value::Num(1.0),
        ]))
        .expect_err("invalid geometry should fail");
        assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
    }

    #[test]
    fn fea_study_requires_profile() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
            geometry_path.to_string_lossy().to_string(),
        ))
        .expect("geometry should load");

        let err = block_on(fea_study_builtin(vec![
            Value::String("missing_profile".to_string()),
            geometry,
        ]))
        .expect_err("missing profile should fail");
        assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
        assert!(err.message().contains("fea.study requires Profile"));
    }

    #[test]
    fn fea_model_requires_profile() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
            geometry_path.to_string_lossy().to_string(),
        ))
        .expect("geometry should load");

        let err = block_on(fea_model_builtin(vec![
            Value::String("missing_profile_model".to_string()),
            geometry,
        ]))
        .expect_err("missing profile should fail");
        assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
        assert!(err.message().contains("fea.model requires Profile"));
    }

    #[test]
    fn fea_load_validate_and_plan_document_workflow() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        std::fs::write(tmp.path().join("part.stl"), TRIANGLE_STL)
            .expect("geometry fixture should write");
        let fea_path = tmp.path().join("bracket.fea");
        std::fs::write(
            &fea_path,
            r#"
version: 1
kind: study
id: bracket_static
geometry:
  path: part.stl
  units: meter
model:
  profile: linear_static_structural
run:
  backend: cpu
"#,
        )
        .expect("FEA fixture should write");

        let study = block_on(fea_load_builtin(fea_path.to_string_lossy().to_string()))
            .expect("FEA document should load");
        let Value::Object(study_object) = study.clone() else {
            panic!("expected loaded FEA study object");
        };
        assert_eq!(study_object.class_name, FEA_STUDY_CLASS);
        assert!(study_object
            .properties
            .contains_key(FEA_STUDY_SPEC_JSON_PROPERTY));

        let validation =
            block_on(fea_validate_builtin(study.clone())).expect("FEA study should validate");
        let Value::Object(validation_object) = validation else {
            panic!("expected validation object");
        };
        assert_eq!(validation_object.class_name, FEA_VALIDATION_CLASS);
        assert_eq!(
            validation_object.properties.get("valid"),
            Some(&Value::Bool(true))
        );

        let plan = block_on(fea_plan_builtin(study)).expect("FEA study should plan");
        let Value::Object(plan_object) = plan else {
            panic!("expected plan object");
        };
        assert_eq!(plan_object.class_name, FEA_PLAN_CLASS);
        assert!(plan_object.properties.contains_key("operation_sequence"));
    }

    #[test]
    fn fea_load_case_accepts_moment_and_torque_alias() {
        for kind in ["moment", "torque"] {
            let load = block_on(fea_load_case_builtin(vec![
                Value::String(format!("tip_{kind}")),
                Value::String("tip_node".to_string()),
                Value::String(kind.to_string()),
                Value::String("Vector".to_string()),
                moment_vector(),
            ]))
            .expect("moment load should build");
            assert_object_class(&load, FEA_LOAD_CASE_CLASS);

            let Value::Object(object) = load else {
                panic!("expected load object");
            };
            let Some(Value::String(payload)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY)
            else {
                panic!("expected load JSON payload");
            };
            let decoded: LoadCase =
                serde_json::from_str(payload).expect("load payload should decode");
            assert_eq!(decoded.load_id, format!("tip_{kind}"));
            assert_eq!(decoded.region_id, "tip_node");
            assert!(matches!(
                decoded.kind,
                LoadKind::Moment {
                    mx: 10.0,
                    my: 20.0,
                    mz: 30.0
                }
            ));
        }
    }

    #[test]
    fn fea_load_case_doc_keywords_include_moment_and_torque() {
        let doc = runmat_builtins::builtin_docs()
            .into_iter()
            .find(|doc| doc.name == "fea.loadCase")
            .expect("fea.loadCase doc metadata should be registered");
        let keywords = doc
            .keywords
            .expect("fea.loadCase should advertise keywords");
        let keyword_set = keywords
            .split(',')
            .map(str::trim)
            .collect::<std::collections::BTreeSet<_>>();

        assert!(keyword_set.contains("moment"));
        assert!(keyword_set.contains("torque"));
    }

    #[test]
    fn fea_boundary_condition_accepts_prescribed_rotation() {
        let boundary = block_on(fea_boundary_condition_builtin(vec![
            Value::String("tip_rotation".to_string()),
            Value::String("tip_node".to_string()),
            Value::String("prescribedRotation".to_string()),
            Value::String("rx".to_string()),
            Value::Num(0.1),
            Value::String("ry".to_string()),
            Value::Num(0.2),
            Value::String("rz".to_string()),
            Value::Num(0.3),
        ]))
        .expect("prescribed rotation boundary condition should build");
        assert_object_class(&boundary, FEA_BOUNDARY_CONDITION_CLASS);

        let Value::Object(object) = boundary else {
            panic!("expected boundary condition object");
        };
        let Some(Value::String(payload)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) else {
            panic!("expected boundary condition JSON payload");
        };
        let decoded: BoundaryCondition =
            serde_json::from_str(payload).expect("boundary condition payload should decode");
        assert_eq!(decoded.bc_id, "tip_rotation");
        assert_eq!(decoded.region_id, "tip_node");
        assert!(matches!(
            decoded.kind,
            BoundaryConditionKind::PrescribedRotation {
                rx: 0.1,
                ry: 0.2,
                rz: 0.3
            }
        ));
    }

    #[test]
    fn fea_boundary_condition_accepts_integer_fields_for_all_numeric_kinds() {
        let rotation = boundary_payload(
            block_on(fea_boundary_condition_builtin(boundary_args(
                "prescribedRotation",
                vec![
                    ("rx", Value::Int(IntValue::I8(1))),
                    ("ry", Value::Int(IntValue::U16(2))),
                    ("rz", Value::Int(IntValue::I32(3))),
                ],
            )))
            .unwrap(),
        );
        assert!(matches!(
            rotation.kind,
            BoundaryConditionKind::PrescribedRotation {
                rx: 1.0,
                ry: 2.0,
                rz: 3.0
            }
        ));

        let cases = [
            (
                "acousticImpedance",
                "specificImpedancePaSPerM",
                IntValue::U32(4),
            ),
            (
                "thermalPrescribedTemperature",
                "temperatureK",
                IntValue::I64(5),
            ),
            ("thermalHeatFlux", "heatFluxWPerM2", IntValue::U64(6)),
            ("cfdInletVelocity", "velocityMPerS", IntValue::I16(7)),
            ("cfdOutletPressure", "pressurePa", IntValue::U8(8)),
        ];
        for (kind, field, value) in cases {
            boundary_payload(
                block_on(fea_boundary_condition_builtin(boundary_args(
                    kind,
                    vec![(field, Value::Int(value))],
                )))
                .unwrap(),
            );
        }

        let convection = boundary_payload(
            block_on(fea_boundary_condition_builtin(boundary_args(
                "thermalConvection",
                vec![
                    ("ambientTemperatureK", Value::Int(IntValue::U8(9))),
                    ("coefficientWPerM2K", Value::Int(IntValue::I16(10))),
                ],
            )))
            .unwrap(),
        );
        assert!(matches!(
            convection.kind,
            BoundaryConditionKind::ThermalConvection {
                ambient_temperature_k: 9.0,
                coefficient_w_per_m2k: 10.0
            }
        ));
    }

    #[test]
    fn fea_boundary_condition_converts_every_integer_class_at_binary64_boundary() {
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ] {
            let expected = boundary_integer_to_f64(&value);
            let boundary = boundary_payload(
                block_on(fea_boundary_condition_builtin(boundary_args(
                    "thermalHeatFlux",
                    vec![("heatFluxWPerM2", Value::Int(value))],
                )))
                .unwrap(),
            );
            let BoundaryConditionKind::ThermalHeatFlux { heat_flux_w_per_m2 } = boundary.kind
            else {
                panic!("expected thermal heat-flux boundary");
            };
            assert_eq!(heat_flux_w_per_m2, expected);
        }
    }

    #[test]
    fn fea_boundary_condition_rejects_nonscalar_numeric_fields() {
        let values =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U8(vec![1, 2]), vec![1, 2])
                .unwrap();
        let err = block_on(fea_boundary_condition_builtin(boundary_args(
            "thermalHeatFlux",
            vec![("heatFluxWPerM2", Value::Tensor(values))],
        )))
        .expect_err("nonscalar field must fail");
        assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
        assert_eq!(
            err.context.builtin.as_deref(),
            Some(BOUNDARY_CONDITION_NAME)
        );
    }

    #[test]
    fn fea_boundary_condition_declares_seven_integer_forms() {
        assert_eq!(FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES.len(), 7);
        assert!(FEA_BOUNDARY_CONDITION_INTEGER_CAPABILITIES
            .iter()
            .flat_map(|capability| capability.inputs)
            .all(|input| input.classes.len() == 8));
    }

    #[test]
    fn typed_constructors_build_full_study_and_sweep_objects() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");

        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
            geometry_path.to_string_lossy().to_string(),
        ))
        .expect("geometry should load");
        let asset = geometry_asset_from_value(MODEL_NAME, &geometry)
            .expect("geometry payload should decode");
        let region_id = asset
            .regions
            .first()
            .expect("fixture should import a region")
            .region_id
            .clone();

        let material = block_on(fea_material_builtin(vec![
            Value::String("steel".to_string()),
            Value::String("YoungsModulusPa".to_string()),
            Value::Num(200e9),
            Value::String("PoissonRatio".to_string()),
            Value::Num(0.30),
        ]))
        .expect("material should build");
        assert_object_class(&material, FEA_MATERIAL_CLASS);

        let assignment = block_on(fea_material_assignment_builtin(vec![
            Value::String(region_id.clone()),
            Value::String("steel".to_string()),
        ]))
        .expect("material assignment should build");
        assert_object_class(&assignment, FEA_MATERIAL_ASSIGNMENT_CLASS);

        let fixed = block_on(fea_boundary_condition_builtin(vec![
            Value::String("fixed_base".to_string()),
            Value::String(region_id.clone()),
            Value::String("fixed".to_string()),
        ]))
        .expect("boundary condition should build");
        assert_object_class(&fixed, FEA_BOUNDARY_CONDITION_CLASS);

        let load = block_on(fea_load_case_builtin(vec![
            Value::String("tip_force".to_string()),
            Value::String(region_id.clone()),
            Value::String("force".to_string()),
            Value::String("Vector".to_string()),
            force_vector(),
        ]))
        .expect("load case should build");
        assert_object_class(&load, FEA_LOAD_CASE_CLASS);

        let step = block_on(fea_step_builtin(vec![
            Value::String("static_step".to_string()),
            Value::String("static".to_string()),
        ]))
        .expect("analysis step should build");
        assert_object_class(&step, FEA_STEP_CLASS);

        let selector = format!("id:{region_id}");
        let mut regional_delta = StructValue::new();
        regional_delta.insert("region_id", Value::String(selector.clone()));
        regional_delta.insert("temperature_delta_k", Value::Int(IntValue::I8(5)));
        let mut field_source = StructValue::new();
        field_source.insert("source_id", Value::String("temperature-map".into()));
        field_source.insert("revision", Value::Int(IntValue::U32(7)));
        field_source.insert("expected_region_ids", cell(vec![Value::String(selector)]));
        let domain = block_on(fea_domain_builtin(vec![
            Value::String("thermoMechanical".into()),
            Value::String("RegionTemperatureDeltas".into()),
            cell(vec![Value::Struct(regional_delta)]),
            Value::String("FieldSource".into()),
            Value::Struct(field_source),
        ]))
        .expect("thermo-mechanical domain should build");

        let model = block_on(fea_model_builtin(vec![
            Value::String("bracket_static_model".to_string()),
            geometry.clone(),
            Value::String("Defaults".to_string()),
            Value::String("none".to_string()),
            Value::String("Profile".to_string()),
            Value::String("linear_static_structural".to_string()),
            Value::String("Materials".to_string()),
            cell(vec![material]),
            Value::String("MaterialAssignments".to_string()),
            cell(vec![assignment]),
            Value::String("BoundaryConditions".to_string()),
            cell(vec![fixed]),
            Value::String("Loads".to_string()),
            cell(vec![load]),
            Value::String("Steps".to_string()),
            cell(vec![step]),
            Value::String("Domains".to_string()),
            cell(vec![domain]),
        ]))
        .expect("model should build");
        assert_object_class(&model, FEA_MODEL_CLASS);
        let decoded_model: AnalysisModel = object_payload(&model);
        let thermo = decoded_model
            .thermo_mechanical
            .expect("thermo-mechanical domain");
        assert_eq!(thermo.region_temperature_deltas[0].region_id, region_id);
        assert_eq!(
            thermo
                .field_source
                .expect("field source")
                .expected_region_ids,
            vec![region_id.clone()]
        );

        let run_options = block_on(fea_run_options_builtin(vec![
            Value::String("linear_static".to_string()),
            Value::String("DeterministicMode".to_string()),
            Value::Bool(true),
            Value::String("PrecisionMode".to_string()),
            Value::String("fp64".to_string()),
            Value::String("QualityPolicy".to_string()),
            Value::String("balanced".to_string()),
        ]))
        .expect("run options should build");
        assert_object_class(&run_options, FEA_RUN_OPTIONS_CLASS);

        let study = block_on(fea_study_builtin(vec![
            Value::String("bracket_static".to_string()),
            geometry,
            Value::String("Profile".to_string()),
            Value::String("linear_static_structural".to_string()),
            Value::String("Backend".to_string()),
            Value::String("cpu".to_string()),
            Value::String("Model".to_string()),
            model,
            Value::String("RunOptions".to_string()),
            run_options,
        ]))
        .expect("study should build");
        assert_object_class(&study, FEA_STUDY_CLASS);

        let sweep = block_on(fea_sweep_builtin(vec![
            Value::String("bracket_sweep".to_string()),
            cell(vec![study]),
            Value::String("FailFast".to_string()),
            Value::Bool(false),
        ]))
        .expect("sweep should build");
        assert_object_class(&sweep, FEA_SWEEP_CLASS);
    }

    #[test]
    fn fea_results_field_exposes_values_metadata_and_plot_context() {
        let (run_value, _study) = synthetic_plot_run_value();

        let results = block_on(fea_results_builtin(vec![run_value])).expect("results should load");
        let Value::Object(results_object) = results.clone() else {
            panic!("expected results object");
        };
        assert_eq!(results_object.class_name, FEA_RESULTS_CLASS);
        assert_eq!(
            results_object.properties.get("run_id"),
            Some(&Value::String("synthetic_plot_run".to_string()))
        );
        assert!(results_object
            .properties
            .contains_key(FEA_STUDY_CONTEXT_JSON_PROPERTY));

        let field = block_on(fea_field_builtin(vec![
            results,
            Value::String("von_mises".to_string()),
        ]))
        .expect("field should resolve by unique suffix");
        let Value::Object(field_object) = field else {
            panic!("expected field object");
        };
        assert_eq!(field_object.class_name, FEA_FIELD_CLASS);
        assert_eq!(
            field_object.properties.get("field_id"),
            Some(&Value::String("structural.von_mises".to_string()))
        );
        assert_eq!(
            field_object.properties.get("unit"),
            Some(&Value::String("Pa".to_string()))
        );
        assert_eq!(
            field_object.properties.get("location"),
            Some(&Value::String("element".to_string()))
        );
        assert_eq!(
            field_object.properties.get("topology_id"),
            Some(&Value::String("analysis_mesh".to_string()))
        );
        assert_eq!(
            field_object.properties.get("element_kind"),
            Some(&Value::String("tetrahedron4".to_string()))
        );
        assert_eq!(
            field_object.properties.get("entity_count"),
            Some(&Value::Int(runmat_builtins::IntValue::U64(1)))
        );
        assert_eq!(
            field_object.properties.get("value_count"),
            Some(&Value::Int(runmat_builtins::IntValue::U64(1)))
        );
        assert_eq!(
            field_object.properties.get("element_count"),
            Some(&Value::Int(runmat_builtins::IntValue::U64(1)))
        );
        let Some(Value::Tensor(values)) = field_object.properties.get("values") else {
            panic!("expected values tensor");
        };
        assert_eq!(values.shape, vec![1]);
        assert_eq!(values.materialize_f64(), vec![42.0]);
        assert!(field_object
            .properties
            .contains_key(FEA_STUDY_CONTEXT_JSON_PROPERTY));
        assert_eq!(
            field_object.properties.get(FEA_RUN_ID_CONTEXT_PROPERTY),
            Some(&Value::String("synthetic_plot_run".to_string()))
        );
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn fea_plot_returns_figure_handle_for_contextual_run_results_and_fields() {
        let (run_value, _study) = synthetic_plot_run_value();

        let run_handle = block_on(fea_plot_builtin(vec![
            run_value.clone(),
            Value::String("von_mises".to_string()),
        ]))
        .expect("run plot should create a figure");
        assert!(matches!(run_handle, Value::Num(handle) if handle >= 1.0));

        let results = block_on(fea_results_builtin(vec![run_value])).expect("results should load");
        let field = block_on(fea_field_builtin(vec![
            results,
            Value::String("structural.von_mises".to_string()),
        ]))
        .expect("field should resolve");
        let field_handle =
            block_on(fea_plot_builtin(vec![field])).expect("field plot should create a figure");
        assert!(matches!(field_handle, Value::Num(handle) if handle >= 1.0));
    }

    #[test]
    fn fea_plot_request_accepts_solver_mesh_edge_option() {
        let (run_value, _study) = synthetic_plot_run_value();

        let request = plot_request_from_args(&[
            run_value,
            Value::String("von_mises".to_string()),
            Value::String("mesh".to_string()),
            Value::String("solver".to_string()),
            Value::String("deformed".to_string()),
            Value::Bool(false),
            Value::String("overlay".to_string()),
            Value::String("cad".to_string()),
        ])
        .expect("plot request should parse mesh, deformation, and overlay options");

        assert_eq!(request.field_id.as_deref(), Some("von_mises"));
        assert!(request.options.show_solver_mesh_edges);
        assert!(!request.options.apply_deformation_overlay);
        assert_eq!(
            request.options.mesh_source,
            crate::analysis::AnalysisFigureMeshSource::CadReference
        );
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn fea_plot_default_prefers_von_mises_scalar_figure() {
        let mut figures = vec![
            generated_test_figure("deformation", vec!["structural.displacement"]),
            generated_test_figure("stress", vec!["structural.von_mises"]),
            generated_test_figure("residual", vec!["structural.residual_norm"]),
        ];

        let selected =
            select_generated_figure(&mut figures, None).expect("default figure should select");

        assert_eq!(selected.title, "stress");
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn fea_plot_default_selects_representative_non_structural_figures() {
        let cases = [
            (
                vec![
                    generated_test_figure("thermal residual", vec!["thermal.residual_norm"]),
                    generated_test_figure("temperature", vec!["thermal.temperature.0"]),
                ],
                "temperature",
            ),
            (
                vec![
                    generated_test_figure("flow residual", vec!["cfd.residual_momentum"]),
                    generated_test_figure("velocity", vec!["fluid.velocity"]),
                ],
                "velocity",
            ),
            (
                vec![
                    generated_test_figure("acoustic phase", vec!["acoustic.phase"]),
                    generated_test_figure("pressure", vec!["acoustic.pressure"]),
                ],
                "pressure",
            ),
            (
                vec![
                    generated_test_figure(
                        "coupling residual",
                        vec!["thermo_mechanical.coupling_residual.0"],
                    ),
                    generated_test_figure(
                        "thermal stress",
                        vec!["thermo_mechanical.thermal_stress.0"],
                    ),
                ],
                "thermal stress",
            ),
        ];

        for (mut figures, expected_title) in cases {
            let selected =
                select_generated_figure(&mut figures, None).expect("default figure should select");
            assert_eq!(selected.title, expected_title);
        }
    }

    #[cfg(feature = "plot-core")]
    fn generated_test_figure(
        title: &str,
        field_ids: Vec<&str>,
    ) -> crate::analysis::AnalysisGeneratedFigure {
        crate::analysis::AnalysisGeneratedFigure {
            kind: crate::analysis::AnalysisGeneratedFigureKind::MeshResult,
            title: title.to_string(),
            field_ids: field_ids.into_iter().map(str::to_string).collect(),
            topology_ids: Vec::new(),
            warnings: Vec::new(),
            figure: runmat_plot::plots::Figure::new(),
        }
    }

    #[test]
    fn fea_usize_parser_reads_typed_integer_storage_exactly_and_rejects_float_boundary() {
        let wide = if usize::BITS == 64 {
            9_007_199_254_740_993
        } else {
            u32::MAX as u64
        };
        let typed =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U64(vec![wide]), vec![1, 1])
                .expect("typed integer");

        assert_eq!(
            usize_from_value(STUDY_NAME, &Value::Tensor(typed)).expect("typed integer"),
            wide as usize
        );

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(usize_from_value(STUDY_NAME, &Value::Num(boundary)).is_err());
    }

    fn synthetic_plot_run_value() -> (Value, Value) {
        crate::analysis::storage::configure_artifact_store(
            crate::analysis::storage::AnalysisArtifactStoreConfig::InMemory,
        )
        .expect("artifact store should configure");

        let tmp = tempfile::tempdir().expect("tempdir should be created");
        std::fs::write(tmp.path().join("part.stl"), TRIANGLE_STL)
            .expect("geometry fixture should write");
        let fea_path = tmp.path().join("plot.fea");
        std::fs::write(
            &fea_path,
            r#"
version: 1
kind: study
id: synthetic_plot
geometry:
  path: part.stl
  units: meter
model:
  profile: linear_static_structural
run:
  backend: cpu
"#,
        )
        .expect("FEA fixture should write");
        let study = block_on(fea_load_builtin(fea_path.to_string_lossy().to_string()))
            .expect("study should load");
        let Value::Object(study_object) = &study else {
            panic!("expected study object");
        };
        let study_json = match study_object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY) {
            Some(Value::String(json)) => json.clone(),
            _ => panic!("expected study spec payload"),
        };

        let run = crate::analysis::AnalysisRunResult {
            run_id: "synthetic_plot_run".to_string(),
            run: runmat_analysis_fea::FeaRunResult {
                backend: ComputeBackend::Cpu,
                solver_backend: "synthetic".to_string(),
                solver_device_apply_k_ratio: 0.0,
                solver_method: "synthetic".to_string(),
                preconditioner: "none".to_string(),
                solver_host_sync_count: 0,
                diagnostics: Vec::new(),
                fields: vec![AnalysisField::host_f64(
                    "structural.von_mises",
                    vec![1],
                    vec![42.0],
                )],
            },
            render_topology: Some(crate::analysis::AnalysisRenderTopology {
                schema_version: "analysis_render_topology/v1".to_string(),
                source: crate::analysis::AnalysisRenderTopologySource::AnalysisMesh,
                meshes: vec![crate::analysis::AnalysisRenderMesh {
                    mesh_id: "synthetic_plot_boundary".to_string(),
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    triangles: vec![[0, 1, 2]],
                    regions: Vec::new(),
                    vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
                    triangle_volume_element_indices: vec![Some(0)],
                }],
            }),
            modal_results: None,
            thermal_results: None,
            transient_results: None,
            nonlinear_results: None,
            electromagnetic_results: None,
            model_validity: crate::analysis::QualityGate::Pass,
            solver_convergence: crate::analysis::QualityGate::Pass,
            result_quality: crate::analysis::QualityGate::Pass,
            run_status: crate::analysis::RunStatus::Publishable,
            publishable: true,
            quality_reasons: Vec::new(),
            provenance: crate::analysis::RunProvenance {
                backend: ComputeBackend::Cpu,
                solver_backend: "synthetic".to_string(),
                solver_device_apply_k_ratio: 0.0,
                solver_host_sync_count: 0,
                precision_mode: "fp64".to_string(),
                deterministic_mode: true,
                solver_method: "synthetic".to_string(),
                preconditioner: "none".to_string(),
                quality_policy: "balanced".to_string(),
                fallback_events: Vec::new(),
            },
        };
        crate::analysis::storage::persist_run_result(&run).expect("run should persist");

        let mut object = ObjectInstance::new(FEA_RUN_RESULT_CLASS.to_string());
        object.properties.insert(
            "run_id".to_string(),
            Value::String("synthetic_plot_run".to_string()),
        );
        object.properties.insert(
            FEA_RUN_ID_CONTEXT_PROPERTY.to_string(),
            Value::String("synthetic_plot_run".to_string()),
        );
        object.properties.insert(
            FEA_STUDY_CONTEXT_JSON_PROPERTY.to_string(),
            Value::String(study_json),
        );
        (Value::Object(object), study)
    }

    fn persist_synthetic_indexed_results() -> String {
        let indexed_fields = |prefix: &str| {
            vec![
                AnalysisField::host_f64(format!("{prefix}.0"), vec![1], vec![10.0]),
                AnalysisField::host_f64(format!("{prefix}.1"), vec![1], vec![20.0]),
            ]
        };
        let run_id = "synthetic_indexed_results".to_string();
        let run = crate::analysis::AnalysisRunResult {
            run_id: run_id.clone(),
            run: runmat_analysis_fea::FeaRunResult {
                backend: ComputeBackend::Cpu,
                solver_backend: "synthetic".to_string(),
                solver_device_apply_k_ratio: 0.0,
                solver_method: "synthetic".to_string(),
                preconditioner: "none".to_string(),
                solver_host_sync_count: 0,
                diagnostics: Vec::new(),
                fields: Vec::new(),
            },
            render_topology: None,
            modal_results: Some(crate::analysis::ModalResultsData {
                modal_payload_version: "modal_results/v1".to_string(),
                eigenvalues_hz: vec![10.0, 20.0],
                mode_shapes: indexed_fields("mode_shape"),
                residual_norms: vec![0.1, 0.2],
                mode_units: crate::analysis::ModalFrequencyUnits::Hz,
                frequency_basis: crate::analysis::ModalFrequencyBasis::NativeEigenSolve,
            }),
            thermal_results: None,
            transient_results: Some(crate::analysis::TransientResultsData {
                transient_payload_version: "transient_results/v1".to_string(),
                time_points_s: vec![0.0, 1.0],
                displacement_snapshots: indexed_fields("displacement"),
                rotation_snapshots: Vec::new(),
                velocity_snapshots: indexed_fields("velocity"),
                angular_velocity_snapshots: Vec::new(),
                acceleration_snapshots: indexed_fields("acceleration"),
                angular_acceleration_snapshots: Vec::new(),
                von_mises_snapshots: indexed_fields("von_mises"),
                kinetic_energy_snapshots: indexed_fields("kinetic_energy"),
                strain_energy_snapshots: indexed_fields("strain_energy"),
                residual_norm_snapshots: indexed_fields("residual_norm"),
                thermo_mechanical_temperature_snapshots: Vec::new(),
                thermo_mechanical_thermal_strain_snapshots: Vec::new(),
                thermo_mechanical_thermal_stress_snapshots: Vec::new(),
                thermo_mechanical_displacement_snapshots: Vec::new(),
                thermo_mechanical_von_mises_snapshots: Vec::new(),
                thermo_mechanical_coupling_residual_snapshots: Vec::new(),
                electro_thermal_temperature_snapshots: Vec::new(),
                electro_thermal_thermal_residual_snapshots: Vec::new(),
                residual_norms: vec![0.25],
                integration_method: crate::analysis::TransientIntegrationMethod::ImplicitEuler,
            }),
            nonlinear_results: None,
            electromagnetic_results: None,
            model_validity: crate::analysis::QualityGate::Pass,
            solver_convergence: crate::analysis::QualityGate::Pass,
            result_quality: crate::analysis::QualityGate::Pass,
            run_status: crate::analysis::RunStatus::Publishable,
            publishable: true,
            quality_reasons: Vec::new(),
            provenance: crate::analysis::RunProvenance {
                backend: ComputeBackend::Cpu,
                solver_backend: "synthetic".to_string(),
                solver_device_apply_k_ratio: 0.0,
                solver_host_sync_count: 0,
                precision_mode: "fp64".to_string(),
                deterministic_mode: true,
                solver_method: "synthetic".to_string(),
                preconditioner: "none".to_string(),
                quality_policy: "balanced".to_string(),
                fallback_events: Vec::new(),
            },
        };
        crate::analysis::storage::persist_run_result(&run).expect("indexed run should persist");
        run_id
    }

    #[test]
    fn fea_results_translates_successful_selectors_and_public_indices_once() {
        let run_id = persist_synthetic_indexed_results();
        let selected = block_on(fea_results_builtin(vec![
            Value::String(run_id.clone()),
            Value::String("ModeIndices".to_string()),
            Value::Int(IntValue::U8(2)),
            Value::String("TransientSnapshotIndices".to_string()),
            Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).expect("double selector")),
        ]))
        .expect("one-based selectors should resolve the second stored entries");
        let Value::Object(selected) = selected else {
            panic!("results object");
        };
        let Some(Value::Struct(modal)) = selected.properties.get("modal_results") else {
            panic!("modal results");
        };
        let Some(Value::Tensor(eigenvalues)) = modal.fields.get("eigenvalues_hz") else {
            panic!("modal eigenvalues");
        };
        assert_eq!(eigenvalues.materialize_f64(), vec![20.0]);
        let Some(Value::Struct(transient)) = selected.properties.get("transient_results") else {
            panic!("transient results");
        };
        let Some(Value::Tensor(time_points)) = transient.fields.get("time_points_s") else {
            panic!("transient time points");
        };
        assert_eq!(time_points.materialize_f64(), vec![1.0]);

        let full = block_on(fea_results_builtin(vec![Value::String(run_id)]))
            .expect("full indexed results");
        let Value::Object(full) = full else {
            panic!("results object");
        };
        let Some(Value::Struct(summary)) = full.properties.get("summary") else {
            panic!("results summary");
        };
        let Some(Value::Tensor(indices)) = summary.fields.get("available_mode_indices") else {
            panic!("available mode indices");
        };
        assert_eq!(
            indices
                .integer_storage()
                .expect("exact public indices")
                .exact_values(),
            vec![IntValue::U64(1), IntValue::U64(2)]
        );
    }

    fn assert_object_class(value: &Value, expected: &str) {
        let Value::Object(object) = value else {
            panic!("expected object value");
        };
        assert_eq!(object.class_name, expected);
        assert!(
            object.properties.contains_key(FEA_PAYLOAD_JSON_PROPERTY)
                || object.properties.contains_key(FEA_STUDY_SPEC_JSON_PROPERTY)
                || object.properties.contains_key(FEA_SWEEP_SPEC_JSON_PROPERTY)
        );
    }
}
