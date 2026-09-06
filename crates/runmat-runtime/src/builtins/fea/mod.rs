use runmat_analysis_core::{
    AcousticConstraint, AcousticConstraintKind, AnalysisBoundary, AnalysisConstraint,
    AnalysisField, AnalysisFieldValues, AnalysisInterface, AnalysisInterfaceKind, AnalysisModel,
    AnalysisModelId, AnalysisProcedure, AnalysisSequence, BoundaryOperator, CfdDomain,
    ComplexPhase, ConjugateHeatTransferInterfaceModel, ContactFriction, ContactInterfaceModel,
    CoordinateFrame, Displacement, DynamicViscosity, ElectricCurrent, ElectricPotential,
    ElectricalConstraint, ElectricalConstraintKind, ElectroThermalDomain, ElectromagneticDomain,
    EvidenceConfidence, ExcitationAction, ExcitationCase, ExcitationTarget, FluidConstraint,
    FluidConstraintKind, FluidStructureInterfaceModel, FluidStructureKinematics, Frequency,
    FrictionCoefficient, HeatFlux, HeatTransferCoefficient, MagneticConstraint,
    MagneticConstraintKind, MagneticVectorPotential, MassDensity, MaterialAcousticModel,
    MaterialAssignment, MaterialElectricalModel, MaterialMechanicalModel, MaterialModel,
    MaterialPlasticModel, MaterialThermalModel, PairedConstraint, PairingPolicy,
    PeriodicConstraint, PeriodicTransform, PhaseShiftedPeriodicConstraint, PhysicalSignal,
    PhysicsFamily, PrescribedStructuralDofs, PrescribedValue, Pressure, ProcedureStep,
    RegionTarget, Rotation, RotationDofValue, SolutionSpace, SpecificAcousticImpedance,
    StructuralConstraint, StructuralConstraintKind, StructuralRotationDof,
    StructuralTranslationDof, Temperature, TemperatureDifference, ThermalConstraint,
    ThermalConstraintKind, ThermalContactResistance, ThermalInterfaceTransfer,
    ThermoMechanicalDomain, Translation, TranslationDofValue, TurbulenceIntensity, Velocity,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_geometry_core::GeometryAsset;
use runmat_macros::runtime_builtin;
use runmat_types::MemberAccess;
use runmat_value::{IntValue, IntegerStorage, NumericScalar, ObjectInstance, Tensor, Value};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[cfg(test)]
use crate::analysis::AnalysisStudySweepValidateData;
use crate::analysis::{
    analysis_create_model_op, analysis_plan_study_op, analysis_plan_study_sweep_op,
    analysis_results_by_run_id_op, analysis_results_compare_op, analysis_run_study_op,
    analysis_run_study_sweep_op, analysis_trends_op, analysis_validate_study_op,
    analysis_validate_study_sweep_op, load_fea_document_from_path_async,
    AnalysisCreateModelIntentSpec, AnalysisExecutionPlan, AnalysisExecutionStep,
    AnalysisFieldDescriptor, AnalysisModelTemplate, AnalysisResultsCompareQuery,
    AnalysisResultsQuery, AnalysisSolverOptions, AnalysisStudySpec, AnalysisStudySweepData,
    AnalysisStudySweepFailureEntry, AnalysisStudySweepPlanData, AnalysisStudySweepSpec,
    AnalysisTrendsQuery, FeaResolvedDocument,
};
use crate::builtins::common::{json::int_value_to_json, tensor as tensor_utils};
use crate::builtins::geometry::{GEOMETRY_ASSET_CLASS, GEOMETRY_ASSET_JSON_PROPERTY};
use crate::builtins::io::json::jsondecode::value_from_json;
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) mod accuracy;
pub(crate) mod activations;
pub(crate) mod boundaries;
pub(crate) mod classes;
pub(crate) mod constraints;
pub(crate) mod domains;
pub(crate) mod excitations;
pub(crate) mod execution;
pub(crate) mod initial_conditions;
pub(crate) mod interfaces;
mod modeling;
pub(crate) mod observations;
pub(crate) mod response_spectra;
pub(crate) mod response_spectrum_policies;
pub(crate) mod sequences;
pub(crate) mod signal_policies;
pub(crate) mod signals;
pub(crate) mod steps;
pub(crate) mod study_policies;
mod study_values;
#[cfg(test)]
mod test_support;
pub(crate) mod values;
mod visualization;
use boundaries::*;
use classes::*;
use constraints::*;
use interfaces::*;
use modeling::*;
use steps::*;
#[cfg(test)]
use visualization::{mesh_request_from_args, plot_request_from_args};

const LOAD_NAME: &str = "fea.load";
const STUDY_NAME: &str = "fea.study";
const SWEEP_NAME: &str = "fea.sweep";
const MODEL_NAME: &str = "fea.model";
const MATERIAL_NAME: &str = "fea.material";
const MATERIAL_ASSIGNMENT_NAME: &str = "fea.materialAssignment";
const VALIDATE_NAME: &str = "fea.validate";
const PLAN_NAME: &str = "fea.plan";
const RUN_NAME: &str = "fea.run";
const RESULTS_NAME: &str = "fea.results";
const FIELD_NAME: &str = "fea.field";
const PLOT_NAME: &str = "plot";
const PLOT_METHOD_BUILTIN_NAME: &str = "__runmat_fea_plot";
const MESH_NAME: &str = "mesh";
const MESH_METHOD_BUILTIN_NAME: &str = "__runmat_fea_mesh";
const COMPARE_NAME: &str = "fea.compare";
const TRENDS_NAME: &str = "fea.trends";

pub(crate) fn ensure_value_classes_registered() {
    classes::ensure_fea_classes_registered();
    activations::ensure_activation_classes_registered();
    accuracy::ensure_accuracy_classes_registered();
    constraints::ensure_constraint_support_classes_registered();
    execution::ensure_execution_classes_registered();
    values::ensure_quantity_classes_registered();
    signal_policies::ensure_signal_policy_classes_registered();
    signals::ensure_signal_classes_registered();
    excitations::ensure_excitation_classes_registered();
    initial_conditions::ensure_initial_condition_classes_registered();
    observations::ensure_observation_classes_registered();
    response_spectrum_policies::ensure_spectrum_policy_classes_registered();
    response_spectra::ensure_response_spectrum_class_registered();
    sequences::ensure_sequence_classes_registered();
    study_policies::ensure_study_policy_classes_registered();
}

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
        description:
            "Required Template plus optional Mesh, Execution, ModelId, and typed model setup options.",
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
        description: "Template, frame, defaults, and typed model-component options.",
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
const PLOT_CONTEXT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "field",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "FEA field carrying result and study context.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Visualization options such as Edges, Overlay, or Deformed.",
    },
];
const MESH_CONTEXT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "context",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "FEA run, results, or field carrying mesh context.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Mesh display options such as Source or Edges.",
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
const COMPONENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "component = fea.component(args, ...)",
    inputs: &IN_VARIADIC_ARGS,
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
const PLOT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "figure = plot(field, Name, Value, ...)",
    inputs: &PLOT_CONTEXT_INPUTS,
    outputs: &OUT_ANY,
}];
const MESH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "figure = mesh(runOrResultsOrField, Name, Value, ...)",
    inputs: &MESH_CONTEXT_INPUTS,
    outputs: &OUT_ANY,
}];
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
pub const FEA_COMPONENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPONENT_SIGNATURES,
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

pub const FEA_STRUCTURAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This RunMat-native FEA API accepts object, text, or enum inputs rather than numeric data; nested typed objects retain their already-defined numeric contracts and no provider gather occurs.",
};

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
        ),
        FeaResolvedDocument::Sweep(spec) => operation_result_to_object(
            VALIDATE_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_SWEEP_VALIDATION_CLASS,
            analysis_validate_study_sweep_op(&spec, OperationContext::new(None, None)),
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
    name = "__runmat_fea_plot",
    category = "fea",
    summary = "Implement plot overloads for FEA result objects.",
    keywords = "fea,plot,visualize,mesh,von_mises,stress,field",
    descriptor(crate::builtins::fea::FEA_PLOT_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_plot_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    visualization::create_plot_from_args(args)
}

#[runtime_builtin(
    name = "__runmat_fea_mesh",
    category = "fea",
    summary = "Implement mesh overloads for FEA result objects.",
    keywords = "fea,mesh,visualize,topology,solver,cad",
    descriptor(crate::builtins::fea::FEA_MESH_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_mesh_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    visualization::create_mesh_from_args(args)
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
        Value::Object(ref object) if object.class_name == FEA_STUDY_CLASS => {
            let spec: AnalysisStudySpec = object_payload(builtin, &input, FEA_STUDY_CLASS)?;
            Ok(FeaResolvedDocument::Study(Box::new(spec)))
        }
        Value::Object(ref object) if object.class_name == FEA_SWEEP_CLASS => {
            let spec: AnalysisStudySweepSpec = object_payload(builtin, &input, FEA_SWEEP_CLASS)?;
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
    )
}

fn object_payload<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
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
    let json = value_to_json(builtin, value)?;
    serde_json::from_value(json).map_err(|err| {
        builtin_error_with_source(
            builtin,
            &ERROR_INPUT,
            format!(
                "{} cannot be decoded from its typed properties: {err}",
                object.class_name
            ),
            err,
        )
    })
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
                    "fea.RunResult does not contain its required run_id",
                )
            })
        }
        Value::Object(object) if object.class_name == FEA_SWEEP_RUN_RESULT_CLASS => {
            Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "fea.SweepRunResult contains multiple run_entries; select one run entry explicitly",
            ))
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
        .get("run_id")
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
            object_payload(builtin, value, FEA_RESULTS_CLASS)
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
    )?;
    insert_study_context(&mut object, spec)?;
    Ok(Value::Object(object))
}

fn insert_study_context(
    object: &mut ObjectInstance,
    spec: &AnalysisStudySpec,
) -> BuiltinResult<()> {
    object.properties.insert(
        FEA_STUDY_CONTEXT_PROPERTY.to_string(),
        study_to_object(spec.clone())?,
    );
    Ok(())
}

fn copy_study_context_property(source: &Value, target: &mut ObjectInstance) {
    if let Some(study) = study_context_value(source) {
        target
            .properties
            .insert(FEA_STUDY_CONTEXT_PROPERTY.to_string(), study);
    }
}

fn copy_run_id_context_property(source: &Value, target: &mut ObjectInstance) {
    if let Some(run_id) = run_id_context_from_value(source) {
        target
            .properties
            .entry("run_id".to_string())
            .or_insert(Value::String(run_id));
    }
}

fn study_context_value(value: &Value) -> Option<Value> {
    let Value::Object(object) = value else {
        return None;
    };
    if object.class_name == FEA_STUDY_CLASS {
        return Some(value.clone());
    }
    object.properties.get(FEA_STUDY_CONTEXT_PROPERTY).cloned()
}

fn study_context_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<AnalysisStudySpec> {
    let Some(study) = study_context_value(value) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin}: FEA visualization requires study geometry context; pass a fea.RunResult from fea.run(study), a derived fea.Results/fea.Field, or call {builtin}(study, run ID)"),
        ));
    };
    object_payload(builtin, &study, FEA_STUDY_CLASS)
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
        "deterministicmode" => "deterministic_mode".to_string(),
        "precisionmode" => "precision_mode".to_string(),
        "preconditionermode" => "preconditioner_mode".to_string(),
        "qualitypolicy" => "quality_policy".to_string(),
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
        Value::StringArray(array) => Ok(serde_json::Value::Array(
            array
                .data
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        )),
        Value::LogicalArray(array) => nested_array_to_json(builtin, &array.shape, |index| {
            Ok(serde_json::Value::Bool(array.data[index] != 0))
        }),
        Value::Tensor(tensor) if tensor.shape == [1, 1] => numeric_scalar_to_json(
            builtin,
            tensor
                .numeric_value_at(0)
                .expect("validated scalar tensor storage"),
        ),
        Value::Tensor(tensor) => nested_array_to_json(builtin, &tensor.shape, |index| {
            numeric_scalar_to_json(
                builtin,
                tensor
                    .numeric_value_at(index)
                    .expect("validated tensor storage"),
            )
        }),
        Value::Cell(cell) => Ok(serde_json::Value::Array(
            cell.data
                .iter()
                .map(|item| value_to_json(builtin, item))
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Value::Struct(fields) => {
            let mut object = serde_json::Map::new();
            for (key, value) in &fields.fields {
                object.insert(key.clone(), value_to_json(builtin, value)?);
            }
            Ok(serde_json::Value::Object(object))
        }
        Value::Object(object) => {
            let mut object_json = serde_json::Map::new();
            for (key, value) in &object.properties {
                object_json.insert(key.clone(), value_to_json(builtin, value)?);
            }
            Ok(serde_json::Value::Object(object_json))
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("cannot convert value to FEA JSON payload: {other:?}"),
        )),
    }
}

fn nested_array_to_json(
    builtin: &'static str,
    shape: &[usize],
    mut element: impl FnMut(usize) -> BuiltinResult<serde_json::Value>,
) -> BuiltinResult<serde_json::Value> {
    if shape.iter().product::<usize>() == 0 {
        return Ok(serde_json::Value::Array(Vec::new()));
    }
    fn build(
        builtin: &'static str,
        shape: &[usize],
        dimension: usize,
        offset: usize,
        element: &mut impl FnMut(usize) -> BuiltinResult<serde_json::Value>,
    ) -> BuiltinResult<serde_json::Value> {
        if dimension == shape.len() {
            return element(offset);
        }
        let stride = shape[..dimension].iter().product::<usize>();
        let values = (0..shape[dimension])
            .map(|coordinate| {
                let next = coordinate
                    .checked_mul(stride)
                    .and_then(|term| offset.checked_add(term));
                let Some(next) = next else {
                    return Err(builtin_error(
                        builtin,
                        &ERROR_INTERNAL,
                        "FEA value dimensions overflow while encoding a typed object",
                    ));
                };
                build(builtin, shape, dimension + 1, next, element)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        Ok(serde_json::Value::Array(values))
    }
    build(builtin, shape, 0, 0, &mut element)
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

fn json_deserialize<T: DeserializeOwned>(
    builtin: &'static str,
    value: serde_json::Value,
    label: &str,
) -> BuiltinResult<T> {
    serde_json::from_value(value)
        .map_err(|err| builtin_error(builtin, &ERROR_INPUT, format!("invalid {label}: {err}")))
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
            if tensor.len() == 1 && tensor.numeric_dtype() == runmat_value::NumericDType::F64 =>
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
            if tensor.numeric_dtype() != runmat_value::NumericDType::F64 {
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

fn resolved_document_to_object(document: FeaResolvedDocument) -> BuiltinResult<Value> {
    match document {
        FeaResolvedDocument::Study(spec) => study_to_object(*spec),
        FeaResolvedDocument::Sweep(spec) => sweep_to_object(spec),
    }
}

fn study_to_object(spec: AnalysisStudySpec) -> BuiltinResult<Value> {
    serializable_to_object(STUDY_NAME, &ERROR_INTERNAL, FEA_STUDY_CLASS, &spec)
}

fn sweep_to_object(spec: AnalysisStudySweepSpec) -> BuiltinResult<Value> {
    serializable_to_object(SWEEP_NAME, &ERROR_INTERNAL, FEA_SWEEP_CLASS, &spec)
}

fn operation_result_to_object<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
) -> BuiltinResult<Value> {
    let envelope =
        result.map_err(|err| operation_error(builtin, operation_error_descriptor, err))?;
    serializable_to_object(
        builtin,
        internal_error_descriptor,
        class_name,
        &envelope.data,
    )
}

fn operation_result_to_object_preserving_integers<T: Serialize>(
    builtin: &'static str,
    operation_error_descriptor: &'static BuiltinErrorDescriptor,
    internal_error_descriptor: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    result: Result<OperationEnvelope<T>, OperationErrorEnvelope>,
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
        FEA_SWEEP_PLAN_CLASS,
        &envelope.data,
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
        FEA_SWEEP_RUN_RESULT_CLASS,
        &envelope.data,
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
) -> BuiltinResult<Value> {
    serializable_to_object_value(builtin, error, class_name, value).map(Value::Object)
}

fn serializable_to_object_preserving_integers<T: Serialize>(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    class_name: &'static str,
    value: &T,
    signed_fields: &[&str],
    unsigned_fields: &[&str],
) -> BuiltinResult<Value> {
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let object = serializable_to_object_value(builtin, error, class_name, value)?;
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
                Tensor::new_integer(IntegerStorage::I64(values), vec![items.len()])
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
                Tensor::new_integer(IntegerStorage::U64(values), vec![items.len()])
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
    promote_json_scalar_kinds(builtin, error, &mut converted, json)?;
    Ok(converted)
}

fn promote_json_scalar_kinds(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &mut Value,
    json: &serde_json::Value,
) -> BuiltinResult<()> {
    match (value, json) {
        (target, serde_json::Value::String(value)) => {
            *target = Value::String(value.clone());
        }
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
                    promote_json_scalar_kinds(builtin, error, target, child)?;
                }
            }
        }
        (Value::Object(object), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                if let Some(target) = object.properties.get_mut(name) {
                    promote_json_scalar_kinds(builtin, error, target, child)?;
                }
            }
        }
        (Value::Cell(cell), serde_json::Value::Array(items)) => {
            for (target, child) in cell.data.iter_mut().zip(items) {
                promote_json_scalar_kinds(builtin, error, target, child)?;
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
) -> BuiltinResult<ObjectInstance> {
    ensure_fea_classes_registered();
    let json = serde_json::to_value(value)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))?;
    let public_json = omit_null_object_fields(json.clone());
    let converted = value_from_json_preserving_integer_kinds(builtin, error, &public_json)?;
    let mut object = ObjectInstance::new(class_name.to_string());
    if let Value::Struct(fields) = converted {
        object.properties = fields.fields.into_iter().collect();
    } else {
        object.properties.insert("value".to_string(), converted);
    }
    Ok(object)
}

fn omit_null_object_fields(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(fields) => serde_json::Value::Object(
            fields
                .into_iter()
                .filter_map(|(name, value)| {
                    (!value.is_null()).then(|| (name, omit_null_object_fields(value)))
                })
                .collect(),
        ),
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(omit_null_object_fields).collect())
        }
        value => value,
    }
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

fn workflow_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    [
        ("validate", VALIDATE_NAME),
        ("plan", PLAN_NAME),
        ("run", RUN_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.to_string(),
            crate::class_registry::RuntimeMethod {
                name: name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: function_name.to_string(),
                implicit_class_argument: None,
            },
        )
    })
    .collect()
}

fn run_result_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    [
        ("results", RESULTS_NAME),
        ("field", FIELD_NAME),
        ("mesh", MESH_METHOD_BUILTIN_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.to_string(),
            crate::class_registry::RuntimeMethod {
                name: name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: function_name.to_string(),
                implicit_class_argument: None,
            },
        )
    })
    .collect()
}

fn results_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    [("field", FIELD_NAME), ("mesh", MESH_METHOD_BUILTIN_NAME)]
        .into_iter()
        .map(|(name, function_name)| {
            (
                name.to_string(),
                crate::class_registry::RuntimeMethod {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: function_name.to_string(),
                    implicit_class_argument: None,
                },
            )
        })
        .collect()
}

fn field_methods() -> HashMap<String, crate::class_registry::RuntimeMethod> {
    [
        ("plot", PLOT_METHOD_BUILTIN_NAME),
        ("mesh", MESH_METHOD_BUILTIN_NAME),
    ]
    .into_iter()
    .map(|(name, function_name)| {
        (
            name.to_string(),
            crate::class_registry::RuntimeMethod {
                name: name.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
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
    use crate::builtins::fea::values::{
        dynamic_viscosity_builtin, frequency_builtin, mass_density_builtin,
        temperature_kelvin_builtin, turbulence_intensity_builtin,
    };
    use futures::executor::block_on;
    use runmat_analysis_fea::ComputeBackend;
    use runmat_meshing_core::CanonicalMeshingContract;
    use runmat_value::CellArray;

    const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
    const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

    fn static_sequence_json(document_id: &str) -> String {
        serde_json::to_string(&AnalysisSequence::from_steps(
            format!("{document_id}.sequence"),
            vec![ProcedureStep::structural_static("static")],
        ))
        .expect("canonical static sequence should serialize")
    }

    fn test_run_identity(
        procedure: runmat_analysis_core::AnalysisProcedure,
    ) -> runmat_analysis_fea::AnalysisRunIdentity {
        let assembly = runmat_analysis_core::StructuralFormulationRealization::new(
            runmat_analysis_core::StructuralOperatorImplementation::Tet4LinearElasticity,
            Some(runmat_analysis_core::StructuralRecoveryImplementation::Tet4ConstantStrain),
        );
        let realization = runmat_analysis_fea::admission::structural_exact_realization(
            runmat_analysis_fea::admission::StructuralRealizationInput {
                assembly,
                procedure,
                coupling: runmat_analysis_core::CouplingImplementation::Uncoupled,
                sources: vec![runmat_analysis_core::SourceImplementation::NotApplicable],
                boundary_transfers: vec![
                    runmat_analysis_core::BoundaryTransferImplementation::NotApplicable,
                ],
            },
        )
        .expect("test structural formulation must be valid");
        runmat_analysis_fea::admission::admit_formulation(
            runmat_analysis_fea::admission::FormulationAdmissionRequest {
                family: runmat_analysis_core::PhysicsFamily::Structural,
                procedure,
                solution_space: runmat_analysis_core::SolutionSpace::Direct,
                requirement: runmat_analysis_core::FormulationRequirement::Experimental,
                realization,
            },
        )
        .expect("test structural formulation must be admitted")
        .into_identity()
    }

    fn cell(values: Vec<Value>) -> Value {
        let shape = if values.is_empty() {
            (0, 0)
        } else {
            (1, values.len())
        };
        Value::Cell(CellArray::new(values, shape.0, shape.1).expect("cell should build"))
    }

    #[test]
    fn study_options_accept_canonical_signal_excitation_and_case_values() {
        crate::builtins::geometry::region::ensure_region_class_registered();
        let mut region =
            ObjectInstance::new(crate::builtins::geometry::region::GEOMETRY_REGION_CLASS.into());
        region
            .properties
            .insert("geometry_id".into(), Value::String("geom".into()));
        region
            .properties
            .insert("geometry_revision".into(), Value::Int(IntValue::U32(2)));
        region
            .properties
            .insert("region_id".into(), Value::String("tip".into()));
        region
            .properties
            .insert("name".into(), Value::String("Tip".into()));
        let signal_value = block_on(signals::force_constant(
            "tip_force.signal".into(),
            Value::Tensor(Tensor::new_2d(vec![0.0, -1_000.0, 0.0], 1, 3).unwrap()),
        ))
        .expect("typed force signal should build");
        let excitation_value = block_on(excitations::structural_force(
            "tip_force".into(),
            "service".into(),
            Value::Object(region),
            signal_value.clone(),
        ))
        .expect("typed structural excitation should build");
        let case_value = block_on(excitations::excitation_case_always(
            "service".into(),
            excitation_value.clone(),
        ))
        .expect("typed excitation case should build");

        let options = StudyConstructorOptions::parse(&[
            Value::String("Signals".into()),
            signal_value,
            Value::String("Excitations".into()),
            excitation_value,
            Value::String("ExcitationCases".into()),
            case_value,
        ])
        .expect("canonical authored components should parse");

        assert_eq!(options.signals.len(), 1);
        assert_eq!(options.excitations.len(), 1);
        assert_eq!(options.excitation_cases.len(), 1);
    }

    fn object_payload<T: DeserializeOwned>(value: &Value) -> T {
        let json = value_to_json("fea.test", value).expect("typed FEA value should encode");
        serde_json::from_value(json).expect("typed FEA value should decode")
    }

    #[test]
    fn fea_usize_parsers_preserve_typed_bounds_and_reject_invalid_values() {
        use runmat_value::{IntValue, IntegerStorage};

        assert_eq!(
            usize_from_value("fea.test", &Value::Int(IntValue::U16(7))).unwrap(),
            7
        );
        assert!(usize_from_value("fea.test", &Value::Int(IntValue::I8(-1))).is_err());
        assert!(usize_from_value("fea.test", &Value::Num(1.5)).is_err());
        assert!(usize_vec_from_value(
            "fea.test",
            &Value::Tensor(Tensor::new_2d(vec![1.0, -1.0], 1, 2).unwrap())
        )
        .is_err());
        let typed_indices =
            Tensor::new_integer(IntegerStorage::U16(vec![2, 4]), vec![1, 2]).unwrap();
        assert_eq!(
            usize_vec_from_value("fea.test", &Value::Tensor(typed_indices)).unwrap(),
            vec![2, 4]
        );

        let maximum = usize_from_value("fea.test", &Value::Int(IntValue::U64(u64::MAX)));
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
            RESULTS_NAME,
            &ERROR_INTERNAL,
            FEA_RESULTS_CLASS,
            &IntegerMirror {
                signed: -9,
                unsigned: u64::MAX,
                matrix: vec![vec![1, 2], vec![3, 4]],
                floating: 1.0,
                floating_vector: vec![2.0, 3.0],
            },
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
        let empty_step = block_on(structural_modal_builtin(Value::String(String::new())))
            .expect_err("step identity must not be empty");
        assert_eq!(empty_step.identifier(), Some("RunMat:fea:InvalidInput"));

        let zero_window = block_on(fea_trends_builtin(vec![
            Value::String("WindowSize".into()),
            Value::Int(IntValue::U8(0)),
        ]))
        .expect_err("trend window must be positive");
        assert_eq!(zero_window.identifier(), Some("RunMat:fea:InvalidInput"));

        assert!(exact_bool_from_value(
            RESULTS_NAME,
            &Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("double flag"))
        )
        .unwrap());
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

        let single_selectors =
            Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], runmat_value::NumericDType::F32)
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
    fn sweep_operation_classes_are_exact_and_do_not_inherit_single_run_methods() {
        ensure_fea_classes_registered();

        let validation = serializable_to_object(
            VALIDATE_NAME,
            &ERROR_INTERNAL,
            FEA_SWEEP_VALIDATION_CLASS,
            &AnalysisStudySweepValidateData {
                sweep_id: "sweep".into(),
                valid: true,
                issue_codes: Vec::new(),
                study_entries: Vec::new(),
                evidence_artifact_path: "validation.json".into(),
            },
        )
        .expect("sweep validation object");
        let plan = serializable_to_object_preserving_integers(
            PLAN_NAME,
            &ERROR_INTERNAL,
            FEA_SWEEP_PLAN_CLASS,
            &AnalysisStudySweepPlanData {
                sweep_id: "sweep".into(),
                study_count: 0,
                planned_count: 0,
                failed_count: 0,
                failure_entries: Vec::new(),
                plan_entries: Vec::new(),
                evidence_artifact_path: "plan.json".into(),
            },
            &[],
            &[
                "study_count",
                "planned_count",
                "failed_count",
                "study_index",
            ],
        )
        .expect("sweep plan object");
        let run = serializable_to_object_preserving_integers(
            RUN_NAME,
            &ERROR_INTERNAL,
            FEA_SWEEP_RUN_RESULT_CLASS,
            &AnalysisStudySweepData {
                sweep_id: "sweep".into(),
                identity: runmat_analysis_fea::execution::SweepIdentity::new(
                    runmat_execution::Digest::sha256(b"sweep"),
                    vec![runmat_analysis_fea::execution::StudyIdentity::new(
                        runmat_execution::Digest::sha256(b"study"),
                        None,
                    )
                    .expect("study identity")],
                    runmat_execution::Digest::sha256(b"composition"),
                )
                .expect("sweep identity"),
                study_count: 0,
                success_count: 0,
                failed_count: 0,
                failure_entries: Vec::new(),
                run_entries: Vec::new(),
                evidence_artifact_path: "run.json".into(),
            },
            &[],
            &[
                "study_count",
                "success_count",
                "failed_count",
                "study_index",
            ],
        )
        .expect("sweep run object");

        for (value, expected_class) in [
            (validation, FEA_SWEEP_VALIDATION_CLASS),
            (plan, FEA_SWEEP_PLAN_CLASS),
            (run, FEA_SWEEP_RUN_RESULT_CLASS),
        ] {
            assert_object_class(&value, expected_class);
            let payload = crate::execution::value_codec::encode_inline_value(&value)
                .expect("sweep value should be portable");
            assert_eq!(
                crate::execution::value_codec::decode_inline_value(&payload)
                    .expect("sweep value should decode"),
                value
            );
        }

        for class_name in [FEA_VALIDATION_CLASS, FEA_PLAN_CLASS, FEA_RUN_RESULT_CLASS] {
            assert!(crate::class_registry::lookup_property(class_name, "sweep_id").is_none());
        }
        assert!(crate::class_registry::lookup_method(FEA_RUN_RESULT_CLASS, "results").is_some());
        assert!(
            crate::class_registry::lookup_method(FEA_SWEEP_RUN_RESULT_CLASS, "results").is_none()
        );
    }

    #[test]
    fn fea_json_preserves_native_integer_scalars_and_tensors() {
        let maximum = runmat_value::IntValue::U64(u64::MAX);
        assert_eq!(
            value_to_json("fea.test", &Value::Int(maximum.clone()))
                .expect("scalar json")
                .to_string(),
            maximum.decimal_string()
        );

        let scalar = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX]),
            vec![1, 1],
        )
        .expect("scalar tensor");
        assert_eq!(
            value_to_json("fea.test", &Value::Tensor(scalar))
                .expect("scalar tensor json")
                .to_string(),
            u64::MAX.to_string()
        );

        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![42, u64::MAX]),
            vec![1, 2],
        )
        .expect("tensor");
        assert_eq!(
            value_to_json("fea.test", &Value::Tensor(tensor))
                .expect("tensor json")
                .to_string(),
            "[[42,18446744073709551615]]"
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
            let expected = integer.to_f64();
            let frequency = block_on(frequency_builtin(Value::Int(integer.clone())))
                .expect("frequency integer field");
            let domain = block_on(domains::electromagnetic_domain_builtin(frequency))
                .expect("typed electromagnetic domain");
            let domain: runmat_analysis_core::ElectromagneticDomain = object_payload(&domain);
            assert_eq!(domain.reference_frequency.hertz, expected);

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
    fn fea_domain_constructors_preserve_typed_physics_boundaries() {
        let temperature =
            block_on(temperature_kelvin_builtin(Value::Num(293.15))).expect("typed temperature");
        let frequency = block_on(frequency_builtin(Value::Num(60.0))).expect("typed frequency");
        let density = block_on(mass_density_builtin(Value::Num(1.225))).expect("typed density");
        let viscosity =
            block_on(dynamic_viscosity_builtin(Value::Num(1.81e-5))).expect("typed viscosity");
        let turbulence = block_on(turbulence_intensity_builtin(Value::Num(0.05)))
            .expect("typed turbulence intensity");

        let thermo = block_on(domains::thermomechanical_domain_builtin(
            temperature.clone(),
        ))
        .expect("thermo-mechanical domain");
        let electro = block_on(domains::electrothermal_domain_builtin(temperature))
            .expect("electro-thermal domain");
        let electromagnetic = block_on(domains::electromagnetic_domain_builtin(frequency.clone()))
            .expect("electromagnetic domain");
        let fluid = block_on(domains::fluid_domain_builtin(
            density, viscosity, turbulence,
        ))
        .expect("fluid domain");

        for (value, class_name) in [
            (thermo, FEA_THERMOMECHANICAL_DOMAIN_CLASS),
            (electro, FEA_ELECTROTHERMAL_DOMAIN_CLASS),
            (electromagnetic, FEA_ELECTROMAGNETIC_DOMAIN_CLASS),
            (fluid, FEA_FLUID_DOMAIN_CLASS),
        ] {
            assert_object_class(&value, class_name);
        }

        let error = block_on(domains::thermomechanical_domain_builtin(frequency))
            .expect_err("frequency must not cross the temperature boundary");
        assert_eq!(error.identifier(), Some("RunMat:fea:InvalidInput"));
    }

    #[test]
    fn fea_domain_classes_expose_only_their_own_constructors() {
        ensure_fea_classes_registered();
        assert!(crate::class_registry::lookup_method(
            FEA_THERMOMECHANICAL_DOMAIN_CLASS,
            "atReferenceTemperature"
        )
        .is_some());
        assert!(crate::class_registry::lookup_method(
            FEA_THERMOMECHANICAL_DOMAIN_CLASS,
            "atFrequency"
        )
        .is_none());
        assert!(
            crate::class_registry::lookup_method(FEA_FLUID_DOMAIN_CLASS, "newtonian").is_some()
        );
        assert!(runmat_builtins::builtin_function_by_name("fea.domain").is_none());
    }

    #[test]
    fn fea_model_rejects_duplicate_domain_families() {
        let temperature =
            block_on(temperature_kelvin_builtin(Value::Num(293.15))).expect("typed temperature");
        let first = block_on(domains::thermomechanical_domain_builtin(
            temperature.clone(),
        ))
        .expect("first domain");
        let second =
            block_on(domains::thermomechanical_domain_builtin(temperature)).expect("second domain");
        let error = modeling::domain_vec_from_value("fea.model", &cell(vec![first, second]))
            .expect_err("duplicate family must reject");
        assert_eq!(error.identifier(), Some("RunMat:fea:InvalidInput"));
        assert!(error
            .message()
            .contains("duplicate thermo-mechanical FEA domain"));
    }

    #[test]
    fn fea_field_metadata_remains_exact_in_public_objects() {
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
        assert_eq!(shape.numeric_dtype(), runmat_value::NumericDType::U64);
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
                descriptor: Default::default(),
            })
        };
        let cases = [
            block_on(frequency_builtin(resident())),
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
            assert!(
                error.message().contains("real numeric scalar")
                    || error.message().contains("cannot convert value")
            );
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
        assert!(!object.properties.contains_key("optional_delta"));
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
    fn fea_study_options_accept_a_jointly_validated_meshing_object() {
        let directory = tempfile::tempdir().expect("temporary artifact directory");
        let artifact = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
            runmat_meshing_core::ElementOrder::Tet4,
        );
        let evidence =
            runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&artifact);
        let mesh_path = directory.path().join("fixture.solver-mesh.cbor");
        let evidence_path = directory.path().join("fixture.meshing-evidence.cbor");
        std::fs::write(
            &mesh_path,
            artifact.canonical_encode().expect("canonical mesh bytes"),
        )
        .expect("mesh fixture");
        std::fs::write(
            &evidence_path,
            evidence
                .canonical_encode()
                .expect("canonical evidence bytes"),
        )
        .expect("evidence fixture");
        let mesh = block_on(crate::builtins::meshing::load_builtin(
            mesh_path.to_string_lossy().to_string(),
            evidence_path.to_string_lossy().to_string(),
        ))
        .expect("validated meshing object");

        let options = StudyConstructorOptions::parse(&[Value::String("Mesh".to_string()), mesh])
            .expect("study mesh option");
        let supplied = options.mesh.expect("typed mesh output");

        assert_eq!(supplied.artifact, artifact);
        assert_eq!(supplied.evidence, evidence);
    }

    #[test]
    fn fea_study_requires_template() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(vec![
            Value::String(geometry_path.to_string_lossy().to_string()),
        ]))
        .expect("geometry should load");

        let err = block_on(fea_study_builtin(vec![
            Value::String("missing_template".to_string()),
            geometry,
        ]))
        .expect_err("missing template should fail");
        assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
        assert!(err.message().contains("fea.study requires Template"));
    }

    #[test]
    fn fea_model_requires_template() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(vec![
            Value::String(geometry_path.to_string_lossy().to_string()),
        ]))
        .expect("geometry should load");

        let err = block_on(fea_model_builtin(vec![
            Value::String("missing_template_model".to_string()),
            geometry,
        ]))
        .expect_err("missing template should fail");
        assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
        assert!(err.message().contains("fea.model requires Template"));
    }

    #[test]
    fn fea_load_validate_and_plan_document_workflow() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        std::fs::write(tmp.path().join("part.stl"), TRIANGLE_STL)
            .expect("geometry fixture should write");
        let fea_path = tmp.path().join("bracket.fea");
        std::fs::write(
            &fea_path,
            format!(
                r#"
version: 1
kind: study
id: bracket_static
geometry:
  path: part.stl
  units: meter
model:
  template: linear_static_structural
constraints:
  - structural:
      constraint_id: fixed_support
      frame: global
      kind:
        fixed:
          target:
            region_id: region_default
            entity_dimension: face
            expected_topology: boundary_face
            cardinality: one_or_more
sequence: {}
execution:
  - step: static
    backend: cpu
    requirement: experimental
"#,
                static_sequence_json("bracket_static")
            ),
        )
        .expect("FEA fixture should write");

        let study = block_on(fea_load_builtin(fea_path.to_string_lossy().to_string()))
            .expect("FEA document should load");
        let Value::Object(study_object) = study.clone() else {
            panic!("expected loaded FEA study object");
        };
        assert_eq!(study_object.class_name, FEA_STUDY_CLASS);
        assert_eq!(
            study_object.properties.get("study_id"),
            Some(&Value::String("bracket_static".into()))
        );
        let inline = crate::execution::value_codec::encode_inline_value(&study)
            .expect("typed FEA study should cross an execution boundary");
        let decoded = crate::execution::value_codec::decode_inline_value(&inline)
            .expect("typed FEA study should decode from an execution boundary");
        assert_eq!(decoded, study);
        let mut invalid = study.clone();
        let Value::Object(invalid_object) = &mut invalid else {
            unreachable!("loaded study is an object")
        };
        invalid_object
            .properties
            .insert("unknown".into(), Value::Bool(true));
        assert!(crate::execution::value_codec::encode_inline_value(&invalid).is_err());
        let (study_id_property, owner) =
            crate::class_registry::lookup_property(FEA_STUDY_CLASS, "study_id")
                .expect("FEA study properties should be registered");
        assert_eq!(owner, FEA_STUDY_CLASS);
        assert!(study_id_property.is_constant);
        assert_eq!(study_id_property.set_access, MemberAccess::Private);

        let validation =
            block_on(fea_validate_builtin(study.clone())).expect("FEA study should validate");
        let Value::Object(validation_object) = validation else {
            panic!("expected validation object");
        };
        assert_eq!(validation_object.class_name, FEA_VALIDATION_CLASS);
        assert_eq!(
            validation_object.properties.get("valid"),
            Some(&Value::Bool(true)),
            "validation properties: {:?}",
            validation_object.properties,
        );

        let plan = block_on(fea_plan_builtin(study)).expect("FEA study should plan");
        let Value::Object(plan_object) = plan else {
            panic!("expected plan object");
        };
        assert_eq!(plan_object.class_name, FEA_PLAN_CLASS);
        assert!(plan_object.properties.contains_key("operation_sequence"));
    }

    #[test]
    fn typed_constructors_build_full_study_and_sweep_objects() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");

        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(vec![
            Value::String(geometry_path.to_string_lossy().to_string()),
        ]))
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

        let fixed_region = block_on(crate::builtins::geometry::region::geometry_region_builtin(
            geometry.clone(),
            region_id.clone(),
        ))
        .expect("fixed region should resolve");
        let fixed = block_on(structural_fixed_builtin(
            "fixed_base".to_string(),
            fixed_region,
        ))
        .expect("structural constraint should build");
        assert_object_class(&fixed, FEA_STRUCTURAL_CONSTRAINT_CLASS);

        let step = block_on(structural_static_builtin(Value::String(
            "static_step".to_string(),
        )))
        .expect("analysis step should build");
        assert_object_class(&step, FEA_STRUCTURAL_STEP_CLASS);

        let reference_temperature = block_on(temperature_kelvin_builtin(Value::Num(293.15)))
            .expect("reference temperature");
        let domain = block_on(domains::thermomechanical_domain_builtin(
            reference_temperature,
        ))
        .expect("thermo-mechanical domain should build");

        let model = block_on(fea_model_builtin(vec![
            Value::String("bracket_static_model".to_string()),
            geometry.clone(),
            Value::String("Defaults".to_string()),
            Value::String("none".to_string()),
            Value::String("Template".to_string()),
            Value::String("linear_static_structural".to_string()),
            Value::String("Materials".to_string()),
            cell(vec![material]),
            Value::String("MaterialAssignments".to_string()),
            cell(vec![assignment]),
            Value::String("Constraints".to_string()),
            cell(vec![fixed]),
            Value::String("Steps".to_string()),
            cell(vec![step.clone()]),
            Value::String("Domains".to_string()),
            cell(vec![domain]),
        ]))
        .expect("model should build");
        assert_object_class(&model, FEA_MODEL_CLASS);
        let decoded_model: AnalysisModel = object_payload(&model);
        let thermo = decoded_model
            .thermo_mechanical
            .expect("thermo-mechanical domain");
        assert_eq!(thermo.reference_temperature.kelvin, 293.15);

        let sequence = block_on(sequences::sequence(vec![
            Value::String("bracket_sequence".to_string()),
            cell(vec![step]),
            cell(Vec::new()),
            cell(Vec::new()),
            cell(Vec::new()),
        ]))
        .expect("sequence should build");
        let execution = block_on(execution::execution_plan_defaults(
            sequence,
            block_on(execution::backend_cpu()).unwrap(),
            block_on(execution::requirement_production()).unwrap(),
        ))
        .expect("execution should build");

        let study = block_on(fea_study_builtin(vec![
            Value::String("bracket_static".to_string()),
            geometry,
            Value::String("Template".to_string()),
            Value::String("linear_static_structural".to_string()),
            Value::String("Execution".to_string()),
            execution,
            Value::String("Model".to_string()),
            model,
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
        let _guard = crate::analysis::storage::artifact_store_test_guard();
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
            .contains_key(FEA_STUDY_CONTEXT_PROPERTY));

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
            Some(&Value::Int(runmat_value::IntValue::U64(1)))
        );
        assert_eq!(
            field_object.properties.get("value_count"),
            Some(&Value::Int(runmat_value::IntValue::U64(1)))
        );
        assert_eq!(
            field_object.properties.get("element_count"),
            Some(&Value::Int(runmat_value::IntValue::U64(1)))
        );
        let Some(Value::Tensor(values)) = field_object.properties.get("values") else {
            panic!("expected values tensor");
        };
        assert_eq!(values.shape, vec![1]);
        assert_eq!(values.materialize_f64(), vec![42.0]);
        assert!(field_object
            .properties
            .contains_key(FEA_STUDY_CONTEXT_PROPERTY));
        assert_eq!(
            field_object.properties.get("run_id"),
            Some(&Value::String("synthetic_plot_run".to_string()))
        );
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn plot_and_mesh_overloads_return_figure_handles_for_fea_objects() {
        let _guard = crate::analysis::storage::artifact_store_test_guard();
        let (run_value, _study) = synthetic_plot_run_value();

        let mesh_handle = block_on(crate::call_builtin_async(
            "mesh",
            &[
                run_value.clone(),
                Value::String("Source".to_string()),
                Value::String("solver".to_string()),
                Value::String("Edges".to_string()),
                Value::Bool(true),
            ],
        ))
        .expect("run mesh should create a figure");
        assert!(matches!(mesh_handle, Value::Num(handle) if handle >= 1.0));

        let results = block_on(fea_results_builtin(vec![run_value])).expect("results should load");
        let field = block_on(fea_field_builtin(vec![
            results,
            Value::String("structural.von_mises".to_string()),
        ]))
        .expect("field should resolve");
        let field_handle = block_on(crate::call_builtin_async("plot", &[field]))
            .expect("field plot should create a figure");
        assert!(matches!(field_handle, Value::Num(handle) if handle >= 1.0));
    }

    #[test]
    fn mesh_request_defaults_to_solver_edges_and_accepts_cad_reference() {
        let _guard = crate::analysis::storage::artifact_store_test_guard();
        let (run_value, _study) = synthetic_plot_run_value();
        let default = mesh_request_from_args(std::slice::from_ref(&run_value))
            .expect("default mesh request should parse");
        assert_eq!(
            default.source,
            crate::analysis::AnalysisFigureMeshSource::Solver
        );
        assert!(default.show_edges);

        let configured = mesh_request_from_args(&[
            run_value,
            Value::String("Source".to_string()),
            Value::String("cad_reference".to_string()),
            Value::String("Edges".to_string()),
            Value::Bool(false),
        ])
        .expect("configured mesh request should parse");
        assert_eq!(
            configured.source,
            crate::analysis::AnalysisFigureMeshSource::CadReference
        );
        assert!(!configured.show_edges);
    }

    #[test]
    fn fea_plot_request_accepts_solver_mesh_edge_option() {
        let _guard = crate::analysis::storage::artifact_store_test_guard();
        let (run_value, _study) = synthetic_plot_run_value();
        let results = block_on(fea_results_builtin(vec![run_value])).expect("results should load");
        let field = block_on(fea_field_builtin(vec![
            results,
            Value::String("structural.von_mises".to_string()),
        ]))
        .expect("field should resolve");

        let request = plot_request_from_args(&[
            field,
            Value::String("Edges".to_string()),
            Value::Bool(true),
            Value::String("deformed".to_string()),
            Value::Bool(false),
            Value::String("overlay".to_string()),
            Value::String("cad".to_string()),
        ])
        .expect("plot request should parse mesh, deformation, and overlay options");

        assert_eq!(request.field_id, "structural.von_mises");
        assert!(request.options.show_solver_mesh_edges);
        assert!(!request.options.apply_deformation_overlay);
        assert_eq!(
            request.options.mesh_source,
            crate::analysis::AnalysisFigureMeshSource::CadReference
        );
    }

    #[test]
    fn fea_usize_parser_reads_typed_integer_storage_exactly_and_rejects_float_boundary() {
        let wide = if usize::BITS == 64 {
            9_007_199_254_740_993
        } else {
            u32::MAX as u64
        };
        let typed = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![wide]), vec![1, 1])
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
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        std::fs::write(tmp.path().join("part.stl"), TRIANGLE_STL)
            .expect("geometry fixture should write");
        let fea_path = tmp.path().join("plot.fea");
        std::fs::write(
            &fea_path,
            format!(
                r#"
version: 1
kind: study
id: synthetic_plot
geometry:
  path: part.stl
  units: meter
model:
  template: linear_static_structural
sequence: {}
execution:
  - step: static
    backend: cpu
    requirement: experimental
"#,
                static_sequence_json("synthetic_plot")
            ),
        )
        .expect("FEA fixture should write");
        let study = block_on(fea_load_builtin(fea_path.to_string_lossy().to_string()))
            .expect("study should load");
        let identity = test_run_identity(runmat_analysis_core::AnalysisProcedure::Static);
        let run = crate::analysis::AnalysisRunResult {
            run_id: "synthetic_plot_run".to_string(),
            run: runmat_analysis_fea::FeaRunResult {
                identity,
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_method: runmat_analysis_fea::SolverMethod::DenseDirect,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
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
            run_status: crate::analysis::RunStatus::Completed,
            numerically_acceptable: true,
            production_publishable: false,
            quality_reasons: Vec::new(),
            production_publication_reasons: vec![
                crate::analysis::ProductionPublicationReason {
                    code: crate::analysis::ProductionPublicationReasonCode::FormulationNotProductionQualified,
                    detail: "the admitted Structural/Static formulation is Experimental, not production-qualified".to_string(),
                },
            ],
            provenance: crate::analysis::RunProvenance {
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_host_sync_count: 0,
                precision_mode: "fp64".to_string(),
                deterministic_mode: true,
                solver_method: runmat_analysis_fea::SolverMethod::DenseDirect,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
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
        object
            .properties
            .insert(FEA_STUDY_CONTEXT_PROPERTY.to_string(), study.clone());
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
        let identity =
            test_run_identity(runmat_analysis_core::AnalysisProcedure::TransientResponse);
        let run = crate::analysis::AnalysisRunResult {
            run_id: run_id.clone(),
            run: runmat_analysis_fea::FeaRunResult {
                identity,
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_method: runmat_analysis_fea::SolverMethod::DenseDirect,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
                solver_host_sync_count: 0,
                diagnostics: Vec::new(),
                fields: Vec::new(),
            },
            render_topology: None,
            modal_results: Some(crate::analysis::ModalResultsData {
                modal_payload_version: "modal_results/v1".to_string(),
                frequencies_hz: vec![10.0, 20.0],
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
            run_status: crate::analysis::RunStatus::Completed,
            numerically_acceptable: true,
            production_publishable: false,
            quality_reasons: Vec::new(),
            production_publication_reasons: vec![
                crate::analysis::ProductionPublicationReason {
                    code: crate::analysis::ProductionPublicationReasonCode::FormulationNotProductionQualified,
                    detail: "the admitted Structural/TransientResponse formulation is Experimental, not production-qualified".to_string(),
                },
            ],
            provenance: crate::analysis::RunProvenance {
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_host_sync_count: 0,
                precision_mode: "fp64".to_string(),
                deterministic_mode: true,
                solver_method: runmat_analysis_fea::SolverMethod::DenseDirect,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
                quality_policy: "balanced".to_string(),
                fallback_events: Vec::new(),
            },
        };
        crate::analysis::storage::persist_run_result(&run).expect("indexed run should persist");
        run_id
    }

    #[test]
    fn fea_results_translates_successful_selectors_and_public_indices_once() {
        let _guard = crate::analysis::storage::artifact_store_test_guard();
        let run_id = persist_synthetic_indexed_results();
        let selected = block_on(fea_results_builtin(vec![
            Value::String(run_id),
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
        let Some(Value::Tensor(eigenvalues)) = modal.fields.get("frequencies_hz") else {
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

        let Some(Value::Struct(summary)) = selected.properties.get("summary") else {
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
        assert!(!object.properties.is_empty());
        assert!(object
            .properties
            .keys()
            .all(|name| !name.starts_with("__runmat_")));
    }
}
