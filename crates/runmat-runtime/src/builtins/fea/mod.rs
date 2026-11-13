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
    MaterialAssignment, MaterialMechanicalModel, MaterialModel, MaterialThermalModel,
    MechanicalDampingModel, PairedConstraint, PairingPolicy, PeriodicConstraint, PeriodicTransform,
    PhaseShiftedPeriodicConstraint, PhysicalSignal, PhysicsFamily, PrescribedStructuralDofs,
    PrescribedValue, Pressure, ProcedureStep, RegionTarget, Rotation, RotationDofValue,
    SolutionSpace, SpecificAcousticImpedance, StructuralConstraint, StructuralConstraintKind,
    StructuralRotationDof, StructuralTranslationDof, Temperature, TemperatureDifference,
    ThermalConstraint, ThermalConstraintKind, ThermalContactResistance, ThermalInterfaceTransfer,
    ThermoMechanicalDomain, Translation, TranslationDofValue, TurbulenceIntensity, Velocity,
};
use runmat_builtins::fea_contract::identities;
use runmat_builtins::{BuiltinDescriptor, BuiltinErrorDescriptor};
use runmat_geometry_core::GeometryAsset;
use runmat_macros::runtime_builtin;
use runmat_types::MethodName;
use runmat_value::{IntValue, IntegerStorage, NumericScalar, ObjectInstance, Tensor, Value};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[cfg(test)]
use crate::analysis::AnalysisStudySweepValidateData;
use crate::analysis::{
    analysis_field_page_by_run_id_op, analysis_plan_study_op, analysis_plan_study_sweep_op,
    analysis_results_by_run_id_op, analysis_results_compare_op, analysis_run_study_op,
    analysis_run_study_sweep_op, analysis_trends_op, analysis_validate_study_op,
    analysis_validate_study_sweep_op, AnalysisExecutionPlan, AnalysisExecutionStep,
    AnalysisFieldDescriptor, AnalysisFieldRequestOptions, AnalysisFieldSelection,
    AnalysisResultsCompareQuery, AnalysisResultsQuery, AnalysisSolverOptions, AnalysisStudySpec,
    AnalysisStudySweepData, AnalysisStudySweepFailureEntry, AnalysisStudySweepPlanData,
    AnalysisStudySweepSpec, AnalysisTrendsQuery,
};
use crate::builtins::common::{json::int_value_to_json, tensor as tensor_utils};
use crate::builtins::geometry::{GEOMETRY_ASSET_CLASS, GEOMETRY_ASSET_JSON_PROPERTY};
use crate::builtins::io::json::jsondecode::value_from_json;
use crate::class_registry::{declaration::instance_methods, RuntimeMethod};
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) mod accuracy;
pub(crate) mod activations;
pub(crate) mod boundaries;
pub(crate) mod classes;
pub(crate) mod constraints;
pub(crate) mod damping;
pub(crate) mod domains;
pub(crate) mod excitations;
pub(crate) mod execution;
mod field_access;
mod artifact_access;
pub(crate) mod field_selectors;
pub(crate) mod generator_policies;
pub(crate) mod initial_conditions;
pub(crate) mod interfaces;
pub(crate) mod material_assignments;
mod modeling;
mod object_api;
mod object_projection;
pub(crate) mod observations;
pub(crate) mod response_spectra;
pub(crate) mod response_spectrum_policies;
pub(crate) mod run_results;
pub(crate) mod sequences;
pub(crate) mod signal_policies;
pub(crate) mod signals;
pub(crate) mod statistical_field_selectors;
pub(crate) mod steps;
pub(crate) mod study_policies;
mod study_values;
mod vector_values;
#[cfg(test)]
mod test_support;
pub(crate) mod values;
mod visualization;
use boundaries::*;
use classes::*;
use constraints::*;
use interfaces::*;
#[cfg(test)]
use material_assignments::*;
use modeling::*;
use steps::*;
use vector_values::{one_based_usize_vec_from_value, string_vec_from_value, usize_vec_from_value};
#[cfg(all(test, feature = "plot-core"))]
use visualization::{mesh_request_from_args, plot_request_from_args};

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
const FIELD_PAGE_NAME: &str = "fea.fieldPage";
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
    damping::ensure_damping_class_registered();
    execution::ensure_execution_classes_registered();
    generator_policies::ensure_generator_policy_classes_registered();
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

// Descriptors, error descriptors, parameter lists, and integer facts for the
// FEA operation surface are owned by the canonical catalog in runmat-builtins.
pub(crate) use runmat_builtins::fea_contract::descriptors::{
    ERROR_INPUT, ERROR_INTERNAL, ERROR_OPERATION,
};
pub use runmat_builtins::fea_contract::integer::FEA_STRUCTURAL_INTEGER_AUDIT;

#[runtime_builtin(
    name = "fea.study",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_study_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    create_study_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.sweep",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_sweep_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    create_sweep_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.model",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_model_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    create_model_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.material",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_material_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    create_material_object_from_args(args)
}

#[runtime_builtin(
    name = "fea.validate",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_validate_builtin(input: Value) -> BuiltinResult<Value> {
    match resolve_document_input(input, VALIDATE_NAME).await? {
        RuntimeResolvedDocument::Study(composed) => operation_result_to_object(
            VALIDATE_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_VALIDATION_CLASS,
            analysis_validate_study_op(&composed.study, OperationContext::new(None, None)),
        ),
        RuntimeResolvedDocument::Sweep(composed) => operation_result_to_object(
            VALIDATE_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_SWEEP_VALIDATION_CLASS,
            analysis_validate_study_sweep_op(&composed.sweep, OperationContext::new(None, None)),
        ),
    }
}

#[runtime_builtin(
    name = "fea.plan",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_plan_builtin(input: Value) -> BuiltinResult<Value> {
    match resolve_document_input(input, PLAN_NAME).await? {
        RuntimeResolvedDocument::Study(composed) => operation_result_to_object(
            PLAN_NAME,
            &ERROR_OPERATION,
            &ERROR_INTERNAL,
            FEA_PLAN_CLASS,
            analysis_plan_study_op(&composed.study, OperationContext::new(None, None)),
        ),
        RuntimeResolvedDocument::Sweep(composed) => sweep_plan_result_to_object(
            analysis_plan_study_sweep_op(&composed.sweep, OperationContext::new(None, None)),
        ),
    }
}

#[runtime_builtin(
    name = "fea.run",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_run_builtin(input: Value) -> BuiltinResult<Value> {
    if matches!(
        input,
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_)
    ) {
        // A `.fea` path runs through the host's public study runner, as `runmat run`.
        let path = scalar_string(&input, RUN_NAME, &ERROR_INPUT)?;
        let service = crate::context::legacy::active()
            .and_then(|runtime| runtime.service_ports().analysis().cloned())
            .ok_or_else(|| {
                builtin_error(
                    RUN_NAME,
                    &ERROR_OPERATION,
                    "running a .fea document requires a host study runner",
                )
            })?;
        let publication = service
            .execute_document(std::path::PathBuf::from(path))
            .await?;
        let object = serializable_to_object_value(
            RUN_NAME,
            &ERROR_INTERNAL,
            FEA_RUN_RESULT_CLASS,
            &publication,
        )?;
        return Ok(Value::Object(object));
    }
    match resolve_document_input(input, RUN_NAME).await? {
        RuntimeResolvedDocument::Study(mut composed) => {
            signals::materialize::materialize_study(
                &mut composed.study,
                &composed.frozen_generators,
            )
            .await?;
            run_study_result_to_object(&composed.study).await
        }
        RuntimeResolvedDocument::Sweep(mut composed) => {
            signals::materialize::materialize_sweep(
                &mut composed.sweep,
                &composed.frozen_generators,
            )
            .await?;
            sweep_run_result_to_object(run_sweep(&composed.sweep).await)
        }
    }
}

#[runtime_builtin(
    name = "fea.results",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_results_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    create_results_object_from_args(args).await
}

#[runtime_builtin(
    name = "fea.field",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_field_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    create_field_object_from_args(args).await
}

#[runtime_builtin(
    name = "fea.fieldPage",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_field_page_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    field_access::create_page_from_args(args).await
}

#[runtime_builtin(
    name = "__runmat_fea_plot",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_plot_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    visualization::create_plot_from_args(args).await
}

#[runtime_builtin(
    name = "__runmat_fea_mesh",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_mesh_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
    visualization::create_mesh_from_args(args).await
}

#[runtime_builtin(
    name = "fea.compare",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_compare_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    create_compare_object_from_args(args).await
}

#[runtime_builtin(
    name = "fea.trends",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea"
)]
pub async fn fea_trends_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    create_trends_object_from_args(args).await
}

async fn resolve_document_input(
    input: Value,
    builtin: &'static str,
) -> BuiltinResult<RuntimeResolvedDocument> {
    match input {
        Value::Object(ref object) if object.class_name().is(identities::FEA_STUDY_CLASS) => {
            modeling::study_from_value_with_generators(builtin, &input)
                .map(Box::new)
                .map(RuntimeResolvedDocument::Study)
        }
        Value::Object(ref object) if object.class_name().is(identities::FEA_SWEEP_CLASS) => {
            modeling::sweep_from_value_with_generators(builtin, &input)
                .map(RuntimeResolvedDocument::Sweep)
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {FEA_STUDY_CLASS} or {FEA_SWEEP_CLASS}; got {other:?}"),
        )),
    }
}

enum RuntimeResolvedDocument {
    Study(Box<modeling::ComposedStudy>),
    Sweep(modeling::ComposedSweep),
}

async fn create_results_object_from_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            RESULTS_NAME,
            &ERROR_INPUT,
            "fea.results requires a run id or fea.RunResult",
        ));
    }
    if let Value::Object(object) = &args[0] {
        if object.class_name().is(identities::FEA_RESULTS_CLASS) && args.len() == 1 {
            return Ok(args[0].clone());
        }
    }
    let requested = run_id_from_value(RESULTS_NAME, &args[0])?;
    let (query, explicit_field_values) = results_query_from_args(args.view().range(1..)?)?;
    artifact_access::read(RESULTS_NAME, || {
        create_persisted_results_object(args, requested, query, explicit_field_values)
    })
    .await
}

fn create_persisted_results_object(
    args: crate::call::BuiltinArguments,
    requested: String,
    query: AnalysisResultsQuery,
    explicit_field_values: bool,
) -> BuiltinResult<Value> {
    // A definition run id names every attempt; the object reports the attempt it holds.
    let run_id = crate::analysis::storage::resolve_run_attempt(&requested).map_err(|err| {
        builtin_error(
            RESULTS_NAME,
            &ERROR_OPERATION,
            format!("failed to resolve the latest FEA run attempt: {err}"),
        )
    })?;
    // Without a field-value option the object opens bounded; an explicit request is answered.
    let query = if explicit_field_values {
        query
    } else {
        crate::analysis::bounded_default_results_query(&run_id, query)
            .map_err(|err| builtin_error(RESULTS_NAME, &ERROR_OPERATION, err))?
    };
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
            "frequency_sample_count",
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
    crate::builtins::fea::object_api::replace(
        RESULTS_NAME,
        &mut object,
        "run_id",
        Value::String(run_id.clone().into()),
    )?;
    copy_study_context_property(&args[0], &mut object)?;
    Ok(Value::Object(object))
}

async fn create_field_object_from_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !(2..=3).contains(&args.len()) {
        return Err(builtin_error(
            FIELD_NAME,
            &ERROR_INPUT,
            "fea.field requires results/run input, a typed field selector, and an optional typed sample",
        ));
    }
    let field_id = field_selectors::from_value(FIELD_NAME, &args[1])?.field_id();
    let selection = match args.get(2) {
        None => AnalysisFieldSelection::new(&field_id, None),
        Some(value) => field_sample_selection(&field_id, value)?,
    };
    let results = results_data_from_value(FIELD_NAME, &args[0]).await?;
    let sampled = selection.sample.is_some() || selection.transient_sample_index.is_some();
    // A shell (declared shape, no values) is paged from the artifact rather than returned empty.
    let inline_field = (!sampled)
        .then(|| find_field(results.fields.into_iter(), &field_id))
        .flatten()
        .filter(|field| {
            field.shape.iter().product::<usize>() == 0
                || field.as_host_f64().is_none_or(|values| !values.is_empty())
        });
    let mut object = if let Some(field) = inline_field {
        let descriptor = find_descriptor(results.field_descriptors.iter(), &field_id)
            .cloned()
            .unwrap_or_else(|| AnalysisFieldDescriptor::from_field(&field));
        field_to_object(&field, &descriptor)?
    } else {
        let run_id = run_id_context_from_value(&args[0]).ok_or_else(|| {
            builtin_error(
                FIELD_NAME,
                &ERROR_INPUT,
                "artifact-backed FEA field selection requires a persisted run identity",
            )
        })?;
        let page = artifact_access::read(FIELD_NAME, || {
            analysis_field_page_by_run_id_op(
                &run_id,
                selection,
                AnalysisFieldRequestOptions {
                    offset: Some(0),
                    limit: Some(0),
                },
                OperationContext::new(None, None),
            )
            .map_err(|error| operation_error(FIELD_NAME, &ERROR_OPERATION, error))
        })
        .await?
        .data;
        field_object(&page.descriptor, None)?
    };
    copy_study_context_property(&args[0], &mut object)?;
    copy_run_id_context_property(&args[0], &mut object)?;
    Ok(Value::Object(object))
}

/// The optional `fea.field` sample: a `fea.Frequency`, a `fea.Time`, or a one-based retained
/// transient state index.
fn field_sample_selection(field_id: &str, value: &Value) -> BuiltinResult<AnalysisFieldSelection> {
    use runmat_analysis_fea::results::field::RecoveredFieldSample;
    match value {
        Value::Object(object) if object.class_name().is(identities::FEA_FREQUENCY_CLASS) => {
            let hertz =
                values::quantity_scalar(FIELD_NAME, value, values::FREQUENCY_CLASS, "hertz")?;
            if !hertz.is_finite() || hertz <= 0.0 {
                return Err(builtin_error(
                    FIELD_NAME,
                    &ERROR_INPUT,
                    "frequency sample must be finite and positive",
                ));
            }
            Ok(AnalysisFieldSelection::new(
                field_id,
                Some(RecoveredFieldSample::FrequencyHertz(hertz)),
            ))
        }
        Value::Object(object) if object.class_name().is(identities::FEA_TIME_CLASS) => {
            let seconds =
                values::quantity_scalar(FIELD_NAME, value, values::TIME_CLASS, "seconds")?;
            if !seconds.is_finite() {
                return Err(builtin_error(
                    FIELD_NAME,
                    &ERROR_INPUT,
                    "time sample must be finite",
                ));
            }
            Ok(AnalysisFieldSelection::new(
                field_id,
                Some(RecoveredFieldSample::TimeSeconds(seconds)),
            ))
        }
        Value::Object(_) => Err(builtin_error(
            FIELD_NAME,
            &ERROR_INPUT,
            "fea.field sample must be a fea.Frequency, a fea.Time, or a positive integer state index",
        )),
        other => {
            let index = usize_from_value(FIELD_NAME, other).map_err(|_| {
                builtin_error(
                    FIELD_NAME,
                    &ERROR_INPUT,
                    "fea.field sample must be a fea.Frequency, a fea.Time such as fea.Time.seconds(0.01), or a positive integer state index",
                )
            })?;
            let index = index.checked_sub(1).ok_or_else(|| {
                builtin_error(
                    FIELD_NAME,
                    &ERROR_INPUT,
                    "transient state indices are one-based",
                )
            })?;
            Ok(AnalysisFieldSelection::transient_index(field_id, index))
        }
    }
}

async fn create_compare_object_from_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() != 2 {
        return Err(builtin_error(
            COMPARE_NAME,
            &ERROR_INPUT,
            "fea.compare requires baseline and candidate run ids",
        ));
    }
    let baseline_run_id = scalar_string(&args[0], COMPARE_NAME, &ERROR_INPUT)?;
    let candidate_run_id = scalar_string(&args[1], COMPARE_NAME, &ERROR_INPUT)?;
    artifact_access::read(COMPARE_NAME, || {
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
    })
    .await
}

async fn create_trends_object_from_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut window_size = AnalysisTrendsQuery::default().window_size;
    let mut window_size_seen = false;
    for pair in expect_name_value_tail(TRENDS_NAME, args.view())? {
        match runmat_builtins::fea_contract::options::FeaTrendsOption::from_normalized(
            pair.key.as_str(),
        ) {
            Some(runmat_builtins::fea_contract::options::FeaTrendsOption::WindowSize) => {
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
            None => {
                return Err(builtin_error(
                    TRENDS_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.trends option `{}`", pair.key),
                ));
            }
        }
    }
    artifact_access::read(TRENDS_NAME, || {
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
    })
    .await
}

fn object_payload<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
) -> BuiltinResult<T> {
    object_payload_omitting(builtin, value, expected_class, &[])
}

fn object_payload_omitting<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
    omitted: &[&str],
) -> BuiltinResult<T> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class} object"),
        ));
    };
    if object.class_name().display_name() != expected_class {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class}, got {}", object.class_name()),
        ));
    }
    let json = object_projection::to_json_omitting(builtin, object, omitted)?;
    serde_json::from_value(json).map_err(|err| {
        builtin_error_with_source(
            builtin,
            &ERROR_INPUT,
            format!(
                "{} cannot be decoded from its typed properties: {err}",
                object.class_name()
            ),
            err,
        )
    })
}

fn results_query_from_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(AnalysisResultsQuery, bool)> {
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
            other => {
                return Err(builtin_error(
                    RESULTS_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.results option `{other}`"),
                ));
            }
        }
    }
    Ok((query, seen.contains("includefieldvalues")))
}

fn run_id_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::Object(object) if object.class_name().is(identities::FEA_RUN_RESULT_CLASS) => {
            let aggregate: crate::analysis::AnalysisStudyRunData =
                object_payload(builtin, value, FEA_RUN_RESULT_CLASS)?;
            match aggregate.steps.as_slice() {
                [step] => Ok(step.run_id.clone()),
                [] => Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "fea.RunResult contains no procedure-step results",
                )),
                _ => Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "fea.RunResult contains multiple procedure-step results; select one with run.step(\"step_id\")",
                )),
            }
        }
        Value::Object(object)
            if object
                .class_name()
                .is(identities::FEA_STEP_RUN_RESULT_CLASS) =>
        {
            run_id_from_object(object).ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "fea.StepRunResult does not contain its required run_id",
                )
            })
        }
        Value::Object(object)
            if object
                .class_name()
                .is(identities::FEA_SWEEP_RUN_RESULT_CLASS) =>
        {
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
            format!("expected run id string, fea.RunResult, or fea.StepRunResult; got {other:?}"),
        )),
    }
}

fn run_id_from_object(object: &ObjectInstance) -> Option<String> {
    object
        .properties()
        .get("run_id")
        .and_then(|value| match value {
            Value::String(runmat_value::StringElement::Text(run_id)) => Some(run_id.to_string()),
            _ => None,
        })
}

async fn results_data_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<crate::analysis::AnalysisResultsData> {
    match value {
        Value::Object(object) if object.class_name().is(identities::FEA_RESULTS_CLASS) => {
            object_payload(builtin, value, FEA_RESULTS_CLASS)
        }
        _ => {
            let run_id = run_id_from_value(builtin, value)?;
            artifact_access::read(builtin, || {
                analysis_results_by_run_id_op(
                    &run_id,
                    AnalysisResultsQuery::default(),
                    OperationContext::new(None, None),
                )
                .map(|envelope| envelope.data)
                .map_err(|err| operation_error(builtin, &ERROR_OPERATION, err))
            })
            .await
        }
    }
}

async fn run_study_result_to_object(spec: &AnalysisStudySpec) -> BuiltinResult<Value> {
    let context = OperationContext::new(None, None);
    let envelope = crate::analysis::fea_run_admission::async_host::run(
        crate::analysis::fea_run_admission::FeaRunKind::MeshAndSolve,
        crate::analysis::analysis_study_content_digest(spec),
        async {
            if let Some(service) = crate::context::legacy::active()
                .and_then(|runtime| runtime.service_ports().analysis().cloned())
            {
                service.execute_study(spec.clone(), context.clone()).await
            } else {
                analysis_run_study_op(spec, context.clone())
            }
        },
        |denied| {
            denied.into_operation_error(
                crate::analysis::ANALYSIS_RUN_STUDY_OPERATION,
                crate::analysis::ANALYSIS_RUN_STUDY_OP_VERSION,
                &context,
            )
        },
        |result, lease_error| {
            crate::analysis::fea_run_admission::entrypoints::merge_lease_error(
                result,
                Some(lease_error),
            )
        },
    )
    .await
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

async fn run_sweep(
    spec: &AnalysisStudySweepSpec,
) -> Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope> {
    let context = OperationContext::new(None, None);
    let identity = crate::analysis::analysis_study_sweep_identity(spec, &context)?;
    crate::analysis::fea_run_admission::async_host::run(
        crate::analysis::fea_run_admission::FeaRunKind::MeshAndSolve,
        identity.digest(),
        async {
            if let Some(service) = crate::context::legacy::active()
                .and_then(|runtime| runtime.service_ports().analysis().cloned())
            {
                service.execute_sweep(spec.clone(), context.clone()).await
            } else {
                analysis_run_study_sweep_op(spec, context.clone())
            }
        },
        |denied| {
            denied.into_operation_error(
                crate::analysis::ANALYSIS_RUN_STUDY_SWEEP_OPERATION,
                crate::analysis::ANALYSIS_RUN_STUDY_SWEEP_OP_VERSION,
                &context,
            )
        },
        |result, lease_error| {
            crate::analysis::fea_run_admission::entrypoints::merge_lease_error(
                result,
                Some(lease_error),
            )
        },
    )
    .await
}

fn insert_study_context(
    object: &mut ObjectInstance,
    spec: &AnalysisStudySpec,
) -> BuiltinResult<()> {
    crate::builtins::fea::object_api::replace(
        RUN_NAME,
        object,
        FEA_STUDY_CONTEXT_PROPERTY,
        study_to_object(spec.clone())?,
    )?;
    Ok(())
}

fn copy_study_context_property(source: &Value, target: &mut ObjectInstance) -> BuiltinResult<()> {
    if let Some(study) = study_context_value(source) {
        crate::builtins::fea::object_api::replace(
            RUN_NAME,
            target,
            FEA_STUDY_CONTEXT_PROPERTY,
            study,
        )?;
    }
    Ok(())
}

fn copy_run_id_context_property(source: &Value, target: &mut ObjectInstance) -> BuiltinResult<()> {
    if target.property("run_id").is_none() {
        if let Some(run_id) = run_id_context_from_value(source) {
            crate::builtins::fea::object_api::replace(
                RUN_NAME,
                target,
                "run_id",
                Value::String(run_id.into()),
            )?;
        }
    }
    Ok(())
}

fn study_context_value(value: &Value) -> Option<Value> {
    let Value::Object(object) = value else {
        return None;
    };
    if object.class_name().is(identities::FEA_STUDY_CLASS) {
        return Some(value.clone());
    }
    object.properties().get(FEA_STUDY_CONTEXT_PROPERTY).cloned()
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
    field_object(descriptor, Some(field_values_value(field)?))
}

fn field_object(
    descriptor: &AnalysisFieldDescriptor,
    values: Option<Value>,
) -> BuiltinResult<ObjectInstance> {
    ensure_fea_classes_registered();
    let materialized = values.is_some();
    let mut object =
        crate::builtins::fea::object_api::ordinary_builder(FIELD_NAME, FEA_FIELD_CLASS)?;
    object.insert(
        "field_id".to_string(),
        Value::String(descriptor.field_id.clone().into()),
    );
    object.insert(
        "shape".to_string(),
        usize_slice_tensor(&descriptor.shape, 1, descriptor.shape.len())?,
    );
    if let Some(values) = values {
        object.insert("values".to_string(), values);
    }
    object.insert("materialized".to_string(), Value::Bool(materialized));
    object.insert(
        "unit".to_string(),
        Value::String(descriptor.unit.clone().unwrap_or_default().into()),
    );
    object.insert(
        "location".to_string(),
        Value::String(
            format!("{:?}", descriptor.location)
                .to_ascii_lowercase()
                .into(),
        ),
    );
    object.insert(
        "kind".to_string(),
        Value::String(format!("{:?}", descriptor.kind).to_ascii_lowercase().into()),
    );
    object.insert(
        "family".to_string(),
        Value::String(descriptor.family.clone().into()),
    );
    object.insert(
        "quantity".to_string(),
        Value::String(
            serde_json::to_value(descriptor.layout.quantity)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_default()
                .into(),
        ),
    );
    object.insert(
        "statistic".to_string(),
        descriptor
            .statistic
            .map(|statistic| {
                Value::String(
                    serde_json::to_value(statistic)
                        .ok()
                        .and_then(|value| value.as_str().map(str::to_string))
                        .unwrap_or_default()
                        .into(),
                )
            })
            .unwrap_or_else(empty_double_value),
    );
    object.insert(
        "topology_id".to_string(),
        descriptor
            .topology_id
            .as_ref()
            .map(|value| Value::String(value.clone().into()))
            .unwrap_or_else(empty_double_value),
    );
    object.insert(
        "element_kind".to_string(),
        descriptor
            .element_kind
            .as_ref()
            .map(|value| Value::String(value.clone().into()))
            .unwrap_or_else(empty_double_value),
    );
    object.insert(
        "component_count".to_string(),
        descriptor
            .component_count
            .map(|value| Value::Int(IntValue::U64(value as u64)))
            .unwrap_or_else(empty_double_value),
    );
    object.insert(
        "element_count".to_string(),
        Value::Int(IntValue::U64(descriptor.element_count as u64)),
    );
    object.insert(
        "entity_count".to_string(),
        Value::Int(IntValue::U64(descriptor.entity_count as u64)),
    );
    object.insert(
        "value_count".to_string(),
        Value::Int(IntValue::U64(descriptor.value_count as u64)),
    );
    object.insert(
        "storage".to_string(),
        Value::String(
            format!("{:?}", descriptor.storage)
                .to_ascii_lowercase()
                .into(),
        ),
    );
    let sample = match descriptor.sample {
        Some(runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(value)) => {
            values::quantity_to_value(FIELD_NAME, values::FREQUENCY_CLASS, "hertz", value)?
        }
        Some(runmat_analysis_fea::results::field::RecoveredFieldSample::TimeSeconds(value)) => {
            values::quantity_to_value(FIELD_NAME, values::TIME_CLASS, "seconds", value)?
        }
        Some(sample) => serializable_to_value_preserving_integers(
            FIELD_NAME,
            &ERROR_INTERNAL,
            &sample,
            &[],
            &["coordinate"],
        )?,
        None => empty_double_value(),
    };
    object.insert("sample".to_string(), sample);
    object.insert(
        "numeric_representation".to_string(),
        field_access::numeric_representation_value(descriptor.numeric_representation.or(
            materialized.then_some(
                runmat_analysis_fea::results::field::RecoveredFieldNumericRepresentation::Real,
            ),
        )),
    );
    Ok(object.finish())
}

fn field_values_value(field: &AnalysisField) -> BuiltinResult<Value> {
    match &field.values {
        AnalysisFieldValues::HostF64(values) => Tensor::new(
            row_major_to_column_major(values, &field.shape),
            field.shape.clone(),
        )
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

/// Analysis fields store their values row-major: the last shape axis varies fastest, so one
/// node's or element's components are contiguous. Runtime tensors are column-major, so the
/// values are reordered so that `values(i, :)` is entity `i` and `values(:, k)` is component `k`.
/// Invalid or overflowing shapes are passed through for `Tensor::new` to reject.
fn row_major_to_column_major(values: &[f64], shape: &[usize]) -> Vec<f64> {
    let count = shape
        .iter()
        .try_fold(1_usize, |count, &extent| count.checked_mul(extent));
    if shape.len() < 2 || count != Some(values.len()) {
        return values.to_vec();
    }
    let mut reordered = vec![0.0; values.len()];
    let mut index = vec![0_usize; shape.len()];
    for &value in values {
        let mut position = 0;
        let mut stride = 1;
        for (axis, &extent) in shape.iter().enumerate() {
            position += index[axis] * stride;
            stride *= extent;
        }
        reordered[position] = value;
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    reordered
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
        if crate::analysis::field_selection::field_id_matches(&field.field_id, requested) {
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
        if crate::analysis::field_selection::field_id_matches(&descriptor.field_id, requested) {
            suffix_matches.push(descriptor);
        }
    }
    if suffix_matches.len() == 1 {
        suffix_matches.pop()
    } else {
        None
    }
}

struct NameValuePair<'a> {
    name: &'a Value,
    key: String,
    value: &'a Value,
}

fn expect_name_value_tail<'a>(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<Vec<NameValuePair<'a>>> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin} options must be Name, Value pairs"),
        ));
    }
    args.chunks_exact(2)
        .map(|pair| {
            let key = option_key(&pair[0], builtin)?;
            Ok(NameValuePair {
                name: pair.get(0).expect("validated name/value pair"),
                key,
                value: pair.get(1).expect("validated name/value pair"),
            })
        })
        .collect()
}

fn json_fields_from_name_values(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'_>,
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
        "massdensitykgperm3" => "mass_density_kg_per_m3".to_string(),
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
        Value::String(s) => Ok(s
            .as_str()
            .map(|text| serde_json::Value::String(text.to_owned()))
            .unwrap_or(serde_json::Value::Null)),
        Value::CharArray(chars) if chars.rows == 1 => Ok(serde_json::Value::String(
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = builtin_error(builtin, &ERROR_INPUT, source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        )),
        Value::StringArray(array) => Ok(serde_json::Value::Array(
            array
                .data()
                .iter()
                .map(|text| {
                    text.as_str()
                        .map(|text| serde_json::Value::String(text.to_owned()))
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect(),
        )),
        Value::LogicalArray(array) => nested_array_to_json(builtin, &array.shape, |index| {
            Ok(serde_json::Value::Bool(array.data[index] != 0))
        }),
        Value::Tensor(tensor) if tensor.shape() == [1, 1] => numeric_scalar_to_json(
            builtin,
            tensor
                .numeric_value_at(0)
                .expect("validated scalar tensor storage"),
        ),
        // A `1xN` row is the typed-object form of a JSON vector.
        Value::Tensor(tensor) if tensor.shape().len() == 2 && tensor.shape()[0] == 1 => (0..tensor
            .shape()[1])
            .map(|index| {
                numeric_scalar_to_json(
                    builtin,
                    tensor
                        .numeric_value_at(index)
                        .expect("validated row tensor storage"),
                )
            })
            .collect::<BuiltinResult<Vec<_>>>()
            .map(serde_json::Value::Array),
        Value::Tensor(tensor) => nested_array_to_json(builtin, tensor.shape(), |index| {
            numeric_scalar_to_json(
                builtin,
                tensor
                    .numeric_value_at(index)
                    .expect("validated tensor storage"),
            )
        }),
        Value::Cell(cell) => Ok(serde_json::Value::Array(
            cell.data()
                .iter()
                .map(|item| value_to_json(builtin, item))
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Value::Struct(fields) => {
            let mut object = serde_json::Map::new();
            for (key, value) in fields.fields.iter() {
                object.insert(key.to_owned(), value_to_json(builtin, value)?);
            }
            Ok(serde_json::Value::Object(object))
        }
        Value::Object(object) => object_projection::to_json(builtin, object),
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
    let keys = fields
        .keys()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>()
        .join(", ");
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

/// A non-negative integer argument to an FEA builtin. Every such value lands on an
/// FEA wire contract, so it is held to the shared wire bound (2^53 - 1) here, at the
/// one place builtins read counts, rejecting exactly what contract decoding rejects.
fn usize_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<usize> {
    let count = platform_usize_from_value(builtin, value)?;
    runmat_types::wire_integer::bounded(count).ok_or_else(|| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "integer {count} exceeds the FEA wire-contract bound of {} (2^53 - 1)",
                runmat_types::wire_integer::MAX_WIRE_INTEGER
            ),
        )
    })
}

fn platform_usize_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<usize> {
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

fn study_to_object(spec: AnalysisStudySpec) -> BuiltinResult<Value> {
    serializable_to_object(STUDY_NAME, &ERROR_INTERNAL, FEA_STUDY_CLASS, &spec)
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
                if object.property(name).is_none() {
                    continue;
                }
                let target = object.try_property_mut_unique(name).ok_or_else(|| {
                    builtin_error(
                        builtin,
                        error,
                        "FEA integer promotion requires a unique object generation",
                    )
                })?;
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
                if !value.fields.contains_key(name) {
                    continue;
                }
                let target = value.fields.try_get_mut_unique(name).ok_or_else(|| {
                    builtin_error(
                        builtin,
                        error,
                        "FEA integer promotion requires a unique struct generation",
                    )
                })?;
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
            let values = cell.try_data_mut_unique().ok_or_else(|| {
                builtin_error(
                    builtin,
                    error,
                    "FEA integer promotion requires a unique cell backing",
                )
            })?;
            for (target, child) in values.iter_mut().zip(items) {
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
    let mut converted = typed_value_from_json(json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.message().to_string(), err))?;
    promote_json_scalar_kinds(builtin, error, &mut converted, json)?;
    orient_json_vectors_as_rows(builtin, error, &mut converted)?;
    Ok(converted)
}

/// Typed FEA payloads keep every JSON array of objects as a cell, element for
/// element. `jsondecode` instead folds such arrays into struct arrays (and a
/// one-element array into a scalar struct), which would lose the array when
/// the object is decoded again by `value_to_json`.
fn typed_value_from_json(json: &serde_json::Value) -> BuiltinResult<Value> {
    fn holds_object(json: &serde_json::Value) -> bool {
        match json {
            serde_json::Value::Object(_) => true,
            // Nullable arrays must retain their slots rather than allowing
            // jsondecode to coerce null into a numerical array value.
            serde_json::Value::Null => true,
            serde_json::Value::Array(items) => items.iter().any(holds_object),
            _ => false,
        }
    }
    match json {
        serde_json::Value::Object(map) => {
            let fields = map
                .iter()
                .map(|(name, child)| Ok((name.clone(), typed_value_from_json(child)?)))
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(Value::Struct(
                runmat_value::StructValue::from_fields_unbounded(fields),
            ))
        }
        serde_json::Value::Array(items) if items.iter().any(holds_object) => {
            let elements = items
                .iter()
                .map(typed_value_from_json)
                .collect::<BuiltinResult<Vec<_>>>()?;
            let (rows, columns) = if items.iter().all(serde_json::Value::is_object) {
                (elements.len(), 1)
            } else {
                (1, elements.len())
            };
            runmat_value::CellArray::new(elements, rows, columns)
                .map(Value::Cell)
                .map_err(|message| build_runtime_error(message).build())
        }
        // A single nested row (`[[a, b, c]]`) would decode to a `1xN` tensor,
        // which re-encodes as the flat vector `[a, b, c]`; keep its nesting.
        serde_json::Value::Array(items)
            if items.len() == 1 && matches!(items[0], serde_json::Value::Array(_)) =>
        {
            runmat_value::CellArray::new(vec![typed_value_from_json(&items[0])?], 1, 1)
                .map(Value::Cell)
                .map_err(|message| build_runtime_error(message).build())
        }
        other => value_from_json(other),
    }
}

/// JSON arrays decode to rank-1 tensors, which report `1xN` from `size` but
/// do not combine with ordinary row vectors in `==` or `isequal`. Typed FEA
/// objects expose authored coordinate vectors as `1xN` rows instead. A
/// single-element array keeps its rank-1 shape so it still re-encodes as a
/// one-element JSON array rather than a scalar (see `value_to_json`).
fn orient_json_vectors_as_rows(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &mut Value,
) -> BuiltinResult<()> {
    match value {
        Value::Tensor(tensor) if tensor.shape().len() == 1 && tensor.shape()[0] >= 2 => {
            let columns = tensor.shape()[0];
            tensor
                .reshape_in_place(vec![1, columns])
                .map_err(|message| builtin_error(builtin, error, message))?;
        }
        Value::StringArray(array)
            if array.shape().len() == 1
                && array.shape()[0] >= 2
                && !array.data()[0].is_missing() =>
        {
            let columns = array.shape()[0];
            *array = runmat_value::StringArray::from_elements_unbounded(
                array.data().to_vec(),
                vec![1, columns],
            )
            .map_err(|message| builtin_error(builtin, error, message))?;
        }
        Value::Struct(structure) => {
            let children = structure.fields.try_values_mut_unique().ok_or_else(|| {
                builtin_error(
                    builtin,
                    error,
                    "FEA vector orientation requires a unique struct generation",
                )
            })?;
            for child in children {
                orient_json_vectors_as_rows(builtin, error, child)?;
            }
        }
        Value::Object(object) => {
            let names = object
                .properties()
                .keys()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            for name in names {
                let child = object.try_property_mut_unique(&name).ok_or_else(|| {
                    builtin_error(
                        builtin,
                        error,
                        "FEA vector orientation requires a unique object generation",
                    )
                })?;
                orient_json_vectors_as_rows(builtin, error, child)?;
            }
        }
        Value::Cell(cell) => {
            let children = cell.try_data_mut_unique().ok_or_else(|| {
                builtin_error(
                    builtin,
                    error,
                    "FEA vector orientation requires a unique cell backing",
                )
            })?;
            for child in children.iter_mut() {
                orient_json_vectors_as_rows(builtin, error, child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn promote_json_scalar_kinds(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    value: &mut Value,
    json: &serde_json::Value,
) -> BuiltinResult<()> {
    match (value, json) {
        (target, serde_json::Value::String(value)) => {
            *target = Value::String(value.clone().into());
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
            if let Some(storage) = exact_json_integer_array(json, tensor.shape()) {
                *tensor = Tensor::new_integer(storage, tensor.shared_shape())
                    .map_err(|message| builtin_error(builtin, error, message))?;
            }
        }
        (Value::Struct(structure), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                if structure.fields.contains_key(name) {
                    let target = structure.fields.try_get_mut_unique(name).ok_or_else(|| {
                        builtin_error(
                            builtin,
                            error,
                            "FEA integer promotion requires a unique struct generation",
                        )
                    })?;
                    promote_json_scalar_kinds(builtin, error, target, child)?;
                }
            }
        }
        (Value::Object(object), serde_json::Value::Object(map)) => {
            for (name, child) in map {
                if object.property(name).is_some() {
                    let target = object.try_property_mut_unique(name).ok_or_else(|| {
                        builtin_error(
                            builtin,
                            error,
                            "FEA integer promotion requires a unique object generation",
                        )
                    })?;
                    promote_json_scalar_kinds(builtin, error, target, child)?;
                }
            }
        }
        (Value::Cell(cell), serde_json::Value::Array(items)) => {
            let values = cell.try_data_mut_unique().ok_or_else(|| {
                builtin_error(
                    builtin,
                    error,
                    "FEA integer promotion requires a unique cell backing",
                )
            })?;
            for (target, child) in values.iter_mut().zip(items) {
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
    if json.as_object().is_some_and(|fields| {
        fields.contains_key(crate::builtins::common::json::typed_object_projection::PROPERTY)
    }) {
        return Err(builtin_error(
            builtin,
            error,
            "typed object projection property is reserved",
        ));
    }
    let mut public_json = if json.is_object() {
        json
    } else {
        serde_json::json!({"value": json})
    };
    let projection = crate::builtins::common::json::typed_object_projection::Projection::capture(
        &mut public_json,
    );
    let converted = value_from_json_preserving_integer_kinds(builtin, error, &public_json)?;
    let mut object = crate::builtins::fea::object_api::ordinary_builder(builtin, class_name)?;
    if let Value::Struct(fields) = converted {
        for (name, value) in fields.into_fields_unbounded() {
            object.insert(name, value);
        }
    } else {
        object.insert("value".to_string(), converted);
    }
    if let Some(projection) = projection {
        let metadata = projection.to_json().map_err(|source| {
            builtin_error_with_source(builtin, error, source.to_string(), source)
        })?;
        object.insert(
            crate::builtins::common::json::typed_object_projection::PROPERTY.to_owned(),
            value_from_json_preserving_integer_kinds(builtin, error, &metadata)?,
        );
    }
    Ok(object.finish())
}

fn geometry_asset_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<GeometryAsset> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin} geometry must be {GEOMETRY_ASSET_CLASS}"),
        ));
    };
    if !object.is_class(GEOMETRY_ASSET_CLASS) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "{builtin} geometry must be {GEOMETRY_ASSET_CLASS}, got {}",
                object.class_name()
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
    let Some(Value::String(runmat_value::StringElement::Text(json))) =
        object.properties().get(property)
    else {
        return Err(builtin_error(
            builtin,
            error,
            format!(
                "{} is missing required runtime payload property `{property}`",
                object.class_name()
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
        Value::String(runmat_value::StringElement::Text(value)) => Ok(value.to_string()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .as_str()
                .expect("nonmissing scalar text")
                .to_owned())
        }
        Value::CharArray(chars) if chars.rows == 1 => Ok(runmat_value::decode_utf16_code_units(
            chars.data(),
        )
        .map_err(|source| {
            let mut error = builtin_error(builtin, error, source.to_string());
            error.source = Some(Box::new(source));
            error
        })?),
        _ => Err(builtin_error(
            builtin,
            error,
            format!("{builtin} expected a text scalar"),
        )),
    }
}

fn parse_scalar_enum_for_builtin<T: DeserializeOwned>(
    builtin: &'static str,
    text: &str,
    label: &str,
) -> BuiltinResult<T> {
    serde_json::from_value::<T>(serde_json::Value::String(text.to_ascii_lowercase())).map_err(
        |err| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("invalid {label} value `{text}`: {err}"),
            )
        },
    )
}

fn workflow_methods() -> HashMap<MethodName, RuntimeMethod> {
    instance_methods([
        ("validate", VALIDATE_NAME),
        ("plan", PLAN_NAME),
        ("run", RUN_NAME),
    ])
}

fn run_result_methods() -> HashMap<MethodName, RuntimeMethod> {
    instance_methods([
        ("step", "fea.stepResult"),
        ("results", RESULTS_NAME),
        ("field", FIELD_NAME),
        ("mesh", MESH_METHOD_BUILTIN_NAME),
    ])
}

fn step_run_result_methods() -> HashMap<MethodName, RuntimeMethod> {
    instance_methods([
        ("results", RESULTS_NAME),
        ("field", FIELD_NAME),
        ("mesh", MESH_METHOD_BUILTIN_NAME),
    ])
}

fn results_methods() -> HashMap<MethodName, RuntimeMethod> {
    instance_methods([("field", FIELD_NAME), ("mesh", MESH_METHOD_BUILTIN_NAME)])
}

fn field_methods() -> HashMap<MethodName, RuntimeMethod> {
    instance_methods([
        ("page", FIELD_PAGE_NAME),
        ("plot", PLOT_METHOD_BUILTIN_NAME),
        ("mesh", MESH_METHOD_BUILTIN_NAME),
    ])
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
    E: crate::runtime_error::RuntimeErrorSourceBound,
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

#[cfg(test)]
mod tests;
