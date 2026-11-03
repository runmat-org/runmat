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

fn string_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|item| scalar_string(item, builtin, &ERROR_INPUT))
            .collect(),
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| scalar_string(&Value::String(text.clone()), builtin, &ERROR_INPUT))
            .collect(),
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
                .shape()
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
mod tests {
    use super::*;
    use crate::builtins::fea::damping::mechanical_damping_rayleigh_builtin;
    use crate::builtins::fea::values::{
        dynamic_viscosity_builtin, frequency_builtin, mass_density_builtin,
        rayleigh_mass_coefficient_builtin, rayleigh_stiffness_coefficient_builtin,
        temperature_kelvin_builtin, turbulence_intensity_builtin,
    };
    use futures::executor::block_on;
    use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};
    use runmat_analysis_fea::ComputeBackend;
    use runmat_meshing_core::CanonicalMeshingContract;
    use runmat_types::MemberAccess;
    use runmat_value::CellArray;

    const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
    const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

    fn test_run_identity(
        procedure: runmat_analysis_core::AnalysisProcedure,
    ) -> runmat_analysis_fea::AnalysisRunIdentity {
        let assembly = runmat_analysis_core::StructuralFormulationRealization::new(
            runmat_analysis_core::StructuralOperatorImplementation::Tet4LinearElasticity,
            Some(runmat_analysis_core::StructuralRecoveryImplementation::Tet4ConstantStrain),
        );
        let realization = runmat_analysis_fea::admission::structural_exact_realization(
            runmat_analysis_fea::admission::StructuralRealizationInput {
                assemblies: vec![assembly],
                procedure,
                kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
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
                realization,
            },
        )
        .expect("test structural formulation must be admitted")
        .into_identity()
    }

    fn synthetic_connector_result(
    ) -> Box<runmat_analysis_fea::connectors::CanonicalStructuralConnectorResult> {
        let topology =
            runmat_analysis_fea::constraints::CanonicalStructuralTopologyIdentity::Solid {
                mesh_digest: runmat_meshing_core::StableDigest::from_bytes([7; 32]),
            };
        let plan =
            runmat_analysis_fea::connectors::CanonicalStructuralConnectorPlan::empty(topology, 0)
                .expect("empty synthetic connector plan");
        Box::new(
            runmat_analysis_fea::connectors::CanonicalStructuralConnectorResult::recover_static(
                plan,
                &runmat_analysis_fea::assembly::dofs::StructuralDofLayout::default(),
                &[],
            )
            .expect("zero synthetic connector state"),
        )
    }

    fn synthetic_static_balance() -> Box<runmat_analysis_fea::solve::StructuralStaticBalanceEvidence>
    {
        Box::new(
            runmat_analysis_fea::solve::StructuralStaticBalanceEvidence {
                internal_work_joules: 0.0,
                applied_load_work_joules: 0.0,
                constraint_work_joules: 0.0,
                work_balance_residual_joules: 0.0,
                work_balance_relative_error: 0.0,
                equilibrium_residual_norm_newtons: 0.0,
                equilibrium_scale_newtons: 1.0,
                equilibrium_relative_error: 0.0,
                relative_tolerance: 1.0e-8,
                accepted: true,
            },
        )
    }

    fn synthetic_constraint_result(
    ) -> Box<runmat_analysis_fea::constraints::CanonicalStructuralConstraintResult> {
        Box::new(
            runmat_analysis_fea::constraints::CanonicalStructuralConstraintResult {
                schema_version: 1,
                constraint_plan_digest: runmat_execution::Digest::sha256(
                    b"synthetic-constraint-plan",
                ),
                topology:
                    runmat_analysis_fea::constraints::CanonicalStructuralTopologyIdentity::Solid {
                        mesh_digest: runmat_meshing_core::StableDigest::from_bytes([7; 32]),
                    },
                reference_points: Vec::new(),
                groups: Vec::new(),
                resultant_force_newtons: [0.0; 3],
                resultant_moment_about_global_origin_newton_meters: [0.0; 3],
                constraint_work_joules: 0.0,
                maximum_displacement_error_meters: 0.0,
                maximum_rotation_error_radians: 0.0,
            },
        )
    }

    fn cell(values: Vec<Value>) -> Value {
        let shape = if values.is_empty() {
            (0, 0)
        } else {
            (1, values.len())
        };
        Value::Cell(CellArray::new(values, shape.0, shape.1).expect("cell should build"))
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

        let bound = runmat_types::wire_integer::MAX_WIRE_INTEGER;
        let maximum = usize_from_value("fea.test", &Value::Int(IntValue::U64(bound)));
        if usize::BITS == 64 {
            assert_eq!(maximum.unwrap(), bound as usize);
        } else {
            assert!(maximum.is_err());
        }
        assert!(usize_from_value("fea.test", &Value::Int(IntValue::U64(bound + 1))).is_err());
        assert!(usize_from_value("fea.test", &Value::Int(IntValue::U64(u64::MAX))).is_err());
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
            object.properties().get("signed"),
            Some(Value::Int(IntValue::I64(-9)))
        ));
        assert!(matches!(
            object.properties().get("unsigned"),
            Some(Value::Int(IntValue::U64(u64::MAX)))
        ));
        let Some(Value::Tensor(matrix)) = object.properties().get("matrix") else {
            panic!("exact integer matrix");
        };
        assert_eq!(matrix.shape(), vec![2, 2]);
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
            object.properties().get("floating"),
            Some(Value::Num(1.0))
        ));
        assert!(matches!(
            object.properties().get("floating_vector"),
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
            let Some(Value::Tensor(values)) = empty.properties().get(name) else {
                panic!("empty exact vector {name}");
            };
            assert!(values.integer_storage().is_some());
            assert!(values.is_empty());
        }
    }

    #[test]
    fn fea_execution_controls_enforce_exact_structural_boundaries() {
        let empty_step = block_on(structural_modal_builtin(Value::String(
            String::new().into(),
        )))
        .expect_err("step identity must not be empty");
        assert_eq!(empty_step.identifier(), Some("RunMat:fea:InvalidInput"));

        let zero_window = block_on(fea_trends_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("WindowSize".into()),
                Value::Int(IntValue::U8(0)),
            ]),
        ))
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
        let wide = runmat_types::wire_integer::MAX_WIRE_INTEGER;
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
        let beyond = Tensor::new_integer(IntegerStorage::U64(vec![wide + 2]), vec![1, 1])
            .expect("selector beyond the wire bound");
        assert_eq!(
            one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(beyond))
                .unwrap_err()
                .identifier(),
            Some("RunMat:fea:InvalidInput")
        );

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
            kind: crate::analysis::AnalysisStudySweepFailureKind::Failed,
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
                    vec![
                        runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(
                            runmat_execution::Digest::sha256(b"study"),
                            None,
                        )
                        .expect("study identity"),
                    ],
                    runmat_analysis_fea::execution::SweepCompositionPolicy::CollectAll,
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
            assert!(crate::class_registry::lookup_property(
                &runmat_types::ClassIdentity::from(class_name),
                &runmat_types::MemberName::from("sweep_id")
            )
            .is_none());
        }
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_RUN_RESULT_CLASS),
            &runmat_types::MethodName::from("results")
        )
        .is_some());
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_RUN_RESULT_CLASS),
            &runmat_types::MethodName::from("step"),
        )
        .is_some());
        assert!(crate::class_registry::lookup_property(
            &runmat_types::ClassIdentity::from(FEA_RUN_RESULT_CLASS),
            &runmat_types::MemberName::from("steps")
        )
        .is_some());
        assert!(crate::class_registry::lookup_property(
            &runmat_types::ClassIdentity::from(FEA_RUN_RESULT_CLASS),
            &runmat_types::MemberName::from("run_id")
        )
        .is_none());
        assert!(crate::class_registry::lookup_property(
            &runmat_types::ClassIdentity::from(FEA_STEP_RUN_RESULT_CLASS),
            &runmat_types::MemberName::from("run_id")
        )
        .is_some());
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_STEP_RUN_RESULT_CLASS),
            &runmat_types::MethodName::from("results")
        )
        .is_some());
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_SWEEP_RUN_RESULT_CLASS),
            &runmat_types::MethodName::from("results")
        )
        .is_none());
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
            "[42,18446744073709551615]"
        );

        let column = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![42, u64::MAX]),
            vec![2, 1],
        )
        .expect("column tensor");
        assert_eq!(
            value_to_json("fea.test", &Value::Tensor(column))
                .expect("column tensor json")
                .to_string(),
            "[[42],[18446744073709551615]]"
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
            assert_eq!(
                domain,
                runmat_analysis_core::ElectromagneticDomain::Harmonic {
                    reference_frequency: runmat_analysis_core::Frequency::new(expected),
                }
            );

            let material = block_on(fea_material_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("material".into()),
                    Value::String("YoungsModulusPa".into()),
                    Value::Int(integer),
                    Value::String("PoissonRatio".into()),
                    Value::Int(IntValue::U8(0)),
                ]),
            ))
            .expect("material integer field");
            let material: MaterialModel = object_payload(&material);
            assert_eq!(
                material
                    .require_mechanical()
                    .expect("authored mechanical capability")
                    .youngs_modulus_pa
                    .canonical_value(),
                expected
            );
        }
    }

    #[test]
    fn fea_material_preserves_typed_rayleigh_damping() {
        let mass =
            block_on(rayleigh_mass_coefficient_builtin(Value::Num(0.2))).expect("mass coefficient");
        let stiffness = block_on(rayleigh_stiffness_coefficient_builtin(Value::Num(0.003)))
            .expect("stiffness coefficient");
        let damping = block_on(mechanical_damping_rayleigh_builtin(mass, stiffness))
            .expect("Rayleigh damping");
        let material = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("steel".into()),
                Value::String("YoungsModulusPa".into()),
                Value::Num(210.0e9),
                Value::String("PoissonRatio".into()),
                Value::Num(0.3),
                Value::String("Damping".into()),
                damping,
            ]),
        ))
        .expect("material");
        let material: MaterialModel = object_payload(&material);
        assert_eq!(
            material.damping,
            Some(MechanicalDampingModel::Rayleigh {
                mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(0.2),
                stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(
                    0.003
                ),
            })
        );

        let error = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("steel".into()),
                Value::String("YoungsModulusPa".into()),
                Value::Num(210.0e9),
                Value::String("PoissonRatio".into()),
                Value::Num(0.3),
                Value::String("Damping".into()),
                Value::String("rayleigh".into()),
            ]),
        ))
        .expect_err("untyped damping must be rejected");
        assert!(error.message().contains("typed fea.MechanicalDamping"));
    }

    #[test]
    fn fea_material_requires_explicit_capabilities_and_canonical_density_name() {
        let material = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("air".into()),
                Value::String("MassDensityKgPerM3".into()),
                Value::Num(1.225),
            ]),
        ))
        .expect("a density-only material is valid for compatible formulations");
        let material: MaterialModel = object_payload(&material);
        assert_eq!(
            material
                .mass_density_kg_per_m3
                .map(|quantity| quantity.canonical_value()),
            Some(1.225)
        );
        assert!(material.elasticity.is_none());

        let empty = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "empty".into(),
            )]),
        ))
        .expect_err("an empty material must fail at construction");
        assert!(empty.message().contains("constitutive capability"));

        let retired = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("air".into()),
                Value::String("DensityKgPerM3".into()),
                Value::Num(1.225),
            ]),
        ))
        .expect_err("the ambiguous retired density spelling must not be accepted");
        assert!(retired.message().contains("unsupported option field"));
        assert!(retired.message().contains("density_kg_per_m3"));
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

        let thermal =
            block_on(domains::thermal_domain_builtin(temperature.clone())).expect("thermal domain");
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
            (thermal, FEA_THERMAL_DOMAIN_CLASS),
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
            &runmat_types::ClassIdentity::from(FEA_THERMAL_DOMAIN_CLASS),
            &runmat_types::MethodName::from("atReferenceTemperature")
        )
        .is_some());
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_THERMOMECHANICAL_DOMAIN_CLASS),
            &runmat_types::MethodName::from("atReferenceTemperature")
        )
        .is_some());
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_THERMOMECHANICAL_DOMAIN_CLASS),
            &runmat_types::MethodName::from("atFrequency")
        )
        .is_none());
        assert!(crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_FLUID_DOMAIN_CLASS),
            &runmat_types::MethodName::from("newtonian")
        )
        .is_some());
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
    fn vector_field_values_have_one_row_per_node_and_one_column_per_component() {
        // Three nodes with displacement u_i = (i, 10 i, 100 i), stored node-major as the
        // analysis producers publish them.
        let field = AnalysisField {
            field_id: "structural.displacement".into(),
            layout: runmat_analysis_core::AnalysisFieldLayout::vector3(
                runmat_analysis_core::ObservationQuantity::Displacement,
                runmat_analysis_core::FieldLocation::Node,
            ),
            shape: vec![3, 3],
            values: AnalysisFieldValues::HostF64(vec![
                1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0,
            ]),
        };
        let descriptor = AnalysisFieldDescriptor::from_field(&field);
        let object = field_to_object(&field, &descriptor).expect("field object");
        let Some(Value::Tensor(values)) = object.properties().get("values") else {
            panic!("expected values tensor");
        };
        assert_eq!(values.shape(), vec![3, 3]);
        let column_major = values.materialize_f64();
        let column = |k: usize| column_major[3 * k..3 * (k + 1)].to_vec();
        assert_eq!(column(0), vec![1.0, 2.0, 3.0], "u(:,1) is the x component");
        assert_eq!(
            column(1),
            vec![10.0, 20.0, 30.0],
            "u(:,2) is the y component"
        );
        assert_eq!(
            column(2),
            vec![100.0, 200.0, 300.0],
            "u(:,3) is the z component"
        );
        let row = |i: usize| [0, 1, 2].map(|k| column_major[i + 3 * k]);
        assert_eq!(row(1), [2.0, 20.0, 200.0], "u(2,:) is node 2's vector");

        // A rank-3 tensor field keeps the same rule on every axis.
        let tensor = AnalysisField {
            field_id: "structural.stress_tensor".into(),
            layout: runmat_analysis_core::AnalysisFieldLayout::components(
                runmat_analysis_core::ObservationQuantity::Stress,
                runmat_analysis_core::FieldLocation::Element,
                4,
            ),
            shape: vec![2, 2, 2],
            values: AnalysisFieldValues::HostF64((0..8).map(f64::from).collect()),
        };
        let Value::Tensor(values) = field_values_value(&tensor).expect("tensor values") else {
            panic!("expected values tensor");
        };
        // Row-major entry (i, j, k) = 4 i + 2 j + k sits at column-major i + 2 j + 4 k.
        assert_eq!(
            values.materialize_f64(),
            vec![0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0]
        );
    }

    #[test]
    fn fea_field_metadata_remains_exact_in_public_objects() {
        let field = AnalysisField {
            field_id: "stress".into(),
            layout: runmat_analysis_core::AnalysisFieldLayout::scalar(
                runmat_analysis_core::ObservationQuantity::Stress,
                runmat_analysis_core::FieldLocation::Element,
            ),
            shape: vec![u32::MAX as usize, 0],
            values: AnalysisFieldValues::HostF64(Vec::new()),
        };
        let descriptor = AnalysisFieldDescriptor::from_field(&field);
        let object = field_to_object(&field, &descriptor).expect("field object");
        let Value::Tensor(shape) = object.properties().get("shape").expect("shape") else {
            panic!("expected integer shape tensor");
        };
        assert_eq!(shape.numeric_dtype(), runmat_value::NumericDType::U64);
        assert_eq!(
            shape.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::from(u32::MAX), 0]))
        );
        assert_eq!(
            object.properties().get("element_count"),
            Some(&Value::Int(IntValue::U64(0)))
        );
        let reference = field_object(&descriptor, None).expect("artifact field reference");
        assert_eq!(
            reference.properties().get("materialized"),
            Some(&Value::Bool(false))
        );
        assert!(!reference.properties().contains_key("values"));

        let device_field = AnalysisField {
            field_id: "device".into(),
            layout: runmat_analysis_core::AnalysisFieldLayout::scalar(
                runmat_analysis_core::ObservationQuantity::Residual,
                runmat_analysis_core::FieldLocation::Global,
            ),
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
    fn structural_field_selectors_are_typed_and_portable() {
        ensure_value_classes_registered();
        let selector = block_on(field_selectors::stress()).expect("typed structural field");
        let Value::Object(object) = &selector else {
            panic!("expected structural field selector object");
        };
        assert_eq!(
            object.class_name().display_name(),
            field_selectors::STRUCTURAL_FIELD_CLASS
        );
        assert_eq!(
            field_selectors::from_value("test", &selector).unwrap(),
            field_selectors::FieldSelector::Structural(
                runmat_analysis_fea::contracts::StructuralFieldSelector::Stress
            )
        );
        let encoded = crate::execution::value_codec::encode_inline_value(&selector)
            .expect("structural field selector should be portable");
        let decoded = crate::execution::value_codec::decode_inline_value(&encoded)
            .expect("portable structural field selector should decode");
        assert_eq!(selector, decoded);
        assert!(field_selectors::from_value("test", &Value::String("stress".into())).is_err());

        let integration_point = block_on(field_selectors::integration_point_stress())
            .expect("typed integration-point field");
        assert_eq!(
            field_selectors::from_value("test", &integration_point).unwrap(),
            field_selectors::FieldSelector::Structural(
                runmat_analysis_fea::contracts::StructuralFieldSelector::IntegrationPointStress
            )
        );
        assert_eq!(
            field_selectors::from_value("test", &integration_point)
                .unwrap()
                .field_id(),
            runmat_analysis_fea::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS
        );
    }

    #[test]
    fn statistical_field_selectors_are_typed_composable_and_portable() {
        ensure_value_classes_registered();
        let stress = block_on(field_selectors::stress()).expect("typed structural field");
        let envelope = block_on(statistical_field_selectors::response_spectrum_envelope(
            stress.clone(),
        ))
        .expect("typed spectrum envelope");
        assert_eq!(
            field_selectors::from_value("test", &envelope)
                .unwrap()
                .field_id(),
            "structural.response_spectrum.envelope.stress"
        );
        let rms = block_on(statistical_field_selectors::random_response_rms(
            stress.clone(),
        ))
        .expect("typed random RMS");
        assert_eq!(
            field_selectors::from_value("test", &rms)
                .unwrap()
                .field_id(),
            "structural.random_response.rms.stress"
        );
        let variance = block_on(statistical_field_selectors::random_response_variance(
            stress,
        ))
        .expect("typed random variance");
        assert_eq!(
            field_selectors::from_value("test", &variance)
                .unwrap()
                .field_id(),
            "structural.random_response.variance.stress"
        );
        let nodal =
            block_on(statistical_field_selectors::response_spectrum_absolute_acceleration())
                .expect("typed nodal envelope");
        assert_eq!(
            field_selectors::from_value("test", &nodal)
                .unwrap()
                .field_id(),
            "structural.response_spectrum.nodal.absolute_acceleration"
        );
        for selector in [envelope, rms, variance, nodal] {
            let encoded = crate::execution::value_codec::encode_inline_value(&selector)
                .expect("statistical selector should be portable");
            assert_eq!(
                crate::execution::value_codec::decode_inline_value(&encoded)
                    .expect("portable selector should decode"),
                selector
            );
        }
        let von_mises = block_on(field_selectors::von_mises()).expect("typed derived field");
        assert!(block_on(statistical_field_selectors::random_response_rms(von_mises)).is_err());
    }

    #[test]
    fn fea_numeric_constructors_reject_resident_fields_without_provider_access() {
        let resident = || {
            Value::GpuTensor(
                runmat_accelerate_api::test_support::TestAllocationOwner::new()
                    .allocate_zeroed(
                        vec![1, 1],
                        runmat_accelerate_api::NumericElementType::F64,
                        runmat_accelerate_api::GpuTensorStorage::Real,
                    )
                    .expect("owned resident fixture")
                    .handle,
            )
        };
        let cases = [
            block_on(frequency_builtin(resident())),
            block_on(fea_material_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("material".into()),
                    Value::String("YoungsModulusPa".into()),
                    resident(),
                    Value::String("PoissonRatio".into()),
                    Value::Num(0.3),
                ]),
            )),
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
            object.properties().get("study_count"),
            Some(&Value::Int(IntValue::U64(3)))
        );
        assert_eq!(
            object.properties().get("quality_reason_count_delta"),
            Some(&Value::Int(IntValue::I64(-4)))
        );
        assert!(!object.properties().contains_key("optional_delta"));
        let Some(Value::Cell(entries)) = object.properties().get("failure_entries") else {
            panic!("expected failure entry cell");
        };
        let Some(Value::Struct(entry)) = entries.data().first() else {
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

        let field_error = futures::executor::block_on(create_field_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(2.0),
            ]),
        ))
        .expect_err("surplus fea.field argument");
        assert_eq!(field_error.context.builtin.as_deref(), Some(FIELD_NAME));

        let compare_error = futures::executor::block_on(create_compare_object_from_args(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("base".into()),
                Value::String("candidate".into()),
                Value::Num(2.0),
            ]),
        ))
        .expect_err("surplus fea.compare argument");
        assert_eq!(compare_error.context.builtin.as_deref(), Some(COMPARE_NAME));

        let model_error = block_on(fea_model_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("model".into()),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("invalid model geometry");
        assert_eq!(model_error.context.builtin.as_deref(), Some(MODEL_NAME));

        let duplicate_error = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("material".into()),
                Value::String("YoungsModulusPa".into()),
                Value::Num(1.0),
                Value::String("youngs_modulus_pa".into()),
                Value::Num(2.0),
                Value::String("PoissonRatio".into()),
                Value::Num(0.3),
            ]),
        ))
        .expect_err("duplicate normalized field");
        assert_eq!(
            duplicate_error.context.builtin.as_deref(),
            Some(MATERIAL_NAME)
        );
        assert!(duplicate_error.message().contains("duplicate"));
    }

    #[test]
    fn fea_study_requires_geometry_asset() {
        let err = block_on(fea_study_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("demo".to_string().into()),
                Value::Num(1.0),
            ]),
        ))
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

        let options =
            StudyConstructorOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
                Value::String("Mesh".to_string().into()),
                mesh,
            ]))
            .expect("study mesh option");
        let supplied = options.mesh.expect("typed mesh output");

        assert_eq!(supplied.artifact, artifact);
        assert_eq!(supplied.evidence, evidence);
    }

    #[test]
    fn fea_study_requires_explicit_model() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
            Value::String(geometry_path.to_string_lossy().to_string().into()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("geometry should load");

        let err = block_on(fea_study_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("missing_model".to_string().into()),
                geometry,
            ]),
        ))
        .expect_err("missing model should fail");
        assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
        assert!(err
            .message()
            .contains("fea.study requires id, geometry, and model arguments"));
    }

    #[test]
    fn fea_model_requires_explicit_procedure_sequence() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
            Value::String(geometry_path.to_string_lossy().to_string().into()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("geometry should load");

        let err = block_on(fea_model_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("missing_sequence_model".to_string().into()),
                geometry,
            ]),
        ))
        .expect_err("missing sequence should fail");
        assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
        assert!(err
            .message()
            .contains("fea.model requires exactly one of Sequence or Steps"));
    }

    #[test]
    fn material_and_plan_objects_match_their_portable_schemas() {
        let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
        ensure_value_classes_registered();
        let material = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("steel".into()),
                Value::String("YoungsModulusPa".into()),
                Value::Num(200e9),
                Value::String("PoissonRatio".into()),
                Value::Num(0.3),
                Value::String("MassDensityKgPerM3".into()),
                Value::Num(7850.0),
            ]),
        ))
        .expect("material");
        let encoded = crate::execution::value_codec::encode_inline_value(&material)
            .expect("a material must satisfy its portable schema");
        assert_eq!(
            crate::execution::value_codec::decode_inline_value(&encoded).expect("decode"),
            material
        );
        for property in ["mass_density_kg_per_m3", "elasticity", "provenance"] {
            assert!(
                crate::class_registry::lookup_property(
                    &runmat_types::ClassIdentity::from(FEA_MATERIAL_CLASS),
                    &runmat_types::MemberName::from(property),
                )
                .is_some(),
                "fea.Material must expose {property}"
            );
        }

        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
            Value::String(geometry_path.to_string_lossy().to_string().into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("geometry should load");
        let region_id = geometry_asset_from_value(MODEL_NAME, &geometry)
            .expect("geometry payload")
            .regions
            .first()
            .expect("fixture region")
            .region_id
            .clone();
        let region = || {
            block_on(crate::builtins::geometry::region::geometry_region_builtin(
                geometry.clone(),
                region_id.clone(),
            ))
            .expect("region")
        };
        let target = block_on(fea_property_region_target_solid_builtin(region())).expect("target");
        let assignment = block_on(fea_material_assignment_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                target,
                material.clone(),
            ]),
        ))
        .expect("assignment");
        let fixed = block_on(structural_fixed_builtin("support".to_string(), region()))
            .expect("constraint");
        let step =
            block_on(structural_static_builtin(Value::String("static".into()))).expect("step");
        let model = block_on(fea_model_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("schema_model".into()),
                geometry.clone(),
                Value::String("Materials".into()),
                cell(vec![material]),
                Value::String("MaterialAssignments".into()),
                cell(vec![assignment]),
                Value::String("Constraints".into()),
                cell(vec![fixed]),
                Value::String("Steps".into()),
                cell(vec![step]),
            ]),
        ))
        .expect("model");
        let study = block_on(fea_study_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("schema_study".into()),
                geometry,
                model,
            ]),
        ))
        .expect("study");
        let plan = block_on(fea_plan_builtin(study)).expect("plan");
        let encoded = crate::execution::value_codec::encode_inline_value(&plan)
            .expect("a plan must satisfy its portable schema");
        assert_eq!(
            crate::execution::value_codec::decode_inline_value(&encoded).expect("decode"),
            plan
        );
        for property in ["sequence_id", "steps", "operation_sequence"] {
            assert!(
                crate::class_registry::lookup_property(
                    &runmat_types::ClassIdentity::from(FEA_PLAN_CLASS),
                    &runmat_types::MemberName::from(property),
                )
                .is_some(),
                "fea.Plan must expose {property}"
            );
        }
    }

    #[test]
    fn scalar_enum_errors_are_attributed_to_the_calling_builtin() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry = test_support::load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
        let step = block_on(structural_static_builtin(Value::String("static".into())))
            .expect("static step");
        let frame_error = block_on(fea_model_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("framed".into()),
                geometry.clone(),
                Value::String("Steps".into()),
                step,
                Value::String("Frame".into()),
                Value::String("not_a_frame".into()),
            ]),
        ))
        .expect_err("an unknown frame must be rejected");
        assert_eq!(frame_error.context.builtin.as_deref(), Some(MODEL_NAME));
        assert_eq!(frame_error.identifier(), Some("RunMat:fea:InvalidInput"));

        let material = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("steel".into()),
                Value::String("YoungsModulusPa".into()),
                Value::Num(200e9),
                Value::String("PoissonRatio".into()),
                Value::Num(0.3),
            ]),
        ))
        .expect("material");
        let target = block_on(fea_property_region_target_all_solids_builtin(geometry))
            .expect("all-solids target");
        let confidence_error = block_on(fea_material_assignment_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                target,
                material,
                Value::String("Confidence".into()),
                Value::String("certain".into()),
            ]),
        ))
        .expect_err("an unknown confidence must be rejected");
        assert_eq!(
            confidence_error.context.builtin.as_deref(),
            Some(MATERIAL_ASSIGNMENT_NAME)
        );
    }

    #[test]
    fn material_assignment_revision_is_uint32_and_round_trips_into_a_model() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry = test_support::load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
        let revision = geometry_asset_from_value(MODEL_NAME, &geometry)
            .expect("geometry payload")
            .revision;
        let material = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("steel".into()),
                Value::String("YoungsModulusPa".into()),
                Value::Num(200e9),
                Value::String("PoissonRatio".into()),
                Value::Num(0.3),
            ]),
        ))
        .expect("material");
        let target = block_on(fea_property_region_target_all_solids_builtin(
            geometry.clone(),
        ))
        .expect("all-solids target");
        let assignment = block_on(fea_material_assignment_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                target,
                material.clone(),
            ]),
        ))
        .expect("assignment");
        let Value::Object(object) = &assignment else {
            panic!("expected fea.MaterialAssignment");
        };
        assert_eq!(
            object.properties().get("geometry_revision"),
            Some(&Value::Int(IntValue::U32(revision)))
        );
        let step = block_on(structural_static_builtin(Value::String("static".into())))
            .expect("static step");
        block_on(fea_model_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("uint32_revision".into()),
                geometry,
                Value::String("Materials".into()),
                cell(vec![material]),
                Value::String("MaterialAssignments".into()),
                cell(vec![assignment]),
                Value::String("Steps".into()),
                step,
            ]),
        ))
        .expect("a uint32 revision must decode when the assignment enters a model");
    }

    #[test]
    fn fea_model_rejects_ambiguous_component_option_aliases() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
            Value::String(geometry_path.to_string_lossy().to_string().into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Units".to_string().into()),
                Value::String("meter".to_string().into()),
            ]),
        ))
        .expect("geometry should load");

        for alias in ["Assignments", "Cases"] {
            let error = block_on(fea_model_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String(format!("rejected_{alias}").into()),
                    geometry.clone(),
                    Value::String("Steps".to_string().into()),
                    modeling::static_test_steps_value(),
                    Value::String(alias.to_string().into()),
                    cell(Vec::new()),
                ]),
            ))
            .expect_err("ambiguous component option alias must be rejected");
            assert!(error.message().contains("unsupported fea.model option"));
        }
    }

    #[test]
    fn typed_constructors_build_full_study_and_sweep_objects() {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        let geometry_path = tmp.path().join("part.step");
        std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");

        let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
            Value::String(geometry_path.to_string_lossy().to_string().into()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
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

        let material = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("steel".to_string().into()),
                Value::String("YoungsModulusPa".to_string().into()),
                Value::Num(200e9),
                Value::String("PoissonRatio".to_string().into()),
                Value::Num(0.30),
            ]),
        ))
        .expect("material should build");
        assert_object_class(&material, FEA_MATERIAL_CLASS);

        let material_region = block_on(crate::builtins::geometry::region::geometry_region_builtin(
            geometry.clone(),
            region_id.clone(),
        ))
        .expect("material region should resolve");
        let beam_target = block_on(fea_property_region_target_beam_builtin(
            material_region.clone(),
        ))
        .expect("beam material target should build");
        let shell_target = block_on(fea_property_region_target_shell_builtin(
            material_region.clone(),
        ))
        .expect("shell material target should build");
        let material_target = block_on(fea_property_region_target_solid_builtin(
            material_region.clone(),
        ))
        .expect("solid material target should build");
        let cell_target = block_on(fea_property_region_target_cell_builtin(material_region))
            .expect("cell material target should build");
        for (target, expected) in [
            (beam_target, "beam"),
            (shell_target, "shell"),
            (material_target.clone(), "solid"),
            (cell_target, "cell"),
        ] {
            let decoded: modeling::GeometryBoundPropertyRegionTarget = object_payload(&target);
            let actual = match decoded.target() {
                runmat_analysis_core::PropertyRegionTarget::Beam(_) => "beam",
                runmat_analysis_core::PropertyRegionTarget::Shell(_) => "shell",
                runmat_analysis_core::PropertyRegionTarget::Solid(_) => "solid",
                runmat_analysis_core::PropertyRegionTarget::Cell(_) => "cell",
            };
            assert_eq!(actual, expected);
            assert!(!decoded.target().selects_all());
        }
        let all_solids_target = block_on(fea_property_region_target_all_solids_builtin(
            geometry.clone(),
        ))
        .expect("all-solids material target should build");
        let all_targets = [
            block_on(fea_property_region_target_all_beams_builtin(
                geometry.clone(),
            ))
            .expect("all-beams material target should build"),
            block_on(fea_property_region_target_all_shells_builtin(
                geometry.clone(),
            ))
            .expect("all-shells material target should build"),
            all_solids_target,
            block_on(fea_property_region_target_all_cells_builtin(
                geometry.clone(),
            ))
            .expect("all-cells material target should build"),
        ];
        for target in all_targets {
            assert_object_class(&target, FEA_PROPERTY_REGION_TARGET_CLASS);
            let decoded: modeling::GeometryBoundPropertyRegionTarget = object_payload(&target);
            assert!(decoded.target().selects_all());
        }
        let assignment = block_on(fea_material_assignment_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                material_target,
                material.clone(),
            ]),
        ))
        .expect("material assignment should build");
        assert_object_class(&assignment, FEA_MATERIAL_ASSIGNMENT_CLASS);
        assert!(block_on(fea_material_assignment_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(region_id.clone().into()),
                Value::String("steel".into()),
            ])
        ))
        .is_err());

        let mut foreign_assignment = assignment.clone();
        let Value::Object(foreign_assignment_object) = &mut foreign_assignment else {
            unreachable!("material assignment is an object")
        };
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            foreign_assignment_object,
            "geometry_id",
            Value::String("different_geometry".into()),
        )
        .expect("ordinary fixture property edit");
        assert!(block_on(fea_model_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("cross_geometry_model".into()),
                geometry.clone(),
                Value::String("Materials".into()),
                cell(vec![material.clone()]),
                Value::String("MaterialAssignments".into()),
                cell(vec![foreign_assignment]),
            ])
        ))
        .is_err());

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
            "static_step".to_string().into(),
        )))
        .expect("analysis step should build");
        assert_object_class(&step, FEA_STRUCTURAL_STEP_CLASS);

        let reference_temperature = block_on(temperature_kelvin_builtin(Value::Num(293.15)))
            .expect("reference temperature");
        let domain = block_on(domains::thermomechanical_domain_builtin(
            reference_temperature,
        ))
        .expect("thermo-mechanical domain should build");

        let signal_domain = block_on(generator_policies::time_domain()).unwrap();
        let signal_sampling = block_on(generator_policies::uniform_sampling(
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Int(IntValue::U64(3)),
        ))
        .unwrap();
        let (force_callable, _catalog) =
            test_support::portable_closure_fixture("loads.force", 41, vec![Value::Num(2.0)]);
        let generated_force = block_on(signals::callable::force(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("generated_force".into()),
                force_callable,
                signal_domain,
                signal_sampling,
            ]),
        ))
        .expect("callable force signal should build");

        let model = block_on(fea_model_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("bracket_static_model".to_string().into()),
                geometry.clone(),
                Value::String("Materials".to_string().into()),
                cell(vec![material]),
                Value::String("MaterialAssignments".to_string().into()),
                cell(vec![assignment]),
                Value::String("Constraints".to_string().into()),
                cell(vec![fixed]),
                Value::String("Steps".to_string().into()),
                cell(vec![step.clone()]),
                Value::String("Domains".to_string().into()),
                cell(vec![domain]),
                Value::String("Signals".to_string().into()),
                cell(vec![generated_force]),
            ]),
        ))
        .expect("model should build");
        assert_object_class(&model, FEA_MODEL_CLASS);
        let encoded_model = crate::execution::value_codec::encode_inline_value(&model)
            .expect("model with a frozen generator should be portable");
        let model = crate::execution::value_codec::decode_inline_value(&encoded_model)
            .expect("portable model should decode");
        let composed_model = modeling::model_from_value("test", &model).unwrap();
        assert_eq!(composed_model.frozen_generators.bindings().len(), 1);
        let thermo = composed_model
            .model
            .thermo_mechanical
            .expect("thermo-mechanical domain");
        assert_eq!(thermo.reference_temperature.canonical_value(), 293.15);

        let mut substituted_model = model.clone();
        let Value::Object(substituted) = &mut substituted_model else {
            unreachable!()
        };
        let Some(Value::Cell(mut bindings)) = substituted
            .property(signals::bindings::FROZEN_GENERATORS_PROPERTY)
            .cloned()
        else {
            unreachable!()
        };
        bindings
            .make_data_unique_unbounded()
            .expect("detach the unclaimed tampered fixture from the original model");
        let Value::Struct(binding) = &mut bindings
            .try_data_mut_unique()
            .expect("tampered fixture owns its bindings")[0]
        else {
            unreachable!()
        };
        *binding = runmat_value::StructValue::from_fields_unbounded(
            binding
                .clone()
                .into_fields_unbounded()
                .map(|(name, value)| {
                    if name == "signal_id" {
                        (name, Value::String("substituted".into()))
                    } else {
                        (name, value)
                    }
                }),
        );
        substituted.replace_property_unbounded(
            signals::bindings::FROZEN_GENERATORS_PROPERTY.to_string(),
            Value::Cell(bindings),
        );
        assert!(modeling::model_from_value("test", &substituted_model).is_err());

        let sequence = block_on(sequences::sequence(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("bracket_sequence".to_string().into()),
                cell(vec![step]),
                cell(Vec::new()),
                cell(Vec::new()),
                cell(Vec::new()),
            ]),
        ))
        .expect("sequence should build");
        let execution = block_on(execution::execution_plan_defaults(
            sequence,
            block_on(execution::backend_cpu()).unwrap(),
        ))
        .expect("execution should build");

        let study = block_on(fea_study_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("bracket_static".to_string().into()),
                geometry,
                model,
                Value::String("Execution".to_string().into()),
                execution,
            ]),
        ))
        .expect("study should build");
        assert_object_class(&study, FEA_STUDY_CLASS);
        let (binding_property, owner) = crate::class_registry::lookup_property(
            &runmat_types::ClassIdentity::from(FEA_STUDY_CLASS),
            &runmat_types::MemberName::from(signals::bindings::FROZEN_GENERATORS_PROPERTY),
        )
        .expect("study frozen-generator property should be registered");
        assert_eq!(owner.display_name(), FEA_STUDY_CLASS);
        assert_eq!(binding_property.get_access, MemberAccess::Private);
        let encoded_study = crate::execution::value_codec::encode_inline_value(&study)
            .expect("study with a frozen generator should be portable");
        let study = crate::execution::value_codec::decode_inline_value(&encoded_study)
            .expect("portable study should decode");
        let composed_study = modeling::study_from_value_with_generators("test", &study).unwrap();
        assert_eq!(composed_study.frozen_generators.bindings().len(), 1);
        let error = block_on(fea_run_builtin(study.clone()))
            .expect_err("unmaterialized callable must not enter a solver");
        assert!(error.message().contains("active RunMat runtime context"));

        let sweep = block_on(fea_sweep_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("bracket_sweep".to_string().into()),
                cell(vec![study.clone(), study]),
                Value::String("FailFast".to_string().into()),
                Value::Bool(false),
            ]),
        ))
        .expect_err("duplicate study identities must be rejected");
        assert!(sweep.message().contains("declared more than once"));

        let study =
            study_to_object_with_generators(composed_study.study, composed_study.frozen_generators)
                .unwrap();
        let sweep = block_on(fea_sweep_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("bracket_sweep".to_string().into()),
                cell(vec![study]),
                Value::String("FailFast".to_string().into()),
                Value::Bool(false),
            ]),
        ))
        .expect("sweep should build");
        assert_object_class(&sweep, FEA_SWEEP_CLASS);
        let encoded_sweep = crate::execution::value_codec::encode_inline_value(&sweep)
            .expect("sweep with frozen generators should be portable");
        let sweep = crate::execution::value_codec::decode_inline_value(&encoded_sweep)
            .expect("portable sweep should decode");
        let sweep = modeling::sweep_from_value_with_generators("test", &sweep).unwrap();
        assert_eq!(
            sweep
                .frozen_generators
                .for_study("bracket_static")
                .bindings()
                .len(),
            1
        );
    }

    #[test]
    fn fea_results_field_exposes_values_metadata_and_plot_context() {
        let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
        let (run_value, _study, run_id) = synthetic_plot_run_value();

        let results = block_on(fea_results_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![run_value]),
        ))
        .expect("results should load");
        let Value::Object(results_object) = results.clone() else {
            panic!("expected results object");
        };
        assert_eq!(
            results_object.class_name().display_name(),
            FEA_RESULTS_CLASS
        );
        assert_eq!(
            results_object.class_name().display_name(),
            FEA_RESULTS_CLASS
        );
        assert_eq!(
            results_object.properties().get("run_id"),
            Some(&Value::String(run_id.clone().into()))
        );
        assert!(results_object
            .properties()
            .contains_key(FEA_STUDY_CONTEXT_PROPERTY));

        // Inline values remain usable even when the configured store cannot
        // provide synchronous access outside a captured transaction.
        let original_store = crate::analysis::storage::replace_artifact_store(std::sync::Arc::new(
            crate::analysis::storage::InMemoryAnalysisArtifactStore::new(),
        ))
        .unwrap();
        let filesystem = runmat_filesystem::replace_provider(std::sync::Arc::new(
            runmat_filesystem::MemoryFsProvider::new(),
        ));
        crate::analysis::storage::configure_artifact_store(
            crate::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
                root: "/unavailable-results".into(),
            },
        )
        .unwrap();
        assert!(crate::analysis::storage::resolve_run_attempt(&run_id).is_err());
        assert_eq!(
            block_on(fea_results_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![results.clone()]),
            ))
            .unwrap(),
            results,
        );
        let field = block_on(fea_field_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                results,
                block_on(field_selectors::von_mises()).expect("typed field selector"),
            ]),
        ))
        .expect("field should resolve by unique suffix");
        crate::analysis::storage::replace_artifact_store(original_store).unwrap();
        drop(filesystem);
        let Value::Object(field_object) = &field else {
            panic!("expected field object");
        };
        assert_eq!(field_object.class_name().display_name(), FEA_FIELD_CLASS);
        assert_eq!(
            field_object.properties().get("field_id"),
            Some(&Value::String("structural.von_mises".to_string().into()))
        );
        assert_eq!(
            field_object.properties().get("unit"),
            Some(&Value::String("Pa".to_string().into()))
        );
        assert_eq!(
            field_object.properties().get("location"),
            Some(&Value::String("element".to_string().into()))
        );
        assert_eq!(
            field_object.properties().get("topology_id"),
            Some(&Value::String("analysis_mesh".to_string().into()))
        );
        assert_eq!(
            field_object.properties().get("element_kind"),
            Some(&Value::String("tetrahedron4".to_string().into()))
        );
        assert_eq!(
            field_object.properties().get("entity_count"),
            Some(&Value::Int(runmat_value::IntValue::U64(1)))
        );
        assert_eq!(
            field_object.properties().get("value_count"),
            Some(&Value::Int(runmat_value::IntValue::U64(1)))
        );
        assert_eq!(
            field_object.properties().get("element_count"),
            Some(&Value::Int(runmat_value::IntValue::U64(1)))
        );
        let Some(Value::Tensor(values)) = field_object.properties().get("values") else {
            panic!("expected values tensor");
        };
        assert_eq!(
            field_object.properties().get("materialized"),
            Some(&Value::Bool(true))
        );
        assert_eq!(values.shape(), vec![1]);
        assert_eq!(values.materialize_f64(), vec![42.0]);
        assert!(field_object
            .properties()
            .contains_key(FEA_STUDY_CONTEXT_PROPERTY));
        assert_eq!(
            field_object.properties().get("run_id"),
            Some(&Value::String(run_id.clone().into()))
        );

        let page = block_on(fea_field_page_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                field,
                Value::Int(IntValue::U8(1)),
                Value::Int(IntValue::U16(1)),
            ]),
        ))
        .expect("persisted field page should load");
        let Value::Object(page) = page else {
            panic!("expected field page object")
        };
        assert_eq!(page.class_name().display_name(), FEA_FIELD_PAGE_CLASS);
        assert_eq!(
            crate::class_registry::lookup_method(
                &runmat_types::ClassIdentity::from(FEA_FIELD_CLASS),
                &runmat_types::MethodName::from("page"),
            )
            .expect("fea.Field.page method")
            .0
            .function_name,
            FIELD_PAGE_NAME
        );
        assert_eq!(
            page.properties().get("start"),
            Some(&Value::Int(IntValue::U64(1)))
        );
        assert_eq!(
            page.properties().get("count"),
            Some(&Value::Int(IntValue::U64(1)))
        );
        assert_eq!(page.properties().get("has_more"), Some(&Value::Bool(false)));
        assert!(matches!(
            page.properties().get("next_start"),
            Some(Value::Tensor(tensor)) if tensor.is_empty()
        ));
        let Some(Value::Tensor(values)) = page.properties().get("values") else {
            panic!("expected real field page values")
        };
        assert_eq!(values.shape(), vec![1, 1]);
        assert_eq!(values.materialize_f64(), vec![42.0]);
        let page = Value::Object(page);
        let encoded = crate::execution::value_codec::encode_inline_value(&page)
            .expect("field page should be portable");
        assert_eq!(
            crate::execution::value_codec::decode_inline_value(&encoded)
                .expect("portable field page should decode"),
            page
        );
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn plot_and_mesh_overloads_return_figure_handles_for_fea_objects() {
        let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
        let (run_value, _study, _run_id) = synthetic_plot_run_value();

        let mesh_handle = block_on(crate::call_builtin_async(
            "mesh",
            &[
                run_value.clone(),
                Value::String("Source".to_string().into()),
                Value::String("solver".to_string().into()),
                Value::String("Edges".to_string().into()),
                Value::Bool(true),
            ],
        ))
        .expect("run mesh should create a figure");
        assert!(matches!(mesh_handle.value(), Value::Num(handle) if *handle >= 1.0));

        let results = block_on(fea_results_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![run_value]),
        ))
        .expect("results should load");
        let field = block_on(fea_field_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                results,
                block_on(field_selectors::von_mises()).expect("typed field selector"),
            ]),
        ))
        .expect("field should resolve");
        let field_handle = block_on(crate::call_builtin_async("plot", &[field]))
            .expect("field plot should create a figure");
        assert!(matches!(field_handle.value(), Value::Num(handle) if *handle >= 1.0));
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn mesh_request_defaults_to_solver_edges_and_accepts_cad_reference() {
        let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
        let (run_value, _study, _run_id) = synthetic_plot_run_value();
        let default = mesh_request_from_args(std::slice::from_ref(&run_value))
            .expect("default mesh request should parse");
        assert_eq!(
            default.source,
            crate::analysis::AnalysisFigureMeshSource::Solver
        );
        assert!(default.show_edges);

        let configured = mesh_request_from_args(&[
            run_value,
            Value::String("Source".to_string().into()),
            Value::String("cad_reference".to_string().into()),
            Value::String("Edges".to_string().into()),
            Value::Bool(false),
        ])
        .expect("configured mesh request should parse");
        assert_eq!(
            configured.source,
            crate::analysis::AnalysisFigureMeshSource::CadReference
        );
        assert!(!configured.show_edges);
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn fea_plot_request_accepts_solver_mesh_edge_option() {
        let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
        let (run_value, _study, _run_id) = synthetic_plot_run_value();
        let results = block_on(fea_results_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![run_value]),
        ))
        .expect("results should load");
        let field = block_on(fea_field_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                results,
                block_on(field_selectors::von_mises()).expect("typed field selector"),
            ]),
        ))
        .expect("field should resolve");

        let request = plot_request_from_args(&[
            field,
            Value::String("Edges".to_string().into()),
            Value::Bool(true),
            Value::String("deformed".to_string().into()),
            Value::Bool(false),
            Value::String("overlay".to_string().into()),
            Value::String("cad".to_string().into()),
        ])
        .expect("plot request should parse mesh, deformation, and overlay options");

        assert_eq!(request.field_id, "structural.von_mises");
        assert_eq!(request.sample, None);
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
            runmat_types::wire_integer::MAX_WIRE_INTEGER
        } else {
            u32::MAX as u64
        };
        let typed = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![wide]), vec![1, 1])
            .expect("typed integer");

        assert_eq!(
            usize_from_value(STUDY_NAME, &Value::Tensor(typed)).expect("typed integer"),
            wide as usize
        );
        let beyond = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![
                runmat_types::wire_integer::MAX_WIRE_INTEGER + 2,
            ]),
            vec![1, 1],
        )
        .expect("typed integer beyond the wire bound");
        assert!(usize_from_value(STUDY_NAME, &Value::Tensor(beyond)).is_err());

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(usize_from_value(STUDY_NAME, &Value::Num(boundary)).is_err());
    }

    fn synthetic_plot_run_value() -> (Value, Value, String) {
        let tmp = tempfile::tempdir().expect("tempdir should be created");
        std::fs::write(tmp.path().join("part.stl"), TRIANGLE_STL)
            .expect("geometry fixture should write");
        let geometry_value = block_on(crate::builtins::geometry::geometry_load_builtin(
            Value::String(
                tmp.path()
                    .join("part.stl")
                    .to_string_lossy()
                    .to_string()
                    .into(),
            ),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("geometry should load");
        let geometry = geometry_asset_from_value(STUDY_NAME, &geometry_value)
            .expect("geometry payload should decode");
        let sequence = AnalysisSequence::from_steps(
            "synthetic_plot.sequence",
            vec![ProcedureStep::structural_static("static")],
        );
        let execution = AnalysisExecutionPlan::for_sequence(&sequence, ComputeBackend::Cpu)
            .expect("execution plan should match the sequence");
        let model = runmat_analysis_core::AnalysisModel {
            model_id: runmat_analysis_core::AnalysisModelId("synthetic_plot.model".to_string()),
            geometry_id: geometry.geometry_id.clone(),
            geometry_revision: geometry.revision,
            units: runmat_types::physical::CoordinateUnits::canonical(),
            frame: runmat_analysis_core::CoordinateFrame::Global,
            materials: Vec::new(),
            material_assignments: Vec::new(),
            structural: None,
            thermal: None,
            thermo_mechanical: None,
            electro_thermal: None,
            electromagnetic: None,
            cfd: None,
            interfaces: Vec::new(),
            constraints: Vec::new(),
            boundaries: Vec::new(),
            connectors: Vec::new(),
            signals: Vec::new(),
            excitations: Vec::new(),
            excitation_cases: Vec::new(),
            initial_conditions: Vec::new(),
            response_spectra: Vec::new(),
            random_excitations: Vec::new(),
            sequence,
        };
        let study = study_to_object(AnalysisStudySpec {
            study_id: "synthetic_plot".to_string(),
            geometry,
            model_source_evidence: None,
            model,
            execution,
            meshing_settings: None,
            immutable_values: Vec::new(),
            solver_mesh_artifact_path: None,
            meshing_evidence_artifact_path: None,
        })
        .expect("study context should encode");
        let identity = test_run_identity(runmat_analysis_core::AnalysisProcedure::Static);
        let run_id = crate::analysis::storage::next_run_id();
        let run = crate::analysis::AnalysisRunResult {
            observation_scalars: Vec::new(),
            run_id: run_id.clone(),
            run: runmat_analysis_fea::FeaRunResult {
                accelerator_evidence: Vec::new(),
                identity,
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_method: runmat_analysis_fea::SolverMethod::SparseDirectCholesky,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
                solver_host_sync_count: 0,
                diagnostics: Vec::new(),
                spatial_projection_evidence: Vec::new(),
                structural_static_convergence: Some(runmat_analysis_fea::contracts::StructuralStaticConvergence {
                    converged: true,
                    iterations: 0,
                    residual_norm: 0.0,
                    relative_residual: Some(0.0),
                    operator_proof: runmat_analysis_fea::contracts::StaticOperatorProof::SparseCholesky,
                    spectral_estimate: None,
                }),
                structural_static_balance: Some(synthetic_static_balance()),
                structural_constraint_result: Some(synthetic_constraint_result()),
                structural_connector_result: Some(synthetic_connector_result()),
                observations: Vec::new(),
                observation_histories: Vec::new(),
                series_reductions: Vec::new(),
                fields: vec![AnalysisField::host_f64(
                    "structural.von_mises", AnalysisFieldLayout::scalar(ObservationQuantity::VonMisesStress, FieldLocation::Element),
                    vec![1],
                    vec![42.0],
                )],
            },
            render_topology: Some(crate::analysis::AnalysisRenderTopology {
                schema_version: "analysis_render_topology/v1".to_string(),
                source: crate::analysis::AnalysisRenderTopologySource::AnalysisMesh,
                volume_element_count: 1,
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
            buckling_results: None,
            thermal_results: None,
            transient_results: None,
            nonlinear_results: None,
            model_validity: crate::analysis::QualityGate::Pass,
            solver_convergence: crate::analysis::QualityGate::Pass,
            result_quality: crate::analysis::QualityGate::Pass,
            run_status: crate::analysis::RunStatus::Completed,
            numerically_acceptable: true,
            quality_reasons: Vec::new(),
            provenance: crate::analysis::RunProvenance {
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_host_sync_count: 0,
                precision_mode: "fp64".to_string(),
                deterministic_mode: true,
                solver_method: runmat_analysis_fea::SolverMethod::SparseDirectCholesky,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
                quality_policy: "balanced".to_string(),
                fallback_events: Vec::new(),
                effective_memory_ceiling_bytes: None,
                static_solve_path: None,
            },
        };
        crate::analysis::storage::persist_run_result(&run).expect("run should persist");

        let mut object = ObjectInstance::new(FEA_STEP_RUN_RESULT_CLASS.to_string());
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            "run_id",
            Value::String(run_id.clone().into()),
        )
        .expect("ordinary fixture property edit");
        crate::builtins::fea::object_api::replace(
            "fea.fixture",
            &mut object,
            FEA_STUDY_CONTEXT_PROPERTY,
            study.clone(),
        )
        .expect("ordinary fixture property edit");
        (Value::Object(object), study, run_id)
    }

    fn persist_synthetic_indexed_results() -> String {
        let indexed_fields = |prefix: &str| {
            vec![
                AnalysisField::host_f64(
                    format!("{prefix}.0"),
                    AnalysisFieldLayout::scalar(
                        ObservationQuantity::Residual,
                        FieldLocation::Global,
                    ),
                    vec![1],
                    vec![10.0],
                ),
                AnalysisField::host_f64(
                    format!("{prefix}.1"),
                    AnalysisFieldLayout::scalar(
                        ObservationQuantity::Residual,
                        FieldLocation::Global,
                    ),
                    vec![1],
                    vec![20.0],
                ),
            ]
        };
        let run_id = crate::analysis::storage::next_run_id();
        let identity =
            test_run_identity(runmat_analysis_core::AnalysisProcedure::TransientResponse);
        let run = crate::analysis::AnalysisRunResult {
            observation_scalars: Vec::new(),
            run_id: run_id.clone(),
            run: runmat_analysis_fea::FeaRunResult {
                accelerator_evidence: Vec::new(),
                identity,
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_method: runmat_analysis_fea::SolverMethod::NewmarkAverageAccelerationPcg,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::Jacobi,
                solver_host_sync_count: 0,
                diagnostics: Vec::new(),
                spatial_projection_evidence: Vec::new(),
                structural_static_convergence: None,
                structural_static_balance: None,
                structural_constraint_result: None,
                structural_connector_result: Some(synthetic_connector_result()),
                observations: Vec::new(),
                observation_histories: Vec::new(),
                series_reductions: Vec::new(),
                fields: Vec::new(),
            },
            render_topology: None,
            modal_results: None,
            buckling_results: None,
            thermal_results: None,
            transient_results: Some(crate::analysis::TransientResultsData {
                transient_payload_version: "transient_results/v1".to_string(),
                initial_state: runmat_analysis_fea::StructuralInitialStateEvidence {
                    procedure_step_id: "synthetic_transient".to_string(),
                    basis: runmat_analysis_fea::StructuralInitialStateEvidenceBasis::OperatorBoundZeroDefault {
                        dof_count: 1,
                        operator_identity: runmat_execution::Digest::sha256(
                            b"runtime-synthetic-transient-fixture",
                        ),
                    },
                    total_dof_count: 1,
                    authored_displacement_dof_count: 0,
                    constraint_derived_displacement_dof_count: 0,
                    zero_default_displacement_dof_count: 1,
                    authored_velocity_dof_count: 0,
                    constraint_derived_velocity_dof_count: 0,
                    zero_default_velocity_dof_count: 1,
                    direct_constrained_dof_count: 0,
                    constraint_equation_count: 0,
                    maximum_displacement_constraint_residual_bits: 0.0_f64.to_bits(),
                    maximum_velocity_constraint_residual_bits: 0.0_f64.to_bits(),
                    applied_conditions: Vec::new(),
                },
                time_points_s: vec![0.0, 1.0],
                displacement_snapshots: indexed_fields("displacement"),
                rotation_snapshots: Vec::new(),
                velocity_snapshots: indexed_fields("velocity"),
                angular_velocity_snapshots: Vec::new(),
                acceleration_snapshots: indexed_fields("acceleration"),
                angular_acceleration_snapshots: Vec::new(),
                strain_snapshots: indexed_fields("strain"),
                stress_snapshots: indexed_fields("stress"),
                strain_energy_density_snapshots: indexed_fields("strain_energy_density"),
                von_mises_snapshots: indexed_fields("von_mises"),
                kinetic_energy_snapshots: indexed_fields("kinetic_energy"),
                strain_energy_snapshots: indexed_fields("strain_energy"),
                residual_norm_snapshots: indexed_fields("residual_norm"),
                reaction_force_snapshots: indexed_fields("reaction_force"),
                reaction_moment_snapshots: Vec::new(),
                external_work_snapshots: indexed_fields("external_work"),
                support_work_snapshots: indexed_fields("support_work"),
                damping_dissipation_snapshots: indexed_fields("damping_dissipation"),
                energy_balance_residual_snapshots: indexed_fields("energy_balance_residual"),
                relative_energy_balance_error_snapshots: indexed_fields(
                    "relative_energy_balance_error",
                ),
                dynamic_balance_snapshots: (0..2)
                    .map(|_| runmat_analysis_fea::StructuralDynamicBalanceSnapshot {
                        applied_load: vec![0.0],
                        inertia_force: vec![0.0],
                        damping_force: vec![0.0],
                        elastic_force: vec![0.0],
                        equilibrium_residual: vec![0.0],
                    })
                    .collect(),
                electro_thermal_temperature_snapshots: Vec::new(),
                electro_thermal_thermal_residual_snapshots: Vec::new(),
                residual_norms: vec![0.5, 0.25],
                integration_method:
                    crate::analysis::TransientIntegrationMethod::NewmarkAverageAcceleration,
            }),
            nonlinear_results: None,
            model_validity: crate::analysis::QualityGate::Pass,
            solver_convergence: crate::analysis::QualityGate::Pass,
            result_quality: crate::analysis::QualityGate::Pass,
            run_status: crate::analysis::RunStatus::Completed,
            numerically_acceptable: true,
            quality_reasons: Vec::new(),
            provenance: crate::analysis::RunProvenance {
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_host_sync_count: 0,
                precision_mode: "fp64".to_string(),
                deterministic_mode: true,
                solver_method: runmat_analysis_fea::SolverMethod::NewmarkAverageAccelerationPcg,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::Jacobi,
                quality_policy: "balanced".to_string(),
                fallback_events: Vec::new(),
                effective_memory_ceiling_bytes: None,
                static_solve_path: None,
            },
        };
        crate::analysis::storage::persist_run_result(&run).expect("indexed run should persist");
        run_id
    }

    #[test]
    fn fea_results_translates_successful_snapshot_selectors_once() {
        let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
        let run_id = persist_synthetic_indexed_results();
        let selected = block_on(fea_results_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(run_id.into()),
                Value::String("TransientSnapshotIndices".to_string().into()),
                Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).expect("double selector")),
            ]),
        ))
        .expect("one-based selectors should resolve the second stored entries");
        let Value::Object(selected) = selected else {
            panic!("results object");
        };
        let Some(Value::Struct(transient)) = selected.properties().get("transient_results") else {
            panic!("transient results");
        };
        let Some(Value::Tensor(time_points)) = transient.fields.get("time_points_s") else {
            panic!("transient time points");
        };
        assert_eq!(time_points.materialize_f64(), vec![1.0]);
        assert!(transient.fields.contains_key("strain_snapshots"));
        assert!(transient.fields.contains_key("stress_snapshots"));
        assert!(transient
            .fields
            .contains_key("strain_energy_density_snapshots"));
        assert!(transient.fields.contains_key("strain_energy_snapshots"));
        assert!(transient.fields.contains_key("reaction_force_snapshots"));
        assert!(transient.fields.contains_key("external_work_snapshots"));
        assert!(transient.fields.contains_key("support_work_snapshots"));
        assert!(transient
            .fields
            .contains_key("damping_dissipation_snapshots"));
        assert!(transient
            .fields
            .contains_key("energy_balance_residual_snapshots"));
        assert!(transient
            .fields
            .contains_key("relative_energy_balance_error_snapshots"));
        assert!(transient.fields.contains_key("dynamic_balance_snapshots"));
        let Some(Value::Struct(initial_state)) = transient.fields.get("initial_state") else {
            panic!("structural initial-state evidence");
        };
        assert_eq!(
            initial_state.fields.get("procedure_step_id"),
            Some(&Value::String("synthetic_transient".to_string().into()))
        );

        let Some(Value::Struct(summary)) = selected.properties().get("summary") else {
            panic!("results summary");
        };
        let Some(Value::Tensor(indices)) = summary.fields.get("available_mode_indices") else {
            panic!("available mode indices");
        };
        assert!(indices
            .integer_storage()
            .expect("exact public indices")
            .exact_values()
            .is_empty());
    }

    fn assert_object_class(value: &Value, expected: &str) {
        let Value::Object(object) = value else {
            panic!("expected object value");
        };
        assert_eq!(object.class_name().display_name(), expected);
        assert!(!object.properties().is_empty());
        assert!(object
            .properties()
            .keys()
            .all(|name| !name.starts_with("__runmat_")
                || name == crate::builtins::common::json::typed_object_projection::PROPERTY));
        object_projection::validate(object).expect("only validated shape provenance is permitted");
    }
}
