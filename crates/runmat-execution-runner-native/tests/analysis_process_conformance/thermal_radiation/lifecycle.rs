//! Shared admitted task pool and cold closure; no direct numerical calls.
use super::*;
use runmat_analysis_execution::{AnalysisOutputPolicy, ThermalStudyPreparation, ThermalStudyScope};
use runmat_analysis_fea::{
    properties::compile_property_map,
    study::{
        plan_thermal_study_with_fields, CompiledAnalysisStudy, ThermalStudyPlanningPolicy,
        ThermalStudyProduct,
    },
};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_execution_runner_native::{
    execute_thermal_study_locally_with_source, NativeExecutionConfig, ThermalStudyExecutionPolicy,
};

pub(super) async fn run(
    input: ThermalWorkload,
    mesh: &SolverMeshArtifact,
) -> CompletedThermalStudy {
    let mut model = input.model;
    let step = &mut model.sequence.steps[0];
    for criterion in &mut step.accuracy.criteria {
        if criterion.contribution == ErrorContributionKind::Algebraic {
            criterion.tolerance = AccuracyTolerance::Relative(TARGET);
        }
    }
    let mut balance = step.accuracy.criteria[0].clone();
    balance.criterion_id = "physical-heat-balance".into();
    balance.evidence = RequiredAccuracyEvidence::ConservationBalance;
    step.accuracy.criteria.push(balance);
    step.accuracy.criteria.push(AccuracyCriterion {
        criterion_id: "mandatory-spatial".into(),
        target: AccuracyTarget::NumericalResult,
        contribution: ErrorContributionKind::SpatialDiscretization,
        evidence: RequiredAccuracyEvidence::ConvergenceStudy,
        tolerance: AccuracyTolerance::Relative(0.1),
        mandatory: true,
    });
    step.observations =
        super::super::thermal::authored_observations(PartialAccessPolicy::ManifestAndChunks);
    if input.procedure == runmat_analysis_fea::ThermalProcedureControl::Steady {
        for request in &mut step.observations {
            if let ObservationRequest::History(spec) = request {
                spec.sampling = SamplingPolicy::Final;
                spec.reduction = None;
                *request = ObservationRequest::Field(spec.clone());
            }
        }
    }
    let ceiling = step.accuracy.resources.resident_byte_budget();
    let wall = step.accuracy.resources.maximum_wall_time_milliseconds;
    let compiled = CompiledAnalysisStudy {
        module_path: "qualification/thermal-radiation.fea".into(),
        study_id: "thermal-radiation".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        properties: compile_property_map(&model, Some(mesh)).unwrap(),
        solver_mesh_digest: mesh.canonical_digest,
        model,
    };
    let workload = plan_thermal_study_with_fields(
        &compiled,
        ThermalStudyPlanningPolicy {
            backend: input.backend,
            element_heat_source: None,
        },
        input.initial_fields,
    )
    .unwrap();
    assert_eq!(workload.residual_target, TARGET);
    let directory = tempfile::tempdir().unwrap();
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = directory.path().join("thermal-radiation");
    config.max_workers = 1;
    // Placement capacity is explicit and equals the unchanged authored resource envelope.
    // Task hints do not resize a local pool.
    config.worker_cpu_millicores = 1000;
    config.worker_memory_bytes = ceiling;
    config.worker_scratch_bytes = ceiling;
    config.worker_capabilities = super::super::capabilities();
    let preparation = ThermalStudyPreparation {
        maximum_resident_bytes: None,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        scope: ThermalStudyScope::Standalone,
        field_source: None,
        product: ThermalStudyProduct { compiled, workload },
        mesh: mesh.clone(),
        frozen_project_digest: None,
        artifact_access: ArtifactAccess {
            authorization_scope: "thermal-radiation-qualification".into(),
            encryption_context: runmat_execution::Digest::sha256("thermal-radiation-qualification"),
        },
        program_revision: super::super::revision(),
        output_policy: AnalysisOutputPolicy::new(8192, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    };
    let resources = runmat_execution::resource::ResourceRequest {
        cpu_millicores: 1000,
        memory_bytes: ceiling,
        scratch_bytes: ceiling,
        max_wall_millis: wall,
        max_artifact_bytes: ObjectInventoryLimits::default().max_total_bytes,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: std::collections::BTreeSet::from([
            runmat_execution::resource::Capability::ProcessIsolation,
        ]),
    };
    let executed = execute_thermal_study_locally_with_source(
        config,
        preparation,
        ThermalStudyExecutionPolicy {
            resources,
            retry: runmat_execution::task::RetryPolicy::Never,
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .await
    .unwrap();
    // The public call drops producer workers. Its source retains storage only.
    let cold = super::super::thermal::cold_reimport::assert_closed_physical_completion(
        &executed.source,
        executed.completed,
        ObjectInventoryLimits::default(),
    );
    drop(executed.source);
    drop(directory);
    assert_eq!(
        cold.accuracy.artifact.evaluation().report().status,
        runmat_analysis_fea::results::accuracy::AccuracyStatus::Indeterminate
    );
    let gate = runmat_analysis_fea::results::accuracy::publication_gate(
        &cold.workload.model.sequence.steps[0].accuracy,
        cold.accuracy.artifact.evaluation(),
    )
    .unwrap();
    assert_eq!(
        gate.decision,
        runmat_analysis_fea::results::accuracy::PublicationDecision::Block
    );
    cold
}
