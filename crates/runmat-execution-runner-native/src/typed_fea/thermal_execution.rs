//! Authored standalone thermal composition; numerical and publication authority stay shared.
use super::{preparation::PreparedAuthoredThermalStudy, LocalFeaExecutionOptions};
use crate::{
    analysis_task_pool::{configure_local_analysis_envelope, LocalAnalysisTaskPool},
    local_structural_study::emit_local_progress,
    LocalStructuralStudyObservers, NativeExecutionConfig, NativeProgramSession,
};
use runmat_analysis_execution::{AnalysisOutputPolicy, ThermalStudyPreparation, ThermalStudyScope};
use runmat_analysis_fea::{
    study::{
        compile_analysis_study, plan_thermal_study_product_with_fields,
        AnalysisStudyCompilationInput,
    },
    FeaProgressPhase, FeaProgressStatus,
};
use runmat_execution::{
    resource::{Capability, ResourceRequest},
    task::RetryPolicy,
};
use runmat_execution_artifact::{cache::CacheExport, object::ObjectInventoryLimits};
use runmat_runtime::analysis::{ThermalStudyCompletion, ThermalStudyObjectSource};
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct CompletedAuthoredThermalStudy {
    pub completion: ThermalStudyCompletion,
    pub source: ThermalStudyObjectSource,
    pub mesh: runmat_meshing_core::SolverMeshArtifact,
    pub meshing_evidence: runmat_meshing_core::MeshingEvidence,
    /// Present for the heat-transfer side of an electrothermal analysis.
    pub electrothermal: Option<ElectrothermalOperatingPoint>,
}

pub(super) async fn execute(
    prepared: PreparedAuthoredThermalStudy,
    workers: u32,
    options: &LocalFeaExecutionOptions,
) -> Result<CompletedAuthoredThermalStudy, String> {
    let source_digest = runmat_analysis_fea::study::structural_study_source_digest(
        &prepared.study,
        &prepared.materials,
    )
    .map_err(|e| e.to_string())?;
    if source_digest != prepared.authored_study_digest {
        return Err("prepared thermal study differs from its authored source identity".into());
    }
    if prepared.planning.backend != runmat_analysis_fea::ComputeBackend::Cpu
        || !prepared.analysis_accelerator_requests.is_empty()
    {
        return Err("authored thermal execution has no admitted accelerator envelope".into());
    }
    let mut config = NativeExecutionConfig::for_current_executable().map_err(|e| e.to_string())?;
    config.max_workers = workers.max(1);
    config.worker_cpu_millicores = prepared
        .local_execution
        .checked_cpu_millicores()
        .map_err(|e| e.to_string())?;
    let observers = LocalStructuralStudyObservers {
        progress: options.progress_observer.clone(),
        mesh: options.mesh_observer.clone(),
    };
    emit_local_progress(
        &observers,
        "fea.typed_study",
        FeaProgressPhase::GeometryLoad,
        FeaProgressStatus::Started,
        "Loading and validating study geometry",
        None,
        None,
    );
    let inventory = ObjectInventoryLimits::default();
    let mut store = crate::FilesystemObjectStore::open(
        config
            .store_root
            .join("whole-studies")
            .join(uuid::Uuid::new_v4().simple().to_string()),
        inventory.max_object_bytes,
    )
    .map_err(|e| e.to_string())?;
    prepared
        .geometry_meshing
        .write_to(&mut store)
        .map_err(|e| e.to_string())?;
    for materialization in &prepared.input_materializations {
        materialization
            .validate_closed()
            .map_err(|e| e.to_string())?;
        for object in materialization.input_objects() {
            let logical = runmat_execution_artifact::LogicalObject::new(
                runmat_execution_artifact::ObjectNamespace::InputValue,
                format!(
                    "analysis/generator-inputs/{}",
                    object.reference.logical_digest
                ),
                object.reference.media_type.clone(),
                object.bytes.clone(),
            )
            .map_err(|e| e.to_string())?;
            if logical.descriptor.digest != object.reference.logical_digest {
                return Err("generator input object differs from its logical reference".into());
            }
            store.write_verified(&logical).map_err(|e| e.to_string())?;
        }
    }
    let geometry = crate::local_study_geometry::execute_local_study_geometry(
        config,
        prepared.geometry_meshing.request(),
        &prepared.artifact_access,
        &mut store,
        &observers,
    )
    .await
    .map_err(|e| e.to_string())?;
    emit_local_progress(
        &observers,
        "fea.typed_study",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "Assembling typed study operators and inputs",
        None,
        None,
    );
    let mut compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: &prepared.study,
        geometry: &geometry.output.document,
        regions: &geometry.output.regions,
        materials: &prepared.materials,
        mesh: &geometry.meshing.artifact,
    })
    .map_err(|e| e.to_string())?;
    let resolved_inputs = prepared
        .input_materializations
        .iter()
        .map(|plan| {
            plan.resolve(&geometry.meshing.artifact)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let materialized = crate::input_materialization::execute_locally(
        geometry.analysis_config.clone(),
        &resolved_inputs,
        &store,
        &prepared.artifact_access,
        workers,
        &geometry.meshing.artifact,
    )
    .await
    .map_err(|e| e.to_string())?;
    crate::input_materialization::apply_to_model(&mut compiled.model, materialized)
        .map_err(|e| e.to_string())?;
    if let Some(heat) = &prepared.heat_source {
        let transient = compiled.model.sequence.steps.iter().any(|step| {
            step.step_id.as_str() == heat.step_id
                && step.procedure == runmat_analysis_core::AnalysisProcedure::TransientResponse
        });
        runmat_analysis_fea::execution::bind_element_heat_source(
            &mut compiled.model,
            &heat.step_id,
            &heat.source_id,
            runmat_execution::Digest::from_bytes(*geometry.meshing.artifact.canonical_digest.bytes()),
            &[(0.0, heat.element_density_w_per_m3.clone())],
            transient,
        )
        .map_err(|e| e.to_string())?;
    }
    let electrothermal = if prepared.electrothermal {
        Some(solve_electrothermal(
            &mut compiled,
            &geometry.meshing.artifact,
            prepared.planning,
        )?)
    } else {
        None
    };
    let [step] = compiled.model.sequence.steps.as_slice() else {
        return Err("authored thermal execution requires a standalone step".into());
    };
    let initial_fields = runmat_runtime::analysis::storage::resolve_initial_fields(
        &compiled.model,
        step,
        step.accuracy.resources.maximum_resident_bytes,
    )
    .map_err(|error| error.to_string())?;
    let product = plan_thermal_study_product_with_fields(
        compiled,
        &geometry.meshing.artifact,
        prepared.planning,
        initial_fields,
    )
    .map_err(|error| error.to_string())?;
    emit_local_progress(
        &observers,
        "fea.typed_study",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "Study operators and inputs assembled",
        Some(1),
        Some(1),
    );
    let [step] = product.compiled.model.sequence.steps.as_slice() else {
        return Err("authored thermal execution requires a standalone step".into());
    };
    let resources = ResourceRequest {
        cpu_millicores: prepared
            .local_execution
            .checked_cpu_millicores()
            .map_err(|e| e.to_string())?,
        memory_bytes: step.accuracy.resources.maximum_resident_bytes,
        scratch_bytes: step.accuracy.resources.maximum_resident_bytes,
        max_wall_millis: step.accuracy.resources.maximum_wall_time_milliseconds,
        max_artifact_bytes: inventory.max_total_bytes,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    };
    resources.validate().map_err(|e| e.to_string())?;
    let mut analysis_config = geometry.analysis_config;
    analysis_config.max_workers = workers.max(1);
    configure_local_analysis_envelope(&mut analysis_config, &resources);
    let session = NativeProgramSession::new(analysis_config).map_err(|e| e.to_string())?;
    // Publication outlives the executing pool. A writable store clone does not retain
    // the session directory; the shared object source carries that lifetime authority.
    let source = session.object_source();
    let mut pool = LocalAnalysisTaskPool::new(session);
    emit_local_progress(
        &observers,
        "fea.typed_study",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "Executing the analysis sequence",
        None,
        None,
    );
    let executed = Box::pin(crate::analysis_thermal::execute_with_pool(
        &mut pool,
        ThermalStudyPreparation {
            local_execution: prepared.local_execution,
            field_source: None,
            scope: ThermalStudyScope::Standalone,
            product,
            mesh: geometry.meshing.artifact.clone(),
            frozen_project_digest: prepared.frozen_project_digest,
            artifact_access: prepared.artifact_access,
            program_revision: prepared.program_revision,
            output_policy: AnalysisOutputPolicy::new(
                runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES,
                inventory,
            )
            .map_err(|e| e.to_string())?,
            required_capabilities: BTreeSet::from([
                runmat_analysis_execution::AnalysisCapabilityRequirement::ProcessIsolation,
            ]),
        },
        crate::ThermalStudyExecutionPolicy {
            resources,
            retry: RetryPolicy::IdempotentInfrastructure,
            deadline_unix_millis: None,
            priority: 0,
        },
    ))
    .await
    .map_err(|e| e.to_string())?;
    let (completion, source) =
        crate::analysis_placement::convert_completed_thermal_study(executed.completed, source)
            .map_err(|e| e.to_string())?;
    emit_local_progress(
        &observers,
        "fea.typed_study",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "Analysis sequence completed",
        Some(1),
        Some(1),
    );
    Ok(CompletedAuthoredThermalStudy {
        completion,
        source,
        mesh: geometry.meshing.artifact,
        meshing_evidence: geometry.meshing.evidence,
        electrothermal,
    })
}

/// The solved operating point of a two-way electrothermal analysis.
#[derive(Debug, Clone)]
pub struct ElectrothermalOperatingPoint {
    pub sweeps: Vec<runmat_analysis_fea::execution::ElectrothermalSweep>,
    /// Element temperatures at the last time point (canonical element order).
    pub element_temperature_k: Vec<f64>,
    /// Last time point of a transient analysis; `None` when steady.
    pub operating_time_s: Option<f64>,
    /// Joule power and terminal power (sum of terminal potential x current) at that point.
    pub joule_power_w: f64,
    pub terminal_power_w: f64,
}

/// Split the compiled electrothermal analysis into conduction and heat transfer, iterate them
/// to the two-way fixed point, and leave `compiled` as the heat-transfer analysis with the
/// converged Joule source bound (it is then solved and published by the thermal lifecycle).
fn solve_electrothermal(
    compiled: &mut runmat_analysis_fea::study::CompiledAnalysisStudy,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    policy: runmat_analysis_fea::study::ThermalStudyPlanningPolicy,
) -> Result<ElectrothermalOperatingPoint, String> {
    use runmat_analysis_fea::execution::{
        electrical_part, element_temperatures, execute_coupled_electrothermal, thermal_part,
        CoupledElectrothermalCompletion, ElectrothermalCouplingControls,
    };
    let [step] = compiled.model.sequence.steps.as_slice() else {
        return Err("electrothermal execution requires one analysis".into());
    };
    let step = step.clone();
    let electrical = electrical_part(&compiled.model, &step);
    compiled.model = thermal_part(&compiled.model, &step);
    runmat_analysis_core::retain_authored_outputs(
        &mut compiled.outputs,
        &mut compiled.assessments,
        |_, quantity| is_thermal_quantity(quantity),
    );
    let template = runmat_analysis_fea::study::plan_thermal_study(compiled, policy)
        .map_err(|error| error.to_string())?;
    let completion = execute_coupled_electrothermal(
        &electrical,
        &template,
        mesh,
        template.residual_target,
        ElectrothermalCouplingControls::default(),
    )
    .map_err(|error| error.to_string())?;
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(mesh)
        .map_err(|error| error.to_string())?;
    let last = completion
        .thermal
        .temperature_snapshots
        .last()
        .and_then(|snapshot| snapshot.as_host_f64())
        .ok_or("electrothermal thermal history is empty")?;
    let element_temperature_k =
        element_temperatures(&projection, last).map_err(|error| error.to_string())?;
    let sample = completion
        .electrical
        .last()
        .ok_or("electrothermal conduction history is empty")?;
    let operating = ElectrothermalOperatingPoint {
        sweeps: completion.sweeps.clone(),
        element_temperature_k,
        operating_time_s: (completion.electrical.len() > 1).then_some(sample.time_s),
        joule_power_w: CoupledElectrothermalCompletion::joule_power_w(sample),
        terminal_power_w: CoupledElectrothermalCompletion::terminal_power_w(sample),
    };
    tracing::info!(
        sweeps = operating.sweeps.len(),
        joule_power_w = operating.joule_power_w,
        terminal_power_w = operating.terminal_power_w,
        last_temperature_change_k = operating.sweeps.last().map(|s| s.maximum_temperature_change_k),
        maximum_temperature_rise_k = operating.sweeps.last().map(|s| s.maximum_temperature_rise_k),
        "electrothermal two-way coupling converged"
    );
    compiled.model = completion.thermal_workload.model;
    Ok(operating)
}

fn is_thermal_quantity(quantity: runmat_analysis_core::ObservationQuantity) -> bool {
    use runmat_analysis_core::ObservationQuantity as Q;
    matches!(
        quantity,
        Q::Temperature
            | Q::TemperatureGradient
            | Q::HeatFlux
            | Q::HeatRate
            | Q::ThermalEnergyStorageRateDensity
    )
}
