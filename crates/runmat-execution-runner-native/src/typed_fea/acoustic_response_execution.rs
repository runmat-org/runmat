//! Authored standalone acoustic response composition; numerical and publication authority stay shared.
use super::{preparation::PreparedAuthoredAcousticResponseStudy, LocalFeaExecutionOptions};
use crate::{
    analysis_task_pool::{configure_local_analysis, LocalAnalysisTaskPool},
    local_structural_study::emit_local_progress,
    LocalStructuralStudyObservers, NativeExecutionConfig, NativeProgramSession,
};
use runmat_analysis_execution::{AnalysisOutputPolicy, AcousticResponseStudyPreparation};
use runmat_analysis_fea::{
    study::{build_acoustic_response_study_product, AcousticResponseStudyProductInput},
    FeaProgressPhase, FeaProgressStatus,
};
use runmat_execution::{
    resource::{Capability, ResourceRequest},
    task::RetryPolicy,
};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_runtime::analysis::{AcousticResponseStudyCompletion, AcousticResponseStudyObjectSource};
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct CompletedAuthoredAcousticResponseStudy {
    pub completion: AcousticResponseStudyCompletion,
    pub source: AcousticResponseStudyObjectSource,
    pub mesh: runmat_meshing_core::SolverMeshArtifact,
    pub meshing_evidence: runmat_meshing_core::MeshingEvidence,
}

pub(super) async fn execute(
    prepared: PreparedAuthoredAcousticResponseStudy,
    workers: u32,
    options: &LocalFeaExecutionOptions,
) -> Result<CompletedAuthoredAcousticResponseStudy, String> {
    let source_digest = runmat_analysis_fea::study::structural_study_source_digest(
        &prepared.study,
        &prepared.materials,
    )
    .map_err(|e| e.to_string())?;
    if source_digest != prepared.authored_study_digest {
        return Err("prepared acoustic response study differs from its authored source identity".into());
    }
    admit_native_envelope(
        options.backend,
        options.accelerators.len(),
        prepared.analysis_accelerator_requests.len(),
        prepared.input_materializations.len(),
    )
    .map_err(|error| error.to_string())?;
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
    let product = build_acoustic_response_study_product(AcousticResponseStudyProductInput {
        study: &prepared.study,
        geometry: &geometry.output.document,
        regions: &geometry.output.regions,
        materials: &prepared.materials,
        mesh: &geometry.meshing.artifact,
        planning: prepared.planning,
    })
    .map_err(|e| e.to_string())?;
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
        return Err("authored acoustic response execution requires a standalone step".into());
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
    configure_local_analysis(&mut analysis_config, &resources);
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
    let executed = Box::pin(crate::analysis_acoustic_response::execute_with_pool(
        &mut pool,
        AcousticResponseStudyPreparation {
            policy: prepared.planning,
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
        crate::analysis_acoustic_response::AcousticResponseStudyExecutionPolicy {
            resources,
            retry: RetryPolicy::IdempotentInfrastructure,
            deadline_unix_millis: None,
            priority: 0,
        },
    ))
    .await
    .map_err(|e| e.to_string())?;
    let (completion, source) =
        crate::analysis_acoustic_response::source::convert(executed, source).map_err(|e| e.to_string())?;
    emit_local_progress(
        &observers,
        "fea.typed_study",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "Analysis sequence completed",
        Some(1),
        Some(1),
    );
    Ok(CompletedAuthoredAcousticResponseStudy {
        completion,
        source,
        mesh: geometry.meshing.artifact,
        meshing_evidence: geometry.meshing.evidence,
    })
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum AcousticResponseNativeEnvelopeError {
    #[error("authored acoustic response execution has no admitted accelerator envelope")]
    Accelerator,
    #[error("authored acoustic response execution does not support signal/source materializations")]
    SourceMaterialization,
}

fn admit_native_envelope(
    backend: runmat_analysis_fea::ComputeBackend,
    requested_accelerators: usize,
    prepared_accelerators: usize,
    source_materializations: usize,
) -> Result<(), AcousticResponseNativeEnvelopeError> {
    if backend != runmat_analysis_fea::ComputeBackend::Cpu
        || requested_accelerators != 0
        || prepared_accelerators != 0
    {
        return Err(AcousticResponseNativeEnvelopeError::Accelerator);
    }
    if source_materializations != 0 {
        return Err(AcousticResponseNativeEnvelopeError::SourceMaterialization);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_fea::ComputeBackend;

    #[test]
    fn acoustic_response_native_intent_rejects_sources_and_accelerators_before_execution() {
        assert_eq!(admit_native_envelope(ComputeBackend::Cpu, 0, 0, 0), Ok(()));
        for (backend, requested, prepared) in [
            (ComputeBackend::Gpu, 0, 0),
            (ComputeBackend::Cpu, 1, 0),
            (ComputeBackend::Cpu, 0, 1),
        ] {
            assert_eq!(
                admit_native_envelope(backend, requested, prepared, 0),
                Err(AcousticResponseNativeEnvelopeError::Accelerator)
            );
        }
        assert_eq!(
            admit_native_envelope(ComputeBackend::Cpu, 0, 0, 1),
            Err(AcousticResponseNativeEnvelopeError::SourceMaterialization)
        );
    }
}
