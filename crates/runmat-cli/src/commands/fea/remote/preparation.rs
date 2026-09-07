use anyhow::{Context, Result};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{
    ArtifactAccess, ProgramExecutionDescriptor, ProgramExecutionInputs, ProgramExecutionObject,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner_native::{
    prepare_exact_meshing_root, prepare_exact_structural_study_root,
    prepare_exact_structural_sweep_root, ExactMeshingRootPreparation,
    ExactStructuralStudyRootPreparation, ExactStructuralSweepRootPreparation,
    NativeMeshingContactDefinition, NativeMeshingPartitionPolicy, PreparedExactStructuralStudyRoot,
    PreparedExactStructuralSweepRoot,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_runtime::analysis::{
    analysis_prepare_structural_study_intent_op, AnalysisStudySpec, PreparedStructuralStudyIntent,
};
use runmat_runtime::operations::OperationContext;

use crate::commands::job::submit::PreparedRemoteExecution;

pub(super) struct RemoteStudySource {
    pub(super) intent: PreparedStructuralStudyIntent,
    pub(super) geometry_name: String,
    pub(super) geometry_bytes: Vec<u8>,
    pub(super) signal_materializations:
        Vec<runmat_analysis_execution::PreparedSignalMaterialization>,
}

pub(super) fn prepare_study_source(
    spec: &AnalysisStudySpec,
    context: OperationContext,
    signal_materializations: Vec<runmat_analysis_execution::PreparedSignalMaterialization>,
) -> Result<RemoteStudySource> {
    let geometry_path = std::path::PathBuf::from(&spec.geometry.source.path);
    let geometry_name = geometry_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("FEA geometry file name is not valid UTF-8")?
        .to_owned();
    let intent = analysis_prepare_structural_study_intent_op(spec, context)
        .map_err(super::super::report_operation_error)?;
    validate_remote_study_backend(single_workload(&intent)?.backend)?;
    let geometry_bytes = std::fs::read(&geometry_path)
        .with_context(|| format!("read FEA geometry {}", geometry_path.display()))?;
    if Digest::sha256(&geometry_bytes) != intent.geometry_source_digest() {
        anyhow::bail!("FEA geometry differs from its authored source SHA-256");
    }
    Ok(RemoteStudySource {
        intent,
        geometry_name,
        geometry_bytes,
        signal_materializations,
    })
}

pub(super) fn prepare_sweep_root(
    studies: &[RemoteStudySource],
    revision: ProgramRevision,
    authority: &str,
    identity: runmat_analysis_fea::execution::SweepIdentity,
    fail_fast: bool,
) -> Result<PreparedExactStructuralSweepRoot> {
    let prepared = studies
        .iter()
        .map(|study| {
            prepare_root(
                &study.intent,
                revision.clone(),
                authority,
                &study.geometry_name,
                &study.geometry_bytes,
                study.signal_materializations.clone(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(prepare_exact_structural_sweep_root(
        ExactStructuralSweepRootPreparation {
            program_revision: revision,
            artifact_access: artifact_access(authority),
            identity,
            fail_fast,
            studies: prepared,
        },
    )?)
}

pub(super) fn prepare_root(
    study: &PreparedStructuralStudyIntent,
    revision: ProgramRevision,
    authority: &str,
    geometry_name: &str,
    geometry_bytes: &[u8],
    signal_materializations: Vec<runmat_analysis_execution::PreparedSignalMaterialization>,
) -> Result<PreparedExactStructuralStudyRoot> {
    let access = artifact_access(authority);
    let import_options = ExactCadImportOptions {
        source_units: study.geometry_units(),
        ..ExactCadImportOptions::default()
    };
    let meshing_request = authored_meshing_request(study)?;
    let contact_definitions = study
        .meshing_contacts()
        .iter()
        .map(|contact| NativeMeshingContactDefinition {
            side_a_face_selectors: contact.side_a_face_selectors.clone(),
            side_b_face_selectors: contact.side_b_face_selectors.clone(),
        })
        .collect();
    let partition = NativeMeshingPartitionPolicy::default();
    let exact_meshing = prepare_exact_meshing_root(ExactMeshingRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        source_name: geometry_name,
        source_bytes: geometry_bytes,
        import_options,
        contact_definitions,
        meshing_request,
        preferred_edges_per_partition: partition.preferred_edges_per_partition,
        preferred_faces_per_partition: partition.preferred_faces_per_partition,
        maximum_chunk_bytes: 4 * 1024 * 1024,
    })?;
    Ok(prepare_exact_structural_study_root(
        ExactStructuralStudyRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access,
            authored_study_digest: study.authored_study_digest(),
            frozen_project_digest: Some(*revision.graph_digest()),
            geometry_source_digest: study.geometry_source_digest(),
            workload_template: single_workload(study)?.clone(),
            signal_materializations,
            exact_meshing,
        },
    )?)
}

fn single_workload(
    study: &PreparedStructuralStudyIntent,
) -> Result<runmat_analysis_fea::execution::StructuralLinearWorkload> {
    let workloads = study.workloads();
    if workloads.len() != 1 {
        anyhow::bail!(
            "remote exact-study packaging requires one procedure step until the sequence root is enabled"
        );
    }
    workloads
        .into_values()
        .next()
        .context("remote structural study contains no workload")
}

pub(super) fn prepare_remote_execution(
    prepared: PreparedExactStructuralStudyRoot,
    revision: ProgramRevision,
    bundle_archive: Vec<u8>,
) -> Result<PreparedRemoteExecution> {
    let objects = prepared.transport_objects()?;
    let request = prepared.into_request();
    prepare_remote_execution_request(request, objects, revision, bundle_archive)
}

pub(super) fn prepare_remote_sweep_execution(
    prepared: PreparedExactStructuralSweepRoot,
    revision: ProgramRevision,
    bundle_archive: Vec<u8>,
) -> Result<PreparedRemoteExecution> {
    let objects = prepared.transport_objects();
    let request = prepared.into_request();
    prepare_remote_execution_request(request, objects, revision, bundle_archive)
}

fn prepare_remote_execution_request(
    request: runmat_execution_artifact::ProgramExecutionRequest,
    objects: Vec<ProgramExecutionObject>,
    revision: ProgramRevision,
    bundle_archive: Vec<u8>,
) -> Result<PreparedRemoteExecution> {
    let descriptor = serde_json::to_vec(&ProgramExecutionDescriptor {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: request.recipe,
        artifact: request.artifact,
        function: request.function,
        requested_outputs: request.requested_outputs,
    })?;
    let inputs = ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        arguments: request.arguments,
        objects: Vec::new(),
    };
    Ok(PreparedRemoteExecution {
        revision,
        bundle_archive,
        descriptor,
        inputs,
        objects,
    })
}

pub(super) fn authored_meshing_request(
    study: &PreparedStructuralStudyIntent,
) -> Result<runmat_meshing_core::MeshingRequest> {
    study
        .meshing_request()
        .cloned()
        .context("remote exact-CAD studies require an authored meshing request")
}

pub(super) fn artifact_access(authority: &str) -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: authority.into(),
        encryption_context: Digest::sha256(
            [
                b"runmat-analysis-access/v1\0".as_slice(),
                authority.as_bytes(),
            ]
            .concat(),
        ),
    }
}

fn validate_remote_study_backend(backend: runmat_analysis_fea::ComputeBackend) -> Result<()> {
    match backend {
        runmat_analysis_fea::ComputeBackend::Cpu => Ok(()),
        runmat_analysis_fea::ComputeBackend::Gpu => anyhow::bail!(
            "remote structural studies do not yet advertise a typed accelerator allocation; use the CPU backend or local execution"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::validate_remote_study_backend;

    #[test]
    fn remote_study_rejects_unallocated_accelerator_execution() {
        assert!(validate_remote_study_backend(runmat_analysis_fea::ComputeBackend::Cpu).is_ok());
        assert!(validate_remote_study_backend(runmat_analysis_fea::ComputeBackend::Gpu).is_err());
    }
}
