use super::source::NativeCheckedFeaSource;
use anyhow::{Context, Result};
use runmat_analysis_core::{AuthoredStudyInputBinding, RealizedStudy};
use runmat_analysis_execution::{
    prepare_geometry_materialization_program, prepare_geometry_materialization_request,
    GeometryMaterializationPreparation,
};
use runmat_analysis_language::ResolvedAnalysisStudy;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::{
    native_geometry_provider_profile, prepare_geometry_meshing_root, prepare_structural_study_root,
    resolve_native_meshing_request, GeometryMeshingRootPreparation, NativeMeshingPartitionPolicy,
    PreparedStructuralStudyRoot, StructuralStudyRootPreparation,
};

pub(super) struct PreparedTypedStudy {
    pub root: PreparedStructuralStudyRoot,
    pub revision: ProgramRevision,
    pub realized: RealizedStudy,
}

#[derive(Clone, Debug)]
pub(super) struct FeaComputeAllocation {
    pub backend: runmat_analysis_fea::ComputeBackend,
    pub accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
}

pub(super) async fn prepare_study(
    source: &NativeCheckedFeaSource,
    resolved: ResolvedAnalysisStudy,
    inputs: &[AuthoredStudyInputBinding],
    authority: &str,
    compute: &FeaComputeAllocation,
    config: &RunMatRuntimeConfig,
) -> Result<PreparedTypedStudy> {
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, inputs)
        .context("realize typed FEA study inputs")?;
    let revision = program_revision(source, config)?;
    let access = artifact_access(authority);
    let geometry_source = super::geometry_source::prepare_geometry_source(
        source,
        &resolved.geometry,
        &resolved.geometry_dependencies,
        &realized.geometry.bindings,
        &revision,
        &access,
        config,
    )
    .await?;
    let provider = native_geometry_provider_profile()
        .context("typed FEA geometry requires a RunMat build with the OCCT feature enabled")?;
    let geometry_request = prepare_geometry_materialization_request(
        resolved.geometry.clone(),
        realized.geometry.bindings.clone(),
        &geometry_source,
        provider,
        GeometryMaterializationPreparation::default(),
    )?;
    let geometry = prepare_geometry_materialization_program(
        geometry_request.clone(),
        revision.clone(),
        &access,
        &geometry_source.roots,
    )?;
    let meshing_request =
        resolve_native_meshing_request(&geometry_request.import_options, resolved.mesh.settings)?;
    let partition = NativeMeshingPartitionPolicy::default();
    let geometry_meshing = prepare_geometry_meshing_root(GeometryMeshingRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        geometry,
        geometry_source_objects: geometry_source.objects,
        meshing_request,
        preferred_edges_per_partition: partition.preferred_edges_per_partition,
        preferred_faces_per_partition: partition.preferred_faces_per_partition,
    })?;
    let authored_study_digest = runmat_analysis_fea::study::structural_study_source_digest(
        &realized,
        &resolved.material_catalog,
    )?;
    let root = prepare_structural_study_root(StructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access,
        authored_study_digest,
        frozen_project_digest: source
            .frozen_project
            .as_ref()
            .map(|project| Digest::from_bytes(*project.graph_digest().bytes())),
        study: realized.clone(),
        materials: resolved.material_catalog,
        planning: runmat_analysis_fea::study::StructuralStudyPlanningPolicy {
            backend: compute.backend,
            formulation_requirement: realized.formulation_requirement,
            numerics: runmat_analysis_fea::study::StructuralNumericalPlanningPolicy::production(),
        },
        analysis_accelerator_requests: compute.accelerators.clone(),
        geometry_meshing,
    })?;
    Ok(PreparedTypedStudy {
        root,
        revision,
        realized,
    })
}

fn program_revision(
    source: &NativeCheckedFeaSource,
    config: &RunMatRuntimeConfig,
) -> Result<ProgramRevision> {
    let (graph, source_digest) = match &source.frozen_project {
        Some(project) => (
            Digest::from_bytes(*project.graph_digest().bytes()),
            Digest::from_bytes(*project.source_revision().bytes()),
        ),
        None => {
            let mut graph = Vec::new();
            for (path, digest) in source.program.bound.graph.source_digests() {
                graph.extend_from_slice(path.as_str().as_bytes());
                graph.push(0);
                graph.extend_from_slice(&digest.0);
            }
            let root = source
                .program
                .bound
                .graph
                .modules
                .get(&source.root_module)
                .context("checked root module is absent")?;
            (
                Digest::sha256(graph),
                Digest::from_bytes(root.source.source_digest.0),
            )
        }
    };
    Ok(ProgramRevision::new(
        graph,
        source_digest,
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?)
}

pub(super) fn artifact_access(authority: &str) -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: authority.to_string(),
        encryption_context: Digest::sha256(
            [
                b"runmat-typed-fea-access/v1\0".as_slice(),
                authority.as_bytes(),
            ]
            .concat(),
        ),
    }
}
