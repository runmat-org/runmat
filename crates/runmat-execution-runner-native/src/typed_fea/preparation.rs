use runmat_analysis_core::{AuthoredStudyInputBinding, RealizedStudy};
use runmat_analysis_execution::{
    prepare_geometry_materialization_program, prepare_geometry_materialization_request,
    GeometryMaterializationPreparation,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;

use super::LocalFeaExecutionOptions;

pub struct PreparedFeaStudy {
    pub root: crate::PreparedStructuralStudyRoot,
    pub workers: u32,
    pub revision: ProgramRevision,
    pub realized: RealizedStudy,
}

pub async fn prepare_study(
    source: &runmat_core::CheckedFeaSessionSource,
    resolved: runmat_analysis_language::ResolvedAnalysisStudy,
    inputs: &[AuthoredStudyInputBinding],
    authority: &str,
    options: &LocalFeaExecutionOptions,
) -> Result<PreparedFeaStudy, String> {
    ensure_structural_execution_boundary(&resolved)?;
    let realized: RealizedStudy = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, inputs)
        .map_err(|error| error.to_string())?;
    let revision = program_revision(source)?;
    let access = checked_fea_artifact_access(authority);
    let geometry_source = super::geometry::prepare_geometry_source(
        source,
        &resolved.geometry,
        &resolved.geometry_dependencies,
        &realized.geometry.bindings,
        &revision,
        &access,
    )
    .await?;
    let provider = crate::native_geometry_provider_profile().ok_or_else(|| {
        "typed FEA geometry requires a RunMat build with the OCCT feature enabled".to_string()
    })?;
    let geometry_request = prepare_geometry_materialization_request(
        resolved.geometry.clone(),
        realized.geometry.bindings.clone(),
        &geometry_source,
        provider,
        GeometryMaterializationPreparation::default(),
    )
    .map_err(|error| error.to_string())?;
    let geometry = prepare_geometry_materialization_program(
        geometry_request.clone(),
        revision.clone(),
        &access,
        &geometry_source.roots,
    )
    .map_err(|error| error.to_string())?;
    let meshing_request = crate::resolve_native_meshing_request(
        &geometry_request.import_options,
        resolved.mesh.settings,
    )
    .map_err(|error| error.to_string())?;
    let partition = crate::NativeMeshingPartitionPolicy::default();
    let geometry_meshing =
        crate::prepare_geometry_meshing_root(crate::GeometryMeshingRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access.clone(),
            geometry,
            geometry_source_objects: geometry_source.objects,
            meshing_request,
            preferred_edges_per_partition: partition.preferred_edges_per_partition,
            preferred_faces_per_partition: partition.preferred_faces_per_partition,
        })
        .map_err(|error| error.to_string())?;
    let workers = resolved
        .execution_policy
        .as_ref()
        .and_then(|policy| policy.resources.workers)
        .map_or(1, |range| range.maximum.max(1));
    let authored_study_digest = runmat_analysis_fea::study::structural_study_source_digest(
        &realized,
        &resolved.material_catalog,
    )
    .map_err(|error| error.to_string())?;
    let input_materializations =
        super::input::prepare_input_materializations(source, &realized).await?;
    let root = crate::prepare_structural_study_root(crate::StructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access,
        authored_study_digest,
        frozen_project_digest: source
            .frozen_project()
            .map(|project| Digest::from_bytes(*project.graph_digest().bytes())),
        study: realized.clone(),
        materials: resolved.material_catalog,
        planning: runmat_analysis_fea::study::StructuralStudyPlanningPolicy {
            backend: options.backend,
            formulation_requirement: resolved.study.formulation_requirement,
            numerics: runmat_analysis_fea::study::StructuralNumericalPlanningPolicy::production(),
        },
        analysis_accelerator_requests: options.accelerators.clone(),
        input_materializations,
        geometry_meshing,
    })
    .map_err(|error| error.to_string())?;
    Ok(PreparedFeaStudy {
        root,
        workers,
        revision,
        realized,
    })
}

fn ensure_structural_execution_boundary(
    resolved: &runmat_analysis_language::ResolvedAnalysisStudy,
) -> Result<(), String> {
    let unsupported = resolved
        .study
        .sequence
        .steps
        .iter()
        .find(|step| step.family != runmat_analysis_core::PhysicsFamily::Structural);
    if let Some(step) = unsupported {
        return Err(format!(
            "RM.FEA.TYPED_EXECUTION.UNSUPPORTED_PHYSICS_FAMILY: typed native execution does not yet dispatch {:?}/{:?} for step `{}`",
            step.family, step.procedure, step.step_id.0
        ));
    }
    Ok(())
}

pub(super) fn program_revision(
    source: &runmat_core::CheckedFeaSessionSource,
) -> Result<ProgramRevision, String> {
    let (graph, source_digest) = match source.frozen_project() {
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
                .ok_or_else(|| "checked root module is absent".to_string())?;
            (
                Digest::sha256(graph),
                Digest::from_bytes(root.source.source_digest.0),
            )
        }
    };
    ProgramRevision::new(graph, source_digest, source.program_environment().clone())
        .map_err(|error| error.to_string())
}

pub fn checked_fea_artifact_access(authority: &str) -> ArtifactAccess {
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
