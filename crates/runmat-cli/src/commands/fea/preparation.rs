use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use runmat_analysis_core::{AuthoredStudyInputBinding, RealizedStudy};
use runmat_analysis_execution::{
    prepare_geometry_materialization_program, prepare_geometry_materialization_request,
    GeometryMaterializationPreparation, PreparedGeometrySource,
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
use runmat_geometry_core::GeometryDefinitionSource;

use super::source::CheckedFeaSource;

pub(super) struct PreparedTypedStudy {
    pub root: PreparedStructuralStudyRoot,
    pub revision: ProgramRevision,
    pub realized: RealizedStudy,
    pub workers: u32,
}

#[derive(Clone, Debug)]
pub(super) struct FeaComputeAllocation {
    pub backend: runmat_analysis_fea::ComputeBackend,
    pub accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
}

pub(super) fn prepare_study(
    source: &CheckedFeaSource,
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
    let geometry_source = prepare_geometry_source(source, &resolved, &access)?;
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
    let workers = resolved
        .execution_policy
        .as_ref()
        .and_then(|policy| policy.resources.workers)
        .map_or(1, |range| range.maximum.max(1));
    let authored_study_digest = typed_study_digest(&realized, &resolved.material_catalog)?;
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
        workers,
    })
}

fn prepare_geometry_source(
    source: &CheckedFeaSource,
    resolved: &ResolvedAnalysisStudy,
    access: &ArtifactAccess,
) -> Result<PreparedGeometrySource> {
    match &resolved.geometry.source {
        GeometryDefinitionSource::Constructed(_) => Ok(PreparedGeometrySource::constructed()),
        GeometryDefinitionSource::Imported(imported) => {
            let path = physical_source_path(&source.source_root, &imported.logical_path)?;
            let bytes = std::fs::read(&path)
                .with_context(|| format!("read imported geometry {}", path.display()))?;
            if Digest::sha256(&bytes).bytes() != imported.content_digest.bytes() {
                bail!(
                    "imported geometry {} changed after typed source checking",
                    path.display()
                );
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("imported geometry file name is not valid UTF-8")?;
            PreparedGeometrySource::imported(
                name,
                &bytes,
                4 * 1024 * 1024,
                access,
            )
            .map_err(anyhow::Error::new)
        }
        GeometryDefinitionSource::Derived(_) => bail!(
            "derived geometry execution requires its exact source-geometry closure; this run surface does not yet compose that prerequisite"
        ),
        GeometryDefinitionSource::Generated(_) => bail!(
            "generated geometry execution requires its frozen callable product; this run surface does not yet compose that prerequisite"
        ),
    }
}

fn physical_source_path(source_root: &Path, logical_path: &str) -> Result<std::path::PathBuf> {
    let relative = logical_path
        .strip_prefix("project/")
        .context("geometry source is outside the checked project root")?;
    if relative.is_empty()
        || relative
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        bail!("geometry source path is not normalized");
    }
    Ok(relative
        .split('/')
        .fold(source_root.to_path_buf(), |path, part| path.join(part)))
}

fn program_revision(
    source: &CheckedFeaSource,
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

fn typed_study_digest(
    study: &RealizedStudy,
    materials: &BTreeMap<
        runmat_analysis_core::AuthoredMaterialRef,
        runmat_analysis_core::MaterialModel,
    >,
) -> Result<Digest> {
    Ok(Digest::sha256(serde_json::to_vec(&(
        study,
        materials.iter().collect::<Vec<_>>(),
    ))?))
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
