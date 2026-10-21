use runmat_analysis_core::{AuthoredStudyInputBinding, RealizedStudy};
use runmat_analysis_execution::{
    prepare_geometry_materialization_program, prepare_geometry_materialization_request,
    GeometryMaterializationPreparation,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;

use super::LocalFeaExecutionOptions;

mod admission;

/// One authored study ready for any placement.
///
/// `root` is the unencoded portable root shared by every family; local and remote execution
/// both encode it with [`crate::prepare_structural_study_root`] and decode the same contract.
pub struct PreparedFeaStudy {
    pub root: crate::StructuralStudyRootPreparation,
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
    let realized: RealizedStudy = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, inputs)
        .map_err(|error| error.to_string())?;
    let family =
        admission::select_family(&realized.sequence, realized.model.electromagnetic.as_ref())?;
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
    let workers = options
        .workers
        .or_else(|| {
            resolved
                .execution_policy
                .as_ref()
                .and_then(|policy| policy.resources.workers)
                .map(|range| range.maximum.max(1))
        })
        .unwrap_or_else(|| (crate::machine_capacity::cpu_millicores() / 1_000).min(32));
    let cpu_threads_per_worker = crate::machine_capacity::study_allocation(
        workers,
        resolved
            .execution_policy
            .as_ref()
            .map(|policy| policy.resources.cpu_threads_per_worker),
    )?;
    let local_thread_count = u16::try_from(cpu_threads_per_worker)
        .map_err(|_| "CPU threads per worker exceed the FEA contract maximum".to_string())?;
    let local_execution =
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_thread_count(
            runmat_analysis_fea::planning::resource::LocalThreadCount::new(local_thread_count)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let authored_study_digest = runmat_analysis_fea::study::structural_study_source_digest(
        &realized,
        &resolved.material_catalog,
    )
    .map_err(|error| error.to_string())?;
    let input_materializations =
        super::input::prepare_input_materializations(source, &realized).await?;
    let frozen_project_digest = source
        .frozen_project()
        .map(|project| Digest::from_bytes(*project.graph_digest().bytes()));
    let planning = match family {
        admission::PreparedFamily::Structural => {
            crate::StudyRootPlanning::Structural(runmat_analysis_fea::study::StructuralStudyPlanningPolicy {
                backend: options.backend,
                formulation_requirement: resolved.study.formulation_requirement,
                numerics: runmat_analysis_fea::study::StructuralNumericalPlanningPolicy::production(),
            })
        }
        admission::PreparedFamily::Thermal => {
            crate::StudyRootPlanning::Thermal(runmat_analysis_fea::study::ThermalStudyPlanningPolicy {
                backend: options.backend,
                formulation_requirement: resolved.study.formulation_requirement,
            })
        }
        admission::PreparedFamily::Acoustic => {
            crate::StudyRootPlanning::Acoustic(runmat_analysis_fea::study::AcousticStudyPlanningPolicy {
                formulation_requirement: resolved.study.formulation_requirement,
                ..runmat_analysis_fea::study::AcousticStudyPlanningPolicy::experimental()
            })
        }
        admission::PreparedFamily::Electrical => {
            crate::StudyRootPlanning::Electrical(runmat_analysis_fea::study::ElectricalStudyPlanningPolicy {
                formulation_requirement: resolved.study.formulation_requirement,
            })
        }
        admission::PreparedFamily::Magnetic => {
            crate::StudyRootPlanning::Magnetic(runmat_analysis_fea::study::MagneticStudyPlanningPolicy {
                formulation_requirement: resolved.study.formulation_requirement,
            })
        }
        admission::PreparedFamily::Maxwell => {
            crate::StudyRootPlanning::Maxwell(runmat_analysis_fea::study::MaxwellStudyPlanningPolicy {
                formulation_requirement: resolved.study.formulation_requirement,
            })
        }
        admission::PreparedFamily::Fluid => {
            crate::StudyRootPlanning::Fluid(runmat_analysis_fea::study::FluidStudyPlanningPolicy {
                formulation_requirement: resolved.study.formulation_requirement,
                numerical: Default::default(),
                geometry: Default::default(),
            })
        }
        admission::PreparedFamily::Nonlinear => {
            crate::StudyRootPlanning::NonlinearSolid(crate::NonlinearSolidStudyPlanning {
                controls: runmat_analysis_fea::solve::nonlinear::NonlinearSolveOptions {
                    formulation_requirement: resolved.study.formulation_requirement,
                    ..Default::default()
                },
                accepted_block_limit:
                    runmat_analysis_execution::DEFAULT_NONLINEAR_SOLID_ACCEPTED_BLOCK_LIMIT as u64,
            })
        }
    };
    if let Some(kind) = crate::StudyFamilyKind::of(&planning) {
        if !input_materializations.is_empty() && !kind.admits_input_materializations() {
            return Err(format!(
                "authored {} execution does not support source materializations",
                kind.label()
            ));
        }
        if planning.backend() != runmat_analysis_fea::ComputeBackend::Cpu
            || !options.accelerators.is_empty()
        {
            return Err(format!(
                "authored {} execution has no admitted accelerator envelope",
                kind.label()
            ));
        }
    }
    let initial_fields = runmat_runtime::analysis::storage::resolve_initial_field_references(
        realized.initial_field_references(),
        runmat_analysis_fea::planning::resource::initial_field_source_ceiling(&realized.sequence)
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let root = crate::StructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access,
        authored_study_digest,
        frozen_project_digest,
        study: realized.clone(),
        materials: resolved.material_catalog,
        planning,
        local_execution,
        analysis_accelerator_requests: options.accelerators.clone(),
        initial_fields,
        input_materializations,
        geometry_meshing,
    };
    Ok(PreparedFeaStudy {
        root,
        workers,
        revision,
        realized,
    })
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
