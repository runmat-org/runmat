use runmat_analysis_core::{AuthoredStudyInputBinding, RealizedStudy};
use runmat_analysis_execution::{
    prepare_geometry_materialization_program, prepare_geometry_materialization_request,
    GeometryMaterializationPreparation,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;

use super::LocalFeaExecutionOptions;

mod acoustic;
mod admission;
mod electrical;
mod fluid;
mod nonlinear;
pub use nonlinear::PreparedAuthoredNonlinearSolidStudy;
mod magnetic;
mod maxwell;
mod thermal;
pub use acoustic::PreparedAuthoredAcousticStudy;
pub use electrical::PreparedAuthoredElectricalStudy;
pub use fluid::PreparedAuthoredFluidStudy;
pub use magnetic::PreparedAuthoredMagneticStudy;
pub use maxwell::PreparedAuthoredMaxwellStudy;
pub use thermal::PreparedAuthoredThermalStudy;

pub enum PreparedFeaStudyRoot {
    Acoustic(PreparedAuthoredAcousticStudy),
    Electrical(PreparedAuthoredElectricalStudy),
    Magnetic(PreparedAuthoredMagneticStudy),
    Maxwell(PreparedAuthoredMaxwellStudy),
    Fluid(PreparedAuthoredFluidStudy),
    Nonlinear(Box<PreparedAuthoredNonlinearSolidStudy>),
    Structural(crate::PreparedStructuralStudyRoot),
    Thermal(PreparedAuthoredThermalStudy),
}

pub struct PreparedFeaStudy {
    pub root: PreparedFeaStudyRoot,
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
    if matches!(
        family,
        admission::PreparedFamily::Acoustic
            | admission::PreparedFamily::Nonlinear
            | admission::PreparedFamily::Fluid
            | admission::PreparedFamily::Magnetic
            | admission::PreparedFamily::Maxwell
    ) && (options.backend != runmat_analysis_fea::ComputeBackend::Cpu
        || !options.accelerators.is_empty())
    {
        return Err(
            "authored J2/acoustic/fluid/magnetic/Maxwell execution has no admitted accelerator envelope"
                .into(),
        );
    }
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
    let root = match family {
        admission::PreparedFamily::Nonlinear => {
            if !input_materializations.is_empty() {
                return Err(
                    "authored J2 execution does not yet admit source materializations".into(),
                );
            }
            PreparedFeaStudyRoot::Nonlinear(Box::new(PreparedAuthoredNonlinearSolidStudy {
                geometry_meshing,
                authored_study_digest,
                frozen_project_digest,
                study: realized.clone(),
                materials: resolved.material_catalog,
                program_revision: revision.clone(),
                artifact_access: access,
                input_materializations,
                local_execution,
                analysis_accelerator_requests: options.accelerators.clone(),
                controls: runmat_analysis_fea::solve::nonlinear::NonlinearSolveOptions {
                    formulation_requirement: resolved.study.formulation_requirement,
                    ..Default::default()
                },
                accepted_block_limit:
                    runmat_analysis_execution::DEFAULT_NONLINEAR_SOLID_ACCEPTED_BLOCK_LIMIT,
            }))
        }
        admission::PreparedFamily::Fluid => {
            if !input_materializations.is_empty() {
                return Err(
                    "authored fluid execution does not support source materializations".into(),
                );
            }
            PreparedFeaStudyRoot::Fluid(PreparedAuthoredFluidStudy {
                geometry_meshing,
                authored_study_digest,
                frozen_project_digest,
                study: realized.clone(),
                materials: resolved.material_catalog,
                program_revision: revision.clone(),
                artifact_access: access,
                input_materializations,
                planning: runmat_analysis_fea::study::FluidStudyPlanningPolicy {
                    formulation_requirement: resolved.study.formulation_requirement,
                    numerical: Default::default(),
                    geometry: Default::default(),
                },
                local_execution,
                analysis_accelerator_requests: options.accelerators.clone(),
            })
        }
        admission::PreparedFamily::Maxwell => {
            if !input_materializations.is_empty() {
                return Err(
                    "authored Maxwell execution does not support source materializations".into(),
                );
            }
            PreparedFeaStudyRoot::Maxwell(PreparedAuthoredMaxwellStudy {
                geometry_meshing,
                authored_study_digest,
                frozen_project_digest,
                study: realized.clone(),
                materials: resolved.material_catalog,
                program_revision: revision.clone(),
                artifact_access: access,
                input_materializations,
                planning: runmat_analysis_fea::study::MaxwellStudyPlanningPolicy {
                    formulation_requirement: resolved.study.formulation_requirement,
                },
                local_execution,
                analysis_accelerator_requests: options.accelerators.clone(),
            })
        }
        admission::PreparedFamily::Magnetic => {
            if !input_materializations.is_empty() {
                return Err(
                    "authored magnetic execution does not support source materializations".into(),
                );
            }
            PreparedFeaStudyRoot::Magnetic(PreparedAuthoredMagneticStudy {
                geometry_meshing,
                authored_study_digest,
                frozen_project_digest,
                study: realized.clone(),
                materials: resolved.material_catalog,
                program_revision: revision.clone(),
                artifact_access: access,
                input_materializations,
                planning: runmat_analysis_fea::study::MagneticStudyPlanningPolicy {
                    formulation_requirement: resolved.study.formulation_requirement,
                },
                local_execution,
                analysis_accelerator_requests: options.accelerators.clone(),
            })
        }
        admission::PreparedFamily::Electrical => {
            if options.backend != runmat_analysis_fea::ComputeBackend::Cpu
                || !options.accelerators.is_empty()
                || !input_materializations.is_empty()
            {
                return Err(
                    "authored electrical execution admits only CPU with no source materializations"
                        .into(),
                );
            }
            PreparedFeaStudyRoot::Electrical(PreparedAuthoredElectricalStudy {
                geometry_meshing,
                authored_study_digest,
                frozen_project_digest,
                study: realized.clone(),
                materials: resolved.material_catalog,
                program_revision: revision.clone(),
                artifact_access: access,
                input_materializations,
                planning: runmat_analysis_fea::study::ElectricalStudyPlanningPolicy {
                    formulation_requirement: resolved.study.formulation_requirement,
                },
                local_execution,
                analysis_accelerator_requests: options.accelerators.clone(),
            })
        }
        admission::PreparedFamily::Acoustic => {
            let input_materializations = input_materializations
                .into_iter()
                .map(|materialization| materialization.rebind(&access).map_err(|e| e.to_string()))
                .collect::<Result<Vec<_>, String>>()?;
            crate::input_materialization_contract::validate_input_materialization_plan_inventory(
                &realized,
                &input_materializations,
                &revision,
            )
            .map_err(|e| e.to_string())?;
            PreparedFeaStudyRoot::Acoustic(PreparedAuthoredAcousticStudy {
                geometry_meshing,
                authored_study_digest,
                frozen_project_digest,
                study: realized.clone(),
                materials: resolved.material_catalog,
                program_revision: revision.clone(),
                artifact_access: access,
                input_materializations,
                planning: runmat_analysis_fea::study::AcousticStudyPlanningPolicy {
                    formulation_requirement: resolved.study.formulation_requirement,
                    ..runmat_analysis_fea::study::AcousticStudyPlanningPolicy::experimental()
                },
                local_execution,
                analysis_accelerator_requests: options.accelerators.clone(),
            })
        }
        admission::PreparedFamily::Thermal => {
            let input_materializations = input_materializations
                .into_iter()
                .map(|materialization| {
                    materialization
                        .rebind(&access)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?;
            crate::input_materialization_contract::validate_input_materialization_plan_inventory(
                &realized,
                &input_materializations,
                &revision,
            )
            .map_err(|error| error.to_string())?;
            PreparedFeaStudyRoot::Thermal(PreparedAuthoredThermalStudy {
                geometry_meshing,
                authored_study_digest,
                frozen_project_digest,
                study: realized.clone(),
                materials: resolved.material_catalog,
                program_revision: revision.clone(),
                artifact_access: access,
                input_materializations,
                planning: runmat_analysis_fea::study::ThermalStudyPlanningPolicy {
                    backend: options.backend,
                    formulation_requirement: resolved.study.formulation_requirement,
                },
                local_execution,
                analysis_accelerator_requests: options.accelerators.clone(),
            })
        }
        admission::PreparedFamily::Structural => PreparedFeaStudyRoot::Structural(
            crate::prepare_structural_study_root(crate::StructuralStudyRootPreparation {
                program_revision: revision.clone(),
                artifact_access: access,
                authored_study_digest,
                frozen_project_digest,
                study: realized.clone(),
                materials: resolved.material_catalog,
                planning: runmat_analysis_fea::study::StructuralStudyPlanningPolicy {
                    backend: options.backend,
                    formulation_requirement: resolved.study.formulation_requirement,
                    numerics:
                        runmat_analysis_fea::study::StructuralNumericalPlanningPolicy::production(),
                },
                local_execution,
                analysis_accelerator_requests: options.accelerators.clone(),
                initial_fields:
                    runmat_runtime::analysis::storage::resolve_initial_field_references(
                        realized.initial_field_references(),
                        runmat_analysis_fea::planning::resource::initial_field_source_ceiling(
                            &realized.sequence,
                        )
                        .map_err(|error| error.to_string())?,
                    )
                    .map_err(|error| error.to_string())?,
                input_materializations,
                geometry_meshing,
            })
            .map_err(|error| error.to_string())?,
        ),
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
