use runmat_analysis_core::{AuthoredStudyInputBinding, RealizedStudy};
use runmat_analysis_execution::{
    prepare_geometry_materialization_program, prepare_geometry_materialization_request,
    GeometryMaterializationPreparation,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;

use super::LocalFeaExecutionOptions;

use runmat_analysis_fea::admission;

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

/// How a prepared root executes an authored study.
#[derive(Clone, Debug)]
pub enum PreparationRole {
    /// The authored study as written. An electrothermal analysis runs as its heat-transfer
    /// side: conduction and heat transfer iterated to the two-way fixed point.
    Authored,
    /// A thermal analysis heated by a solved upstream loss field.
    Heated(runmat_analysis_fea::study::ElementHeatSource),
    /// The conduction side of an electrothermal analysis at the temperatures the heat case
    /// published (element means, canonical element order; the last time of a transient);
    /// publishes potential, current density and Joule loss.
    ElectrothermalConduction {
        element_temperature_k: Vec<f64>,
        operating_time_s: Option<f64>,
    },
}

pub async fn prepare_study(
    source: &runmat_core::CheckedFeaSessionSource,
    resolved: runmat_analysis_language::ResolvedAnalysisStudy,
    inputs: &[AuthoredStudyInputBinding],
    authority: &str,
    options: &LocalFeaExecutionOptions,
) -> Result<PreparedFeaStudy, String> {
    prepare_study_with_role(
        source,
        resolved,
        inputs,
        authority,
        options,
        PreparationRole::Authored,
    )
    .await
}

pub async fn prepare_study_with_role(
    source: &runmat_core::CheckedFeaSessionSource,
    resolved: runmat_analysis_language::ResolvedAnalysisStudy,
    inputs: &[AuthoredStudyInputBinding],
    authority: &str,
    options: &LocalFeaExecutionOptions,
    role: PreparationRole,
) -> Result<PreparedFeaStudy, String> {
    let allocation = super::run_admission::study_allocation(&resolved, options)?;
    let realized: RealizedStudy = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, inputs)
        .map_err(|error| error.to_string())?;
    let family = match &role {
        PreparationRole::ElectrothermalConduction { .. } => {
            if !admission::is_electrothermal(&realized.sequence) {
                return Err(
                    "the conduction role requires one fresh electrothermal analysis".into(),
                );
            }
            admission::AdmittedFamily::Electrical
        }
        // The same static admission `runmat check` reports, before any geometry or mesh work.
        _ => admission::admit_study(&realized).map_err(|issues| {
            issues
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        })?,
    };
    if family == admission::AdmittedFamily::Maxwell
        && realized.sequence.steps.iter().any(|step| {
            !matches!(&step.independent_domain,
                runmat_analysis_core::IndependentDomain::FrequencyHertz(frequencies) if frequencies.len() == 1)
        })
    {
        return Err("a prepared Maxwell harmonic case carries exactly one frequency; multi-frequency studies are fanned out first".into());
    }
    let (element_heat_source, operating_temperatures) = match role {
        PreparationRole::Heated(source) => (Some(source), None),
        PreparationRole::Authored => (None, None),
        PreparationRole::ElectrothermalConduction {
            element_temperature_k,
            operating_time_s,
        } => (None, Some((element_temperature_k, operating_time_s))),
    };
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
    let workers = allocation.workers;
    let cpu_threads_per_worker = allocation.threads;
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
        admission::AdmittedFamily::Structural => crate::StudyRootPlanning::Structural(
            runmat_analysis_fea::study::StructuralStudyPlanningPolicy {
                backend: options.backend,
                numerics: runmat_analysis_fea::study::StructuralNumericalPlanningPolicy::production(
                ),
            },
        ),
        admission::AdmittedFamily::Thermal => crate::StudyRootPlanning::Thermal(
            runmat_analysis_fea::study::ThermalStudyPlanningPolicy {
                backend: options.backend,
                element_heat_source,
            },
        ),
        admission::AdmittedFamily::Acoustic => crate::StudyRootPlanning::Acoustic(
            runmat_analysis_fea::study::AcousticStudyPlanningPolicy::standard(),
        ),
        admission::AdmittedFamily::AcousticResponse => crate::StudyRootPlanning::AcousticResponse(
            runmat_analysis_fea::study::AcousticResponseStudyPlanningPolicy::default(),
        ),
        admission::AdmittedFamily::Electrical => {
            let (element_temperature_k, operating_time_s) = match operating_temperatures {
                Some((temperatures, time)) => (Some(temperatures), time),
                None => (None, None),
            };
            crate::StudyRootPlanning::Electrical(
                runmat_analysis_fea::study::ElectricalStudyPlanningPolicy {
                    element_temperature_k,
                    operating_time_s,
                },
            )
        }
        admission::AdmittedFamily::Magnetic => crate::StudyRootPlanning::Magnetic(
            runmat_analysis_fea::study::MagneticStudyPlanningPolicy {},
        ),
        admission::AdmittedFamily::Maxwell => crate::StudyRootPlanning::Maxwell(
            runmat_analysis_fea::study::MaxwellStudyPlanningPolicy {},
        ),
        admission::AdmittedFamily::Fluid => {
            crate::StudyRootPlanning::Fluid(runmat_analysis_fea::study::FluidStudyPlanningPolicy {
                numerical: Default::default(),
                geometry: Default::default(),
            })
        }
        admission::AdmittedFamily::Nonlinear => {
            crate::StudyRootPlanning::NonlinearSolid(Box::new(crate::NonlinearSolidStudyPlanning {
                controls: runmat_analysis_fea::solve::nonlinear::NonlinearSolveOptions {
                    ..Default::default()
                },
                accepted_block_limit:
                    runmat_analysis_execution::DEFAULT_NONLINEAR_SOLID_ACCEPTED_BLOCK_LIMIT as u64,
            }))
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
        // Execution policy, not study definition: it travels with the root so every
        // placement plans each worker within it, and never enters the authored study digest.
        memory_per_worker_bytes: resolved
            .execution_policy
            .as_ref()
            .and_then(|policy| policy.resources.maximum_memory_bytes_per_worker),
        // The authored admission policy travels the same way, so the worker admits a strict
        // placement before it meshes or solves.
        accelerator_admission: resolved
            .execution_policy
            .as_ref()
            .map(|policy| policy.resources.accelerator_admission())
            .unwrap_or_default(),
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
    source.program_revision()
}

pub fn checked_fea_artifact_access(authority: &str) -> ArtifactAccess {
    runmat_analysis_execution::checked_fea_artifact_access(authority)
}
