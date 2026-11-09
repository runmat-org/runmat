//! Placement-neutral composition of every non-structural authored family.
//!
//! Local and remote execution decode the same portable study root and run exact geometry and
//! meshing through their own placement. They then share this planning, coordinator dispatch and
//! completion encoding, so a family behaves identically wherever its tasks run.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{AnalysisModel, AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_analysis_execution::{
    AcousticResponseStudyPreparation, AcousticStudyPreparation, AnalysisCapabilityRequirement,
    AnalysisOutputPolicy, CompletedAcousticResponseStudy, CompletedAcousticStudy,
    CompletedElectricalStudy, CompletedMagneticStudy, CompletedMaxwellStudy,
    CompletedNonlinearSolidStudy, CompletedThermalStudy, ElectricalStudyPreparation,
    FluidStudyPreparationParts, MagneticStudyPreparation, MaxwellStudyPreparation,
    NonlinearSolidStudyPreparation, PlannedFluidStudyPreparation, ThermalStudyPreparation,
    ThermalStudyScope,
};
use runmat_analysis_fea::execution::InitialFieldBinding;
use runmat_analysis_fea::planning::resource::LocalExecutionPlan;
use runmat_analysis_fea::study::{
    build_electrical_study_product, build_magnetic_study_product, build_maxwell_study_product,
    build_nonlinear_solid_study_product, compile_analysis_study,
    plan_acoustic_response_study_product, plan_acoustic_study_product,
    plan_electrical_study_product, plan_thermal_study_product_with_fields,
    AnalysisStudyCompilationInput, ElectricalStudyProductInput, MagneticStudyProductInput,
    MaxwellStudyProductInput, NonlinearSolidStudyProductInput,
};
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{PublishedObjectCatalog, ValueRef};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, LogicalObject};
use runmat_geometry_core::{GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use crate::analysis_task_pool::AnalysisTaskPool;
use crate::{NativeExecutionError, NativeExecutionResult, StudyRootPlanning};

/// The authored non-structural families that share this composition.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StudyFamilyKind {
    Thermal,
    Electrical,
    Magnetic,
    Maxwell,
    Acoustic,
    AcousticResponse,
    Fluid,
    NonlinearSolid,
}

impl StudyFamilyKind {
    pub fn of(planning: &StudyRootPlanning) -> Option<Self> {
        Some(match planning {
            StudyRootPlanning::Structural(_) => return None,
            StudyRootPlanning::Thermal(_) => Self::Thermal,
            StudyRootPlanning::Electrical(_) => Self::Electrical,
            StudyRootPlanning::Magnetic(_) => Self::Magnetic,
            StudyRootPlanning::Maxwell(_) => Self::Maxwell,
            StudyRootPlanning::Acoustic(_) => Self::Acoustic,
            StudyRootPlanning::AcousticResponse(_) => Self::AcousticResponse,
            StudyRootPlanning::Fluid(_) => Self::Fluid,
            StudyRootPlanning::NonlinearSolid(_) => Self::NonlinearSolid,
        })
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Thermal => "thermal",
            Self::Electrical => "electrical",
            Self::Magnetic => "magnetic",
            Self::Maxwell => "Maxwell",
            Self::Acoustic => "acoustic",
            Self::AcousticResponse => "acoustic response",
            Self::Fluid => "fluid",
            Self::NonlinearSolid => "nonlinear solid",
        }
    }

    /// Families whose authored signals or spectra are generated before planning.
    pub const fn admits_input_materializations(self) -> bool {
        matches!(self, Self::Thermal | Self::Acoustic)
    }
}

/// Exact geometry, mesh and authored source for one family study after meshing completed.
pub(crate) struct FamilyStudyContext<'a> {
    pub study: &'a RealizedStudy,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub planning: &'a StudyRootPlanning,
    pub document: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub mesh: &'a SolverMeshArtifact,
    pub initial_fields: &'a [InitialFieldBinding],
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: &'a ArtifactAccess,
    pub program_revision: &'a ProgramRevision,
    pub local_execution: LocalExecutionPlan,
    /// Reads the memory one analysis worker of this run may use (locally: the run's pool less
    /// what the coordinating process holds). Planning calls it once, after the family compile,
    /// so the read sees the coordinator's compiled study. Path selection, the output policy and
    /// the worker envelope all use that one value.
    pub worker_memory: &'a dyn Fn() -> Result<u64, String>,
}

/// Coordinator preparation for one family, ready for any analysis task pool.
pub(crate) enum FamilyStudyPreparation {
    Thermal(Box<ThermalStudyPreparation>),
    Electrical(Box<ElectricalStudyPreparation>),
    Magnetic(Box<MagneticStudyPreparation>),
    Maxwell(Box<MaxwellStudyPreparation>),
    Acoustic(Box<AcousticStudyPreparation>),
    AcousticResponse(Box<AcousticResponseStudyPreparation>),
    Fluid(Box<PlannedFluidStudyPreparation>),
    NonlinearSolid(Box<NonlinearSolidStudyPreparation>),
}

pub(crate) struct PlannedFamilyStudy {
    pub preparation: FamilyStudyPreparation,
    /// Authored step envelope; local placement sizes its worker processes from it.
    pub maximum_resident_bytes: u64,
    pub maximum_wall_millis: u64,
    /// The run's worker memory, as read once during planning.
    pub worker_memory_bytes: u64,
}

impl PlannedFamilyStudy {
    /// Process-isolated task envelope derived from the authored step accuracy resources. Its
    /// memory is the effective ceiling, min(authored, the run's worker memory).
    pub fn local_resources(
        &self,
        local_execution: LocalExecutionPlan,
        worker_memory_bytes: u64,
    ) -> Result<ResourceRequest, String> {
        let memory = self.maximum_resident_bytes.min(worker_memory_bytes);
        let resources = ResourceRequest {
            cpu_millicores: local_execution
                .checked_cpu_millicores()
                .map_err(|error| error.to_string())?,
            memory_bytes: memory,
            scratch_bytes: memory,
            max_wall_millis: self.maximum_wall_millis,
            max_artifact_bytes: ObjectInventoryLimits::admitted(memory).max_total_bytes,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
        };
        resources.validate().map_err(|error| error.to_string())?;
        Ok(resources)
    }
}

/// Compile and plan one family study over its exact mesh.
///
/// `apply_inputs` seals generated signals or spectra into the compiled model; it is invoked only
/// for families that admit input materializations. Planning failures are study failures.
pub(crate) fn plan_family_study(
    context: FamilyStudyContext<'_>,
    apply_inputs: &mut dyn FnMut(&mut AnalysisModel) -> Result<(), String>,
) -> Result<PlannedFamilyStudy, String> {
    let kind = StudyFamilyKind::of(context.planning)
        .ok_or("structural planning is not a family composition")?;
    if !kind.admits_input_materializations() && !context.initial_fields.is_empty() {
        return Err(format!(
            "authored {} execution does not admit initial fields",
            kind.label()
        ));
    }
    // One read per run, at the first use after the compile.
    let read = std::cell::Cell::new(None::<u64>);
    let worker_memory = || -> Result<u64, String> {
        if let Some(bytes) = read.get() {
            return Ok(bytes);
        }
        let bytes = (context.worker_memory)()?;
        read.set(Some(bytes));
        Ok(bytes)
    };
    let output_policy = || admitted_output_policy(context.study, worker_memory()?);
    // Every family plans within the run's admitted budget: each step's resident ceiling becomes
    // min(authored, the run's worker memory). A plan then selects a solve path that fits the
    // worker (the proving factor or an iterative path) or fails here, before any solve, naming
    // the memory it needs and the memory admitted. The worker envelope uses the same value, so
    // no planned task can ask the pool for more than it holds.
    let admit = |steps: &mut [runmat_analysis_core::ProcedureStep]| -> Result<(), String> {
        let machine = worker_memory()?;
        for step in steps {
            let budget = step.accuracy.resources.resident_byte_budget().min(machine);
            step.accuracy.resources.set_resident_byte_budget(budget);
        }
        Ok(())
    };
    // Families whose product builder compiles the study itself plan from an admitted copy.
    let admitted_study = || -> Result<RealizedStudy, String> {
        let mut study = context.study.clone();
        admit(&mut study.sequence.steps)?;
        Ok(study)
    };
    let capabilities = BTreeSet::from([AnalysisCapabilityRequirement::ProcessIsolation]);
    let mesh = context.mesh.clone();
    let access = context.artifact_access.clone();
    let revision = context.program_revision.clone();
    let frozen = context.frozen_project_digest;
    let compile = |apply_inputs: &mut dyn FnMut(&mut AnalysisModel) -> Result<(), String>| {
        let mut compiled = compile_analysis_study(AnalysisStudyCompilationInput {
            study: context.study,
            geometry: context.document,
            regions: context.regions,
            materials: context.materials,
            mesh: context.mesh,
        })
        .map_err(|error| error.to_string())?;
        apply_inputs(&mut compiled.model)?;
        Ok::<_, String>(compiled)
    };
    let (preparation, step) = match context.planning {
        StudyRootPlanning::Structural(_) => unreachable!("checked above"),
        StudyRootPlanning::Thermal(policy) => {
            let mut compiled = compile(apply_inputs)?;
            if let Some(heat) = &policy.element_heat_source {
                heat.bind(&mut compiled.model, context.mesh)
                    .map_err(|error| error.to_string())?;
            }
            if runmat_analysis_fea::execution::is_electrothermal_study(&compiled) {
                compiled = electrothermal_heat_study(&compiled, context.mesh, policy)?;
            }
            // The thermal plan selects its solve path (the proving factor, or the conduction
            // multigrid when the factor does not fit) within the admitted step ceiling, so the
            // plan, the worker's re-derivation and the worker envelope agree.
            admit(&mut compiled.model.sequence.steps)?;
            let product = plan_thermal_study_product_with_fields(
                compiled,
                context.mesh,
                policy.clone(),
                context.initial_fields.to_vec(),
            )
            .map_err(|error| error.to_string())?;
            let step = single_step(&product.compiled.model, kind)?;
            (
                FamilyStudyPreparation::Thermal(Box::new(ThermalStudyPreparation {
                    local_execution: context.local_execution,
                    field_source: None,
                    scope: ThermalStudyScope::Standalone,
                    product,
                    mesh,
                    frozen_project_digest: frozen,
                    artifact_access: access,
                    program_revision: revision,
                    output_policy: output_policy()?,
                    required_capabilities: capabilities,
                })),
                step,
            )
        }
        StudyRootPlanning::Acoustic(policy) => {
            let mut compiled = compile(apply_inputs)?;
            admit(&mut compiled.model.sequence.steps)?;
            let product = plan_acoustic_study_product(compiled, context.mesh, *policy)
                .map_err(|error| error.to_string())?;
            let step = single_step(&product.compiled.model, kind)?;
            (
                FamilyStudyPreparation::Acoustic(Box::new(AcousticStudyPreparation {
                    local_execution: context.local_execution,
                    policy: *policy,
                    product,
                    mesh,
                    frozen_project_digest: frozen,
                    artifact_access: access,
                    program_revision: revision,
                    output_policy: output_policy()?,
                    required_capabilities: capabilities,
                })),
                step,
            )
        }
        StudyRootPlanning::AcousticResponse(policy) => {
            let mut compiled = compile(apply_inputs)?;
            admit(&mut compiled.model.sequence.steps)?;
            let product = plan_acoustic_response_study_product(compiled, context.mesh, *policy)
                .map_err(|error| error.to_string())?;
            let step = single_step(&product.compiled.model, kind)?;
            (
                FamilyStudyPreparation::AcousticResponse(Box::new(
                    AcousticResponseStudyPreparation {
                        local_execution: context.local_execution,
                        policy: *policy,
                        product,
                        mesh,
                        frozen_project_digest: frozen,
                        artifact_access: access,
                        program_revision: revision,
                        output_policy: output_policy()?,
                        required_capabilities: capabilities,
                    },
                )),
                step,
            )
        }
        StudyRootPlanning::Electrical(policy) => {
            let policy = policy.clone();
            let mut compiled = compile(&mut |_| Ok(()))?;
            admit(&mut compiled.model.sequence.steps)?;
            let product = if runmat_analysis_fea::execution::is_electrothermal_study(&compiled) {
                // The conduction side of an electrothermal analysis, at the temperatures its
                // heat case published; the coupled loop is solved once, by the heat case.
                if policy.element_temperature_k.is_none() {
                    return Err("the conduction side of an electrothermal analysis needs the temperatures its heat case published".into());
                }
                let conduction =
                    runmat_analysis_fea::execution::electrothermal_conduction_study(&compiled)
                        .map_err(|error| error.to_string())?;
                plan_electrical_study_product(conduction, context.mesh, policy.clone())
            } else {
                build_electrical_study_product(ElectricalStudyProductInput {
                    study: &admitted_study()?,
                    geometry: context.document,
                    regions: context.regions,
                    materials: context.materials,
                    mesh: context.mesh,
                    planning: policy.clone(),
                })
            }
            .map_err(|error| error.to_string())?;
            let step = single_step(&product.compiled.model, kind)?;
            (
                FamilyStudyPreparation::Electrical(Box::new(ElectricalStudyPreparation {
                    local_execution: context.local_execution,
                    machine_memory_bytes: worker_memory()?,
                    policy,
                    product,
                    mesh,
                    frozen_project_digest: frozen,
                    artifact_access: access,
                    program_revision: revision,
                    output_policy: output_policy()?,
                    required_capabilities: capabilities,
                })),
                step,
            )
        }
        StudyRootPlanning::Magnetic(policy) => {
            let product = build_magnetic_study_product(MagneticStudyProductInput {
                study: &admitted_study()?,
                geometry: context.document,
                regions: context.regions,
                materials: context.materials,
                mesh: context.mesh,
                planning: *policy,
            })
            .map_err(|error| error.to_string())?;
            let step = single_step(&product.compiled.model, kind)?;
            (
                FamilyStudyPreparation::Magnetic(Box::new(MagneticStudyPreparation {
                    local_execution: context.local_execution,
                    policy: *policy,
                    product,
                    mesh,
                    frozen_project_digest: frozen,
                    artifact_access: access,
                    program_revision: revision,
                    output_policy: output_policy()?,
                    required_capabilities: capabilities,
                })),
                step,
            )
        }
        StudyRootPlanning::Maxwell(policy) => {
            let product = build_maxwell_study_product(MaxwellStudyProductInput {
                study: &admitted_study()?,
                geometry: context.document,
                regions: context.regions,
                materials: context.materials,
                mesh: context.mesh,
                planning: *policy,
            })
            .map_err(|error| error.to_string())?;
            let step = single_step(&product.compiled.model, kind)?;
            (
                FamilyStudyPreparation::Maxwell(Box::new(MaxwellStudyPreparation {
                    local_execution: context.local_execution,
                    policy: *policy,
                    product,
                    mesh,
                    frozen_project_digest: frozen,
                    artifact_access: access,
                    program_revision: revision,
                    output_policy: output_policy()?,
                    required_capabilities: capabilities,
                })),
                step,
            )
        }
        StudyRootPlanning::Fluid(policy) => {
            let mut compiled = compile(&mut |_| Ok(()))?;
            admit(&mut compiled.model.sequence.steps)?;
            let planned = PlannedFluidStudyPreparation::plan(
                compiled,
                FluidStudyPreparationParts {
                    local_execution: context.local_execution,
                    policy: *policy,
                    mesh,
                    frozen_project_digest: frozen,
                    artifact_access: access,
                    program_revision: revision,
                    output_policy: output_policy()?,
                    required_capabilities: capabilities,
                },
            )
            .map_err(|error| error.to_string())?;
            let product = &planned.preparation().product;
            let mut step = single_step(&product.compiled().model, kind)?;
            step.0 = step.0.min(product.workload().policy.resident_byte_budget());
            (FamilyStudyPreparation::Fluid(Box::new(planned)), step)
        }
        StudyRootPlanning::NonlinearSolid(policy) => {
            let product = build_nonlinear_solid_study_product(NonlinearSolidStudyProductInput {
                study: &admitted_study()?,
                geometry: context.document,
                regions: context.regions,
                materials: context.materials,
                mesh: context.mesh,
                controls: policy.controls.clone(),
            })
            .map_err(|error| error.to_string())?;
            let step = single_step(&product.compiled.model, kind)?;
            (
                FamilyStudyPreparation::NonlinearSolid(Box::new(NonlinearSolidStudyPreparation {
                    local_execution: context.local_execution,
                    accepted_block_limit: usize::try_from(policy.accepted_block_limit)
                        .map_err(|error| error.to_string())?,
                    product,
                    mesh,
                    frozen_project_digest: frozen,
                    artifact_access: access,
                    program_revision: revision,
                    output_policy: output_policy()?,
                })),
                step,
            )
        }
    };
    Ok(PlannedFamilyStudy {
        preparation,
        maximum_resident_bytes: step.0,
        maximum_wall_millis: step.1,
        worker_memory_bytes: worker_memory()?,
    })
}

/// Iterate conduction and heat transfer to the two-way fixed point and return the heat-transfer
/// side with the converged Joule source bound.
fn electrothermal_heat_study(
    compiled: &runmat_analysis_fea::study::CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    policy: &runmat_analysis_fea::study::ThermalStudyPlanningPolicy,
) -> Result<runmat_analysis_fea::study::CompiledAnalysisStudy, String> {
    let (thermal, point) = runmat_analysis_fea::execution::solve_electrothermal_heat_study(
        compiled,
        mesh,
        policy.clone(),
    )
    .map_err(|error| error.to_string())?;
    tracing::info!(
        sweeps = point.sweeps.len(),
        converged = point.converged,
        joule_power_w = point.joule_power_w,
        terminal_power_w = point.terminal_power_w,
        last_temperature_change_k = point.sweeps.last().map(|s| s.maximum_temperature_change_k),
        maximum_temperature_rise_k = point.sweeps.last().map(|s| s.maximum_temperature_rise_k),
        "electrothermal two-way coupling converged"
    );
    if !point.converged {
        return Err(format!(
            "electrothermal coupling did not converge in {} sweeps",
            point.sweeps.len()
        ));
    }
    Ok(thermal)
}

/// Output inventory admitted by the study's memory budget: min(authored step resident ceiling,
/// the run's worker memory), within the platform ceilings. The output reservation then checks
/// each task's exact requirement against the same budget.
fn admitted_output_policy(
    study: &runmat_analysis_core::RealizedStudy,
    machine: u64,
) -> Result<AnalysisOutputPolicy, String> {
    let authored = study
        .sequence
        .steps
        .iter()
        .map(|step| step.accuracy.resources.resident_byte_budget())
        .min()
        .unwrap_or(u64::MAX);
    AnalysisOutputPolicy::new(
        runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES,
        ObjectInventoryLimits::admitted(authored.min(machine)),
    )
    .map_err(|error| error.to_string())
}

fn single_step(model: &AnalysisModel, kind: StudyFamilyKind) -> Result<(u64, u64), String> {
    let [step] = model.sequence.steps.as_slice() else {
        return Err(format!(
            "authored {} execution requires a standalone step",
            kind.label()
        ));
    };
    Ok((
        step.accuracy.resources.resident_byte_budget(),
        step.accuracy.resources.maximum_wall_time_milliseconds,
    ))
}

/// Scheduling policy shared by every family coordinator.
#[derive(Clone, Debug)]
pub(crate) struct FamilyStudyExecutionPolicy {
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

/// Bridge-level completion of one family study, independent of where its tasks ran.
pub enum CompletedFamilyStudy {
    Thermal(Box<CompletedThermalStudy>),
    Electrical(Box<CompletedElectricalStudy>),
    Magnetic(Box<CompletedMagneticStudy>),
    Maxwell(Box<CompletedMaxwellStudy>),
    Acoustic(Box<CompletedAcousticStudy>),
    AcousticResponse(Box<CompletedAcousticResponseStudy>),
    Fluid(Box<runmat_analysis_execution::VerifiedFluidStudy>),
    NonlinearSolid(Box<CompletedNonlinearSolidStudy>),
}

/// Run every governed phase of one family through the supplied pool.
pub(crate) async fn execute_family_study(
    pool: &mut impl AnalysisTaskPool,
    planned: PlannedFamilyStudy,
    policy: FamilyStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedFamilyStudy> {
    let FamilyStudyExecutionPolicy {
        resources,
        retry,
        deadline_unix_millis,
        priority,
    } = policy;
    // Each family keeps its own boxed poll frame so large unrelated coordinator futures never
    // share one worker stack.
    Ok(match planned.preparation {
        FamilyStudyPreparation::Thermal(preparation) => CompletedFamilyStudy::Thermal(Box::new(
            Box::pin(crate::analysis_thermal::execute_with_pool(
                pool,
                *preparation,
                crate::ThermalStudyExecutionPolicy {
                    resources,
                    retry,
                    deadline_unix_millis,
                    priority,
                },
            ))
            .await?
            .completed,
        )),
        FamilyStudyPreparation::Electrical(preparation) => {
            CompletedFamilyStudy::Electrical(Box::new(
                Box::pin(crate::analysis_electrical::execute_with_pool(
                    pool,
                    *preparation,
                    crate::analysis_electrical::ElectricalStudyExecutionPolicy {
                        resources,
                        retry,
                        deadline_unix_millis,
                        priority,
                    },
                ))
                .await?,
            ))
        }
        FamilyStudyPreparation::Magnetic(preparation) => CompletedFamilyStudy::Magnetic(Box::new(
            Box::pin(crate::analysis_magnetic::execute_with_pool(
                pool,
                *preparation,
                crate::analysis_magnetic::MagneticStudyExecutionPolicy {
                    resources,
                    retry,
                    deadline_unix_millis,
                    priority,
                },
            ))
            .await?,
        )),
        FamilyStudyPreparation::Maxwell(preparation) => CompletedFamilyStudy::Maxwell(Box::new(
            Box::pin(crate::analysis_maxwell::execute_with_pool(
                pool,
                *preparation,
                crate::analysis_maxwell::MaxwellStudyExecutionPolicy {
                    resources,
                    retry,
                    deadline_unix_millis,
                    priority,
                },
            ))
            .await?,
        )),
        FamilyStudyPreparation::Acoustic(preparation) => CompletedFamilyStudy::Acoustic(Box::new(
            Box::pin(crate::analysis_acoustic::execute_with_pool(
                pool,
                *preparation,
                crate::analysis_acoustic::AcousticStudyExecutionPolicy {
                    resources,
                    retry,
                    deadline_unix_millis,
                    priority,
                },
            ))
            .await?,
        )),
        FamilyStudyPreparation::AcousticResponse(preparation) => {
            CompletedFamilyStudy::AcousticResponse(Box::new(
                Box::pin(crate::analysis_acoustic_response::execute_with_pool(
                    pool,
                    *preparation,
                    crate::analysis_acoustic_response::AcousticResponseStudyExecutionPolicy {
                        resources,
                        retry,
                        deadline_unix_millis,
                        priority,
                    },
                ))
                .await?,
            ))
        }
        FamilyStudyPreparation::Fluid(preparation) => CompletedFamilyStudy::Fluid(Box::new(
            Box::pin(crate::analysis_fluid::execute_with_pool(
                pool,
                *preparation,
                crate::analysis_fluid::FluidStudyExecutionPolicy {
                    resources,
                    retry,
                    deadline_unix_millis,
                    priority,
                },
            ))
            .await?,
        )),
        FamilyStudyPreparation::NonlinearSolid(preparation) => {
            CompletedFamilyStudy::NonlinearSolid(Box::new(
                Box::pin(crate::analysis_nonlinear::execute_with_pool(
                    pool,
                    *preparation,
                    crate::analysis_nonlinear::NonlinearSolidStudyExecutionPolicy {
                        resources,
                        retry,
                        deadline_unix_millis,
                        priority,
                    },
                ))
                .await?,
            ))
        }
    })
}

/// Canonical completion root of one family: the closed object catalog that reproduces it.
pub(crate) struct FamilyCompletionRoot {
    pub root: ValueRef,
    pub objects: Vec<LogicalObject>,
    pub catalog: PublishedObjectCatalog,
}

impl std::fmt::Debug for CompletedFamilyStudy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompletedFamilyStudy")
            .field("family", &self.kind())
            .finish_non_exhaustive()
    }
}

impl CompletedFamilyStudy {
    pub fn kind(&self) -> StudyFamilyKind {
        match self {
            Self::Thermal(_) => StudyFamilyKind::Thermal,
            Self::Electrical(_) => StudyFamilyKind::Electrical,
            Self::Magnetic(_) => StudyFamilyKind::Magnetic,
            Self::Maxwell(_) => StudyFamilyKind::Maxwell,
            Self::Acoustic(_) => StudyFamilyKind::Acoustic,
            Self::AcousticResponse(_) => StudyFamilyKind::AcousticResponse,
            Self::Fluid(_) => StudyFamilyKind::Fluid,
            Self::NonlinearSolid(_) => StudyFamilyKind::NonlinearSolid,
        }
    }

    /// Encode the canonical completion root and the new objects it introduces.
    pub(crate) fn prepare_root(&self) -> NativeExecutionResult<FamilyCompletionRoot> {
        macro_rules! root {
            ($prepared:expr) => {{
                let prepared = $prepared.map_err(protocol)?;
                FamilyCompletionRoot {
                    root: prepared.root,
                    objects: prepared.objects,
                    catalog: prepared.catalog,
                }
            }};
        }
        Ok(match self {
            Self::Thermal(completed) => {
                root!(runmat_analysis_execution::prepare_thermal_study_root(
                    completed
                ))
            }
            Self::Electrical(completed) => {
                root!(runmat_analysis_execution::prepare_electrical_study_root(
                    completed
                ))
            }
            Self::Magnetic(completed) => {
                root!(runmat_analysis_execution::prepare_magnetic_study_root(
                    completed
                ))
            }
            Self::Maxwell(completed) => {
                root!(runmat_analysis_execution::prepare_maxwell_study_root(
                    completed
                ))
            }
            Self::Acoustic(completed) => {
                root!(runmat_analysis_execution::prepare_acoustic_study_root(
                    completed
                ))
            }
            Self::AcousticResponse(completed) => {
                root!(runmat_analysis_execution::prepare_acoustic_response_study_root(completed))
            }
            Self::Fluid(completed) => {
                root!(runmat_analysis_execution::prepare_fluid_study_root_for(
                    completed,
                    completed.preparation()
                ))
            }
            Self::NonlinearSolid(completed) => {
                root!(runmat_analysis_execution::prepare_nonlinear_solid_study_root(completed))
            }
        })
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
