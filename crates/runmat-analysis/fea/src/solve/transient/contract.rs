use runmat_analysis_core::SignalSampleApproach;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    assembly::AssemblySummary,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::FeaDiagnostic,
    solve::backend::kind::LinearAlgebraBackendKind,
    ComputeBackend,
};

use super::{state::SecondOrderState, TransientStepBoundary};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientSolveOptions {
    pub time_step_policy: StructuralTimeStepPolicy,
    pub residual_target: f64,
    #[serde(default = "default_progress_operation")]
    pub progress_operation: String,
    #[serde(default)]
    pub solver_mesh: Option<SolverMeshArtifact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralTimeStepPolicy {
    Fixed { interval_count: usize },
}

impl StructuralTimeStepPolicy {
    pub fn interval_count(self) -> usize {
        match self {
            Self::Fixed { interval_count } => interval_count,
        }
    }
}

fn default_progress_operation() -> String {
    "fea.run_transient".to_string()
}

impl Default for TransientSolveOptions {
    fn default() -> Self {
        Self {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 10 },
            residual_target: 1.0e-6,
            progress_operation: default_progress_operation(),
            solver_mesh: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientSolveResult {
    pub schedule: crate::execution::StructuralTransientSchedule,
    pub converged_steps: usize,
    pub total_steps: usize,
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<Vec<f64>>,
    pub velocity_snapshots: Vec<Vec<f64>>,
    pub acceleration_snapshots: Vec<Vec<f64>>,
    pub residual_norms: Vec<f64>,
    pub accepted_time_steps_s: Vec<f64>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: SolverMethod,
    pub solver_backend: LinearAlgebraBackendKind,
    pub solver_host_sync_count: u32,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub preconditioner: SolverPreconditioner,
    pub accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
    #[serde(skip)]
    pub(crate) assembly_snapshots: Vec<AssemblySummary>,
    #[serde(skip)]
    pub(crate) strain_energy_snapshots: Vec<f64>,
    #[serde(skip)]
    pub(crate) kinetic_energy_snapshots: Vec<f64>,
    #[serde(skip)]
    pub(crate) reaction_snapshots: Vec<Vec<f64>>,
    #[serde(skip)]
    pub(crate) integration_intervals: Vec<TransientIntegrationInterval>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub external_work_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub support_work_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub damping_dissipation_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub energy_balance_residual_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, Vec<f64>>,
    pub relative_energy_balance_error: Vec<f64>,
    pub dynamic_balance_snapshots: Vec<crate::contracts::StructuralDynamicBalanceSnapshot>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TransientIntegrationInterval {
    pub(crate) start_time_s: runmat_types::physical::Time,
    pub(crate) start_approach: SignalSampleApproach,
    pub(crate) start_summary: AssemblySummary,
    pub(crate) start_displacement: Vec<f64>,
    pub(crate) start_velocity: Vec<f64>,
    pub(crate) start_acceleration: Vec<f64>,
    pub(crate) end_time_s: runmat_types::physical::Time,
    pub(crate) end_approach: SignalSampleApproach,
    pub(crate) end_summary: AssemblySummary,
    pub(crate) end_displacement: Vec<f64>,
    pub(crate) end_velocity: Vec<f64>,
    pub(crate) end_acceleration: Vec<f64>,
}

impl TransientIntegrationInterval {
    pub(super) fn empty(
        start_time_s: f64,
        start_approach: SignalSampleApproach,
        start_summary: AssemblySummary,
        end_time_s: f64,
        end_approach: SignalSampleApproach,
        end_summary: AssemblySummary,
    ) -> Self {
        Self {
            start_time_s: runmat_types::physical::Time::new(start_time_s),
            start_approach,
            start_summary,
            start_displacement: Vec::new(),
            start_velocity: Vec::new(),
            start_acceleration: Vec::new(),
            end_time_s: runmat_types::physical::Time::new(end_time_s),
            end_approach,
            end_summary,
            end_displacement: Vec::new(),
            end_velocity: Vec::new(),
            end_acceleration: Vec::new(),
        }
    }

    pub(super) fn start_state(&self) -> Result<SecondOrderState, String> {
        SecondOrderState::try_new(
            self.start_displacement.clone(),
            self.start_velocity.clone(),
            self.start_acceleration.clone(),
        )
        .map_err(|error| error.to_string())
    }

    pub(super) fn end_state(&self) -> Result<SecondOrderState, String> {
        SecondOrderState::try_new(
            self.end_displacement.clone(),
            self.end_velocity.clone(),
            self.end_acceleration.clone(),
        )
        .map_err(|error| error.to_string())
    }
}

#[derive(Debug, Error)]
pub enum TransientSolveError<E> {
    #[error("transient numerical usage: {0}")]
    Usage(runmat_simulation_usage::LedgerError),
    #[error("transient solve options are invalid: {0}")]
    InvalidOptions(&'static str),
    #[error("transient solve was cancelled before a complete result was available")]
    Cancelled,
    #[error("structural transient schedule is invalid: {detail}")]
    Schedule { detail: String },
    #[error("initial structural acceleration could not be established: {detail}")]
    InitialAcceleration { detail: String },
    #[error(
        "structural acceleration could not be re-established after a discontinuity at {time_s} s: {detail}", time_s = .time_s.canonical_value()
    )]
    AccelerationRefresh {
        time_s: runmat_types::physical::Time,
        detail: String,
    },
    #[error("structural transient step ending at {time_s} s failed: {detail}", time_s = .time_s.canonical_value())]
    Step {
        time_s: runmat_types::physical::Time,
        detail: String,
    },
    #[error(
        "structural transient state at {time_s} s exceeded its projected residual target: {residual_norm} > {target}", time_s = .time_s.canonical_value()
    )]
    ResidualTargetExceeded {
        time_s: runmat_types::physical::Time,
        residual_norm: f64,
        target: f64,
    },
    #[error("transient reassembly failed: {0}")]
    Assembly(E),
    #[error("transient physical-state reconstruction failed: {detail}")]
    PhysicalMapping { detail: String },
    #[error("transient reassembly changed the structural degree-of-freedom count")]
    DofCountChanged,
    #[error("transient reassembly changed the invariant structural {component}")]
    StructuralSystemChanged { component: &'static str },
    #[error("structural dynamic-balance evaluation failed: {detail}")]
    DynamicBalance { detail: String },
    #[error("structural transient solver accounting exceeded its supported range")]
    SolverCounterOverflow,
    #[error("structural transient schedule ordinal exceeds the platform index range")]
    ScheduleOrdinalLimit,
    #[error("structural transient block bounds ({start}, {terminal}] are invalid")]
    InvalidBlockBounds { start: u64, terminal: u64 },
    #[error("structural transient continuation side does not match its schedule point")]
    InvalidContinuation,
    #[error("structural transient continuation state is invalid: {detail}")]
    InvalidContinuationState { detail: String },
    #[error("structural transient execution contract is invalid: {detail}")]
    ExecutionContract { detail: String },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TransientTimeDomain<'a> {
    pub(crate) start_time_s: runmat_types::physical::Time,
    pub(crate) end_time_s: runmat_types::physical::Time,
    pub(crate) step_boundaries: &'a [TransientStepBoundary],
}

pub(crate) struct TransientPhysicalMappingInput<'a> {
    pub(crate) summary: &'a AssemblySummary,
    pub(crate) options: TransientSolveOptions,
    pub(crate) backend: ComputeBackend,
    pub(crate) time_domain: TransientTimeDomain<'a>,
    pub(crate) initial_displacement: Vec<f64>,
    pub(crate) initial_velocity: Vec<f64>,
    pub(crate) interval_retention: TransientIntervalRetention,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TransientIntervalRetention {
    /// Ordinary reporting and portable blocks consume each interval's balance
    /// during advancement; their final results need no duplicate endpoints.
    DiscardAfterBalance,
    /// Prescribed-motion reconstruction needs the physical endpoint histories
    /// after the reduced solve to rebuild reactions and work.
    RetainForMotion,
}
