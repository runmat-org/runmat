use super::{
    solve_average_acceleration_step_reusing, solve_initial_acceleration_reusing, AssemblySummary,
    EffectiveDynamicsSolveError, EffectiveDynamicsSolveResult, LinearAlgebraBackendKind,
    SecondOrderState, SolverMethod, SolverPreconditioner, StructuralTransientSolverCounters,
    TransientSolveError,
};

pub(in crate::solve::transient) enum TransientStepFailure {
    Usage(runmat_simulation_usage::LedgerError),
    Detail(String),
}

impl TransientStepFailure {
    pub(in crate::solve::transient) fn effective(error: EffectiveDynamicsSolveError) -> Self {
        match error {
            EffectiveDynamicsSolveError::Usage(error) => Self::Usage(error),
            EffectiveDynamicsSolveError::DynamicBalance(
                crate::solve::transient::balance::DynamicBalanceError::Usage(error),
            ) => Self::Usage(error),
            EffectiveDynamicsSolveError::SparseSolve(crate::algebra::ScopedSpmvError::Usage(
                error,
            )) => Self::Usage(error),
            EffectiveDynamicsSolveError::Newmark(
                crate::solve::transient::newmark::NewmarkError::Usage(error),
            ) => Self::Usage(error),
            other => Self::Detail(other.to_string()),
        }
    }

    pub(in crate::solve::transient) fn projection(
        error: crate::solve::dynamics::ModalProjectionError,
    ) -> Self {
        match error {
            crate::solve::dynamics::ModalProjectionError::Usage(error) => Self::Usage(error),
            other => Self::Detail(other.to_string()),
        }
    }

    pub(in crate::solve::transient) fn modal(
        error: crate::solve::dynamics::ModalTransientError,
    ) -> Self {
        match error {
            crate::solve::dynamics::ModalTransientError::Usage(error) => Self::Usage(error),
            other => Self::Detail(other.to_string()),
        }
    }
}

pub(in crate::solve::transient) struct TransientInitialVectors {
    pub(in crate::solve::transient) displacement: Vec<f64>,
    pub(in crate::solve::transient) velocity: Vec<f64>,
}

pub(in crate::solve::transient) struct TransientNumericalStepOutcome {
    pub(in crate::solve::transient) state: SecondOrderState,
    pub(in crate::solve::transient) projected_relative_residual: f64,
    pub(in crate::solve::transient) accounting: TransientSolverAccounting,
}

pub(in crate::solve::transient) trait TransientStepStrategy {
    fn coordinate_count(&self) -> usize;
    fn solver_method(&self) -> SolverMethod;
    fn resumed_accounting(
        &self,
        counters: StructuralTransientSolverCounters,
    ) -> TransientSolverAccounting;
    fn initialize(
        &mut self,
        summary: &AssemblySummary,
        initial: TransientInitialVectors,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure>;
    fn advance(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
        time_step_s: f64,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure>;
    fn refresh_acceleration(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure>;
}

pub(in crate::solve::transient) struct DirectTransientStepStrategy<'a> {
    initial_summary: &'a AssemblySummary,
    backend_kind: LinearAlgebraBackendKind,
    residual_target: f64,
    preparation: crate::algebra::LinearPreparationReuse,
}

impl<'a> DirectTransientStepStrategy<'a> {
    pub(in crate::solve::transient) fn new(
        initial_summary: &'a AssemblySummary,
        backend_kind: LinearAlgebraBackendKind,
        residual_target: f64,
    ) -> Self {
        Self {
            initial_summary,
            backend_kind,
            residual_target,
            preparation: Default::default(),
        }
    }

    fn outcome(
        solved: EffectiveDynamicsSolveResult,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure> {
        Ok(TransientNumericalStepOutcome {
            state: solved.state().clone(),
            projected_relative_residual: solved.projected_residual().relative_norm(),
            accounting: TransientSolverAccounting::from_effective(&solved)
                .map_err(TransientStepFailure::Detail)?,
        })
    }
}

impl TransientStepStrategy for DirectTransientStepStrategy<'_> {
    fn coordinate_count(&self) -> usize {
        self.initial_summary.dof_count
    }

    fn solver_method(&self) -> SolverMethod {
        SolverMethod::NewmarkAverageAccelerationPcg
    }

    fn resumed_accounting(
        &self,
        counters: StructuralTransientSolverCounters,
    ) -> TransientSolverAccounting {
        TransientSolverAccounting {
            solver_backend: self.backend_kind,
            counters,
            preconditioner: SolverPreconditioner::None,
            accelerator_evidence: Vec::new(),
        }
    }

    fn initialize(
        &mut self,
        summary: &AssemblySummary,
        initial: TransientInitialVectors,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure> {
        solve_initial_acceleration_reusing(
            summary,
            initial.displacement,
            initial.velocity,
            self.backend_kind,
            self.residual_target,
            &mut self.preparation,
        )
        .map_err(TransientStepFailure::effective)
        .and_then(Self::outcome)
    }

    fn advance(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
        time_step_s: f64,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure> {
        solve_average_acceleration_step_reusing(
            summary,
            previous,
            time_step_s,
            self.backend_kind,
            self.residual_target,
            &mut self.preparation,
        )
        .map_err(TransientStepFailure::effective)
        .and_then(Self::outcome)
    }

    fn refresh_acceleration(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
    ) -> Result<TransientNumericalStepOutcome, TransientStepFailure> {
        solve_initial_acceleration_reusing(
            summary,
            previous.displacement().to_vec(),
            previous.velocity().to_vec(),
            self.backend_kind,
            self.residual_target,
            &mut self.preparation,
        )
        .map_err(TransientStepFailure::effective)
        .and_then(Self::outcome)
    }
}

#[derive(Clone, Debug)]
pub(in crate::solve::transient) struct TransientSolverAccounting {
    pub(in crate::solve::transient) solver_backend: LinearAlgebraBackendKind,
    pub(in crate::solve::transient) counters: StructuralTransientSolverCounters,
    pub(in crate::solve::transient) preconditioner: SolverPreconditioner,
    pub(in crate::solve::transient) accelerator_evidence:
        Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
}

impl Default for TransientSolverAccounting {
    fn default() -> Self {
        Self {
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            counters: StructuralTransientSolverCounters::default(),
            preconditioner: SolverPreconditioner::None,
            accelerator_evidence: Vec::new(),
        }
    }
}

impl TransientSolverAccounting {
    fn from_effective(solved: &EffectiveDynamicsSolveResult) -> Result<Self, String> {
        let mut accounting = Self::default();
        for linear in solved.linear_solves() {
            accounting.solver_backend = linear.solver_backend;
            accounting.counters.host_sync_count = accounting
                .counters
                .host_sync_count
                .checked_add(u64::from(linear.host_sync_count))
                .ok_or_else(|| "transient host synchronization counter overflow".to_string())?;
            accounting.counters.device_apply_k_count = accounting
                .counters
                .device_apply_k_count
                .checked_add(u64::from(linear.device_apply_k_count))
                .ok_or_else(|| "transient device apply counter overflow".to_string())?;
            accounting.counters.device_apply_k_attempt_count = accounting
                .counters
                .device_apply_k_attempt_count
                .checked_add(u64::from(linear.device_apply_k_attempt_count))
                .ok_or_else(|| "transient device apply-attempt counter overflow".to_string())?;
            accounting.preconditioner = linear.preconditioner;
            accounting
                .accelerator_evidence
                .extend(linear.accelerator_evidence.iter().cloned());
        }
        Ok(accounting)
    }

    pub(in crate::solve::transient) fn modal_dense() -> Self {
        Self {
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            counters: StructuralTransientSolverCounters {
                dense_factorization_count: 1,
                dense_solve_count: 1,
                ..StructuralTransientSolverCounters::default()
            },
            preconditioner: SolverPreconditioner::None,
            accelerator_evidence: Vec::new(),
        }
    }

    pub(super) fn merge<E>(&mut self, increment: Self) -> Result<(), TransientSolveError<E>> {
        self.solver_backend = increment.solver_backend;
        self.preconditioner = increment.preconditioner;
        macro_rules! add_counter {
            ($field:ident) => {
                self.counters.$field = self
                    .counters
                    .$field
                    .checked_add(increment.counters.$field)
                    .ok_or(TransientSolveError::SolverCounterOverflow)?;
            };
        }
        add_counter!(host_sync_count);
        add_counter!(device_apply_k_count);
        add_counter!(device_apply_k_attempt_count);
        add_counter!(dense_factorization_count);
        add_counter!(dense_solve_count);
        self.accelerator_evidence
            .extend(increment.accelerator_evidence);
        Ok(())
    }
}
