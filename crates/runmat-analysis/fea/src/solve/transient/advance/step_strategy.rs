use super::{
    solve_average_acceleration_step, solve_initial_acceleration, AssemblySummary,
    EffectiveDynamicsSolveResult, LinearAlgebraBackendKind, SecondOrderState, SolverMethod,
    SolverPreconditioner, StructuralTransientSolverCounters, TransientSolveError,
};

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
    ) -> Result<TransientNumericalStepOutcome, String>;
    fn advance(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
        time_step_s: f64,
    ) -> Result<TransientNumericalStepOutcome, String>;
    fn refresh_acceleration(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
    ) -> Result<TransientNumericalStepOutcome, String>;
}

pub(in crate::solve::transient) struct DirectTransientStepStrategy<'a> {
    initial_summary: &'a AssemblySummary,
    backend_kind: LinearAlgebraBackendKind,
    residual_target: f64,
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
        }
    }

    fn outcome(
        solved: EffectiveDynamicsSolveResult,
    ) -> Result<TransientNumericalStepOutcome, String> {
        Ok(TransientNumericalStepOutcome {
            state: solved.state().clone(),
            projected_relative_residual: solved.projected_residual().relative_norm(),
            accounting: TransientSolverAccounting::from_effective(&solved)?,
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
    ) -> Result<TransientNumericalStepOutcome, String> {
        solve_initial_acceleration(
            summary,
            initial.displacement,
            initial.velocity,
            self.backend_kind,
            self.residual_target,
        )
        .map_err(|error| error.to_string())
        .and_then(Self::outcome)
    }

    fn advance(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
        time_step_s: f64,
    ) -> Result<TransientNumericalStepOutcome, String> {
        solve_average_acceleration_step(
            summary,
            previous,
            time_step_s,
            self.backend_kind,
            self.residual_target,
        )
        .map_err(|error| error.to_string())
        .and_then(Self::outcome)
    }

    fn refresh_acceleration(
        &mut self,
        summary: &AssemblySummary,
        previous: &SecondOrderState,
    ) -> Result<TransientNumericalStepOutcome, String> {
        solve_initial_acceleration(
            summary,
            previous.displacement().to_vec(),
            previous.velocity().to_vec(),
            self.backend_kind,
            self.residual_target,
        )
        .map_err(|error| error.to_string())
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
