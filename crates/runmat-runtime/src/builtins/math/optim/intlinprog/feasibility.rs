//! Internal original-model feasibility repair for population proposals.
//! Numerical relaxations are never rounded into integer-feasible witnesses.
use super::{common, controls, engine, invalid, model, publication, ScalarCooperation};
use crate::BuiltinResult;
use common::solver_timing::SolverTiming;
use num_rational::BigRational;
use num_traits::Zero;
mod geometry;
mod independent;
mod original;
#[cfg(test)]
mod tests;
pub(in crate::builtins::math::optim) use geometry::Input;
pub(in crate::builtins::math::optim) use original::OriginalLinear;

fn workspace_admission(cooperation: &ScalarCooperation) -> BuiltinResult<()> {
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        "intlinprog",
        "finite host-memory execution is unsupported; sparse feasibility repair and exact-rational workspaces have no complete finite bound",
    )
}

pub(in crate::builtins::math::optim) struct Membership {
    pub violation: f64,
    pub feasible: bool,
}

pub(in crate::builtins::math::optim) enum Repair {
    Unchanged,
    Point(Vec<f64>),
    Infeasible,
    TimeLimit,
    Unverified,
}

enum Storage {
    Independent(independent::Domain),
    // Exact-capacity one-element owner avoids a large inline enum variant.
    // Empty only while the same workspace is moved through its continuation.
    Coupled { slot: Vec<model::Workspace> },
}
pub(in crate::builtins::math::optim) struct Domain {
    storage: Storage,
    controls: controls::Controls,
    variables: usize,
    first_distance_row: usize,
    equalities: usize,
    tolerance: f64,
}
impl Domain {
    pub async fn new(
        input: Input,
        timing: SolverTiming,
        max_time: common::global_time::TimeLimit,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if cooperation.runtime().is_none() {
            let mut error = crate::context::RuntimeCapabilityError::new(
                crate::context::RuntimeCapability::Builtin,
                "linear feasibility repair requires its selected execution context",
            )
            .into_runtime_error();
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(error);
        }
        workspace_admission(cooperation)?;
        let max_time = max_time.validate(common::global_time::Policy {
            identity: runmat_builtins::INTLINPROG_CATALOG_ENTRY.identity,
            error: &runmat_builtins::INTLINPROG_ERROR_INVALID_ARGUMENT,
        })?;
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(invalid("invalid internal feasibility controls"));
        }
        let variables = input.lower.len();
        let first_distance_row = input.rhs.len();
        let equalities = input.equalities;
        let controls = controls::Controls::feasibility(timing, max_time, tolerance)?;
        geometry::validate(&input, cooperation).await?;
        let storage = if input.matrix.m == 0 {
            Storage::Independent(independent::Domain {
                lower: input.lower,
                upper: input.upper,
                integers: input.integers,
            })
        } else {
            let workspace = geometry::lift(input, cooperation).await?;
            let mut slot = runmat_value::exact_capacity_vec(1).map_err(super::allocation)?;
            slot.push(workspace);
            Storage::Coupled { slot }
        };
        Ok(Self {
            storage,
            controls,
            variables,
            first_distance_row,
            equalities,
            tolerance,
        })
    }
    /// Retune controls without replacing the original model, clock or budget.
    pub fn retune(
        &mut self,
        maximum: common::global_time::TimeLimit,
        tolerance: f64,
    ) -> BuiltinResult<()> {
        let maximum = maximum.validate(common::global_time::Policy {
            identity: runmat_builtins::INTLINPROG_CATALOG_ENTRY.identity,
            error: &runmat_builtins::INTLINPROG_ERROR_INVALID_ARGUMENT,
        })?;
        if !tolerance.is_finite() || tolerance <= 0. {
            return Err(invalid("invalid internal feasibility tolerance"));
        }
        self.controls.retune_feasibility(maximum, tolerance);
        self.tolerance = tolerance;
        Ok(())
    }
    pub fn bounds(&self) -> (&[f64], &[f64]) {
        match &self.storage {
            Storage::Independent(domain) => (&domain.lower, &domain.upper),
            Storage::Coupled { slot } => {
                let (lower, upper) = slot
                    .first()
                    .expect("workspace retained outside repair")
                    .original_bounds();
                (&lower[..self.variables], &upper[..self.variables])
            }
        }
    }
    pub fn original_linear(&self) -> OriginalLinear<'_> {
        let (matrix, rhs) = match &self.storage {
            Storage::Independent(_) => (None, &[][..]),
            Storage::Coupled { slot } => {
                let model = slot
                    .first()
                    .expect("workspace retained outside repair")
                    .view();
                (Some(model.a), &model.b[..self.first_distance_row])
            }
        };
        OriginalLinear {
            matrix,
            rhs,
            variables: self.variables,
            equalities: self.equalities,
        }
    }
    pub fn integers(&self) -> &[usize] {
        match &self.storage {
            Storage::Independent(domain) => &domain.integers,
            Storage::Coupled { slot } => {
                &slot
                    .first()
                    .expect("workspace retained outside repair")
                    .integers
            }
        }
    }
    /// Check the returned binary64 coordinate generation against the same
    /// original rows. Distance rows are retargeted to this point with zero
    /// auxiliaries, so they contribute zero residual rather than another model.
    pub async fn classify(
        &mut self,
        point: &[f64],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Membership> {
        use num_traits::ToPrimitive;
        if point.len() != self.variables || !tolerance.is_finite() || tolerance < 0. {
            return Err(invalid("invalid original membership query"));
        }
        let (lower, upper) = self.bounds();
        let integers = self.integers();
        let mut violation = 0.0f64;
        let mut integral = true;
        let mut bounds_feasible = true;
        let mut integer = 0;
        for (column, &value) in point.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid("membership point must be finite"));
            }
            bounds_feasible &= value >= lower[column] && value <= upper[column];
            violation = violation
                .max((lower[column] - value).max(0.))
                .max((value - upper[column]).max(0.));
            if integers.get(integer) == Some(&column) {
                integral &= value.fract() == 0.;
                integer += 1;
            }
            cooperation.completed(1).await?;
        }
        if matches!(&self.storage, Storage::Independent(_)) {
            return Ok(Membership {
                violation,
                feasible: integral && bounds_feasible,
            });
        }
        let Storage::Coupled { slot } = &mut self.storage else {
            unreachable!("independent classification returned")
        };
        let workspace = slot.first_mut().expect("owned repair workspace");
        workspace
            .retarget_distance(self.first_distance_row, point, cooperation)
            .await?;
        let mut exact =
            runmat_value::exact_capacity_vec(workspace.n()).map_err(super::allocation)?;
        for &value in point {
            exact.push(BigRational::from_float(value).expect("finite coordinates"));
            cooperation.completed(1).await?;
        }
        for _ in 0..self.variables {
            exact.push(BigRational::zero());
            cooperation.completed(1).await?;
        }
        let Some((_, violation)) =
            common::conic::linear::evaluate(&workspace.view(), &exact, cooperation).await?
        else {
            return Err(invalid("original linear membership could not be evaluated"));
        };
        let feasible = integral
            && bounds_feasible
            && violation <= BigRational::from_float(tolerance).expect("finite tolerance");
        Ok(Membership {
            violation: violation.to_f64().unwrap_or(f64::INFINITY),
            feasible,
        })
    }
    pub fn repair_nodes(&self) -> usize {
        self.controls.nodes
    }
    pub async fn repair(
        &mut self,
        proposal: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Repair> {
        if proposal.len() != self.variables || proposal.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "repair proposal must be finite with original variable extent",
            ));
        }
        cooperation.completed(0).await?;
        if self.controls.time_limit_reached() {
            return Ok(Repair::TimeLimit);
        }
        if let Storage::Independent(domain) = &self.storage {
            return domain.repair(proposal, cooperation).await;
        }
        let Storage::Coupled { slot } = &mut self.storage else {
            unreachable!("independent repair returned")
        };
        let workspace = slot.first_mut().expect("one owned repair workspace");
        workspace
            .retarget_distance(self.first_distance_row, proposal, cooperation)
            .await?;
        let (lower, upper) = workspace.original_bounds();
        let mut exact =
            runmat_value::exact_capacity_vec(workspace.n()).map_err(super::allocation)?;
        let mut domains_admitted = true;
        for (column, &value) in proposal.iter().enumerate() {
            domains_admitted &= value >= lower[column] && value <= upper[column];
            exact.push(BigRational::from_float(value).expect("finite proposal"));
            cooperation.completed(1).await?;
        }
        for &column in &workspace.integers {
            domains_admitted &= exact[column].is_integer();
            cooperation.completed(1).await?;
        }
        for _ in 0..self.variables {
            exact.push(BigRational::zero());
            cooperation.completed(1).await?;
        }
        // This is an original-row feasibility check, not a recovery solve.
        // In particular an already valid point consumes no MIP node at all.
        let evaluated =
            common::conic::linear::evaluate(&workspace.view(), &exact, cooperation).await?;
        let tolerance = BigRational::from_float(self.tolerance).expect("finite admitted tolerance");
        if domains_admitted && evaluated.is_some_and(|(_, violation)| violation <= tolerance) {
            return Ok(Repair::Unchanged);
        }
        drop(exact);
        if self.controls.time_limit_reached() {
            return Ok(Repair::TimeLimit);
        }
        self.controls.begin_feasibility_repair();
        let runtime = cooperation
            .runtime()
            .cloned()
            .expect("admitted original feasibility context");
        let workspace = slot.pop().expect("one owned repair workspace");
        let mut outcome = None;
        let controls = &mut self.controls;
        let retained_workspace = slot;
        let completed = &mut outcome;
        let repair_cooperation = &mut *cooperation;
        // Admit the concrete continuation before constructing it. Only
        // move-only CPU numerical state crosses this small factory boundary;
        // there is no second payload, allocator or solver lifecycle.
        crate::execution::admit_domain_continuation_with(&runtime, move || async move {
            let (outcome, workspace) =
                engine::solve(workspace, controls, repair_cooperation).await?;
            debug_assert!(retained_workspace.is_empty());
            retained_workspace.push(workspace);
            *completed = Some(outcome);
            Ok(())
        })?
        .await?;
        let outcome = outcome.expect("successful continuation published one numerical outcome");
        let Storage::Coupled { slot } = &self.storage else {
            unreachable!("coupled repair owner")
        };
        let workspace = slot.first().expect("returned original workspace");
        if self.controls.time_limit_reached()
            || matches!(
                outcome.terminal,
                engine::Terminal::Stopped(engine::Stop::Time)
            )
        {
            return Ok(Repair::TimeLimit);
        }
        let Some(witness) = outcome.incumbent else {
            return Ok(
                if matches!(outcome.terminal, engine::Terminal::Infeasible) {
                    Repair::Infeasible
                } else {
                    Repair::Unverified
                },
            );
        };
        let Some(binary) = publication::BinaryPoint::prepare(
            &witness.point,
            &workspace.integers,
            &[],
            cooperation,
        )
        .await?
        else {
            return Ok(Repair::Unverified);
        };
        let mut scratch = witness.point;
        let evaluated = binary
            .evaluate(
                &mut scratch,
                &workspace.node(),
                &self.controls,
                &mut super::compiled::Supplied,
                cooperation,
            )
            .await?;
        drop(scratch);
        if !evaluated.admitted {
            return Ok(Repair::Unverified);
        }
        let mut values = binary.values;
        values.truncate(self.variables);
        // A witness was consumed, not copied into population history. Actual
        // objective/nonlinear callbacks remain entirely outside this owner.
        Ok(Repair::Point(values))
    }
}
