//! Shared, assembly-free structural transient preparation.
//!
//! The checkpoint-scope and direct-host paths must use one active-constraint and schedule
//! authority. Resource projection can inspect this result before either path builds an operator.

use runmat_analysis_core::{validate_model, AnalysisModel, AnalysisProcedure, ProcedureStep};

use crate::{
    contracts::FeaRunError,
    excitation::{time_breakpoints, StructuralMotionSet},
    execution::StructuralTransientWorkload,
    pipeline::observation_history::required_time_history_samples,
    solve::transient::{
        canonical_transient_schedule, canonical_transient_step_boundaries,
        StructuralTimeStepPolicy, TransientSolveOptions, TransientStepBoundary,
    },
};

use super::constraint_admission;

pub(crate) struct PreparedTransientPreassembly {
    pub(crate) execution_step: ProcedureStep,
    pub(crate) active_model: AnalysisModel,
    pub(crate) motions: StructuralMotionSet,
    pub(crate) schedule: crate::execution::StructuralTransientSchedule,
    pub(crate) step_boundaries: Vec<TransientStepBoundary>,
}

/// Inputs needed to validate the active model and schedule before any mesh-owned solve options
/// or structural operators are constructed.
#[derive(Clone, Copy)]
pub(crate) struct TransientPreassemblyOptions {
    pub(crate) time_step_policy: StructuralTimeStepPolicy,
}

impl From<&TransientSolveOptions> for TransientPreassemblyOptions {
    fn from(options: &TransientSolveOptions) -> Self {
        Self {
            time_step_policy: options.time_step_policy,
        }
    }
}

impl From<&StructuralTransientWorkload> for TransientPreassemblyOptions {
    fn from(workload: &StructuralTransientWorkload) -> Self {
        Self {
            time_step_policy: workload.time_step_policy,
        }
    }
}

pub(crate) fn prepare_transient_preassembly(
    model: &AnalysisModel,
    options: TransientPreassemblyOptions,
) -> Result<PreparedTransientPreassembly, FeaRunError> {
    let execution_step =
        super::super::admission::require_structural(model, AnalysisProcedure::TransientResponse)?
            .clone();
    validate_model(model).map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    admit_transient_minimum_samples(
        options.time_step_policy.interval_count(),
        execution_step.accuracy.resources.maximum_samples,
    )?;
    let mut required_history_samples = required_time_history_samples(&execution_step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let time_grid = crate::excitation::ProcedureTimeGrid::from_interval_count(
        &execution_step,
        options.time_step_policy.interval_count(),
    )
    .map_err(FeaRunError::InvalidModel)?;
    let start_time_s = time_grid.start_s();
    let end_time_s = time_grid.end_s();
    let active_model =
        super::super::activation::model_with_active_constraints(model, &execution_step);
    constraint_admission::validate(&active_model)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let motions =
        StructuralMotionSet::materialize(&active_model, &execution_step, start_time_s, end_time_s)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    required_history_samples.extend(
        motions
            .breakpoints_s()
            .iter()
            .copied()
            .filter(|time| *time >= start_time_s && *time <= end_time_s),
    );
    required_history_samples.sort_by(f64::total_cmp);
    required_history_samples.dedup_by(|left, right| left.to_bits() == right.to_bits());
    let discontinuities =
        time_breakpoints(&active_model, &execution_step, start_time_s, end_time_s)
            .map_err(|error| error.into_fea_error())?;
    let step_boundaries = canonical_transient_step_boundaries(
        &discontinuities,
        &required_history_samples,
        start_time_s,
        end_time_s,
    )
    .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let schedule = canonical_transient_schedule(
        options.time_step_policy.interval_count(),
        start_time_s,
        end_time_s,
        &step_boundaries,
    )
    .map_err(FeaRunError::Solve)?;
    // Breakpoints and history coordinates can add states beyond the authored interval count.
    admit_transient_schedule_samples(
        schedule.points.len(),
        execution_step.accuracy.resources.maximum_samples,
    )?;
    Ok(PreparedTransientPreassembly {
        execution_step,
        active_model: active_model.into_owned(),
        motions,
        schedule,
        step_boundaries,
    })
}

fn admit_transient_minimum_samples(
    interval_count: usize,
    maximum_samples: usize,
) -> Result<(), FeaRunError> {
    let minimum_samples = interval_count.checked_add(1).ok_or_else(|| {
        FeaRunError::InvalidModel("structural transient sample count overflows".into())
    })?;
    if minimum_samples > maximum_samples {
        return Err(FeaRunError::InvalidModel(format!(
            "structural transient schedule requires at least {minimum_samples} samples, exceeding the authored ceiling of {maximum_samples}"
        )));
    }
    Ok(())
}

fn admit_transient_schedule_samples(
    actual_samples: usize,
    maximum_samples: usize,
) -> Result<(), FeaRunError> {
    if actual_samples > maximum_samples {
        return Err(FeaRunError::InvalidModel(format!(
            "structural transient schedule requires {actual_samples} samples, exceeding the authored ceiling of {maximum_samples}"
        )));
    }
    Ok(())
}
