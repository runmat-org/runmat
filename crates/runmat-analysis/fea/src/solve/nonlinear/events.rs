use runmat_analysis_core::SolverObservationEvent;
use runmat_canonical_codec::encode_contract;
use runmat_execution::Digest;
use serde::Serialize;

use super::NonlinearSolveResult;

const SOURCE_PREFIX: &[u8] = b"runmat-fea-nonlinear-event-source/v4\0";

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearCompletedSolverEvent {
    pub event: SolverObservationEvent,
    /// State index in the history inventory: zero is the initial state and `i + 1` is accepted
    /// nonlinear snapshot `i`.
    pub state_index: usize,
    pub load_factor: f64,
    pub occurrence_index: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NonlinearEventDetectionError {
    Invalid(&'static str),
    Canonical(String),
}

impl std::fmt::Display for NonlinearEventDetectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(reason) => formatter.write_str(reason),
            Self::Canonical(reason) => write!(
                formatter,
                "cannot identify nonlinear event source: {reason}"
            ),
        }
    }
}

impl std::error::Error for NonlinearEventDetectionError {}

#[derive(Serialize)]
struct NonlinearEventSource<'a> {
    accepted_state: &'a super::AcceptedSolidState,
    converged_increments: usize,
    total_increments: usize,
    initial_load_factor: f64,
    load_factors: &'a [f64],
    initial_displacement: &'a [f64],
    initial_equivalent_plastic_strain: &'a [f64],
    initial_contact_pressure: &'a [f64],
    initial_contact_gap: &'a [f64],
    displacement_snapshots: &'a [Vec<f64>],
    internal_force_snapshots: &'a [Vec<f64>],
    stress_snapshots: &'a [Vec<f64>],
    total_strain_snapshots: &'a [Vec<f64>],
    von_mises_snapshots: &'a [Vec<f64>],
    plastic_strain_snapshots: &'a [Vec<f64>],
    equivalent_plastic_strain_snapshots: &'a [Vec<f64>],
    contact_pressure_snapshots: &'a [Vec<f64>],
    contact_gap_snapshots: &'a [Vec<f64>],
    residual_norms: &'a [f64],
    increment_norms: &'a [f64],
    relative_increment_norms: &'a [f64],
    iteration_counts: &'a [usize],
    failed_increments: usize,
    line_search_backtracks: usize,
    max_line_search_backtracks_per_increment: usize,
    tangent_rebuild_count: usize,
    iteration_spike_count: usize,
    convergence_stall_count: usize,
    backtrack_burst_count: usize,
    solver_method: crate::contracts::SolverMethod,
    solver_backend: crate::solve::backend::kind::LinearAlgebraBackendKind,
    solver_host_sync_count: u32,
    device_apply_k_count: u32,
    device_apply_k_attempt_count: u32,
    preconditioner: crate::contracts::SolverPreconditioner,
}

pub fn detect_completed_nonlinear_events(
    result: &NonlinearSolveResult,
) -> Result<Vec<NonlinearCompletedSolverEvent>, NonlinearEventDetectionError> {
    validate_event_source(result)?;
    let mut events = Vec::new();
    let mut yielded = has_positive(&result.initial_equivalent_plastic_strain);
    if yielded {
        events.push(NonlinearCompletedSolverEvent {
            event: SolverObservationEvent::FirstYield,
            state_index: 0,
            load_factor: result.initial_load_factor,
            occurrence_index: 0,
        });
    }

    let mut contact_status = active_contact_status(
        &result.initial_contact_pressure,
        &result.initial_contact_gap,
    );
    let mut contact_occurrence = 0_u64;
    for index in 0..result.load_factors.len() {
        let state_index = index + 1;
        let load_factor = result.load_factors[index];
        if !yielded && has_positive(&result.equivalent_plastic_strain_snapshots[index]) {
            yielded = true;
            events.push(NonlinearCompletedSolverEvent {
                event: SolverObservationEvent::FirstYield,
                state_index,
                load_factor,
                occurrence_index: 0,
            });
        }
        // A run without contact has no contact history and no contact status changes.
        let next_contact_status = match (
            result.contact_pressure_snapshots.get(index),
            result.contact_gap_snapshots.get(index),
        ) {
            (Some(pressure), Some(gap)) => active_contact_status(pressure, gap),
            _ => contact_status.clone(),
        };
        if next_contact_status != contact_status {
            events.push(NonlinearCompletedSolverEvent {
                event: SolverObservationEvent::ContactStatusChange,
                state_index,
                load_factor,
                occurrence_index: contact_occurrence,
            });
            contact_occurrence =
                contact_occurrence
                    .checked_add(1)
                    .ok_or(NonlinearEventDetectionError::Invalid(
                        "contact event occurrence count overflowed",
                    ))?;
        }
        contact_status = next_contact_status;
    }
    Ok(events)
}

/// Digest of deterministic numerical solver content used to bind in-process event attribution.
/// Diagnostics and wall-clock measurements are intentionally excluded; they are execution
/// evidence rather than the numerical state from which the event was detected.
///
/// `limits` is the history inventory the resource plan admitted for this result.
pub fn nonlinear_solver_content_digest(
    result: &NonlinearSolveResult,
    limits: crate::planning::resource::NonlinearHistoryLimits,
) -> Result<Digest, NonlinearEventDetectionError> {
    validate_event_source(result)?;
    let limits = limits
        .canonical()
        .map_err(NonlinearEventDetectionError::Canonical)?;
    let bytes = encode_contract(
        SOURCE_PREFIX,
        "completed-nonlinear-solver-content",
        &NonlinearEventSource {
            accepted_state: &result.accepted_state,
            converged_increments: result.converged_increments,
            total_increments: result.total_increments,
            initial_load_factor: result.initial_load_factor,
            load_factors: &result.load_factors,
            initial_displacement: &result.initial_displacement,
            initial_equivalent_plastic_strain: &result.initial_equivalent_plastic_strain,
            initial_contact_pressure: &result.initial_contact_pressure,
            initial_contact_gap: &result.initial_contact_gap,
            displacement_snapshots: &result.displacement_snapshots,
            internal_force_snapshots: &result.internal_force_snapshots,
            stress_snapshots: &result.stress_snapshots,
            total_strain_snapshots: &result.total_strain_snapshots,
            von_mises_snapshots: &result.von_mises_snapshots,
            plastic_strain_snapshots: &result.plastic_strain_snapshots,
            equivalent_plastic_strain_snapshots: &result.equivalent_plastic_strain_snapshots,
            contact_pressure_snapshots: &result.contact_pressure_snapshots,
            contact_gap_snapshots: &result.contact_gap_snapshots,
            residual_norms: &result.residual_norms,
            increment_norms: &result.increment_norms,
            relative_increment_norms: &result.relative_increment_norms,
            iteration_counts: &result.iteration_counts,
            failed_increments: result.failed_increments,
            line_search_backtracks: result.line_search_backtracks,
            max_line_search_backtracks_per_increment: result
                .max_line_search_backtracks_per_increment,
            tangent_rebuild_count: result.tangent_rebuild_count,
            iteration_spike_count: result.iteration_spike_count,
            convergence_stall_count: result.convergence_stall_count,
            backtrack_burst_count: result.backtrack_burst_count,
            solver_method: result.solver_method,
            solver_backend: result.solver_backend,
            solver_host_sync_count: result.solver_host_sync_count,
            device_apply_k_count: result.device_apply_k_count,
            device_apply_k_attempt_count: result.device_apply_k_attempt_count,
            preconditioner: result.preconditioner,
        },
        limits,
    )
    .map_err(|error| NonlinearEventDetectionError::Canonical(error.to_string()))?;
    Ok(Digest::sha256(bytes))
}

fn validate_event_source(
    result: &NonlinearSolveResult,
) -> Result<(), NonlinearEventDetectionError> {
    let count = result.load_factors.len();
    let snapshot_counts = [
        result.displacement_snapshots.len(),
        result.internal_force_snapshots.len(),
        result.stress_snapshots.len(),
        result.total_strain_snapshots.len(),
        result.von_mises_snapshots.len(),
        result.plastic_strain_snapshots.len(),
        result.equivalent_plastic_strain_snapshots.len(),
        result.residual_norms.len(),
        result.increment_norms.len(),
        result.relative_increment_norms.len(),
        result.iteration_counts.len(),
    ];
    if !result.initial_load_factor.is_finite()
        || snapshot_counts.iter().any(|candidate| *candidate != count)
        // Contact history is complete for a contact run and absent otherwise.
        || result.contact_pressure_snapshots.len() != result.contact_gap_snapshots.len()
        || !matches!(result.contact_pressure_snapshots.len(), 0)
            && result.contact_pressure_snapshots.len() != count
        || result
            .load_factors
            .iter()
            .scan(result.initial_load_factor, |previous, value| {
                let valid = value.is_finite() && *value > *previous;
                *previous = *value;
                Some(valid)
            })
            .any(|valid| !valid)
    {
        return Err(NonlinearEventDetectionError::Invalid(
            "nonlinear event source has inconsistent state coordinates or snapshot counts",
        ));
    }
    for values in std::iter::once(result.initial_displacement.as_slice())
        .chain(std::iter::once(
            result.initial_equivalent_plastic_strain.as_slice(),
        ))
        .chain(std::iter::once(result.initial_contact_pressure.as_slice()))
        .chain(std::iter::once(result.initial_contact_gap.as_slice()))
        .chain(result.displacement_snapshots.iter().map(Vec::as_slice))
        .chain(result.internal_force_snapshots.iter().map(Vec::as_slice))
        .chain(result.stress_snapshots.iter().map(Vec::as_slice))
        .chain(result.total_strain_snapshots.iter().map(Vec::as_slice))
        .chain(result.von_mises_snapshots.iter().map(Vec::as_slice))
        .chain(result.plastic_strain_snapshots.iter().map(Vec::as_slice))
        .chain(
            result
                .equivalent_plastic_strain_snapshots
                .iter()
                .map(Vec::as_slice),
        )
        .chain(result.contact_pressure_snapshots.iter().map(Vec::as_slice))
        .chain(result.contact_gap_snapshots.iter().map(Vec::as_slice))
    {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(NonlinearEventDetectionError::Invalid(
                "nonlinear event source contains non-finite numerical state",
            ));
        }
    }
    if result
        .residual_norms
        .iter()
        .chain(&result.increment_norms)
        .chain(&result.relative_increment_norms)
        .any(|value| !value.is_finite())
    {
        return Err(NonlinearEventDetectionError::Invalid(
            "nonlinear event source contains non-finite convergence state",
        ));
    }
    if result.initial_contact_pressure.len() != result.initial_contact_gap.len()
        || result
            .contact_pressure_snapshots
            .iter()
            .zip(&result.contact_gap_snapshots)
            .any(|(pressure, gap)| {
                pressure.len() != gap.len()
                    || pressure.len() != result.initial_contact_pressure.len()
            })
    {
        return Err(NonlinearEventDetectionError::Invalid(
            "nonlinear contact event source has inconsistent entity layouts",
        ));
    }
    let plastic_width = result.initial_equivalent_plastic_strain.len();
    if result
        .equivalent_plastic_strain_snapshots
        .iter()
        .any(|snapshot| snapshot.len() != plastic_width)
    {
        return Err(NonlinearEventDetectionError::Invalid(
            "nonlinear yield event source has inconsistent entity layouts",
        ));
    }
    Ok(())
}

fn has_positive(values: &[f64]) -> bool {
    values.iter().any(|value| *value > 0.0)
}

fn active_contact_status(pressure: &[f64], gap: &[f64]) -> Vec<bool> {
    pressure
        .iter()
        .zip(gap)
        .map(|(pressure, gap)| *pressure > 0.0 || *gap <= 0.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{
        contracts::{SolverMethod, SolverPreconditioner},
        diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
        solve::backend::kind::LinearAlgebraBackendKind,
    };

    use super::*;

    fn result() -> NonlinearSolveResult {
        // Synthetic event histories below are not a physical/restart qualification.
        let (model, mesh, summary) = crate::assembly::nonlinear_solid::tests::fixture(
            runmat_meshing_core::ElementOrder::Tet4,
            false,
        );
        let basis = crate::assembly::nonlinear_solid::CanonicalNonlinearSolidBasis::prepare(
            &model, &mesh, &summary,
        )
        .unwrap();
        NonlinearSolveResult {
            accepted_state: super::super::AcceptedSolidState {
                scope_identity: Digest::sha256(b"synthetic-event-test"),
                contact: None,
                contact_energy: None,
                load_factor: 1.0,
                displacement: vec![0.0; summary.dof_count],
                constitutive: basis.zero_state(),
                accumulated_plastic_dissipation_j: 0.0,
            },
            converged_increments: 2,
            total_increments: 2,
            initial_load_factor: 0.0,
            load_factors: vec![0.5, 1.0],
            initial_displacement: vec![0.0],
            initial_equivalent_plastic_strain: vec![0.0],
            initial_contact_pressure: vec![0.0],
            initial_contact_gap: vec![1.0],
            displacement_snapshots: vec![vec![0.5], vec![1.0]],
            internal_force_snapshots: vec![vec![1.0], vec![2.0]],
            stress_snapshots: vec![vec![1.0; 6], vec![2.0; 6]],
            total_strain_snapshots: vec![vec![0.001; 6], vec![0.002; 6]],
            von_mises_snapshots: vec![vec![1.0], vec![2.0]],
            plastic_strain_snapshots: vec![vec![0.0], vec![0.1]],
            equivalent_plastic_strain_snapshots: vec![vec![0.0], vec![0.1]],
            contact_pressure_snapshots: vec![vec![2.0], vec![0.0]],
            contact_gap_snapshots: vec![vec![-0.1], vec![0.2]],
            residual_norms: vec![1.0e-8, 1.0e-9],
            increment_norms: vec![1.0e-7, 1.0e-8],
            relative_increment_norms: vec![1.0e-8, 1.0e-9],
            iteration_counts: vec![3, 4],
            failed_increments: 0,
            line_search_backtracks: 0,
            max_line_search_backtracks_per_increment: 0,
            tangent_rebuild_count: 2,
            iteration_spike_count: 0,
            convergence_stall_count: 0,
            backtrack_burst_count: 0,
            diagnostics: Vec::new(),
            solver_method: SolverMethod::IncrementalNewtonRaphson,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            preconditioner: SolverPreconditioner::Jacobi,
            accelerator_evidence: Vec::new(),
        }
    }

    #[test]
    fn completed_events_retain_exact_state_and_occurrence_order() {
        let events = detect_completed_nonlinear_events(&result()).unwrap();
        assert_eq!(
            events,
            vec![
                NonlinearCompletedSolverEvent {
                    event: SolverObservationEvent::ContactStatusChange,
                    state_index: 1,
                    load_factor: 0.5,
                    occurrence_index: 0,
                },
                NonlinearCompletedSolverEvent {
                    event: SolverObservationEvent::FirstYield,
                    state_index: 2,
                    load_factor: 1.0,
                    occurrence_index: 0,
                },
                NonlinearCompletedSolverEvent {
                    event: SolverObservationEvent::ContactStatusChange,
                    state_index: 2,
                    load_factor: 1.0,
                    occurrence_index: 1,
                },
            ]
        );
    }

    const LIMITS: crate::planning::resource::NonlinearHistoryLimits =
        crate::planning::resource::NonlinearHistoryLimits {
            encoded_bytes: crate::planning::resource::ByteCount(1 << 24),
            collection_items: crate::planning::resource::Count(1 << 24),
        };

    #[test]
    fn content_digest_is_bounded_by_the_planned_history_inventory() {
        let baseline = result();
        let digest = nonlinear_solver_content_digest(&baseline, LIMITS).unwrap();
        // The bound comes from the plan, not a fixed ceiling: a plan that admits less
        // history than the solver produced fails closed with a typed codec error.
        let short = crate::planning::resource::NonlinearHistoryLimits {
            encoded_bytes: crate::planning::resource::ByteCount(64),
            ..LIMITS
        };
        assert!(matches!(
            nonlinear_solver_content_digest(&baseline, short),
            Err(NonlinearEventDetectionError::Canonical(_))
        ));
        assert_eq!(
            nonlinear_solver_content_digest(&baseline, LIMITS).unwrap(),
            digest
        );
    }

    #[test]
    fn source_identity_binds_numerical_state_but_not_execution_diagnostics() {
        let baseline = result();
        let baseline_digest = nonlinear_solver_content_digest(&baseline, LIMITS).unwrap();
        let mut substituted_strain = baseline.clone();
        substituted_strain.total_strain_snapshots[1][0] += 0.001;
        assert_ne!(
            baseline_digest,
            nonlinear_solver_content_digest(&substituted_strain, LIMITS).unwrap()
        );

        let mut diagnostic_only = baseline.clone();
        diagnostic_only.diagnostics.push(FeaDiagnostic {
            code: "EXECUTION_NOTE".into(),
            severity: FeaDiagnosticSeverity::Info,
            message: "provider timing changed".into(),
        });
        assert_eq!(
            nonlinear_solver_content_digest(&diagnostic_only, LIMITS).unwrap(),
            baseline_digest
        );

        let mut substituted_state = baseline;
        substituted_state.equivalent_plastic_strain_snapshots[1][0] = 0.2;
        assert_ne!(
            nonlinear_solver_content_digest(&substituted_state, LIMITS).unwrap(),
            baseline_digest
        );
    }

    #[test]
    fn malformed_snapshot_inventory_fails_closed() {
        let mut malformed = result();
        malformed.contact_gap_snapshots.pop();
        assert!(detect_completed_nonlinear_events(&malformed).is_err());
        assert!(nonlinear_solver_content_digest(&malformed, LIMITS).is_err());
    }
}
