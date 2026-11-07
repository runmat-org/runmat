//! The conduction multigrid path: its structural proof and its backward-error acceptance.
use runmat_analysis_core::{AnalysisInterfaceKind, AnalysisModel, BoundaryOperator};
use runmat_simulation_usage::{LedgerError, OperationCount};

use crate::{
    algebra::{
        capacity::real_spmv_work, AssembledRealSystem, LinearPreparationReuse,
        ScalarConductionError, ScalarConductionProof,
    },
    contracts::FeaRunError,
    physics::thermal::assembly::ThermalOperatorAssembly,
    pipeline::homogeneous_static::{metered, norm_work, vector_work},
    planning::resource::{ThermalOperatorResourceFacts, ThermalSolvePath},
};

use super::ThermalSystem;

/// The preparation the planned path solves with: `None` on the proving factorization, the
/// structural conduction proof on the conduction multigrid path.
pub(super) fn preparation(
    facts: &ThermalOperatorResourceFacts,
    assembly: &ThermalOperatorAssembly,
    model: &AnalysisModel,
    operator: &ThermalSystem,
) -> Result<Option<LinearPreparationReuse>, FeaRunError> {
    if facts.path() != ThermalSolvePath::ConductionMultigrid {
        return Ok(None);
    }
    if !assembly.relations.is_empty() {
        return Err(FeaRunError::Solve(
            "the thermal conduction multigrid path does not solve tie, periodic or jump relations; \
             their reduced solve needs the proving factorization, which exceeds this run's budget"
                .into(),
        ));
    }
    let refused = |error: ScalarConductionError| FeaRunError::Solve(error.to_string());
    for boundary in &model.boundaries {
        if let BoundaryOperator::ThermalConvection { coefficient, .. } = &boundary.operator {
            let value = coefficient.canonical_value();
            if !value.is_finite() || value < 0.0 {
                return Err(refused(ScalarConductionError::NegativeConvection {
                    boundary_id: boundary.boundary_id.clone(),
                    value,
                }));
            }
        }
    }
    for interface in &model.interfaces {
        if let AnalysisInterfaceKind::ConjugateHeatTransfer(transfer) = &interface.kind {
            let conductance = match &transfer.transfer {
                runmat_analysis_core::ThermalInterfaceTransfer::Conductance(value) => {
                    value.canonical_value()
                }
                runmat_analysis_core::ThermalInterfaceTransfer::Resistance(value) => {
                    1.0 / value.canonical_value()
                }
                _ => continue,
            };
            if !conductance.is_finite() || conductance < 0.0 {
                return Err(refused(ScalarConductionError::NegativeInterfaceConductance {
                    interface_id: interface.interface_id.clone(),
                    value: conductance,
                }));
            }
        }
    }
    let proof = operator.with_canonical_system(|numerical| {
        let stiffness = numerical.operators().stiffness();
        let work = OperationCount::from_operations(ScalarConductionProof::operations_for(
            stiffness.rows() as u64,
            stiffness.values().len() as u64,
        ));
        metered(work, || {
            ScalarConductionProof::prove(
                stiffness,
                numerical.constrained_dofs(),
                assembly.positive_robin_weight.canonical_values(),
                assembly.element_conductivity_w_per_mk.canonical_values(),
            )
            .map_err(refused)
        })
    })?;
    Ok(Some(LinearPreparationReuse::with_scalar_conduction(proof)))
}

/// Accepts a conduction multigrid temperature by its normwise backward error on the prepared
/// system `A x = b` (prescribed rows and columns replaced by identity):
/// `‖A x − b‖ / (‖A‖_F ‖x‖ + ‖b‖) ≤ tolerance`, the CG stopping tolerance. The residual is
/// recomputed from the assembled operator, independent of the CG recurrence. Returns the error.
pub(super) fn require_backward_error(
    system: &AssembledRealSystem,
    solution: &[f64],
    tolerance: f64,
) -> Result<f64, FeaRunError> {
    let stiffness = system.operators().stiffness();
    let constrained = system.constrained_dofs();
    let load = system.load();
    let n = stiffness.rows();
    if solution.len() != n || load.len() != n || constrained.len() != n {
        return Err(FeaRunError::Solve(
            "thermal backward-error check dimensions are inconsistent".into(),
        ));
    }
    let entries = stiffness.values().len();
    // One product with the operator, two operations per entry for the prepared Frobenius norm
    // and the constrained-column lift, and three per row for the residual and right-hand side.
    let work = real_spmv_work(stiffness)?
        .checked_add(vector_work(entries, 2)?)
        .map_err(LedgerError::from)?
        .checked_add(vector_work(n, 3)?)
        .map_err(LedgerError::from)?;
    let (residual, rhs, frobenius_squared) = metered(work, || {
        let pattern = stiffness.pattern();
        let values = stiffness.values();
        let mut residual = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        let mut frobenius_squared = 0.0;
        for row in 0..n {
            if constrained[row] {
                residual[row] = solution[row] - load[row];
                rhs[row] = load[row];
                frobenius_squared += 1.0;
                continue;
            }
            let (mut all, mut lifted) = (0.0, 0.0);
            for entry in pattern.row_range(row) {
                let column = pattern.column(entry);
                let value = values[entry];
                all += value * solution[column];
                if constrained[column] {
                    lifted += value * load[column];
                } else {
                    frobenius_squared += value * value;
                }
            }
            // The prescribed columns of x carry the prescribed values, so A x − b reduces to
            // K x − f on a free row.
            residual[row] = all - load[row];
            rhs[row] = load[row] - lifted;
        }
        Ok((residual, rhs, frobenius_squared))
    })?;
    let norm = |values: &[f64]| values.iter().map(|value| value * value).sum::<f64>().sqrt();
    let residual_norm = metered(norm_work(n)?, || Ok(norm(&residual)))?;
    let solution_norm = metered(norm_work(n)?, || Ok(norm(solution)))?;
    let rhs_norm = metered(norm_work(n)?, || Ok(norm(&rhs)))?;
    // One square root, one product, one sum and one quotient.
    let backward_error = metered(OperationCount::from_operations(4), || {
        Ok(residual_norm
            / (frobenius_squared.sqrt() * solution_norm + rhs_norm).max(f64::MIN_POSITIVE))
    })?;
    if !backward_error.is_finite() || backward_error > tolerance {
        return Err(FeaRunError::Solve(format!(
            "thermal conduction multigrid solve failed its backward-error acceptance: \
             ‖A x − b‖ / (‖A‖_F ‖x‖ + ‖b‖) = {backward_error:.3e}, tolerance {tolerance:.3e}"
        )));
    }
    Ok(backward_error)
}
