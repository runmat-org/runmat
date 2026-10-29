//! Conductors at an unknown common potential: `electrical.equipotential`,
//! `electrical.floating_conductor` (optionally carrying an `electric_charge` load) and
//! `electrical.integral_current`.
//!
//! Each such terminal holds one unknown potential U and one condition on its net outward
//! flux: zero for an equipotential or uncharged floating conductor, -Q for a conductor
//! carrying charge Q (electrostatics), and -I for a terminal injecting current I (steady
//! conduction). The problem is linear in the terminal potentials, so the outward fluxes are
//! F = f0 + C U, with f0 from the problem with every such terminal at 0 V and column j of C
//! from the source-free problem with terminal j at 1 V. Solving C U = target - f0 gives the
//! potentials, and a final solve with those potentials prescribed is the published result
//! (its terminal fluxes meet the conditions to solver tolerance).
use super::{CanonicalElectricalResponse, ElectricalStaticError};
use crate::excitation::{RealizedExcitationKind, RealizedExcitationTarget};
use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, ElectricPotential, ElectricalConstraint,
    ElectricalConstraintKind, ExcitationAction, PrescribedValue, RegionTarget,
};

struct FloatingTerminal {
    constraint_id: String,
    target: RegionTarget,
    /// Required net outward flux (A in conduction, C in electrostatics).
    outward_flux: f64,
}

/// Whether the model holds a conductor at an unknown common potential.
pub(super) fn has_floating_terminals(model: &AnalysisModel) -> bool {
    model.constraints.iter().any(|constraint| {
        matches!(constraint, AnalysisConstraint::Electrical(constraint)
            if matches!(constraint.kind,
                ElectricalConstraintKind::Equipotential { .. }
                | ElectricalConstraintKind::FloatingConductor { .. }
                | ElectricalConstraintKind::IntegralCurrent { .. }))
    })
}

/// Solve with every floating terminal at the potential that meets its flux condition.
pub(super) fn solve(
    model: &AnalysisModel,
    step_id: &str,
    sample: crate::excitation::ExcitationSamplePoint<'_>,
    mut solve_fixed: impl FnMut(
        &AnalysisModel,
    ) -> Result<CanonicalElectricalResponse, ElectricalStaticError>,
) -> Result<CanonicalElectricalResponse, ElectricalStaticError> {
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == step_id)
        .ok_or_else(|| ElectricalStaticError::InvalidInput("selected step absent".into()))?;
    // Charges carried by floating conductors, by the region they are applied to.
    let mut charges = std::collections::BTreeMap::<String, f64>::new();
    if model
        .excitations
        .iter()
        .any(|action| matches!(action, ExcitationAction::ElectricCharge(_)))
    {
        let loads = crate::excitation::realize_step_excitations(model, step, sample.clone())
            .map_err(|error| ElectricalStaticError::InvalidInput(error.to_string()))?;
        for load in loads.iter() {
            let RealizedExcitationKind::ElectricCharge(value) = load.kind else {
                continue;
            };
            let RealizedExcitationTarget::Region(target) = &load.target else {
                return Err(ElectricalStaticError::Boundary(
                    "electric charge belongs on a floating conductor's faces".into(),
                ));
            };
            if value.imaginary != 0.0 || !value.real.is_finite() {
                return Err(ElectricalStaticError::Boundary(
                    "static charge must be finite real".into(),
                ));
            }
            *charges.entry(target.region_id.clone()).or_default() += value.real;
        }
    }
    let mut terminals = Vec::new();
    for constraint in &model.constraints {
        let AnalysisConstraint::Electrical(constraint) = constraint else {
            continue;
        };
        let (target, outward_flux) = match &constraint.kind {
            ElectricalConstraintKind::Equipotential { target } => (target, 0.0),
            ElectricalConstraintKind::FloatingConductor { target } => {
                (target, -charges.remove(&target.region_id).unwrap_or(0.0))
            }
            ElectricalConstraintKind::IntegralCurrent { target, value } => {
                let current = crate::excitation::sample_prescribed_current(
                    value,
                    &model.signals,
                    sample.clone(),
                )
                .map_err(|error| ElectricalStaticError::Boundary(error.to_string()))?;
                if current.imaginary != 0.0 || !current.real.is_finite() {
                    return Err(ElectricalStaticError::Boundary(
                        "static terminal current must be finite real".into(),
                    ));
                }
                (target, -current.real)
            }
            _ => continue,
        };
        terminals.push(FloatingTerminal {
            constraint_id: constraint.constraint_id.clone(),
            target: target.clone(),
            outward_flux,
        });
    }
    if let Some(region) = charges.keys().next() {
        return Err(ElectricalStaticError::Boundary(format!(
            "electric charge on {region} needs an electrical.floating_conductor on the same faces"
        )));
    }
    let without_charges = |model: &AnalysisModel| {
        let mut model = model.clone();
        let removed: std::collections::BTreeSet<_> = model
            .excitations
            .iter()
            .filter(|action| matches!(action, ExcitationAction::ElectricCharge(_)))
            .map(|action| action.excitation().excitation_id.clone())
            .collect();
        model
            .excitations
            .retain(|action| !removed.contains(&action.excitation().excitation_id));
        for case in &mut model.excitation_cases {
            case.action_ids.retain(|id| !removed.contains(id));
        }
        model
            .excitation_cases
            .retain(|case| !case.action_ids.is_empty());
        model
    };
    let fixed = |model: &AnalysisModel, potentials: &[f64], keep_sources: bool| {
        let mut model = if keep_sources {
            without_charges(model)
        } else {
            // The unit problems are source-free: every prescribed potential at 0 V and no loads.
            let mut model = model.clone();
            model.excitations.clear();
            model.excitation_cases.clear();
            for constraint in &mut model.constraints {
                if let AnalysisConstraint::Electrical(ElectricalConstraint {
                    kind: ElectricalConstraintKind::PrescribedPotential { value, .. },
                    ..
                }) = constraint
                {
                    *value = PrescribedValue::Constant(ElectricPotential::new(0.0));
                }
            }
            model
        };
        for constraint in &mut model.constraints {
            let AnalysisConstraint::Electrical(constraint) = constraint else {
                continue;
            };
            if let Some(index) = terminals
                .iter()
                .position(|terminal| terminal.constraint_id == constraint.constraint_id)
            {
                constraint.kind = ElectricalConstraintKind::PrescribedPotential {
                    target: terminals[index].target.clone(),
                    value: PrescribedValue::Constant(ElectricPotential::new(potentials[index])),
                };
            }
        }
        model
    };
    let flux = |response: &CanonicalElectricalResponse| -> Result<Vec<f64>, ElectricalStaticError> {
        terminals
            .iter()
            .map(|terminal| {
                response
                    .terminals
                    .iter()
                    .find(|t| t.constraint_id == terminal.constraint_id)
                    .map(|t| t.outward_flux.canonical_value())
                    .ok_or_else(|| {
                        ElectricalStaticError::Boundary(format!(
                            "terminal {} has no flux",
                            terminal.constraint_id
                        ))
                    })
            })
            .collect()
    };
    let count = terminals.len();
    let base = flux(&solve_fixed(&fixed(model, &vec![0.0; count], true))?)?;
    let mut coupling = vec![vec![0.0; count]; count];
    for j in 0..count {
        let mut unit = vec![0.0; count];
        unit[j] = 1.0;
        let column = flux(&solve_fixed(&fixed(model, &unit, false))?)?;
        for i in 0..count {
            coupling[i][j] = column[i];
        }
    }
    let rhs: Vec<f64> = (0..count)
        .map(|i| terminals[i].outward_flux - base[i])
        .collect();
    let potentials = solve_dense(coupling, rhs).ok_or_else(|| {
        ElectricalStaticError::Boundary(
            "a floating conductor is not coupled to any prescribed potential".into(),
        )
    })?;
    solve_fixed(&fixed(model, &potentials, true))
}

/// Gaussian elimination with partial pivoting for the small terminal system.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    let scale = a.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    if !(scale > 0.0) {
        return None;
    }
    for k in 0..n {
        let pivot = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[pivot][k].abs() <= 1e-14 * scale {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for k in (0..n).rev() {
        let sum: f64 = (k + 1..n).map(|j| a[k][j] * x[j]).sum();
        x[k] = (b[k] - sum) / a[k][k];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}
