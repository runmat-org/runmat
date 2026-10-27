//! Evaluate retained physical fields without a solve, replacement field or RHS.
use super::{
    geometry_ops::*, momentum, pressure, CanonicalFluidError as Error, CanonicalFluidProblem,
    CanonicalFluidResult,
};
use crate::algebra::spmv;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidResidual {
    pub face_forces: super::flux_evidence::CanonicalFluidFaceForces,
    /// Cell vorticity ∇×u, 1/s, from the boundary-aware gradient of the viscous stress.
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub cell_vorticity_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Vorticity, Vec<V>>,
    /// Sorted exact authored boundary IDs, with original outward force/flow terms.
    pub boundary_balances: Vec<super::CanonicalFluidBoundaryBalance>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub cell_momentum_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Vec<V>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// Unrelaxed sparse-action residual, distinct from original face-flux balance.
    pub cell_algebraic_momentum_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Vec<V>>,
    pub cell_momentum_relative: Vec<f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub cell_continuity_m3_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::VolumeFlowRate, Vec<f64>>,
    pub cell_continuity_relative: Vec<f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub global_momentum_n: runmat_types::physical::Quantity<runmat_types::physical::kind::Force, V>,
    pub global_momentum_relative: f64,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub global_boundary_flux_m3_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::VolumeFlowRate, f64>,
    pub momentum_relative: f64,
    pub continuity_relative: f64,
    pub flux_constitutive_relative: f64,
}
impl CanonicalFluidResidual {
    pub(super) fn accepted(&self, tolerance: f64) -> bool {
        self.momentum_relative.is_finite()
            && self.momentum_relative <= tolerance
            && self.continuity_relative.is_finite()
            && self.continuity_relative <= tolerance
            && self.flux_constitutive_relative.is_finite()
            && self.flux_constitutive_relative <= tolerance
            && self.global_momentum_relative.is_finite()
            && self.global_momentum_relative <= tolerance
            && self
                .cell_momentum_relative
                .iter()
                .all(|r| r.is_finite() && *r <= tolerance)
    }
}

pub(super) fn recover(
    p: &CanonicalFluidProblem<'_>,
    original: &CanonicalFluidResult,
) -> Result<CanonicalFluidResidual, Error> {
    recover_with_time(p, original, p.numerical_input_identity, None)
}

pub(super) fn recover_with_time(
    p: &CanonicalFluidProblem<'_>,
    original: &CanonicalFluidResult,
    numerical_input_identity: runmat_execution::Digest,
    time: Option<super::transient::TimeDerivative<'_>>,
) -> Result<CanonicalFluidResidual, Error> {
    if original.numerical_input_identity != numerical_input_identity
        || original.authored_authority_identity != p.authored_authority_identity
    {
        return Err(Error::Invalid("retained original input binding"));
    }
    if original.outer_iterations == 0
        || original.outer_iterations > p.policy.maximum_outer_iterations
        || original.linear_iterations > p.reservation.linear_iterations_upper_bound
    {
        return Err(Error::Invalid("retained iteration authority"));
    }
    if original.pressure_reference_pa.canonical_values()
        != p.pressure_reference_pa.canonical_values()
    {
        return Err(Error::Invalid(
            "retained original pressure reference substitution",
        ));
    }
    if original.pressure_pa.canonical_values().len() != p.topology.cells.len()
        || original.pressure_fluctuation_pa.canonical_values().len() != p.topology.cells.len()
        || original
            .pressure_pa
            .canonical_values()
            .iter()
            .zip(original.pressure_fluctuation_pa.canonical_values())
            .zip(p.pressure_reference_pa.canonical_values())
            .any(|((absolute, fluctuation), reference)| {
                !absolute.is_finite()
                    || !fluctuation.is_finite()
                    || absolute.to_bits() != (reference + fluctuation).to_bits()
            })
    {
        return Err(Error::Invalid("retained original pressure projection"));
    }
    p.pressure_reference_plan
        .validate(p, original.pressure_fluctuation_pa.canonical_values())?;
    let r = evaluate_with_time(
        p,
        original.velocity_m_per_s.canonical_values(),
        original.pressure_fluctuation_pa.canonical_values(),
        original.face_volume_flux_m3_per_s.canonical_values(),
        time,
    )?;
    if !r.accepted(p.policy.physical_relative_tolerance) {
        return Err(Error::Invalid("retained original physical equations"));
    }
    if original.residual != r {
        return Err(Error::Invalid("retained residual summary substitution"));
    }
    Ok(r)
}

#[cfg(test)]
pub(super) fn evaluate(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_fluctuation_pa: &[f64],
    q: &[f64],
) -> Result<CanonicalFluidResidual, Error> {
    evaluate_with_time(p, u, pressure_fluctuation_pa, q, None)
}

pub(super) fn evaluate_with_time(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_fluctuation_pa: &[f64],
    q: &[f64],
    time: Option<super::transient::TimeDerivative<'_>>,
) -> Result<CanonicalFluidResidual, Error> {
    let n = p.topology.cells.len();
    if u.len() != n
        || pressure_fluctuation_pa.len() != n
        || q.len() != p.topology.faces.len()
        || u.iter()
            .flatten()
            .chain(pressure_fluctuation_pa)
            .chain(q)
            .any(|v| !v.is_finite())
    {
        return Err(Error::Invalid("retained field shape/value"));
    }
    let m = momentum::assemble_with_time(
        p,
        u,
        pressure_fluctuation_pa,
        q,
        momentum::MomentumMode::Original,
        time,
    )?;
    let mut rm = vec![[0.0; 3]; n];
    let mut momentum_norm = 0.0f64;
    let mut momentum_scale = 0.0f64;
    for k in 0..3 {
        let uk = u.iter().map(|v| v[k]).collect::<Vec<_>>();
        let action = spmv(&m.matrix, &uk).map_err(|e| Error::Operator(e.to_string()))?;
        for c in 0..n {
            rm[c][k] = action[c] - m.rhs[c][k];
            momentum_norm = momentum_norm.hypot(rm[c][k]);
            momentum_scale = momentum_scale.hypot(action[c].abs() + m.rhs[c][k].abs());
        }
    }
    let rau = (0..n)
        .map(|c| p.topology.cells[c].volume_m3 * p.policy.momentum_relaxation / m.diagonal[c])
        .collect::<Vec<_>>();
    let gp = pressure_gradient(p, pressure_fluctuation_pa)?;
    let pf = pressure_force(p, pressure_fluctuation_pa, &gp)?;
    let pressure_term = (0..n)
        .map(|c| scale(pf[c], rau[c] / p.topology.cells[c].volume_m3))
        .collect::<Vec<_>>();
    let qh = pressure::h_flux(p, u, &pressure_term)?;
    let rf = pressure::coefficients(p, &rau)?;
    let constitutive = pressure::corrected_flux(p, &qh, &rf, pressure_fluctuation_pa)?;
    let continuity = super::continuity::evaluate(p, q)?;
    let mut discrepancy = 0.0f64;
    let mut discrepancy_scale = 0.0f64;
    for (actual, expected) in q.iter().zip(&constitutive) {
        discrepancy = discrepancy.hypot(actual - expected);
        discrepancy_scale = discrepancy_scale.hypot(actual.abs() + expected.abs());
    }
    let continuity_relative = continuity.relative.iter().copied().fold(0.0, f64::max);
    let momentum_relative = relative(momentum_norm, momentum_scale);
    let flux_constitutive_relative = relative(discrepancy, discrepancy_scale);
    let (face_forces, cell_vorticity_per_s) = super::flux_evidence::evaluate(
        p,
        u,
        pressure_fluctuation_pa,
        q,
        time.and_then(|t| t.motion)
            .map(|m| m.face_mesh_flux_m3_per_s),
    )?;
    let mut independent_cells = vec![[0.0; 3]; n];
    let mut local_scales = vec![0.0; n];
    let mut global_momentum = [0.0; 3];
    let mut global_scale = 0.0;
    for (i, f) in p.topology.faces.iter().enumerate() {
        let force = sub(
            sub(
                face_forces.advective_momentum_n.canonical_values()[i],
                face_forces.viscous_traction_n.canonical_values()[i],
            ),
            face_forces
                .pressure_fluctuation_traction_n
                .canonical_values()[i],
        );
        let scale = norm(face_forces.advective_momentum_n.canonical_values()[i])
            + norm(face_forces.viscous_traction_n.canonical_values()[i])
            + norm(
                face_forces
                    .pressure_fluctuation_traction_n
                    .canonical_values()[i],
            );
        independent_cells[f.owner] = add(independent_cells[f.owner], force);
        local_scales[f.owner] += scale;
        if let Some(c) = f.neighbor {
            independent_cells[c] = sub(independent_cells[c], force);
            local_scales[c] += scale;
        } else {
            global_momentum = add(global_momentum, force);
            global_scale += scale;
        }
    }
    for c in 0..n {
        let body = scale(p.body_force_n_per_m3[c], p.topology.cells[c].volume_m3);
        independent_cells[c] = sub(independent_cells[c], body);
        local_scales[c] += norm(body);
        global_momentum = sub(global_momentum, body);
        global_scale += norm(body);
    }
    for c in 0..n {
        if let Some((linear, quadratic)) = p.porous[c] {
            let drag = scale(
                u[c],
                p.topology.cells[c].volume_m3 * (linear + quadratic * norm(u[c])),
            );
            independent_cells[c] = add(independent_cells[c], drag);
            local_scales[c] += norm(drag);
            global_momentum = add(global_momentum, drag);
            global_scale += norm(drag);
        }
    }
    if let Some(time) = time {
        for c in 0..n {
            let inertia = time.force(p, c, u[c])?;
            independent_cells[c] = add(independent_cells[c], inertia);
            local_scales[c] += norm(inertia);
            global_momentum = add(global_momentum, inertia);
            global_scale += norm(inertia);
        }
    }
    let cell_momentum_relative = independent_cells
        .iter()
        .zip(&local_scales)
        .map(|(r, s)| relative(norm(*r), *s))
        .collect();
    let global_momentum_relative = relative(norm(global_momentum), global_scale);
    let boundary_balances = p.boundary_balance_plan.evaluate(p, q, &face_forces)?;
    Ok(CanonicalFluidResidual {
        face_forces,
        cell_vorticity_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Vorticity,
            _,
        >::new(cell_vorticity_per_s),
        boundary_balances,
        cell_momentum_n:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Force, _>::new(
                independent_cells,
            ),
        cell_algebraic_momentum_n: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Force,
            _,
        >::new(rm),
        cell_momentum_relative,
        cell_continuity_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(continuity.cell),
        cell_continuity_relative: continuity.relative,
        global_momentum_n:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Force, _>::new(
                global_momentum,
            ),
        global_momentum_relative,
        global_boundary_flux_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(continuity.global),
        momentum_relative,
        continuity_relative,
        flux_constitutive_relative,
    })
}

pub(super) fn relative(numerator: f64, denominator: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() {
        f64::INFINITY
    } else if denominator == 0.0 {
        if numerator == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        numerator / denominator
    }
}

/// The cell with the largest independent momentum residual: index, centroid, volume, its
/// exterior face laws and relative residual, for exhaustion diagnostics.
pub(super) fn describe_worst_cell(
    p: &CanonicalFluidProblem<'_>,
    residual: &CanonicalFluidResidual,
) -> (f64, String) {
    let (cell, relative) =
        residual
            .cell_momentum_relative
            .iter()
            .enumerate()
            .fold(
                (0, 0.0f64),
                |acc, (c, r)| if *r > acc.1 { (c, *r) } else { acc },
            );
    let Some(geometry) = p.topology.cells.get(cell) else {
        return (relative, "no cells".into());
    };
    let laws = p
        .topology
        .faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.owner == cell && f.neighbor.is_none())
        .map(|(i, _)| match &p.boundaries[i] {
            super::boundary::FaceBoundary::Interior => "interior",
            super::boundary::FaceBoundary::Velocity {
                moving_wall: true, ..
            } => "moving_wall",
            super::boundary::FaceBoundary::Velocity { .. } => "velocity",
            super::boundary::FaceBoundary::PressureOutlet { .. } => "outlet",
            super::boundary::FaceBoundary::Symmetry { .. } => "symmetry",
        })
        .collect::<Vec<_>>();
    (
        relative,
        format!(
            "cell {cell} at {:?} m, volume {:e} m^3, exterior faces {laws:?}, relative residual {relative:e}",
            geometry.centroid_m, geometry.volume_m3
        ),
    )
}
