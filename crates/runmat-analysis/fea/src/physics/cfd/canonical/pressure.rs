//! Compact pressure Laplacian with geometric nonorthogonal correction.
//! All numerical pressure arrays are original component-relative Pa, never absolute Pa.
use super::{
    boundary::FaceBoundary, geometry_ops::*, linear, CanonicalFluidError as Error,
    CanonicalFluidProblem,
};

pub(super) fn h_flux(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_term: &[V],
) -> Result<Vec<f64>, Error> {
    // Reconstruct velocity geometrically, but interpolate rAU*grad(p) separately:
    // the compact pressure action must cancel this original Rhie-Chow term.
    let gu = velocity_gradients(p, u)?;
    indexed(p.topology.faces.len(), |i| {
        let f = &p.topology.faces[i];
        if let Some(n) = f.neighbor {
            let (_, _, w) = face_geometry(p, f)?;
            Ok(dot(
                add(
                    interpolate(
                        face_velocity_trace(p, i, f.owner, u, &gu),
                        face_velocity_trace(p, i, n, u, &gu),
                        w,
                    ),
                    interpolate(pressure_term[f.owner], pressure_term[n], w),
                ),
                f.owner_area_vector_m2,
            ))
        } else {
            match p.boundaries[i] {
                FaceBoundary::Velocity { value, .. } => Ok(dot(value, f.owner_area_vector_m2)),
                FaceBoundary::Symmetry { .. } => Ok(0.0),
                FaceBoundary::PressureOutlet { .. } => Ok(dot(
                    add(
                        face_velocity_trace(p, i, f.owner, u, &gu),
                        pressure_term[f.owner],
                    ),
                    f.owner_area_vector_m2,
                )),
                _ => Err(Error::Invalid("missing face law")),
            }
        }
    })
}

pub(super) fn coefficients(p: &CanonicalFluidProblem<'_>, rau: &[f64]) -> Result<Vec<f64>, Error> {
    indexed(p.topology.faces.len(), |i| {
        let f = &p.topology.faces[i];
        let (_, _, w) = face_geometry(p, f)?;
        Ok(f.neighbor
            .map_or(rau[f.owner], |n| w * rau[f.owner] + (1.0 - w) * rau[n]))
    })
}

pub(super) fn corrected_flux(
    p: &CanonicalFluidProblem<'_>,
    qh: &[f64],
    rf: &[f64],
    pressure: &[f64],
) -> Result<Vec<f64>, Error> {
    corrected_flux_with(p, qh, rf, pressure, &pressure_gradient(p, pressure)?)
}

/// [`corrected_flux`] with the pressure gradient `gp` of `pressure` already evaluated.
pub(super) fn corrected_flux_with(
    p: &CanonicalFluidProblem<'_>,
    qh: &[f64],
    rf: &[f64],
    pressure: &[f64],
    gp: &[V],
) -> Result<Vec<f64>, Error> {
    indexed(p.topology.faces.len(), |i| {
        let f = &p.topology.faces[i];
        let (alpha, tangent, w) = face_geometry(p, f)?;
        let action = if let Some(n) = f.neighbor {
            alpha * (pressure[n] - pressure[f.owner])
                + dot(interpolate(gp[f.owner], gp[n], w), tangent)
        } else {
            match p.boundaries[i] {
                FaceBoundary::Velocity { .. } | FaceBoundary::Symmetry { .. } => return Ok(qh[i]),
                FaceBoundary::PressureOutlet { pressure: pb, .. } => {
                    alpha
                        * ((pb - p.pressure_reference_pa.canonical_values()[f.owner])
                            - pressure[f.owner])
                        + dot(gp[f.owner], tangent)
                }
                _ => return Err(Error::Invalid("missing face law")),
            }
        };
        let flux = qh[i] - rf[i] * action;
        if !flux.is_finite() {
            return Err(Error::Invalid("nonfinite pressure face flux"));
        }
        Ok(flux)
    })
}

pub(super) fn solve(
    p: &CanonicalFluidProblem<'_>,
    qh: &[f64],
    rf: &[f64],
    old_pressure: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, u64), Error> {
    let mut pressure = old_pressure.to_vec();
    let mut linear_iterations = 0;
    for _ in 0..p.policy.maximum_nonorthogonal_iterations {
        let gp = pressure_gradient(p, &pressure)?;
        let mut a = linear::assembler(p, "incompressible-pressure-rhie-chow")?;
        let mut b = vec![0.0; pressure.len()];
        let mut source_scale = vec![0.0; pressure.len()];
        for (i, f) in p.topology.faces.iter().enumerate() {
            let (alpha, tangent, w) = face_geometry(p, f)?;
            let d = rf[i] * alpha;
            let face_gradient = if let Some(n) = f.neighbor {
                interpolate(gp[f.owner], gp[n], w)
            } else {
                gp[f.owner]
            };
            // Bound source-construction rounding against the actual face terms,
            // not the cancellation-small assembled RHS. This does not repair b.
            let scale = qh[i].abs()
                + rf[i].abs()
                    * face_gradient
                        .iter()
                        .zip(tangent)
                        .map(|(g, t)| g.abs() * t.abs())
                        .sum::<f64>();
            source_scale[f.owner] += scale;
            if let Some(n) = f.neighbor {
                source_scale[n] += scale;
            }
            if let Some(n) = f.neighbor {
                a.add(f.owner, f.owner, d);
                a.add(f.owner, n, -d);
                a.add(n, n, d);
                a.add(n, f.owner, -d);
                let explicit = -qh[i] + rf[i] * dot(interpolate(gp[f.owner], gp[n], w), tangent);
                b[f.owner] += explicit;
                b[n] -= explicit;
            } else {
                match p.boundaries[i] {
                    FaceBoundary::Velocity { .. } | FaceBoundary::Symmetry { .. } => {
                        b[f.owner] -= qh[i]
                    }
                    FaceBoundary::PressureOutlet { pressure: pb, .. } => {
                        a.add(f.owner, f.owner, d);
                        b[f.owner] += -qh[i]
                            + d * (pb - p.pressure_reference_pa.canonical_values()[f.owner])
                            + rf[i] * dot(gp[f.owner], tangent);
                    }
                    _ => return Err(Error::Invalid("missing face law")),
                }
            }
        }
        let a = a.finish().map_err(|e| Error::Operator(e.to_string()))?;
        let roots = p.pressure_reference_plan.roots();
        let (mut next, count) = if roots.is_empty() {
            linear::solve(
                p,
                &a,
                &b,
                &pressure,
                linear::LinearSystem::PressureCorrection,
            )?
        } else {
            let (prepared, rhs) = crate::algebra::prepare_scalar_component_gauges(
                &a,
                &b,
                &roots,
                &source_scale
                    .iter()
                    .map(|s| 64.0 * f64::EPSILON * s)
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| Error::Operator(e.to_string()))?;
            linear::solve(
                p,
                &prepared,
                &rhs,
                &pressure,
                linear::LinearSystem::PressureCorrection,
            )?
        };
        p.pressure_reference_plan.normalize(p, &mut next)?;
        pressure = next;
        linear_iterations += count;
    }
    let q = corrected_flux(p, qh, rf, &pressure)?;
    Ok((pressure, q, linear_iterations))
}
