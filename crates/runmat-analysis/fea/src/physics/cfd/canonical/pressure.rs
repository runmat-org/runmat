//! Compact pressure Laplacian with geometric nonorthogonal correction.
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
    p.topology
        .faces
        .iter()
        .enumerate()
        .map(|(i, f)| {
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
        .collect()
}

pub(super) fn coefficients(p: &CanonicalFluidProblem<'_>, rau: &[f64]) -> Result<Vec<f64>, Error> {
    p.topology
        .faces
        .iter()
        .map(|f| {
            let (_, _, w) = face_geometry(p, f)?;
            Ok(f.neighbor
                .map_or(rau[f.owner], |n| w * rau[f.owner] + (1.0 - w) * rau[n]))
        })
        .collect()
}

pub(super) fn corrected_flux(
    p: &CanonicalFluidProblem<'_>,
    qh: &[f64],
    rf: &[f64],
    pressure: &[f64],
) -> Result<Vec<f64>, Error> {
    let gp = gradient(p, pressure, None)?;
    p.topology
        .faces
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let (alpha, tangent, w) = face_geometry(p, f)?;
            let action = if let Some(n) = f.neighbor {
                alpha * (pressure[n] - pressure[f.owner])
                    + dot(interpolate(gp[f.owner], gp[n], w), tangent)
            } else {
                match p.boundaries[i] {
                    FaceBoundary::Velocity { .. } => return Ok(qh[i]),
                    FaceBoundary::PressureOutlet { pressure: pb, .. } => {
                        alpha * (pb - pressure[f.owner]) + dot(gp[f.owner], tangent)
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
        .collect()
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
        let gp = gradient(p, &pressure, None)?;
        let mut a = linear::assembler(p, "incompressible-pressure-rhie-chow")?;
        let mut b = vec![0.0; pressure.len()];
        for (i, f) in p.topology.faces.iter().enumerate() {
            let (alpha, tangent, w) = face_geometry(p, f)?;
            let d = rf[i] * alpha;
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
                    FaceBoundary::Velocity { .. } => b[f.owner] -= qh[i],
                    FaceBoundary::PressureOutlet { pressure: pb, .. } => {
                        a.add(f.owner, f.owner, d);
                        b[f.owner] += -qh[i]
                            + d * (pb - p.pressure_reference_pa[f.owner])
                            + rf[i] * dot(gp[f.owner], tangent);
                    }
                    _ => return Err(Error::Invalid("missing face law")),
                }
            }
        }
        let a = a.finish().map_err(|e| Error::Operator(e.to_string()))?;
        let (next, count) = linear::solve(p, &a, &b)?;
        pressure = next
            .into_iter()
            .zip(&p.pressure_reference_pa)
            .map(|(v, r)| v + r)
            .collect();
        linear_iterations += count;
    }
    let q = corrected_flux(p, qh, rf, &pressure)?;
    Ok((pressure, q, linear_iterations))
}
