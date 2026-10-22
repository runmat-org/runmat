//! Physical-coordinate least-squares gradients and over-relaxed nonorthogonal faces.
use super::{boundary::FaceBoundary, CanonicalFluidError as Error, CanonicalFluidProblem};
use crate::topology::FiniteVolumeFace;

pub(super) type V = [f64; 3];
pub(super) fn add(a: V, b: V) -> V {
    std::array::from_fn(|k| a[k] + b[k])
}
pub(super) fn sub(a: V, b: V) -> V {
    std::array::from_fn(|k| a[k] - b[k])
}
pub(super) fn scale(a: V, b: f64) -> V {
    a.map(|x| x * b)
}
pub(super) fn dot(a: V, b: V) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
pub(super) fn norm(a: V) -> f64 {
    a.into_iter().fold(0.0, f64::hypot)
}

pub(super) fn face_geometry(
    p: &CanonicalFluidProblem<'_>,
    f: &FiniteVolumeFace,
) -> Result<(f64, V, f64), Error> {
    let x = p.topology.cells[f.owner].centroid_m;
    let y = f
        .neighbor
        .map_or(f.centroid_m, |n| p.topology.cells[n].centroid_m);
    let d = sub(y, x);
    let sd = dot(f.owner_area_vector_m2, d);
    if !sd.is_finite() || sd <= 0.0 {
        return Err(Error::Invalid("nonpositive face conductance"));
    }
    let alpha = dot(f.owner_area_vector_m2, f.owner_area_vector_m2) / sd;
    let tangent = sub(f.owner_area_vector_m2, scale(d, alpha));
    // Perpendicular centroid-to-face distances determine interpolation, not indices.
    let w = if let Some(n) = f.neighbor {
        let owner_distance = dot(sub(f.centroid_m, x), f.owner_area_vector_m2).abs();
        let neighbor_distance = dot(
            sub(p.topology.cells[n].centroid_m, f.centroid_m),
            f.owner_area_vector_m2,
        )
        .abs();
        neighbor_distance / (owner_distance + neighbor_distance)
    } else {
        1.0
    };
    if !alpha.is_finite() || !w.is_finite() || !(0.0..=1.0).contains(&w) {
        return Err(Error::Invalid("face metric"));
    }
    Ok((alpha, tangent, w))
}

pub(super) fn interpolate(a: V, b: V, w: f64) -> V {
    add(scale(a, w), scale(b, 1.0 - w))
}

pub(super) fn pressure_gradient(
    p: &CanonicalFluidProblem<'_>,
    values: &[f64],
) -> Result<Vec<V>, Error> {
    super::pressure_gradient::evaluate(p, values)
}

pub(super) fn velocity_gradients(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
) -> Result<Vec<[V; 3]>, Error> {
    super::velocity_gradient::evaluate(p, u)
}

pub(super) fn pressure_force(
    p: &CanonicalFluidProblem<'_>,
    pressure: &[f64],
    g: &[V],
) -> Result<Vec<V>, Error> {
    let mut force = vec![[0.0; 3]; pressure.len()];
    for (i, f) in p.topology.faces.iter().enumerate() {
        let pf = face_pressure_offset(p, i, pressure, g)?;
        let flux = scale(f.owner_area_vector_m2, pf);
        force[f.owner] = add(force[f.owner], flux);
        if let Some(n) = f.neighbor {
            force[n] = sub(force[n], flux);
        }
    }
    Ok(force)
}

pub(super) fn face_pressure_offset(
    p: &CanonicalFluidProblem<'_>,
    i: usize,
    pressure: &[f64],
    g: &[V],
) -> Result<f64, Error> {
    let f = &p.topology.faces[i];
    Ok(if let Some(n) = f.neighbor {
        let (_, _, w) = face_geometry(p, f)?;
        let pp = pressure[f.owner]
            + dot(
                g[f.owner],
                sub(f.centroid_m, p.topology.cells[f.owner].centroid_m),
            );
        let pn = pressure[n] + dot(g[n], sub(f.centroid_m, p.topology.cells[n].centroid_m));
        w * pp + (1.0 - w) * pn
    } else {
        match p.boundaries[i] {
            FaceBoundary::PressureOutlet { pressure, .. } => {
                pressure - p.pressure_reference_pa[f.owner]
            }
            _ => {
                pressure[f.owner]
                    + dot(
                        g[f.owner],
                        sub(f.centroid_m, p.topology.cells[f.owner].centroid_m),
                    )
            }
        }
    })
}

/// Symmetry-plane viscous traction on the owner fluid: zero shear and the normal
/// stress 2µ ∂u_n/∂n with zero normal velocity on the plane. Returns the implicit
/// per-component coefficient 2µα and the traction −2µα (u·n̂) n̂ in newtons.
pub(super) fn symmetry_traction(
    p: &CanonicalFluidProblem<'_>,
    i: usize,
    owner_velocity: V,
    owner_gradient: &[V; 3],
) -> Result<(f64, V), Error> {
    let f = &p.topology.faces[i];
    let (alpha, _, _) = face_geometry(p, f)?;
    let normal = scale(f.owner_area_vector_m2, 1.0 / f.area_m2);
    let coefficient = 2.0 * face_viscosity(p, owner_gradient) * alpha;
    let traction = scale(normal, -coefficient * dot(owner_velocity, normal));
    if !coefficient.is_finite() || traction.iter().any(|x| !x.is_finite()) {
        return Err(Error::Invalid("symmetry face traction"));
    }
    Ok((coefficient, traction))
}

/// Shear rate √(2 D:D) of a velocity gradient (row k is ∇u_k), D = ½(∇u + ∇uᵀ).
pub(super) fn shear_rate(gradient: &[V; 3]) -> f64 {
    let mut sum = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let d = gradient[i][j] + gradient[j][i];
            sum += 0.5 * d * d;
        }
    }
    sum.sqrt()
}

/// Viscosity at a face or cell: the Newtonian constant, or the generalized-Newtonian law at
/// the local shear rate (lagged on the current velocity in the outer iteration).
pub(super) fn face_viscosity(p: &CanonicalFluidProblem<'_>, gradient: &[V; 3]) -> f64 {
    match p.rheology {
        None => p.viscosity,
        Some(law) => law.viscosity(shear_rate(gradient)),
    }
}

/// Single face-physics authority (Newtonian or generalized-Newtonian stress) shared by
/// assembly and original recovery.
pub(super) fn viscous_parts(
    p: &CanonicalFluidProblem<'_>,
    i: usize,
    gu: &[[V; 3]],
) -> Result<(f64, V), Error> {
    let f = &p.topology.faces[i];
    let (alpha, tangent, w) = face_geometry(p, f)?;
    let grad = if let Some(other) = f.neighbor {
        std::array::from_fn(|k| interpolate(gu[f.owner][k], gu[other][k], w))
    } else {
        gu[f.owner]
    };
    let normal_diffusion = !matches!(p.boundaries[i], FaceBoundary::PressureOutlet { .. });
    let viscosity = face_viscosity(p, &grad);
    if !viscosity.is_finite() || viscosity <= 0.0 {
        return Err(Error::Invalid("nonpositive or nonfinite face viscosity"));
    }
    let remainder = std::array::from_fn(|k| {
        viscosity
            * ((if normal_diffusion {
                dot(grad[k], tangent)
            } else {
                0.0
            }) + (0..3)
                .map(|j| grad[j][k] * f.owner_area_vector_m2[j])
                .sum::<f64>())
    });
    Ok((
        if normal_diffusion {
            viscosity * alpha
        } else {
            0.0
        },
        remainder,
    ))
}

/// Original cell gradient extrapolated to the actual face centroid, not the
/// centroid-line interpolation point of a skew tetrahedral face.
pub(super) fn face_velocity_trace(
    p: &CanonicalFluidProblem<'_>,
    i: usize,
    c: usize,
    u: &[V],
    gu: &[[V; 3]],
) -> V {
    let d = sub(
        p.topology.faces[i].centroid_m,
        p.topology.cells[c].centroid_m,
    );
    add(u[c], std::array::from_fn(|k| dot(gu[c][k], d)))
}

pub(super) fn face_upwind_velocity(
    p: &CanonicalFluidProblem<'_>,
    i: usize,
    u: &[V],
    q: f64,
    gu: &[[V; 3]],
) -> Result<V, Error> {
    let f = &p.topology.faces[i];
    if f.neighbor.is_none() {
        if let FaceBoundary::Velocity { value, .. } = p.boundaries[i] {
            return Ok(value);
        }
    }
    if q >= 0.0 {
        return Ok(face_velocity_trace(p, i, f.owner, u, gu));
    }
    if let Some(n) = f.neighbor {
        return Ok(face_velocity_trace(p, i, n, u, gu));
    }
    if let FaceBoundary::Velocity { value, .. } = p.boundaries[i] {
        Ok(value)
    } else {
        Err(Error::Unsupported("pressure outlet backflow"))
    }
}

/// Conservative deferred linear-upwind correction; the implicit matrix retains
/// its positive first-order upwind base. Assembly and recovery use this same law.
pub(super) fn advection_remainder(
    p: &CanonicalFluidProblem<'_>,
    i: usize,
    u: &[V],
    q: f64,
    gu: &[[V; 3]],
) -> Result<V, Error> {
    let f = &p.topology.faces[i];
    let base = if q >= 0.0 {
        u[f.owner]
    } else if let Some(n) = f.neighbor {
        u[n]
    } else if let FaceBoundary::Velocity { value, .. } = p.boundaries[i] {
        value
    } else {
        return Err(Error::Unsupported("pressure outlet backflow"));
    };
    Ok(scale(
        sub(face_upwind_velocity(p, i, u, q, gu)?, base),
        p.density * q,
    ))
}
