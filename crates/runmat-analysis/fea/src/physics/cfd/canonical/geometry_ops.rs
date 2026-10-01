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

/// Each local 3x3 stencil uses neighbour differences or the exact boundary law.
/// Neumann rows use unit physical normals. No pseudo-inverse, diagonal floor or
/// Cartesian projection silently repairs a rank-deficient stencil.
pub(super) fn gradient(
    p: &CanonicalFluidProblem<'_>,
    values: &[f64],
    velocity_component: Option<usize>,
) -> Result<Vec<V>, Error> {
    if velocity_component.is_none() {
        return super::pressure_gradient::evaluate(p, values);
    }
    let mut out = Vec::with_capacity(p.topology.cells.len());
    for (c, cell) in p.topology.cells.iter().enumerate() {
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        for incidence in cell.faces {
            let f = &p.topology.faces[incidence.face_index];
            let other = if c == f.owner {
                f.neighbor
            } else {
                Some(f.owner)
            };
            let (d, delta) = if let Some(n) = other {
                (
                    sub(p.topology.cells[n].centroid_m, cell.centroid_m),
                    values[n] - values[c],
                )
            } else {
                match (&p.boundaries[incidence.face_index], velocity_component) {
                    (FaceBoundary::Velocity { value, .. }, Some(k)) => {
                        (sub(f.centroid_m, cell.centroid_m), value[k] - values[c])
                    }
                    (FaceBoundary::PressureOutlet { pressure, .. }, None) => {
                        (sub(f.centroid_m, cell.centroid_m), pressure - values[c])
                    }
                    _ => (scale(f.owner_area_vector_m2, 1.0 / f.area_m2), 0.0),
                }
            };
            let dsq = dot(d, d);
            if !dsq.is_finite() || dsq <= 0.0 {
                return Err(Error::Invalid("gradient stencil distance"));
            }
            for i in 0..3 {
                b[i] += d[i] * delta / dsq;
                for j in 0..3 {
                    a[i][j] += d[i] * d[j] / dsq;
                }
            }
        }
        out.push(solve_three(a, b)?);
    }
    Ok(out)
}

pub(super) fn solve_three(mut a: [[f64; 3]; 3], mut b: V) -> Result<V, Error> {
    let scale = a.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    for k in 0..3 {
        let pivot = (k..3)
            .max_by(|i, j| a[*i][k].abs().total_cmp(&a[*j][k].abs()))
            .unwrap();
        if a[pivot][k].abs() <= scale * f64::EPSILON * 128.0 {
            return Err(Error::Invalid("rank-deficient physical gradient stencil"));
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..3 {
            let q = a[i][k] / a[k][k];
            for j in k..3 {
                a[i][j] -= q * a[k][j];
            }
            b[i] -= q * b[k];
        }
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        x[i] = (b[i] - (i + 1..3).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("nonfinite physical gradient"));
    }
    Ok(x)
}

pub(super) fn velocity_gradients(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
) -> Result<Vec<[V; 3]>, Error> {
    let g = (0..3)
        .map(|k| gradient(p, &u.iter().map(|v| v[k]).collect::<Vec<_>>(), Some(k)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((0..u.len())
        .map(|c| std::array::from_fn(|k| g[k][c]))
        .collect())
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
        let pp = pressure[f.owner] - p.pressure_reference_pa[f.owner]
            + dot(
                g[f.owner],
                sub(f.centroid_m, p.topology.cells[f.owner].centroid_m),
            );
        let pn = pressure[n] - p.pressure_reference_pa[n]
            + dot(g[n], sub(f.centroid_m, p.topology.cells[n].centroid_m));
        w * pp + (1.0 - w) * pn
    } else {
        match p.boundaries[i] {
            FaceBoundary::PressureOutlet { pressure, .. } => {
                pressure - p.pressure_reference_pa[f.owner]
            }
            _ => {
                pressure[f.owner] - p.pressure_reference_pa[f.owner]
                    + dot(
                        g[f.owner],
                        sub(f.centroid_m, p.topology.cells[f.owner].centroid_m),
                    )
            }
        }
    })
}

/// Single Newtonian face-physics authority shared by assembly and original recovery.
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
    let remainder = std::array::from_fn(|k| {
        p.viscosity
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
            p.viscosity * alpha
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
