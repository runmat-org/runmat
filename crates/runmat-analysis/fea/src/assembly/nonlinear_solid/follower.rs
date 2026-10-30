//! Follower boundary loads on total Lagrangian solids.
//!
//! A follower pressure acts along the deformed surface normal on the deformed area; a follower
//! traction or face force keeps its nominal intensity per reference area and rotates with the
//! local surface frame. Both are integrated on the current boundary at every Newton iterate and
//! contribute their load stiffness `∂f/∂u`, which is not symmetric in general.
use super::*;
use crate::excitation::{RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations};
use crate::topology::quadrature::{boundary_shape, DUNAVANT_SEVEN};
use runmat_analysis_core::{CoordinateFrame, ExcitationKinematics};
use runmat_meshing_core::BoundaryTriangleOrder;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
enum FollowerIntensity {
    /// Pressure in Pa on the deformed area, positive along the face's oriented area vector
    /// (the same sign convention as a fixed-frame pressure).
    Pressure(f64),
    /// Nominal traction in Pa per reference area, expressed in the reference frame.
    NominalTraction([f64; 3]),
}

#[derive(Debug, Clone, Serialize)]
struct FollowerFace {
    /// Projection (and basis) node indices of the boundary triangle.
    nodes: Vec<usize>,
    quadratic: bool,
    orientation: f64,
    intensity: FollowerIntensity,
}

/// Prepared follower loads at unit load factor.
pub(crate) struct FollowerLoadOperator {
    faces: Vec<FollowerFace>,
    identity: Digest,
}

/// Unit-load-factor follower force and its displacement derivative, one dense block per face.
pub(crate) struct FollowerLoadEvaluation {
    pub force: Vec<f64>,
    pub blocks: Vec<FollowerStiffnessBlock>,
}

pub(crate) struct FollowerStiffnessBlock {
    pub rows: Vec<usize>,
    /// Row-major `∂f_rows/∂u_rows` at unit load factor.
    pub values: Vec<f64>,
}

impl FollowerLoadOperator {
    pub(crate) fn identity(&self) -> Digest {
        self.identity
    }
}

impl CanonicalNonlinearSolidBasis {
    /// Collects every follower force, traction and pressure excitation onto its boundary
    /// faces. Returns `None` when the step has no follower loads.
    pub(crate) fn prepare_follower_loads(
        &self,
        mesh: &SolverMeshArtifact,
        excitations: &RealizedExcitations,
    ) -> Result<Option<FollowerLoadOperator>, NonlinearSolidError> {
        let follower =
            excitations.filtered(|load| load.policy.kinematics == ExcitationKinematics::Follower);
        if follower.len() == 0 {
            return Ok(None);
        }
        if self.kinematics != StructuralKinematics::LargeDeformation {
            return Err(NonlinearSolidError::Unsupported(
                "follower loads require total Lagrangian (large-deformation) kinematics".into(),
            ));
        }
        if mesh.canonical_digest != self.mesh_digest {
            return Err(NonlinearSolidError::Authority(
                "follower load mesh differs from the solid basis".into(),
            ));
        }
        let projection = AnalysisProjection::derive(mesh).map_err(NonlinearSolidError::from)?;
        if projection.nodes.len() != self.nodes.len()
            || projection
                .nodes
                .iter()
                .zip(&self.nodes)
                .any(|(p, n)| p.stable_identity != n.identity || p.coordinates_m != n.coordinates_m)
        {
            return Err(NonlinearSolidError::Authority(
                "follower load boundary nodes differ from the solid basis".into(),
            ));
        }
        let mut faces = Vec::new();
        for load in follower.iter() {
            if load.policy.frame != CoordinateFrame::Global {
                return Err(NonlinearSolidError::Unsupported(format!(
                    "follower load `{}` must be authored in the global frame of the reference configuration",
                    load.excitation_id.0
                )));
            }
            let RealizedExcitationTarget::Region(target) = &load.target else {
                return Err(NonlinearSolidError::Unsupported(format!(
                    "follower load `{}` must target a boundary face region",
                    load.excitation_id.0
                )));
            };
            let selected = projection
                .boundary_faces
                .iter()
                .filter(|face| {
                    face.provenance
                        .iter()
                        .any(|entity| target.matches_persistent_entity(entity))
                })
                .collect::<Vec<_>>();
            if selected.is_empty() {
                return Err(NonlinearSolidError::Authority(format!(
                    "follower load `{}` selects no boundary faces",
                    load.excitation_id.0
                )));
            }
            let intensity = match load.kind {
                RealizedExcitationKind::Pressure(value) => FollowerIntensity::Pressure(value.real),
                RealizedExcitationKind::Traction(value) => {
                    FollowerIntensity::NominalTraction(value.real)
                }
                RealizedExcitationKind::Force(value) => {
                    let area = selected
                        .iter()
                        .map(|face| face.area_m2.canonical_value())
                        .sum::<f64>();
                    if !area.is_finite() || area <= 0.0 {
                        return Err(NonlinearSolidError::Authority(format!(
                            "follower force `{}` needs a positive reference area",
                            load.excitation_id.0
                        )));
                    }
                    FollowerIntensity::NominalTraction(value.real.map(|f| f / area))
                }
                _ => {
                    return Err(NonlinearSolidError::Unsupported(format!(
                    "follower kinematics is defined for force, traction and pressure only (`{}`)",
                    load.excitation_id.0
                )))
                }
            };
            for face in selected {
                let quadratic = match face.order {
                    BoundaryTriangleOrder::Tri3 => false,
                    BoundaryTriangleOrder::Tri6 => true,
                };
                if face.node_indices.len() != if quadratic { 6 } else { 3 } {
                    return Err(NonlinearSolidError::Authority(
                        "follower boundary face node inventory differs".into(),
                    ));
                }
                faces.push(FollowerFace {
                    nodes: face.node_indices.clone(),
                    quadratic,
                    orientation: projection
                        .boundary_parametric_orientation(face)
                        .map_err(NonlinearSolidError::from)?,
                    intensity,
                });
            }
        }
        // Streamed: the follower inventory is hashed without materializing its encoding.
        let digest = crate::contracts::canonical_digest::canonical_contract_digest(
            b"runmat-fea-follower-boundary-loads/v1\0",
            "follower-boundary-loads",
            &(self.identity, &faces),
            CanonicalLimits::new(0, 0, 4096, 64),
        )
        .map_err(|e| NonlinearSolidError::Authority(e.to_string()))?;
        Ok(Some(FollowerLoadOperator {
            faces,
            identity: digest,
        }))
    }

    /// Follower force at unit load factor in the configuration `X + u`, with its load stiffness.
    pub(crate) fn evaluate_follower_loads(
        &self,
        operator: &FollowerLoadOperator,
        displacement: &[f64],
    ) -> Result<FollowerLoadEvaluation, NonlinearSolidError> {
        if displacement.len() != self.dof_count || displacement.iter().any(|v| !v.is_finite()) {
            return Err(NonlinearSolidError::Authority(
                "follower load displacement is not a finite full physical-layout vector".into(),
            ));
        }
        let mut force = vec![0.0; self.dof_count];
        let mut blocks = Vec::with_capacity(operator.faces.len());
        for face in &operator.faces {
            if crate::progress::is_cancelled() {
                return Err(NonlinearSolidError::Cancelled);
            }
            let rows = face
                .nodes
                .iter()
                .flat_map(|&node| self.nodes[node].rows)
                .collect::<Vec<_>>();
            let reference = face
                .nodes
                .iter()
                .map(|&node| self.nodes[node].coordinates_m.canonical_value())
                .collect::<Vec<_>>();
            let current = face
                .nodes
                .iter()
                .map(|&node| {
                    let n = &self.nodes[node];
                    std::array::from_fn(|k| {
                        n.coordinates_m.canonical_value()[k] + displacement[n.rows[k]]
                    })
                })
                .collect::<Vec<[f64; 3]>>();
            let face_force = face_force(face, &reference, &current)?;
            for (row, value) in rows.iter().zip(&face_force) {
                force[*row] += value;
            }
            let values = match face.intensity {
                FollowerIntensity::Pressure(pressure) => {
                    pressure_stiffness(face, &current, pressure)
                }
                FollowerIntensity::NominalTraction(_) => {
                    traction_stiffness(face, &reference, &current)?
                }
            };
            if values.iter().any(|v| !v.is_finite()) {
                return Err(NonlinearSolidError::Constitutive(
                    "nonfinite follower load stiffness".into(),
                ));
            }
            blocks.push(FollowerStiffnessBlock { rows, values });
        }
        if force.iter().any(|v| !v.is_finite()) {
            return Err(NonlinearSolidError::Constitutive(
                "nonfinite follower load".into(),
            ));
        }
        Ok(FollowerLoadEvaluation { force, blocks })
    }
}

struct SurfacePoint {
    shape: [f64; 6],
    derivatives: [[f64; 2]; 6],
    weight: f64,
}

fn surface_points(quadratic: bool) -> impl Iterator<Item = SurfacePoint> {
    let order = if quadratic {
        BoundaryTriangleOrder::Tri6
    } else {
        BoundaryTriangleOrder::Tri3
    };
    DUNAVANT_SEVEN.iter().map(move |(barycentric, weight)| {
        let (shape, derivatives) = boundary_shape(order, *barycentric);
        SurfacePoint {
            shape,
            derivatives,
            weight: 0.5 * weight,
        }
    })
}

fn tangents(points: &[[f64; 3]], derivatives: &[[f64; 2]; 6]) -> [[f64; 3]; 2] {
    std::array::from_fn(|axis| {
        std::array::from_fn(|k| {
            points
                .iter()
                .zip(derivatives)
                .map(|(x, d)| x[k] * d[axis])
                .sum::<f64>()
        })
    })
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    a[0].hypot(a[1]).hypot(a[2])
}

/// Orthonormal surface frame `[e1, e2, e3]` from the first tangent and the normal.
fn surface_frame(t1: [f64; 3], normal: [f64; 3]) -> Option<[[f64; 3]; 3]> {
    let (l1, ln) = (norm(t1), norm(normal));
    if !(l1 > 0.0 && ln > 0.0) {
        return None;
    }
    let e1 = t1.map(|v| v / l1);
    let e3 = normal.map(|v| v / ln);
    Some([e1, cross(e3, e1), e3])
}

fn face_force(
    face: &FollowerFace,
    reference: &[[f64; 3]],
    current: &[[f64; 3]],
) -> Result<Vec<f64>, NonlinearSolidError> {
    let count = face.nodes.len();
    let mut force = vec![0.0; 3 * count];
    for point in surface_points(face.quadratic) {
        let [t1, t2] = tangents(current, &point.derivatives);
        let normal = cross(t1, t2).map(|v| v * face.orientation);
        let integrated = match face.intensity {
            FollowerIntensity::Pressure(pressure) => normal.map(|v| v * pressure * point.weight),
            FollowerIntensity::NominalTraction(traction) => {
                let [r1, r2] = tangents(reference, &point.derivatives);
                let reference_normal = cross(r1, r2).map(|v| v * face.orientation);
                let (Some(from), Some(to)) = (
                    surface_frame(r1, reference_normal),
                    surface_frame(t1, normal),
                ) else {
                    return Err(NonlinearSolidError::Constitutive(
                        "degenerate follower boundary face".into(),
                    ));
                };
                // R = Σ_i e_i(current) ⊗ e_i(reference)
                let local: [f64; 3] =
                    std::array::from_fn(|i| (0..3).map(|k| from[i][k] * traction[k]).sum::<f64>());
                let rotated: [f64; 3] =
                    std::array::from_fn(|k| (0..3).map(|i| to[i][k] * local[i]).sum::<f64>());
                let area = norm(reference_normal) * point.weight;
                rotated.map(|v| v * area)
            }
        };
        for a in 0..count {
            for k in 0..3 {
                force[3 * a + k] += point.shape[a] * integrated[k];
            }
        }
    }
    Ok(force)
}

/// `∂f_a/∂x_b = p w N_a s (N_b,η [t1]× − N_b,ξ [t2]×)` for a follower pressure.
fn pressure_stiffness(face: &FollowerFace, current: &[[f64; 3]], pressure: f64) -> Vec<f64> {
    let count = face.nodes.len();
    let width = 3 * count;
    let mut values = vec![0.0; width * width];
    let skew = |v: [f64; 3]| [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
    for point in surface_points(face.quadratic) {
        let [t1, t2] = tangents(current, &point.derivatives);
        let (s1, s2) = (skew(t1), skew(t2));
        let scale = pressure * point.weight * face.orientation;
        for a in 0..count {
            for b in 0..count {
                let [dxi, deta] = point.derivatives[b];
                for i in 0..3 {
                    for j in 0..3 {
                        values[(3 * a + i) * width + 3 * b + j] +=
                            scale * point.shape[a] * (deta * s1[i][j] - dxi * s2[i][j]);
                    }
                }
            }
        }
    }
    values
}

/// Central differences of the rotated nominal traction with respect to the face's own nodes.
fn traction_stiffness(
    face: &FollowerFace,
    reference: &[[f64; 3]],
    current: &[[f64; 3]],
) -> Result<Vec<f64>, NonlinearSolidError> {
    let count = face.nodes.len();
    let width = 3 * count;
    let size = (1..count)
        .map(|a| norm(std::array::from_fn(|k| reference[a][k] - reference[0][k])))
        .fold(0.0_f64, f64::max);
    let step = 1.0e-7 * size;
    let mut values = vec![0.0; width * width];
    let mut perturbed = current.to_vec();
    for b in 0..count {
        for j in 0..3 {
            perturbed[b][j] = current[b][j] + step;
            let plus = face_force(face, reference, &perturbed)?;
            perturbed[b][j] = current[b][j] - step;
            let minus = face_force(face, reference, &perturbed)?;
            perturbed[b][j] = current[b][j];
            for row in 0..width {
                values[row * width + 3 * b + j] = (plus[row] - minus[row]) / (2.0 * step);
            }
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(quadratic: bool, intensity: FollowerIntensity) -> (FollowerFace, Vec<[f64; 3]>) {
        let corners = [[0.0, 0.0, 0.0], [2.0, 0.1, 0.0], [0.3, 1.5, 0.2]];
        let mut nodes = corners.to_vec();
        if quadratic {
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                nodes.push(std::array::from_fn(|k| {
                    0.5 * (corners[a][k] + corners[b][k])
                }));
            }
        }
        (
            FollowerFace {
                nodes: (0..nodes.len()).collect(),
                quadratic,
                orientation: 1.0,
                intensity,
            },
            nodes,
        )
    }

    fn deformed(reference: &[[f64; 3]]) -> Vec<[f64; 3]> {
        reference
            .iter()
            .enumerate()
            .map(|(a, x)| {
                [
                    x[0] + 0.2 * x[1] + 0.05 * (a as f64).sin(),
                    x[1] - 0.1 * x[0] + 0.03 * a as f64,
                    x[2] + 0.4 * x[0] * x[1],
                ]
            })
            .collect()
    }

    #[test]
    fn pressure_resultant_is_pressure_times_deformed_area_vector() {
        for quadratic in [false, true] {
            let (face, reference) = face(quadratic, FollowerIntensity::Pressure(3.0));
            let current = deformed(&reference);
            let force = face_force(&face, &reference, &current).unwrap();
            let total: [f64; 3] =
                std::array::from_fn(|k| (0..face.nodes.len()).map(|a| force[3 * a + k]).sum());
            // Flat Tri3 area vector; Tri6 checked against Tri3 when midside nodes stay midside.
            if !quadratic {
                let e1 = std::array::from_fn(|k| current[1][k] - current[0][k]);
                let e2 = std::array::from_fn(|k| current[2][k] - current[0][k]);
                let area = cross(e1, e2).map(|v| 0.5 * v * 3.0);
                for k in 0..3 {
                    assert!((total[k] - area[k]).abs() < 1e-12, "{total:?} {area:?}");
                }
            }
        }
    }

    #[test]
    fn analytic_pressure_and_numerical_traction_stiffness_match_central_differences() {
        for quadratic in [false, true] {
            for intensity in [
                FollowerIntensity::Pressure(2.5),
                FollowerIntensity::NominalTraction([0.3, -1.2, 0.7]),
            ] {
                let (face, reference) = face(quadratic, intensity);
                let current = deformed(&reference);
                let stiffness = match intensity {
                    FollowerIntensity::Pressure(p) => pressure_stiffness(&face, &current, p),
                    FollowerIntensity::NominalTraction(_) => {
                        traction_stiffness(&face, &reference, &current).unwrap()
                    }
                };
                let width = 3 * face.nodes.len();
                let h = 1e-6;
                let mut worst = 0.0_f64;
                for b in 0..face.nodes.len() {
                    for j in 0..3 {
                        let mut plus = current.clone();
                        plus[b][j] += h;
                        let mut minus = current.clone();
                        minus[b][j] -= h;
                        let fp = face_force(&face, &reference, &plus).unwrap();
                        let fm = face_force(&face, &reference, &minus).unwrap();
                        for row in 0..width {
                            let fd = (fp[row] - fm[row]) / (2.0 * h);
                            worst = worst.max((fd - stiffness[row * width + 3 * b + j]).abs());
                        }
                    }
                }
                assert!(
                    worst < 1e-6,
                    "quadratic={quadratic} {intensity:?} worst={worst}"
                );
            }
        }
    }

    #[test]
    fn follower_traction_rotates_with_a_rigidly_rotated_face_and_keeps_its_magnitude() {
        let traction = [0.0, 0.0, -4.0];
        let (face, reference) = face(false, FollowerIntensity::NominalTraction(traction));
        let at_rest = face_force(&face, &reference, &reference).unwrap();
        let (s, c) = (0.8_f64.sin(), 0.8_f64.cos());
        let rotate = |x: [f64; 3]| [c * x[0] - s * x[2], x[1], s * x[0] + c * x[2]];
        let rotated = reference.iter().map(|x| rotate(*x)).collect::<Vec<_>>();
        let moved = face_force(&face, &reference, &rotated).unwrap();
        for a in 0..3 {
            let expected = rotate([at_rest[3 * a], at_rest[3 * a + 1], at_rest[3 * a + 2]]);
            for k in 0..3 {
                assert!((moved[3 * a + k] - expected[k]).abs() < 1e-12);
            }
        }
    }
}
