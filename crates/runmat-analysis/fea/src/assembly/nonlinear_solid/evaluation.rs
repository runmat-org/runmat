use super::*;

/// Point kinematics of one integration point in the reference configuration.
///
/// Small strain: `F = I` in the strain-displacement operator and the engineering strain is the
/// symmetric displacement gradient. Total Lagrangian: `F = I + H`, the strain is the
/// Green-Lagrange strain `E = (H + Hᵀ + HᵀH)/2`, the work-conjugate stress is the second
/// Piola-Kirchhoff stress, and the strain-displacement operator is `δE = sym(Fᵀ ∇δu)`.
pub(super) struct PointKinematics {
    pub deformation_gradient: [[f64; 3]; 3],
    /// Engineering order xx, yy, zz, yz, xz, xy (shear components doubled).
    pub strain: [f64; 6],
}

pub(super) fn point_kinematics(
    kinematics: StructuralKinematics,
    element: &PreparedElement,
    point: &PreparedPoint,
    displacement: &[f64],
) -> PointKinematics {
    // H_ij = ∂u_i/∂X_j
    let mut h = [[0.0; 3]; 3];
    for (rows, g) in element.dofs.iter().zip(&point.gradients) {
        for i in 0..3 {
            let u = displacement[rows[i]];
            for j in 0..3 {
                h[i][j] += u * g[j];
            }
        }
    }
    match kinematics {
        StructuralKinematics::SmallStrain => PointKinematics {
            deformation_gradient: IDENTITY,
            strain: [
                h[0][0],
                h[1][1],
                h[2][2],
                h[1][2] + h[2][1],
                h[0][2] + h[2][0],
                h[0][1] + h[1][0],
            ],
        },
        StructuralKinematics::LargeDeformation => {
            let f = std::array::from_fn(|i| std::array::from_fn(|j| IDENTITY[i][j] + h[i][j]));
            // C = FᵀF; E = (C - I)/2, written without cancellation as (H + Hᵀ + HᵀH)/2.
            let e = |i: usize, j: usize| {
                0.5 * (h[i][j] + h[j][i] + (0..3).map(|k| h[k][i] * h[k][j]).sum::<f64>())
            };
            PointKinematics {
                deformation_gradient: f,
                strain: [
                    e(0, 0),
                    e(1, 1),
                    e(2, 2),
                    2.0 * e(1, 2),
                    2.0 * e(0, 2),
                    2.0 * e(0, 1),
                ],
            }
        }
    }
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Strain-displacement columns of one node: row `k` is `∂E/∂u_k` for that node.
pub(super) fn columns(gradient: &[f64; 3], f: &[[f64; 3]; 3]) -> [[f64; 6]; 3] {
    let [x, y, z] = *gradient;
    std::array::from_fn(|k| {
        let r = f[k];
        [
            r[0] * x,
            r[1] * y,
            r[2] * z,
            r[1] * z + r[2] * y,
            r[0] * z + r[2] * x,
            r[0] * y + r[1] * x,
        ]
    })
}

/// Cauchy stress `σ = F S Fᵀ / J` from second Piola-Kirchhoff stress, both in physical-shear order.
pub(super) fn cauchy_stress(f: &[[f64; 3]; 3], s: &[f64; 6]) -> Option<[f64; 6]> {
    let tensor = voigt_tensor(s);
    let det = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
        - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
        + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
    if !det.is_finite() || det <= 0.0 {
        return None;
    }
    let sigma = |i: usize, j: usize| {
        let mut value = 0.0;
        for (a, row) in tensor.iter().enumerate() {
            for (b, s_ab) in row.iter().enumerate() {
                value += f[i][a] * s_ab * f[j][b];
            }
        }
        value / det
    };
    Some([
        sigma(0, 0),
        sigma(1, 1),
        sigma(2, 2),
        sigma(1, 2),
        sigma(0, 2),
        sigma(0, 1),
    ])
}

fn voigt_tensor(s: &[f64; 6]) -> [[f64; 3]; 3] {
    [[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]]
}

impl CanonicalNonlinearSolidBasis {
    pub(crate) fn evaluate(
        &self,
        displacement: &[f64],
        committed: &NonlinearSolidState,
    ) -> Result<NonlinearSolidEvaluation, NonlinearSolidError> {
        if crate::progress::is_cancelled() {
            return Err(NonlinearSolidError::Cancelled);
        }
        self.validate_state(committed)?;
        if displacement.len() != self.dof_count || displacement.iter().any(|u| !u.is_finite()) {
            return Err(NonlinearSolidError::Authority(
                "displacement is not a finite full physical-layout vector".into(),
            ));
        }
        let large = self.kinematics == StructuralKinematics::LargeDeformation;
        let mut tangent = RealSparseAssembler::with_pattern(
            self.tangent_pattern
                .try_clone()
                .map_err(|e| NonlinearSolidError::Tangent(e.to_string()))?,
            OperatorRole::EffectiveTangent,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|e| NonlinearSolidError::Tangent(e.to_string()))?;
        tangent.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Structural,
            contribution_id: ContributionId(
                if large {
                    "physical_j2_total_lagrangian_solid_tangent"
                } else {
                    "physical_j2_solid_tangent"
                }
                .into(),
            ),
        });
        let mut internal_force = vec![0.0; self.dof_count];
        let mut trial_state = self.zero_state();
        let mut point_evidence = Vec::with_capacity(self.point_count());
        let mut element_evidence = Vec::with_capacity(self.elements.len());
        let mut ordinal = 0;
        for element in &self.elements {
            let mut evidence = NonlinearSolidElementEvidence {
                element_identity: element.identity,
                element_id: element.element_id,
                volume_m3: 0.0,
                mean_total_engineering_strain: [0.0; 6],
                mean_stress_pa: [0.0; 6],
                elastic_energy_j: runmat_types::physical::Energy::new(0.0),
                hardening_energy_j: runmat_types::physical::Energy::new(0.0),
                plastic_dissipation_increment_j: 0.0,
            };
            // The admitted Tet4/Tet10 envelope has at most 30 translation columns.
            // Point scratch lives on the stack and is reused for every quadrature point.
            let mut point_columns = [(0usize, [0.0; 6]); 30];
            let mut point_products = [[0.0; 6]; 30];
            let column_count = element.dofs.len() * 3;
            if !matches!(element.dofs.len(), 4 | 10) {
                return Err(NonlinearSolidError::Authority(
                    "unsupported element column inventory".into(),
                ));
            }
            for point in &element.points {
                if crate::progress::is_cancelled() {
                    return Err(NonlinearSolidError::Cancelled);
                }
                if point.gradients.len() != element.dofs.len() {
                    return Err(NonlinearSolidError::Authority(
                        "point gradient inventory differs".into(),
                    ));
                }
                let PointKinematics {
                    deformation_gradient: f,
                    strain,
                } = point_kinematics(self.kinematics, element, point, displacement);
                for (slot, value) in point_columns[..column_count].iter_mut().zip(
                    element
                        .dofs
                        .iter()
                        .zip(&point.gradients)
                        .flat_map(|(rows, g)| rows.iter().copied().zip(columns(g, &f))),
                ) {
                    *slot = value;
                }
                let basis = &point_columns[..column_count];
                let response = element
                    .constitutive
                    .update(strain, &committed.points[ordinal])
                    .map_err(|e| NonlinearSolidError::Constitutive(e.to_string()))?;
                if response
                    .stress_pa
                    .iter()
                    .chain(response.tangent_pa.iter().flatten())
                    .any(|v| !v.is_finite())
                    || [
                        response.elastic_energy_density_pa,
                        response.hardening_energy_density_pa,
                        response.plastic_dissipation_increment_pa,
                    ]
                    .iter()
                    .any(|e| !e.is_finite() || *e < 0.0)
                {
                    return Err(NonlinearSolidError::Constitutive(
                        "nonfinite stress/tangent or invalid constitutive energy".into(),
                    ));
                }
                let cb = &mut point_products[..column_count];
                for (product, (_, b)) in cb.iter_mut().zip(basis) {
                    *product = std::array::from_fn::<_, 6, _>(|i| {
                        response.tangent_pa[i]
                            .iter()
                            .zip(b)
                            .map(|(c, v)| c * v)
                            .sum::<f64>()
                    });
                }
                for (row, b) in basis {
                    internal_force[*row] += point.weight_m3
                        * b.iter()
                            .zip(response.stress_pa)
                            .map(|(v, s)| v * s)
                            .sum::<f64>();
                    for (right, (column, _)) in basis.iter().enumerate() {
                        tangent.add(
                            *row,
                            *column,
                            point.weight_m3
                                * b.iter().zip(cb[right]).map(|(a, c)| a * c).sum::<f64>(),
                        );
                    }
                }
                let stress_pa = if large {
                    // Initial-stress (geometric) stiffness: (∇N_aᵀ S ∇N_b) I for every node pair.
                    let s = voigt_tensor(&response.stress_pa);
                    for (rows_a, ga) in element.dofs.iter().zip(&point.gradients) {
                        let sga: [f64; 3] =
                            std::array::from_fn(|i| (0..3).map(|j| s[i][j] * ga[j]).sum::<f64>());
                        for (rows_b, gb) in element.dofs.iter().zip(&point.gradients) {
                            let value =
                                point.weight_m3 * (0..3).map(|i| sga[i] * gb[i]).sum::<f64>();
                            for k in 0..3 {
                                tangent.add(rows_a[k], rows_b[k], value);
                            }
                        }
                    }
                    cauchy_stress(&f, &response.stress_pa).ok_or_else(|| {
                        NonlinearSolidError::Constitutive(
                            "deformation gradient is not orientation preserving".into(),
                        )
                    })?
                } else {
                    response.stress_pa
                };
                evidence.volume_m3 += point.weight_m3;
                for (i, value) in strain.iter().enumerate() {
                    evidence.mean_total_engineering_strain[i] += point.weight_m3 * value;
                    evidence.mean_stress_pa[i] += point.weight_m3 * stress_pa[i];
                }
                evidence.elastic_energy_j += runmat_types::physical::Energy::new(
                    point.weight_m3 * response.elastic_energy_density_pa,
                );
                evidence.hardening_energy_j += runmat_types::physical::Energy::new(
                    point.weight_m3 * response.hardening_energy_density_pa,
                );
                evidence.plastic_dissipation_increment_j +=
                    point.weight_m3 * response.plastic_dissipation_increment_pa;
                point_evidence.push(NonlinearSolidPointEvidence {
                    identity: point.identity.clone(),
                    position_m: point.position_m,
                    volume_weight_m3: point.weight_m3,
                    total_engineering_strain: strain,
                    stress_pa,
                    elastic_energy_density_pa: response.elastic_energy_density_pa,
                    hardening_energy_density_pa: response.hardening_energy_density_pa,
                    plastic_dissipation_increment_pa: response.plastic_dissipation_increment_pa,
                });
                trial_state.points[ordinal] = response.state;
                ordinal += 1;
            }
            for i in 0..6 {
                evidence.mean_total_engineering_strain[i] /= evidence.volume_m3;
                evidence.mean_stress_pa[i] /= evidence.volume_m3;
            }
            element_evidence.push(evidence);
        }
        if internal_force.iter().any(|v| !v.is_finite()) {
            return Err(NonlinearSolidError::Constitutive(
                "nonfinite assembled internal force".into(),
            ));
        }
        if crate::progress::is_cancelled() {
            return Err(NonlinearSolidError::Cancelled);
        }
        let tangent = tangent
            .finish()
            .map_err(|e| NonlinearSolidError::Tangent(e.to_string()))?;
        if crate::progress::is_cancelled() {
            return Err(NonlinearSolidError::Cancelled);
        }
        Ok(NonlinearSolidEvaluation {
            internal_force,
            tangent,
            trial_state,
            point_evidence,
            element_evidence,
        })
    }
}
