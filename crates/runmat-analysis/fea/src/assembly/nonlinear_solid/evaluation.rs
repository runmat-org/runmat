use super::*;

pub(super) fn columns(gradient: &[f64; 3]) -> [[f64; 6]; 3] {
    let [x, y, z] = *gradient;
    [
        [x, 0.0, 0.0, 0.0, z, y],
        [0.0, y, 0.0, z, 0.0, x],
        [0.0, 0.0, z, y, x, 0.0],
    ]
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
            contribution_id: ContributionId("physical_j2_solid_tangent".into()),
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
                elastic_energy_j: 0.0,
                hardening_energy_j: 0.0,
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
                for (slot, value) in point_columns[..column_count].iter_mut().zip(
                    element
                        .dofs
                        .iter()
                        .zip(&point.gradients)
                        .flat_map(|(rows, g)| rows.iter().copied().zip(columns(g))),
                ) {
                    *slot = value;
                }
                let basis = &point_columns[..column_count];
                let strain: [f64; 6] = std::array::from_fn(|component| {
                    basis
                        .iter()
                        .map(|(row, b)| b[component] * displacement[*row])
                        .sum()
                });
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
                evidence.volume_m3 += point.weight_m3;
                for (i, value) in strain.iter().enumerate() {
                    evidence.mean_total_engineering_strain[i] += point.weight_m3 * value;
                    evidence.mean_stress_pa[i] += point.weight_m3 * response.stress_pa[i];
                }
                evidence.elastic_energy_j += point.weight_m3 * response.elastic_energy_density_pa;
                evidence.hardening_energy_j +=
                    point.weight_m3 * response.hardening_energy_density_pa;
                evidence.plastic_dissipation_increment_j +=
                    point.weight_m3 * response.plastic_dissipation_increment_pa;
                point_evidence.push(NonlinearSolidPointEvidence {
                    identity: point.identity.clone(),
                    position_m: point.position_m,
                    volume_weight_m3: point.weight_m3,
                    total_engineering_strain: strain,
                    stress_pa: response.stress_pa,
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
