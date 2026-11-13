//! Physical departure-coordinate residual and shared-CSR Newton tangent.
use super::*;
use crate::algebra::NullspaceEliminationEvidence;

mod radiation;

pub(crate) struct ThermalNonlinearResidual {
    pub(crate) residual_w:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Power, Vec<f64>>,
    /// Largest original absolute physical transfer sum, without a numerical floor.
    pub(crate) physical_power_scale_w: runmat_types::physical::Power,
}

impl ThermalOperatorAssembly {
    pub(super) fn reject_radiation_in_linear_route(
        &self,
        model: &AnalysisModel,
    ) -> Result<(), ThermalAssemblyError> {
        if let Some(boundary) = model
            .boundaries
            .iter()
            .find(|b| matches!(b.operator, BoundaryOperator::ThermalRadiation { .. }))
        {
            return Err(ThermalAssemblyError::RadiationNotAdmitted {
                operator_id: boundary.boundary_id.clone(),
            });
        }
        Ok(())
    }

    pub(crate) fn nonlinear_residual(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        reference: &[f64],
        departure: &[f64],
        previous: Option<&[f64]>,
        dt: f64,
    ) -> Result<ThermalNonlinearResidual, ThermalAssemblyError> {
        let source = self.relative_source(model, sample, reference)?;
        self.validate_nonlinear_state(reference, departure, previous, dt)?;
        let mut residual = apply_matrix(&self.conductivity, departure);
        for (value, source) in residual.iter_mut().zip(source) {
            *value -= source;
        }
        let mut storage_scale = 0.0;
        if let Some(previous) = previous {
            let rate = departure
                .iter()
                .zip(previous)
                .map(|(t, p)| (t - p) / dt)
                .collect::<Vec<_>>();
            let storage = apply_matrix(&self.capacity, &rate);
            for (value, power) in residual.iter_mut().zip(storage) {
                *value += power;
                storage_scale += power.abs();
            }
        }
        let mut boundary_scale = 0.0;
        self.visit_radiation(model, sample, reference, departure, |face, form| {
            boundary_scale += form.absolute_transfer_w;
            for (local, &node) in face.node_indices.iter().enumerate() {
                residual[node] += form.outward_nodal_power_w[local];
            }
        })?;
        for boundary in &model.boundaries {
            let BoundaryOperator::ThermalConvection {
                ambient_temperature,
                coefficient,
            } = &boundary.operator
            else {
                continue;
            };
            let ambient = sample_ambient_temperature(
                model,
                &boundary.boundary_id,
                ambient_temperature,
                sample,
            )?;
            for face in self.projection.boundary_faces_for_target(&boundary.target) {
                self.projection
                    .visit_boundary_quadrature(face, |point| {
                        let difference = face
                            .node_indices
                            .iter()
                            .zip(point.shape_values())
                            .map(|(&node, shape)| {
                                shape * (departure[node] + (reference[node] - ambient))
                            })
                            .sum::<f64>();
                        boundary_scale +=
                            (coefficient.canonical_value() * difference * point.area_weight_m2)
                                .abs();
                    })
                    .map_err(ThermalAssemblyError::from)?;
            }
        }
        let prescribed_scale = residual
            .iter()
            .zip(self.prescribed_temperature_k.canonical_values())
            .filter(|(_, prescribed)| prescribed.is_some())
            .map(|(power, _)| power.abs())
            .sum::<f64>();
        let applied_scale = self
            .applied_source
            .canonical_values()
            .iter()
            .map(|power| power.abs())
            .sum::<f64>();
        let scale = storage_scale
            .max(boundary_scale)
            .max(prescribed_scale)
            .max(applied_scale);
        if residual.iter().any(|v| !v.is_finite()) || !scale.is_finite() {
            return Err(ThermalAssemblyError::InvalidProjection(
                "nonfinite nonlinear physical power".into(),
            ));
        }
        Ok(ThermalNonlinearResidual {
            residual_w: runmat_types::physical::Quantity::new(residual),
            physical_power_scale_w: runmat_types::physical::Power::new(scale),
        })
    }

    pub(crate) fn nonlinear_increment_operator(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        reference: &[f64],
        departure: &[f64],
        previous: Option<&[f64]>,
        dt: f64,
    ) -> Result<
        crate::algebra::PhysicalScalarSystem<runmat_types::physical::kind::TemperatureDifference>,
        ThermalAssemblyError,
    > {
        let physical =
            self.nonlinear_residual(model, sample, reference, departure, previous, dt)?;
        let support = self
            .conductivity
            .pattern()
            .union(self.capacity.pattern())
            .map_err(|e| ThermalAssemblyError::SparseOperator(e.to_string()))?;
        let properties = if previous.is_some() {
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE
        } else {
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE
        };
        let role = if previous.is_some() {
            OperatorRole::EffectiveTangent
        } else {
            OperatorRole::Conductivity
        };
        let mut tangent = RealSparseAssembler::with_pattern(support, role, properties)
            .map_err(|e| ThermalAssemblyError::SparseOperator(e.to_string()))?;
        add_scaled_matrix(&mut tangent, &self.conductivity, 1.0);
        if previous.is_some() {
            add_scaled_matrix(&mut tangent, &self.capacity, dt.recip());
        }
        tangent.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Thermal,
            contribution_id: ContributionId("physical_radiation_exact_newton_tangent".into()),
        });
        let mut anchors = self.positive_robin_weight.canonical_values().clone();
        self.visit_radiation(model, sample, reference, departure, |face, form| {
            for (i, &row) in face.node_indices.iter().enumerate() {
                anchors[row] += form.tangent_w_per_k[i][i];
                for (j, &column) in face.node_indices.iter().enumerate() {
                    tangent.add(row, column, form.tangent_w_per_k[i][j]);
                }
            }
        })?;
        let stiffness = tangent
            .finish()
            .map_err(|e| ThermalAssemblyError::SparseOperator(e.to_string()))?;
        let mut rhs = physical
            .residual_w
            .canonical_values()
            .iter()
            .map(|power| -power)
            .collect::<Vec<_>>();
        let constrained = self
            .prescribed_temperature_k
            .canonical_values()
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        for (node, prescribed) in self
            .prescribed_temperature_k
            .canonical_values()
            .iter()
            .enumerate()
        {
            if let Some(temperature) = prescribed {
                rhs[node] = (temperature - reference[node]) - departure[node];
            }
        }
        let damping = RealSparseMatrix::diagonal(
            vec![0.0; rhs.len()],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|e| ThermalAssemblyError::SparseOperator(e.to_string()))?;
        let operators = CanonicalRealOperators::try_new(stiffness, self.capacity.clone(), damping)
            .map_err(|e| ThermalAssemblyError::SparseOperator(e.to_string()))?;
        let evidence = if previous.is_some() {
            None
        } else {
            match NullspaceEliminationEvidence::prove_scalar_diffusion(
                &self.projection,
                operators.stiffness(),
                &constrained,
                &anchors,
            ) {
                Ok(evidence) => Some(evidence),
                Err(crate::algebra::NullspaceEvidenceError::UnderconstrainedComponent) => None,
                Err(error) => return Err(ThermalAssemblyError::SparseOperator(error.to_string())),
            }
        };
        AssembledRealSystem::try_new_with_nullspace_evidence(operators, constrained, rhs, evidence)
            .map_err(|e| ThermalAssemblyError::SparseOperator(e.to_string()))
            .and_then(|numerical| {
                crate::algebra::PhysicalScalarSystem::try_new(
                    numerical,
                    runmat_types::physical::DynamicDimension::quantity_type(
                        runmat_types::physical::QuantityKind::Power,
                    )
                    .with_semantic_kind(Some(runmat_types::physical::QuantityKind::Power))
                    .expect("thermal equation"),
                )
                .map_err(|error| ThermalAssemblyError::SparseOperator(error.into()))
            })
    }

    fn validate_nonlinear_state(
        &self,
        reference: &[f64],
        departure: &[f64],
        previous: Option<&[f64]>,
        dt: f64,
    ) -> Result<(), ThermalAssemblyError> {
        let n = self.projection.nodes.len();
        if reference.len() != n
            || departure.len() != n
            || previous.is_some_and(|p| p.len() != n || !dt.is_finite() || dt <= 0.0)
        {
            return Err(ThermalAssemblyError::InvalidProjection("nonlinear thermal state requires finite nonnegative Kelvin samples and admitted BE coordinates".into()));
        }
        if reference
            .iter()
            .zip(departure)
            .any(|(r, d)| !r.is_finite() || !d.is_finite() || !(r + d).is_finite() || r + d < 0.0)
            || previous.is_some_and(|p| {
                reference
                    .iter()
                    .zip(p)
                    .any(|(r, d)| !d.is_finite() || !(r + d).is_finite() || r + d < 0.0)
            })
        {
            return Err(ThermalAssemblyError::Radiation(
                crate::physics::thermal::radiation::RadiationError::InvalidState,
            ));
        }
        Ok(())
    }
}
