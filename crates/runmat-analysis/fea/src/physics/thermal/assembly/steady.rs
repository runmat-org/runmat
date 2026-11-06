use runmat_analysis_core::{AnalysisModel, BoundaryOperator};

use super::{sample_ambient_temperature, ThermalAssemblyError, ThermalOperatorAssembly};
use crate::{
    algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties,
        NullspaceEliminationEvidence, OperatorRole, RealSparseMatrix,
    },
    excitation::ExcitationSamplePoint,
};

impl ThermalOperatorAssembly {
    /// Solve for T-reference, with physical loads formed before absolute-temperature cancellation.
    pub(crate) fn steady_increment_operator(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        reference_temperature_k: &[f64],
    ) -> Result<
        crate::algebra::PhysicalScalarSystem<runmat_types::physical::kind::TemperatureDifference>,
        ThermalAssemblyError,
    > {
        self.reject_radiation_in_linear_route(model)?;
        let stiffness = self.conductivity.clone();
        let mut rhs = self.relative_source(model, sample, reference_temperature_k)?;
        for (node, value) in self
            .prescribed_temperature_k
            .canonical_values()
            .iter()
            .enumerate()
        {
            if let Some(value) = value {
                rhs[node] = value - reference_temperature_k[node];
            }
        }
        let constrained = self
            .prescribed_temperature_k
            .canonical_values()
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        let mass = self.capacity.clone();
        let damping = RealSparseMatrix::diagonal(
            vec![0.0; self.projection.nodes.len()],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))?;
        let operators = CanonicalRealOperators::try_new(stiffness, mass, damping)
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))?;
        let evidence = match NullspaceEliminationEvidence::prove_scalar_diffusion(
            &self.projection,
            operators.stiffness(),
            &constrained,
            self.positive_robin_weight.canonical_values(),
        ) {
            Ok(evidence) => Some(evidence),
            Err(crate::algebra::NullspaceEvidenceError::UnderconstrainedComponent) => None,
            Err(error) => return Err(ThermalAssemblyError::SparseOperator(error.to_string())),
        };
        AssembledRealSystem::try_new_with_nullspace_evidence(operators, constrained, rhs, evidence)
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))
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

    /// Shared steady/transient forcing in component-constant temperature coordinates.
    pub(super) fn relative_source(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        reference_temperature_k: &[f64],
    ) -> Result<Vec<f64>, ThermalAssemblyError> {
        if reference_temperature_k.len() != self.projection.nodes.len()
            || reference_temperature_k
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(ThermalAssemblyError::InvalidProjection(
                "thermal reference must contain one finite value per node".into(),
            ));
        }
        if self
            .projection
            .node_neighbors
            .iter()
            .enumerate()
            .any(|(node, neighbors)| {
                neighbors.iter().any(|&neighbor| {
                    reference_temperature_k.get(neighbor) != Some(&reference_temperature_k[node])
                })
            })
        {
            return Err(ThermalAssemblyError::InvalidProjection(
                "thermal reference must be constant on every connected component".into(),
            ));
        }
        let mut rhs = self.applied_source.canonical_values().clone();
        // Internal conduction annihilates a constant by partition of unity. Only Robin
        // terms need a reference correction; form ambient-reference before multiplication.
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
                        let power = coefficient.canonical_value()
                            * (ambient - reference_temperature_k[face.node_indices[0]])
                            * point.area_weight_m2;
                        for (&node, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                            rhs[node] += shape * power;
                        }
                    })
                    .map_err(ThermalAssemblyError::from)?;
            }
        }
        Ok(rhs)
    }
}
