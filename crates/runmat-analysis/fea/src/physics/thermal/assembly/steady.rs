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
    ) -> Result<AssembledRealSystem, ThermalAssemblyError> {
        if reference_temperature_k.len() != self.projection.nodes.len()
            || reference_temperature_k
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(ThermalAssemblyError::InvalidProjection(
                "steady thermal reference must contain one finite value per node".into(),
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
                "steady thermal reference must be constant on every connected component".into(),
            ));
        }
        let stiffness = self.conductivity.clone();
        let mut rhs = self.applied_source.clone();
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
                        let power = coefficient.watts_per_square_meter_kelvin
                            * (ambient - reference_temperature_k[face.node_indices[0]])
                            * point.area_weight_m2;
                        for (&node, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                            rhs[node] += shape * power;
                        }
                    })
                    .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
            }
        }
        for (node, value) in self.prescribed_temperature_k.iter().enumerate() {
            if let Some(value) = value {
                rhs[node] = value - reference_temperature_k[node];
            }
        }
        let constrained = self
            .prescribed_temperature_k
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
        let evidence = match NullspaceEliminationEvidence::prove_thermal(
            &self.projection,
            operators.stiffness(),
            &constrained,
            &self.positive_robin_weight,
        ) {
            Ok(evidence) => Some(evidence),
            Err(crate::algebra::NullspaceEvidenceError::UnderconstrainedComponent) => None,
            Err(error) => return Err(ThermalAssemblyError::SparseOperator(error.to_string())),
        };
        AssembledRealSystem::try_new_with_nullspace_evidence(operators, constrained, rhs, evidence)
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))
    }
}
