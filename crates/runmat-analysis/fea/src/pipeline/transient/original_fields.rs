//! Describe original nodal histories without reconstructing or copying their numerical values.
use super::*;
use crate::execution::{
    FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
    SolverResultIdentity, StepIdentity,
};
use crate::results::field::{
    OriginalFieldSourceEntry, OriginalFieldSourcePlan, RecoveredFieldDefinition,
    RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    StructuralRecoveredFieldTopology,
};
use runmat_analysis_core::{FieldLocation, ObservationQuantity, ObservationValueShape};

pub fn original_transient_field_source(
    model: &AnalysisModel,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    producer: StepIdentity,
    source: SolverResultIdentity,
    result: &FeaTransientRunResult,
) -> Result<OriginalFieldSourcePlan, FeaRunError> {
    let build = || {
        let topology = StructuralRecoveredFieldTopology::derive(model, Some(mesh))?;
        let request = FieldRecoveryRequest::new(
            producer,
            FieldRecoverySource::SolverResult(source),
            FieldTopologyIdentity::Structural {
                topology: topology.identity,
            },
            FieldRecoveryPlan::default(),
        )
        .map_err(|error| {
            crate::results::field::RecoveredFieldError::InvalidArtifact(error.to_string())
        })?;
        let mut fields = Vec::new();
        for (quantity, history) in [
            (
                ObservationQuantity::Displacement,
                &result.displacement_snapshots,
            ),
            (ObservationQuantity::Velocity, &result.velocity_snapshots),
        ] {
            if history.len() != result.time_points_s.len() {
                return Err(crate::results::field::RecoveredFieldError::InvalidArtifact(
                    "original transient channel differs from its accepted schedule".into(),
                ));
            }
            for (field, time) in history.iter().zip(&result.time_points_s) {
                let definition = RecoveredFieldDefinition::new(
                    &field.field_id,
                    quantity,
                    FieldLocation::Node,
                    RecoveredFieldEntityDomain::Nodes,
                    ObservationValueShape::Vector3,
                    RecoveredFieldNumericRepresentation::Real,
                )?;
                if field.shape != definition.storage_shape(topology.entity_counts.nodes)? {
                    return Err(crate::results::field::RecoveredFieldError::StorageShapeMismatch);
                }
                fields.push(OriginalFieldSourceEntry {
                    definition,
                    sample: RecoveredFieldSample::TimeSeconds(*time),
                });
            }
        }
        let plan = OriginalFieldSourcePlan {
            request,
            entity_counts: topology.entity_counts,
            fields,
        };
        plan.validate()?;
        Ok(plan)
    };
    build().map_err(|error: crate::results::field::RecoveredFieldError| {
        FeaRunError::Solve(error.to_string())
    })
}
