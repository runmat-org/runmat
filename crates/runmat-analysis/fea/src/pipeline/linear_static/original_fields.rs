//! Original static displacement channel; output omission never triggers another recovery.
use crate::{
    execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
        SequenceIdentity, SolverResultIdentity, StepIdentity, StructuralLinearWorkload,
    },
    results::field::{
        OriginalFieldSourceEntry, OriginalFieldSourcePlan, RecoveredFieldDefinition,
        RecoveredFieldEntityDomain, RecoveredFieldError, RecoveredFieldNumericRepresentation,
        RecoveredFieldSample, StructuralRecoveredFieldTopology,
    },
    FeaRunResult, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
};
use runmat_analysis_core::{FieldLocation, ObservationQuantity, ObservationValueShape};
use runmat_meshing_core::SolverMeshArtifact;

pub fn original_static_displacement_source(
    workload: &StructuralLinearWorkload,
    mesh: Option<&SolverMeshArtifact>,
    source: SolverResultIdentity,
    result: &FeaRunResult,
) -> Result<Option<OriginalFieldSourcePlan>, RecoveredFieldError> {
    let Some(field) = result.field(FEA_FIELD_STRUCTURAL_DISPLACEMENT) else {
        return Ok(None);
    };
    if field.as_host_f64().is_none() {
        return Ok(None);
    }
    let invalid = |error: String| RecoveredFieldError::InvalidArtifact(error);
    let step = workload
        .procedure_step()
        .map_err(|error| invalid(error.to_string()))?;
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence)
        .map_err(|error| invalid(error.to_string()))?;
    let producer = StepIdentity::from_solver_contract(sequence, step)
        .map_err(|error| invalid(error.to_string()))?;
    let topology = StructuralRecoveredFieldTopology::derive(&workload.model, mesh)?;
    let definition = RecoveredFieldDefinition::new(
        FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        ObservationQuantity::Displacement,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )?;
    if field.shape != definition.storage_shape(topology.entity_counts.nodes)? {
        return Err(RecoveredFieldError::StorageShapeMismatch);
    }
    let request = FieldRecoveryRequest::new(
        producer,
        FieldRecoverySource::SolverResult(source),
        FieldTopologyIdentity::Structural {
            topology: topology.identity,
        },
        FieldRecoveryPlan::default(),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let plan = OriginalFieldSourcePlan {
        request,
        entity_counts: topology.entity_counts,
        fields: vec![OriginalFieldSourceEntry {
            definition,
            sample: RecoveredFieldSample::Final,
        }],
    };
    plan.validate()?;
    Ok(Some(plan))
}
