use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
};

use crate::{
    connectors::CanonicalStructuralConnectorResult,
    contracts::{
        FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE, FEA_FIELD_STRUCTURAL_CONNECTOR_STORED_ENERGY,
    },
    results::field::{
        RecoveredField, RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldError,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
};

/// Publishes one force vector and one stored-energy scalar for each canonical connector.
///
/// Entry order is the connector plan's stable identity order. This preserves exact ownership for
/// connector-target observations and avoids reconstructing connector response from nodal fields.
pub fn recover_connector_observable_fields(
    result: &CanonicalStructuralConnectorResult,
) -> Result<Vec<RecoveredField>, RecoveredFieldError> {
    result
        .validate()
        .map_err(|error| RecoveredFieldError::InvalidArtifact(error.to_string()))?;
    if result.responses.is_empty() {
        return Ok(Vec::new());
    }
    let connector_count = result.responses.len();
    let forces = result
        .responses
        .iter()
        .flat_map(|response| response.resultant_force_newtons())
        .collect();
    let energies = result
        .responses
        .iter()
        .map(|response| response.stored_energy_joules())
        .collect();
    [
        (
            FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
            ObservationQuantity::ReactionForce,
            ObservationValueShape::Vector3,
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
                vec![connector_count, 3],
                forces,
            ),
        ),
        (
            FEA_FIELD_STRUCTURAL_CONNECTOR_STORED_ENERGY,
            ObservationQuantity::StrainEnergy,
            ObservationValueShape::Scalar,
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_CONNECTOR_STORED_ENERGY,
                vec![connector_count],
                energies,
            ),
        ),
    ]
    .into_iter()
    .map(|(field_id, quantity, shape, storage)| {
        RecoveredField::new(
            RecoveredFieldDefinition::new(
                field_id,
                quantity,
                FieldLocation::Connector,
                RecoveredFieldEntityDomain::Connectors,
                shape,
                RecoveredFieldNumericRepresentation::Real,
            )?,
            RecoveredFieldSample::Final,
            storage,
        )
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use runmat_execution::Digest;
    use runmat_meshing_core::StableDigest;

    use super::*;
    use crate::{
        assembly::dofs::{StructuralDofLayout, StructuralNodeDofSet},
        connectors::{
            CanonicalConnectorEntry, CanonicalConnectorNode, CanonicalStructuralConnectorPlan,
        },
        physics::structural::topology::CanonicalStructuralTopologyIdentity,
    };

    #[test]
    fn recovery_preserves_canonical_connector_order_and_exact_values() {
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational(),
            StructuralNodeDofSet::translational(),
        ]);
        let plan = CanonicalStructuralConnectorPlan::new(
            CanonicalStructuralTopologyIdentity::Explicit {
                structural_topology_digest: Digest::sha256(b"topology"),
            },
            2,
            Digest::sha256(b"authored"),
            vec![
                CanonicalConnectorEntry::IsotropicGroundSpring {
                    connector_id: "z-ground".into(),
                    source_digest: Digest::sha256(b"ground"),
                    target: node(1, 2),
                    stiffness_newtons_per_meter: 10.0,
                },
                CanonicalConnectorEntry::AxialSpring {
                    connector_id: "a-axial".into(),
                    source_digest: Digest::sha256(b"axial"),
                    primary: node(0, 1),
                    secondary: node(1, 2),
                    unit_axis: [1.0, 0.0, 0.0],
                    stiffness_newtons_per_meter: 100.0,
                },
            ],
        )
        .unwrap();
        let result = CanonicalStructuralConnectorResult::recover_static(
            plan,
            &layout,
            &[0.0, 0.0, 0.0, 0.25, -0.5, 1.0],
        )
        .unwrap();

        let fields = recover_connector_observable_fields(&result).unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(
            fields[0].storage().as_host_f64().unwrap(),
            &[25.0, 0.0, 0.0, -2.5, 5.0, -10.0]
        );
        assert_eq!(fields[1].storage().as_host_f64().unwrap(), &[3.125, 6.5625]);
        assert_eq!(fields[0].storage().shape, vec![2, 3]);
        assert_eq!(fields[1].storage().shape, vec![2]);
    }

    fn node(node_index: usize, byte: u8) -> CanonicalConnectorNode {
        CanonicalConnectorNode {
            node_index,
            node_identity: StableDigest::from_bytes([byte; 32]),
        }
    }
}
