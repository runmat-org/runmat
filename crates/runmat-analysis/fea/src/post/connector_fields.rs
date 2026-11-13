use runmat_analysis_core::AnalysisFieldLayout;
use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
};

use crate::{
    assembly::dofs::StructuralDofLayout,
    connectors::{CanonicalStructuralConnectorPlan, CanonicalStructuralConnectorResult},
    contracts::{
        FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE, FEA_FIELD_STRUCTURAL_CONNECTOR_STORED_ENERGY,
    },
    results::field::{
        RecoveredField, RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldError,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
};

use super::ComplexStructuralFieldRecoveryError;

pub fn connector_observable_field_definitions(
) -> Result<[RecoveredFieldDefinition; 2], RecoveredFieldError> {
    Ok([
        RecoveredFieldDefinition::new(
            FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
            ObservationQuantity::ReactionForce,
            FieldLocation::Connector,
            RecoveredFieldEntityDomain::Connectors,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::Real,
        )?,
        RecoveredFieldDefinition::new(
            FEA_FIELD_STRUCTURAL_CONNECTOR_STORED_ENERGY,
            ObservationQuantity::StrainEnergy,
            FieldLocation::Connector,
            RecoveredFieldEntityDomain::Connectors,
            ObservationValueShape::Scalar,
            RecoveredFieldNumericRepresentation::Real,
        )?,
    ])
}

/// Recovers the peak-amplitude connector-force phasor from one harmonic displacement state.
///
/// The canonical real connector law remains authoritative. Evaluating it for the real and
/// imaginary displacement/velocity parts preserves the exact spring, foundation, and dashpot
/// semantics used by static and transient recovery without introducing a second complex law.
pub fn recover_complex_connector_force_field(
    plan: CanonicalStructuralConnectorPlan,
    layout: &StructuralDofLayout,
    frequency_hz: f64,
    displacement_real: &[f64],
    displacement_imaginary: &[f64],
) -> Result<Option<RecoveredField>, ComplexStructuralFieldRecoveryError> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(RecoveredFieldError::InvalidArtifact(
            "connector phasor recovery requires a positive finite frequency".into(),
        )
        .into());
    }
    let omega = std::f64::consts::TAU * frequency_hz;
    let velocity_real = displacement_imaginary
        .iter()
        .map(|value| -omega * value)
        .collect::<Vec<_>>();
    let velocity_imaginary = displacement_real
        .iter()
        .map(|value| omega * value)
        .collect::<Vec<_>>();
    let real = CanonicalStructuralConnectorResult::recover_dynamic(
        plan.clone(),
        layout,
        displacement_real,
        &velocity_real,
    )?;
    let imaginary = CanonicalStructuralConnectorResult::recover_dynamic(
        plan,
        layout,
        displacement_imaginary,
        &velocity_imaginary,
    )?;
    if real.responses.is_empty() {
        return Ok(None);
    }
    let connector_count = real.responses.len();
    let real = AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
        AnalysisFieldLayout::vector3(
            ObservationQuantity::ConnectorForce,
            FieldLocation::Connector,
        ),
        vec![connector_count, 3],
        real.responses
            .iter()
            .flat_map(|response| response.resultant_force_newtons())
            .collect(),
    );
    let imaginary = AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
        AnalysisFieldLayout::vector3(
            ObservationQuantity::ConnectorForce,
            FieldLocation::Connector,
        ),
        vec![connector_count, 3],
        imaginary
            .responses
            .iter()
            .flat_map(|response| response.resultant_force_newtons())
            .collect(),
    );
    Ok(Some(RecoveredField::new(
        RecoveredFieldDefinition::new(
            FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
            ObservationQuantity::ReactionForce,
            FieldLocation::Connector,
            RecoveredFieldEntityDomain::Connectors,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::ComplexInterleaved,
        )?,
        RecoveredFieldSample::FrequencyHertz(frequency_hz),
        super::complex_fields::interleave(&real, &imaginary)?,
    )?))
}

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
    connector_observable_field_definitions()?
        .into_iter()
        .zip([
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
                AnalysisFieldLayout::vector3(
                    ObservationQuantity::ConnectorForce,
                    FieldLocation::Connector,
                ),
                vec![connector_count, 3],
                forces,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_CONNECTOR_STORED_ENERGY,
                AnalysisFieldLayout::scalar(
                    ObservationQuantity::StrainEnergy,
                    FieldLocation::Connector,
                ),
                vec![connector_count],
                energies,
            ),
        ])
        .map(|(definition, storage)| {
            RecoveredField::new(definition, RecoveredFieldSample::Final, storage)
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

    #[test]
    fn harmonic_recovery_uses_displacement_for_springs_and_velocity_for_dashpots() {
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
                CanonicalConnectorEntry::AxialSpring {
                    connector_id: "spring".into(),
                    source_digest: Digest::sha256(b"spring"),
                    primary: node(0, 1),
                    secondary: node(1, 2),
                    unit_axis: [1.0, 0.0, 0.0],
                    stiffness_newtons_per_meter: 20.0,
                },
                CanonicalConnectorEntry::AxialDashpot {
                    connector_id: "dashpot".into(),
                    source_digest: Digest::sha256(b"dashpot"),
                    primary: node(0, 1),
                    secondary: node(1, 2),
                    unit_axis: [1.0, 0.0, 0.0],
                    damping_newton_seconds_per_meter: 5.0,
                },
            ],
        )
        .unwrap();
        let field = recover_complex_connector_force_field(
            plan,
            &layout,
            2.0,
            &[0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
            &[0.0, 0.0, 0.0, -0.5, 0.0, 0.0],
        )
        .unwrap()
        .unwrap();
        let omega = std::f64::consts::TAU * 2.0;
        assert_eq!(field.storage().shape, vec![2, 3, 2]);
        assert_eq!(
            field.storage().as_host_f64().unwrap(),
            &[
                -5.0 * omega * -0.5,
                5.0 * omega * 0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                5.0,
                -10.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        );
    }

    fn node(node_index: usize, byte: u8) -> CanonicalConnectorNode {
        CanonicalConnectorNode {
            node_index,
            node_identity: StableDigest::from_bytes([byte; 32]),
        }
    }
}
