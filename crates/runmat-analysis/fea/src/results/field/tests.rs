use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
};
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use crate::topology::AnalysisProjection;

use super::*;

fn projection() -> AnalysisProjection {
    AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet4)).unwrap()
}

fn temperature_definition(label: &str) -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        label,
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::StructuralNodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap()
}

fn temperature_field(label: &str) -> AnalysisField {
    AnalysisField::host_f64(label, vec![4], vec![300.0, 301.0, 302.0, 303.0])
}

fn source_result() -> crate::execution::SolverResultIdentity {
    let sequence = crate::execution::SequenceIdentity::new(Digest::sha256("sequence")).unwrap();
    let step = crate::execution::StepIdentity::new(sequence, Digest::sha256("step")).unwrap();
    crate::execution::SolverResultIdentity::from_step_result(step, Digest::sha256("result"))
        .unwrap()
}

fn recovery_request() -> crate::execution::FieldRecoveryRequestIdentity {
    let sequence = crate::execution::SequenceIdentity::new(Digest::sha256("sequence")).unwrap();
    let step = crate::execution::StepIdentity::new(sequence, Digest::sha256("step")).unwrap();
    crate::execution::FieldRecoveryRequest::new(
        step,
        source_result(),
        crate::constraints::CanonicalStructuralTopologyIdentity::Explicit {
            structural_topology_digest: Digest::sha256("topology"),
        },
    )
    .unwrap()
    .identity()
}

#[test]
fn definition_identity_is_semantic_not_a_display_label() {
    let first = temperature_definition("temperature.primary");
    let second = temperature_definition("temperature.for_display");

    assert_eq!(first.identity(), second.identity());
}

#[test]
fn catalog_rejects_semantic_ambiguity_and_topology_shape_mismatch() {
    let projection = projection();
    let first_definition = temperature_definition("temperature.primary");
    let second_definition = temperature_definition("temperature.duplicate");
    let first = RecoveredField::new(
        first_definition,
        RecoveredFieldSample::Final,
        temperature_field("temperature.primary"),
    )
    .unwrap();
    let second = RecoveredField::new(
        second_definition,
        RecoveredFieldSample::Final,
        temperature_field("temperature.duplicate"),
    )
    .unwrap();
    assert_eq!(
        RecoveredFieldCatalog::new(&projection, vec![first, second]).unwrap_err(),
        RecoveredFieldError::CatalogDuplicateSample
    );

    let malformed = RecoveredField::new(
        temperature_definition("temperature.short"),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64("temperature.short", vec![3], vec![1.0, 2.0, 3.0]),
    )
    .unwrap();
    assert_eq!(
        RecoveredFieldCatalog::new(&projection, vec![malformed]).unwrap_err(),
        RecoveredFieldError::StorageShapeMismatch
    );
}

#[test]
fn constructors_and_deserialization_reject_shape_identity_and_sample_substitution() {
    assert_eq!(
        RecoveredFieldDefinition::new(
            "temperature.invalid",
            ObservationQuantity::Temperature,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::StructuralNodes,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap_err(),
        RecoveredFieldError::QuantityShapeMismatch
    );
    assert_eq!(
        RecoveredField::new(
            temperature_definition("temperature.negative-zero"),
            RecoveredFieldSample::TimeSeconds(-0.0),
            temperature_field("temperature.negative-zero"),
        )
        .unwrap_err(),
        RecoveredFieldError::InvalidSample
    );

    let definition = temperature_definition("temperature.identity");
    let mut encoded = serde_json::to_value(definition).unwrap();
    encoded["quantity"] = serde_json::json!("heat_flux");
    assert!(serde_json::from_value::<RecoveredFieldDefinition>(encoded).is_err());
}

#[test]
fn recovered_field_artifact_binds_source_topology_and_content() {
    let projection = projection();
    let field = RecoveredField::new(
        temperature_definition("temperature.primary"),
        RecoveredFieldSample::Final,
        temperature_field("temperature.primary"),
    )
    .unwrap();
    let artifact = RecoveredFieldArtifact::new(
        recovery_request(),
        source_result(),
        projection.projection_digest,
        RecoveredFieldEntityCounts {
            structural_nodes: projection.nodes.len(),
            solid_elements: projection.elements.len(),
            beam_elements: 0,
            shell_elements: 0,
            constraint_groups: 0,
        },
        vec![field],
    )
    .unwrap();
    let decoded =
        RecoveredFieldArtifact::from_canonical_bytes(&artifact.canonical_bytes().unwrap()).unwrap();
    assert_eq!(decoded, artifact);
    assert!(RecoveredFieldCatalog::from_artifact(&projection, decoded).is_ok());

    let mut encoded = serde_json::to_value(&artifact).unwrap();
    encoded["topology_digest"] = serde_json::to_value(Digest::sha256("substituted")).unwrap();
    assert!(serde_json::from_value::<RecoveredFieldArtifact>(encoded).is_err());
}

#[test]
fn field_manifest_resolves_semantics_without_values_and_reconstructs_exact_content() {
    let projection = projection();
    let final_field = RecoveredField::new(
        temperature_definition("temperature.final"),
        RecoveredFieldSample::Final,
        temperature_field("temperature.final"),
    )
    .unwrap();
    let time_field = RecoveredField::new(
        temperature_definition("temperature.time"),
        RecoveredFieldSample::TimeSeconds(1.0),
        temperature_field("temperature.time"),
    )
    .unwrap();
    let artifact = RecoveredFieldArtifact::new(
        recovery_request(),
        source_result(),
        projection.projection_digest,
        RecoveredFieldEntityCounts {
            structural_nodes: projection.nodes.len(),
            solid_elements: projection.elements.len(),
            beam_elements: 0,
            shell_elements: 0,
            constraint_groups: 0,
        },
        vec![final_field.clone(), time_field.clone()],
    )
    .unwrap();
    let manifest = artifact.manifest().unwrap();
    assert_eq!(manifest.identity, artifact.identity());
    assert_eq!(manifest.fields.len(), 2);
    assert_eq!(manifest.fields[0].field, final_field.identity());
    assert_eq!(
        manifest.fields[1].sample,
        RecoveredFieldSample::TimeSeconds(1.0)
    );
    let decoded_manifest =
        RecoveredFieldArtifactManifest::from_canonical_bytes(&manifest.canonical_bytes().unwrap())
            .unwrap();
    let decoded_field =
        RecoveredField::from_canonical_bytes(&final_field.canonical_bytes().unwrap()).unwrap();
    assert_eq!(decoded_field, final_field);
    assert_eq!(
        RecoveredFieldArtifact::from_manifest_and_fields(
            decoded_manifest.clone(),
            vec![final_field.clone(), time_field.clone()],
        )
        .unwrap(),
        artifact
    );

    let substituted = RecoveredField::new(
        temperature_definition("temperature.time"),
        RecoveredFieldSample::TimeSeconds(1.0),
        AnalysisField::host_f64(
            "temperature.time",
            vec![4],
            vec![900.0, 901.0, 902.0, 903.0],
        ),
    )
    .unwrap();
    assert!(RecoveredFieldArtifact::from_manifest_and_fields(
        decoded_manifest,
        vec![final_field, substituted],
    )
    .is_err());
}

#[test]
fn field_storage_partitions_are_canonical_complete_and_reconstruct_exactly() {
    let entity_count = RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 3;
    let definition = temperature_definition("temperature.large");
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            "temperature.large",
            vec![entity_count],
            (0..entity_count).map(|index| index as f64).collect(),
        ),
    )
    .unwrap();
    let partitions = canonical_field_partitions(&field).unwrap();
    assert_eq!(partitions.len(), 2);
    assert_eq!(
        partitions[0].source_indices.len(),
        RECOVERED_FIELD_PARTITION_ENTITY_COUNT
    );
    assert_eq!(
        partitions[1].source_indices,
        vec![entity_count - 3, entity_count - 2, entity_count - 1]
    );

    let reconstructed = reconstruct_recovered_field(
        field.identity(),
        definition,
        RecoveredFieldSample::Final,
        partitions,
        entity_count,
    )
    .unwrap();
    assert_eq!(reconstructed, field);

    let artifact = RecoveredFieldArtifact::new(
        recovery_request(),
        source_result(),
        Digest::sha256("large-topology"),
        RecoveredFieldEntityCounts {
            structural_nodes: entity_count,
            solid_elements: 1,
            beam_elements: 0,
            shell_elements: 0,
            constraint_groups: 0,
        },
        vec![field],
    )
    .unwrap();
    let manifest = artifact.manifest().unwrap();
    assert_eq!(manifest.fields[0].partitions.len(), 2);
    let mut invalid = manifest;
    invalid.fields[0].partitions[1].first_source_index = 1;
    assert!(invalid.validate().is_err());
}
