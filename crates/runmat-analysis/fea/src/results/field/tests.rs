use runmat_analysis_core::{AnalysisFieldLayout};
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
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap()
}

fn temperature_field(label: &str) -> AnalysisField {
    AnalysisField::host_f64(label, AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node), vec![4], vec![300.0, 301.0, 302.0, 303.0])
}

#[test]
fn partition_shape_profile_bounds_actual_contract_without_numeric_storage() {
    for count in [1, 4, RECOVERED_FIELD_PARTITION_ENTITY_COUNT] {
        let definition = temperature_definition("temperature.profile");
        let sample = RecoveredFieldSample::TimeSeconds(1.25);
        let field = RecoveredField::new(
            definition.clone(),
            sample,
            AnalysisField::host_f64(definition.field_id(), definition.layout(), vec![count], vec![f64::MAX; count]),
        )
        .unwrap();
        let partition = RecoveredFieldPartition::select(&field, (0..count).collect()).unwrap();
        let profile = RecoveredFieldPartition::profile_shape(&definition, sample, count).unwrap();
        let actual = runmat_canonical_codec::profile_serialized_shape(
            &partition,
            runmat_canonical_codec::CanonicalLimits::new(
                64 * 1024 * 1024,
                4_000_000,
                4_000_000,
                32,
            ),
        )
        .unwrap();
        assert_eq!(profile, actual);
        let bytes = partition.canonical_bytes().unwrap();
        assert!(bytes.len() as u64 <= profile.encoded_bytes_upper_bound);
        assert_eq!(
            RecoveredFieldPartition::from_canonical_bytes(&bytes).unwrap(),
            partition
        );
        // Explicit pre-refactor wire contract protects field names, identity transparency and storage.
        let legacy = serde_json::json!({
            "schema_version": 1, "identity": partition.identity,
            "source_field": partition.source_field, "definition": definition,
            "sample": sample, "source_indices": partition.source_indices,
            "storage": partition.storage,
        });
        assert_eq!(serde_json::to_value(&partition).unwrap(), legacy);
    }
    let definition = temperature_definition("temperature.profile");
    for count in [0, RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 1, usize::MAX] {
        assert!(RecoveredFieldPartition::profile_shape(
            &definition,
            RecoveredFieldSample::Final,
            count
        )
        .is_err());
    }
    assert!(RecoveredFieldPartition::profile_shape(
        &definition,
        RecoveredFieldSample::TimeSeconds(f64::NAN),
        1
    )
    .is_err());
}

#[test]
fn partition_shape_profile_covers_vector_and_tensor_widths() {
    for (quantity, shape) in [
        (
            ObservationQuantity::HeatFlux,
            ObservationValueShape::Vector3,
        ),
        (
            ObservationQuantity::Stress,
            ObservationValueShape::SymmetricTensor3,
        ),
    ] {
        let definition = RecoveredFieldDefinition::new(
            "profile.components",
            quantity,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            shape,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap();
        let count = 7;
        let storage = AnalysisField::host_f64(
            definition.field_id(), definition.layout(),
            definition.storage_shape(count).unwrap(),
            vec![-f64::MAX; count * definition.storage_component_count().unwrap()],
        );
        let field =
            RecoveredField::new(definition.clone(), RecoveredFieldSample::Final, storage).unwrap();
        let partition = RecoveredFieldPartition::select(&field, vec![0, 2, 6]).unwrap();
        let profile =
            RecoveredFieldPartition::profile_shape(&definition, RecoveredFieldSample::Final, 3)
                .unwrap();
        let actual = runmat_canonical_codec::profile_serialized_shape(
            &partition,
            runmat_canonical_codec::CanonicalLimits::new(
                64 * 1024 * 1024,
                4_000_000,
                4_000_000,
                32,
            ),
        )
        .unwrap();
        assert_eq!(profile, actual);
        assert!(
            partition.canonical_bytes().unwrap().len() as u64 <= profile.encoded_bytes_upper_bound
        );
    }
}

#[test]
fn borrowed_field_identity_matches_owned_contract_and_rejects_bad_storage() {
    let definition = temperature_definition("temperature");
    let storage = temperature_field("temperature");
    let sample = RecoveredFieldSample::TimeSeconds(0.5);
    let identity = RecoveredField::identity_for(&definition, sample, &storage).unwrap();
    let owned = RecoveredField::new(definition.clone(), sample, storage.clone()).unwrap();
    assert_eq!(identity, owned.identity());
    let roundtrip =
        RecoveredField::from_canonical_bytes(&owned.canonical_bytes().unwrap()).unwrap();
    assert_eq!(roundtrip, owned);
    let mut malformed = storage.clone();
    malformed.field_id = "other".into();
    assert!(RecoveredField::identity_for(&definition, sample, &malformed).is_err());
    malformed = storage;
    malformed.shape = vec![2, 2];
    assert!(RecoveredField::identity_for(&definition, sample, &malformed).is_err());
    assert!(RecoveredField::new(definition, sample, malformed).is_err());
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
        crate::execution::FieldRecoverySource::SolverResult(source_result()),
        crate::execution::FieldTopologyIdentity::Structural {
            topology: crate::constraints::CanonicalStructuralTopologyIdentity::Explicit {
                structural_topology_digest: Digest::sha256("topology"),
            },
        },
        crate::execution::FieldRecoveryPlan::default(),
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
fn transferred_field_identity_retains_source_provenance() {
    let native = RecoveredFieldDefinition::new(
        "stress.native",
        ObservationQuantity::VonMisesStress,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let transferred = RecoveredFieldDefinition::new_derived(
        "stress.recovered",
        ObservationQuantity::VonMisesStress,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
        RecoveredFieldDerivation::LocationTransfer {
            source_location: FieldLocation::Element,
            recovery: runmat_analysis_core::RecoveryPolicy::VolumeWeightedAverage,
        },
    )
    .unwrap();

    assert_ne!(native.identity(), transferred.identity());
    assert!(!native.satisfies_recovery(
        FieldLocation::Element,
        FieldLocation::Node,
        runmat_analysis_core::RecoveryPolicy::VolumeWeightedAverage,
    ));
    assert!(transferred.satisfies_recovery(
        FieldLocation::Element,
        FieldLocation::Node,
        runmat_analysis_core::RecoveryPolicy::VolumeWeightedAverage,
    ));
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
        AnalysisField::host_f64("temperature.short", AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node), vec![3], vec![1.0, 2.0, 3.0]),
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
            RecoveredFieldEntityDomain::Nodes,
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
        crate::execution::FieldRecoverySource::SolverResult(source_result()),
        projection.projection_digest,
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: projection.elements.len() * 4,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
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
        crate::execution::FieldRecoverySource::SolverResult(source_result()),
        projection.projection_digest,
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: projection.elements.len() * 4,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
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
            "temperature.time", AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node),
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
            "temperature.large", definition.layout(),
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
        crate::execution::FieldRecoverySource::SolverResult(source_result()),
        Digest::sha256("large-topology"),
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: entity_count,
            volume_elements: 1,
            volume_integration_points: 4,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
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

#[test]
fn integration_point_tensor_storage_pages_and_reconstructs_exactly() {
    let entity_count = RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 1;
    let definition = RecoveredFieldDefinition::new(
        crate::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
        ObservationQuantity::Stress,
        FieldLocation::IntegrationPoint,
        RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ObservationValueShape::SymmetricTensor3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let values = (0..entity_count * 6)
        .map(|index| index as f64 * 0.25)
        .collect::<Vec<_>>();
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            crate::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS, definition.layout(),
            vec![entity_count, 6],
            values,
        ),
    )
    .unwrap();
    let partitions = canonical_field_partitions(&field).unwrap();

    assert_eq!(partitions.len(), 2);
    assert_eq!(
        partitions[0].source_indices.len(),
        RECOVERED_FIELD_PARTITION_ENTITY_COUNT
    );
    assert_eq!(partitions[1].source_indices, vec![entity_count - 1]);
    assert_eq!(partitions[1].storage.shape, vec![1, 6]);
    assert_eq!(
        reconstruct_recovered_field(
            field.identity(),
            definition,
            RecoveredFieldSample::Final,
            partitions,
            entity_count,
        )
        .unwrap(),
        field
    );
}

#[test]
fn physical_storage_admission_retains_quantity_and_rejects_nonfinite_values() {
    let definition = temperature_definition("temperature.physical");
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(definition.field_id(), definition.layout(), vec![2], vec![273.15, 300.0]),
    )
    .unwrap();
    let physical = field.physical_host_field().unwrap();
    assert_eq!(
        physical.component(0).unwrap().semantic_kind(),
        Some(runmat_types::physical::QuantityKind::Temperature)
    );
    assert!(RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(definition.field_id(), definition.layout(), vec![1], vec![f64::NAN])
    )
    .is_err());
    // A diagnostic name alone does not define physical units: modal eigenvalues
    // and buckling factors, for example, cannot share an invented Ratio header.
    assert!(RecoveredFieldDefinition::new(
        "eigenvalue",
        ObservationQuantity::Eigenvalue,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real
    )
    .is_err());
}
