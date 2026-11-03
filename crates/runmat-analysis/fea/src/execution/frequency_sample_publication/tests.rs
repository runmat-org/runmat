use runmat_analysis_core::{AnalysisFieldLayout, ComplexRepresentation};
use super::*;
mod observations;
use crate::execution::{
    FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequirement, SequenceIdentity,
    StructuralFrequencyEntryResultIdentity,
};
use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;
use crate::results::field::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition, RecoveredFieldEntityCounts,
    RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
};
use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape, RecoveryPolicy,
};

fn producer(label: &str) -> StepIdentity {
    StepIdentity::new(
        SequenceIdentity::new(Digest::sha256("frequency sequence")).unwrap(),
        Digest::sha256(label),
    )
    .unwrap()
}

fn authority(structural: bool, frequencies: &[f64]) -> FrequencySamplePublicationAuthority {
    let producer = producer("harmonic step");
    let plan = FrequencyResponsePlan::new(frequencies[0], frequencies, &[]).unwrap();
    let entries = plan
        .entries()
        .unwrap()
        .into_iter()
        .map(|entry| {
            let label = format!("numerical entry {}", entry.ordinal());
            FrequencySampleNumericalEntry {
                entry,
                workload: AnalysisWorkloadIdentity::for_test(&label).unwrap(),
                source: if structural {
                    FieldRecoverySource::StructuralFrequencyEntry(
                        StructuralFrequencyEntryResultIdentity::derive(
                            "test-frequency-result",
                            &label,
                        )
                        .unwrap(),
                    )
                } else {
                    FieldRecoverySource::SolverResult(
                        SolverResultIdentity::from_step_result(producer, Digest::sha256(&label))
                            .unwrap(),
                    )
                },
            }
        })
        .collect();
    let topology = if structural {
        FieldTopologyIdentity::Structural {
            topology: CanonicalStructuralTopologyIdentity::Explicit {
                structural_topology_digest: Digest::sha256("topology"),
            },
        }
    } else {
        FieldTopologyIdentity::VolumeMesh {
            mesh_digest: runmat_meshing_core::StableDigest::from_bytes(
                *Digest::sha256("topology").bytes(),
            ),
        }
    };
    FrequencySamplePublicationAuthority::new(
        AnalysisWorkloadIdentity::for_test("whole authored frequency workload").unwrap(),
        SolverResultIdentity::from_step_result(producer, Digest::sha256("whole numerical result"))
            .unwrap(),
        producer,
        topology,
        plan,
        entries,
    )
    .unwrap()
}

fn sample(
    authority: &FrequencySamplePublicationAuthority,
    index: usize,
    coordinate: RecoveredFieldSample,
) -> FrequencySamplePublicationSample {
    let numerical = &authority.entries[index];
    let request = FieldRecoveryRequest::new(
        authority.producer,
        numerical.source,
        authority.topology,
        FieldRecoveryPlan {
            requirements: vec![FieldRecoveryRequirement {
                support: Default::default(),
                quantity: ObservationQuantity::AcousticPressure,
                source_location: FieldLocation::Node,
                output_location: FieldLocation::Node,
                recovery: RecoveryPolicy::Native,
                value_shape: ObservationValueShape::Scalar,
                entity_domain: RecoveredFieldEntityDomain::Nodes,
                cost: FieldRecoveryCost {
                    source_entity_count: 1,
                    output_entity_count: 1,
                    topology_incidence_count: 0,
                    arithmetic_operation_count: 1,
                },
            }],
        },
    )
    .unwrap();
    // Synthetic catalog contents exercise only the contract, never a physics qualification.
    let field = RecoveredField::new(
        RecoveredFieldDefinition::new(
            "pressure",
            ObservationQuantity::AcousticPressure,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Scalar,
            RecoveredFieldNumericRepresentation::ComplexInterleaved,
        )
        .unwrap(),
        coordinate,
        AnalysisField::host_f64("pressure", AnalysisFieldLayout::scalar(ObservationQuantity::AcousticPressure, FieldLocation::Node).representation(ComplexRepresentation::Complex), vec![1, 2], vec![1.0, 0.4]),
    )
    .unwrap();
    let fields = RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        authority.topology.discretization_digest(),
        RecoveredFieldEntityCounts {
            nodes: 1,
            volume_elements: 1,
            volume_integration_points: 0,
            oriented_edges: 0,
            oriented_faces: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field],
    )
    .unwrap()
    .manifest()
    .unwrap();
    FrequencySamplePublicationSample {
        entry: numerical.entry,
        recovery_request: request,
        fields,
        observations: None,
    }
}

fn samples(
    authority: &FrequencySamplePublicationAuthority,
) -> Vec<FrequencySamplePublicationSample> {
    authority
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            sample(
                authority,
                index,
                RecoveredFieldSample::FrequencyHertz(entry.entry.frequency_hz()),
            )
        })
        .collect()
}

#[test]
fn complete_catalog_is_canonical_for_both_shared_numerical_source_kinds() {
    for structural in [false, true] {
        let authority = authority(structural, &[1.0, 2.0, 4.0]);
        let manifest =
            FrequencySamplePublicationManifest::new(authority.clone(), samples(&authority))
                .unwrap();
        let bytes = manifest.canonical_bytes().unwrap();
        let cold = FrequencySamplePublicationManifest::from_canonical_bytes(&bytes).unwrap();
        cold.validate_for(&authority).unwrap();
        assert_eq!(cold, manifest);
        assert_eq!(cold.canonical_bytes().unwrap(), bytes);
        assert_eq!(
            serde_json::from_value::<FrequencySamplePublicationManifest>(
                serde_json::to_value(&manifest).unwrap()
            )
            .unwrap(),
            manifest
        );
        // The wire inventory contains metadata catalogs, never field values or paths.
        let wire = serde_json::to_value(&manifest).unwrap();
        assert!(wire["samples"][0]["fields"]["fields"][0]
            .get("storage")
            .is_none());
        assert!(wire.get("receipt").is_none());
    }
}

#[test]
fn numerical_inventory_rejects_missing_duplicate_unordered_and_foreign_entries() {
    let original = authority(false, &[1.0, 2.0]);
    for mutation in 0..5 {
        let mut invalid_authority = original.clone();
        match mutation {
            0 => {
                invalid_authority.entries.pop();
            }
            1 => invalid_authority.entries.swap(0, 1),
            2 => invalid_authority.entries[1].source = invalid_authority.entries[0].source,
            3 => invalid_authority.entries[1].workload = invalid_authority.entries[0].workload,
            4 => {
                invalid_authority.entries[1].entry = FrequencyResponsePlan::new(3.0, &[3.0], &[])
                    .unwrap()
                    .entries()
                    .unwrap()[0]
            }
            _ => unreachable!(),
        }
        assert!(invalid_authority.validate().is_err());
        assert!(
            serde_json::from_value::<FrequencySamplePublicationAuthority>(
                serde_json::to_value(&invalid_authority).unwrap()
            )
            .is_err()
        );
    }
}

#[test]
fn authority_rejects_placeholder_aggregate_and_entry_identities() {
    for structural in [false, true] {
        for digest in [Digest::from_bytes([0; 32]), Digest::sha256([])] {
            let original = authority(structural, &[1.0, 2.0]);
            let placeholder = serde_json::to_value(digest).unwrap();
            for mutation in 0..5 {
                let mut invalid = original.clone();
                match mutation {
                    0 => invalid.workload = serde_json::from_value(placeholder.clone()).unwrap(),
                    1 => invalid.result = serde_json::from_value(placeholder.clone()).unwrap(),
                    2 => invalid.producer = serde_json::from_value(placeholder.clone()).unwrap(),
                    3 => {
                        invalid.entries[0].workload =
                            serde_json::from_value(placeholder.clone()).unwrap()
                    }
                    4 => {
                        invalid.entries[0].source = if structural {
                            FieldRecoverySource::StructuralFrequencyEntry(
                                serde_json::from_value(placeholder.clone()).unwrap(),
                            )
                        } else {
                            FieldRecoverySource::SolverResult(
                                serde_json::from_value(placeholder.clone()).unwrap(),
                            )
                        }
                    }
                    _ => unreachable!(),
                }
                assert!(invalid.validate().is_err());
                assert!(
                    serde_json::from_value::<FrequencySamplePublicationAuthority>(
                        serde_json::to_value(&invalid).unwrap()
                    )
                    .is_err()
                );
            }
        }
    }
}

#[test]
fn different_definitions_cannot_publish_the_same_semantic_field_at_one_frequency() {
    let authority = authority(false, &[1.0]);
    let mut sample = samples(&authority).remove(0);
    let mut plan = sample.recovery_request.plan.clone();
    let mut velocity_requirement = plan.requirements[0].clone();
    velocity_requirement.quantity = ObservationQuantity::AcousticParticleVelocity;
    velocity_requirement.source_location = FieldLocation::Element;
    velocity_requirement.output_location = FieldLocation::Element;
    velocity_requirement.value_shape = ObservationValueShape::Vector3;
    velocity_requirement.entity_domain = RecoveredFieldEntityDomain::VolumeElements;
    plan.requirements.push(velocity_requirement);
    sample.recovery_request = FieldRecoveryRequest::new(
        authority.producer,
        sample.fields.source,
        authority.topology,
        plan,
    )
    .unwrap();
    let coordinate = RecoveredFieldSample::FrequencyHertz(1.0);
    let pressure = RecoveredField::new(
        sample.fields.fields[0].definition.clone(),
        coordinate,
        AnalysisField::host_f64("pressure", sample.fields.fields[0].definition.layout(), vec![1, 2], vec![1.0, 0.4]),
    )
    .unwrap();
    let misleading_velocity = RecoveredField::new(
        RecoveredFieldDefinition::new(
            "pressure",
            ObservationQuantity::AcousticParticleVelocity,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::ComplexInterleaved,
        )
        .unwrap(),
        coordinate,
        AnalysisField::host_f64("pressure", AnalysisFieldLayout::vector3(ObservationQuantity::AcousticParticleVelocity, FieldLocation::Element).representation(ComplexRepresentation::Complex), vec![1, 3, 2], vec![0.0; 6]),
    )
    .unwrap();
    sample.fields = RecoveredFieldArtifact::new(
        sample.recovery_request.identity(),
        sample.fields.source,
        sample.fields.topology_digest,
        sample.fields.entity_counts,
        vec![pressure, misleading_velocity],
    )
    .unwrap()
    .manifest()
    .unwrap();
    // Each underlying artifact/definition is independently valid. The shared semantic namespace
    // must still reject two different definitions advertised under one field ID and coordinate.
    sample.fields.validate().unwrap();
    assert!(FrequencySamplePublicationManifest::new(authority, vec![sample]).is_err());
}

#[test]
fn admission_rejects_missing_duplicate_foreign_producer_topology_source_and_coordinate() {
    let original = authority(false, &[1.0, 2.0]);
    for mutation in 0..7 {
        let mut entries = samples(&original);
        match mutation {
            0 => {
                entries.pop();
            }
            1 => entries[1] = entries[0].clone(),
            2 => entries.swap(0, 1),
            3..=5 => {
                let mut foreign = original.clone();
                match mutation {
                    3 => foreign.producer = producer("foreign producer"),
                    4 => {
                        foreign.topology = FieldTopologyIdentity::VolumeMesh {
                            mesh_digest: runmat_meshing_core::StableDigest::from_bytes(
                                *Digest::sha256("foreign topology").bytes(),
                            ),
                        }
                    }
                    5 => {
                        foreign.entries[0].source = FieldRecoverySource::SolverResult(
                            SolverResultIdentity::from_step_result(
                                original.producer,
                                Digest::sha256("foreign result"),
                            )
                            .unwrap(),
                        )
                    }
                    _ => unreachable!(),
                }
                entries[0] = sample(&foreign, 0, RecoveredFieldSample::FrequencyHertz(1.0));
            }
            6 => entries[0] = sample(&original, 0, RecoveredFieldSample::FrequencyHertz(9.0)),
            _ => unreachable!(),
        }
        assert!(FrequencySamplePublicationManifest::new(original.clone(), entries).is_err());
    }
}

#[test]
fn validated_decode_and_external_authority_reject_substitution() {
    let authority = authority(false, &[1.0, 2.0]);
    let manifest =
        FrequencySamplePublicationManifest::new(authority.clone(), samples(&authority)).unwrap();
    for key in ["identity", "schema_version", "samples"] {
        let mut wire = serde_json::to_value(&manifest).unwrap();
        match key {
            "identity" => {
                wire[key] = serde_json::to_value(FrequencySamplePublicationIdentity(
                    Digest::sha256("tamper"),
                ))
                .unwrap()
            }
            "schema_version" => wire[key] = 2.into(),
            "samples" => {
                wire[key].as_array_mut().unwrap().pop();
            }
            _ => unreachable!(),
        }
        assert!(serde_json::from_value::<FrequencySamplePublicationManifest>(wire).is_err());
    }
    let mut foreign = authority.clone();
    foreign.workload = AnalysisWorkloadIdentity::for_test("foreign aggregate workload").unwrap();
    assert!(manifest.validate_for(&foreign).is_err());
    foreign = authority.clone();
    foreign.result = SolverResultIdentity::from_step_result(
        authority.producer,
        Digest::sha256("foreign aggregate result"),
    )
    .unwrap();
    assert!(manifest.validate_for(&foreign).is_err());
}

#[test]
fn selection_is_complete_exact_and_never_implicitly_reference_first_or_last() {
    let authority = authority(false, &[1.0, 2.0, 4.0]);
    let manifest =
        FrequencySamplePublicationManifest::new(authority.clone(), samples(&authority)).unwrap();
    assert!(manifest.select_sample(None).is_err());
    for frequency in [1.0, 2.0, 4.0] {
        assert_eq!(
            manifest
                .select_sample(Some(RecoveredFieldSample::FrequencyHertz(frequency)))
                .unwrap()
                .entry
                .frequency_hz(),
            frequency
        );
    }
    for coordinate in [
        RecoveredFieldSample::Final,
        RecoveredFieldSample::TimeSeconds(1.0),
        RecoveredFieldSample::FrequencyHertz(3.0),
        RecoveredFieldSample::FrequencyHertz(-0.0),
        RecoveredFieldSample::FrequencyHertz(f64::NAN),
    ] {
        assert!(manifest.select_sample(Some(coordinate)).is_err());
    }
    let singleton = self::authority(false, &[2.0]);
    let singleton =
        FrequencySamplePublicationManifest::new(singleton.clone(), samples(&singleton)).unwrap();
    assert_eq!(
        singleton.select_sample(None).unwrap().entry.frequency_hz(),
        2.0
    );
}
