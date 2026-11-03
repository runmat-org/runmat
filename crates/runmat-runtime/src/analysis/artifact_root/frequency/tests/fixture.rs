//! Synthetic transport contract, not a numerical or formulation qualification.
use super::*;
use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape, RecoveryPolicy,
};
use runmat_analysis_core::{AnalysisFieldLayout, ComplexRepresentation};
use runmat_analysis_fea::{execution::*, frequency::FrequencyResponsePlan, results::field::*};
use runmat_execution::value::{artifact_object_value_id, ValueRefKind};
use std::collections::BTreeMap;

pub(super) struct Fixture {
    pub publication: FrequencySamplePublicationManifest,
    pub authority: FrequencySamplePublicationAuthority,
    pub inventory: FrequencySampleArtifactRootInventory,
    pub catalog: PublishedObjectCatalog,
    pub objects: BTreeMap<Digest, Vec<u8>>,
    pub second_partition_chunks: Vec<Digest>,
}

impl Fixture {
    pub fn access(&self) -> FrequencySamplePublicationAccess<'_> {
        FrequencySamplePublicationAccess {
            publication: &self.publication,
            authority: &self.authority,
            object_catalog: &self.catalog,
        }
    }
}

fn contract_identity<T: serde::de::DeserializeOwned>(label: &str) -> T {
    serde_json::from_value(serde_json::to_value(Digest::sha256(label)).unwrap()).unwrap()
}

fn reference(bytes: &[u8], media: &str, schema: &str) -> runmat_execution::value::ValueRef {
    let logical_digest = Digest::sha256(bytes);
    let encryption_context = Digest::sha256("frequency contract encryption");
    let authorization_scope = "analysis:frequency-contract".to_string();
    runmat_execution::value::ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: artifact_object_value_id(logical_digest, encryption_context, &authorization_scope),
        logical_digest,
        encoded_length: bytes.len() as u64,
        media_type: media.into(),
        value_schema: schema.into(),
        encryption_context,
        kind: ValueRefKind::ResultObject,
        authorization_scope,
        resident_fence: None,
    }
}

fn retain(
    chunked: ChunkedAnalysisPayload,
    objects: &mut BTreeMap<Digest, Vec<u8>>,
    references: &mut BTreeMap<
        runmat_execution::identity::ValueId,
        runmat_execution::value::ValueRef,
    >,
) -> AnalysisArtifactRoot {
    let bytes = chunked.manifest.canonical_bytes().unwrap();
    let root = reference(
        &bytes,
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        ANALYSIS_MANIFEST_VALUE_SCHEMA,
    );
    objects.insert(root.logical_digest, bytes);
    references.insert(root.id, root.clone());
    for bytes in chunked.chunks {
        let child = reference(
            &bytes,
            ANALYSIS_CHUNK_MEDIA_TYPE,
            ANALYSIS_CHUNK_VALUE_SCHEMA,
        );
        objects.insert(child.logical_digest, bytes);
        references.insert(child.id, child);
    }
    AnalysisArtifactRoot::new(
        root,
        chunked.manifest.kind,
        chunked.manifest.logical_identity,
        chunked.manifest.payload_digest,
    )
    .unwrap()
}

pub(super) fn fixture() -> Fixture {
    let producer = StepIdentity::new(
        SequenceIdentity::new(Digest::sha256("sequence")).unwrap(),
        Digest::sha256("frequency step"),
    )
    .unwrap();
    let plan = FrequencyResponsePlan::new(1.0, &[1.0, 2.0], &[]).unwrap();
    let entries = plan
        .entries()
        .unwrap()
        .into_iter()
        .map(|entry| FrequencySampleNumericalEntry {
            entry,
            workload: contract_identity(&format!("workload {}", entry.ordinal())),
            source: FieldRecoverySource::SolverResult(
                SolverResultIdentity::from_step_result(
                    producer,
                    Digest::sha256(format!("result {}", entry.ordinal())),
                )
                .unwrap(),
            ),
        })
        .collect::<Vec<_>>();
    let authority = FrequencySamplePublicationAuthority::new(
        contract_identity("aggregate workload"),
        SolverResultIdentity::from_step_result(producer, Digest::sha256("aggregate result"))
            .unwrap(),
        producer,
        FieldTopologyIdentity::VolumeMesh {
            mesh_digest: runmat_meshing_core::StableDigest::from_bytes(
                *Digest::sha256("topology").bytes(),
            ),
        },
        plan,
        entries,
    )
    .unwrap();
    let count = RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 2;
    let mut objects = BTreeMap::new();
    let mut references = BTreeMap::new();
    let mut sample_roots = Vec::new();
    let mut samples = Vec::new();
    let mut second_partition_chunks = Vec::new();
    for numerical in authority.entries() {
        let request = FieldRecoveryRequest::new(
            producer,
            numerical.source,
            authority.topology(),
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
                        source_entity_count: count as u64,
                        output_entity_count: count as u64,
                        topology_incidence_count: 0,
                        arithmetic_operation_count: count as u64,
                    },
                }],
            },
        )
        .unwrap();
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
            RecoveredFieldSample::FrequencyHertz(numerical.entry.frequency_hz()),
            AnalysisField::host_f64(
                "pressure",
                AnalysisFieldLayout::scalar(
                    ObservationQuantity::AcousticPressure,
                    FieldLocation::Node,
                )
                .representation(ComplexRepresentation::Complex),
                vec![count, 2],
                (0..count * 2)
                    .map(|index| numerical.entry.frequency_hz() * 100_000.0 + index as f64)
                    .collect(),
            ),
        )
        .unwrap();
        let partitions = canonical_field_partitions(&field).unwrap();
        let artifact = RecoveredFieldArtifact::new(
            request.identity(),
            numerical.source,
            authority.topology().discretization_digest(),
            RecoveredFieldEntityCounts {
                nodes: count,
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
        .unwrap();
        let fields = artifact.manifest().unwrap();
        let mut payloads = vec![fields.canonical_bytes().unwrap()];
        payloads.extend(
            partitions
                .iter()
                .map(|partition| partition.canonical_bytes().unwrap()),
        );
        let mut identities = vec![fields.identity.digest()];
        identities.extend(
            partitions
                .iter()
                .map(|partition| partition.identity.digest()),
        );
        let inputs = identities
            .iter()
            .zip(&payloads)
            .map(|(identity, payload)| AnalysisPayloadPartition {
                logical_identity: *identity,
                payload,
            })
            .collect::<Vec<_>>();
        let chunked = chunk_partitioned_numerical_payload(
            AnalysisNumericalObjectKind::Field,
            fields.identity.digest(),
            &inputs,
            4096,
        )
        .unwrap();
        if numerical.entry.ordinal() == 1 {
            let partition = chunked.manifest.partitions.last().unwrap();
            let first = partition.first_chunk_ordinal as usize;
            second_partition_chunks = chunked.manifest.chunks
                [first..first + partition.chunk_count as usize]
                .iter()
                .map(|chunk| chunk.digest)
                .collect();
        }
        sample_roots.push(FrequencySampleArtifactRoots::new(
            retain(chunked, &mut objects, &mut references),
            None,
        ));
        samples.push(FrequencySamplePublicationSample {
            entry: numerical.entry,
            recovery_request: request,
            fields,
            observations: None,
        });
    }
    let publication = FrequencySamplePublicationManifest::new(authority.clone(), samples).unwrap();
    let payload = publication.canonical_bytes().unwrap();
    let metadata = retain(
        chunk_numerical_payload(
            AnalysisNumericalObjectKind::FrequencySamplePublication,
            publication.identity().digest(),
            &payload,
            4096,
        )
        .unwrap(),
        &mut objects,
        &mut references,
    );
    let roots = std::iter::once(metadata.reference().clone())
        .chain(
            sample_roots
                .iter()
                .map(|sample| sample.fields().reference().clone()),
        )
        .collect();
    let catalog = PublishedObjectCatalog::new(roots, references.into_values().collect()).unwrap();
    let inventory = FrequencySampleArtifactRootInventory::new(
        metadata,
        sample_roots,
        &FrequencySamplePublicationAccess {
            publication: &publication,
            authority: &authority,
            object_catalog: &catalog,
        },
    )
    .unwrap();
    Fixture {
        publication,
        authority,
        inventory,
        catalog,
        objects,
        second_partition_chunks,
    }
}
