use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::results::field::{
    select_canonical_field_partitions, RecoveredFieldArtifactEntry, RecoveredFieldPartition,
    RecoveredFieldPartitionSet,
};
use runmat_execution::value::ValueRef;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;

use crate::{
    import_analysis_partition, AnalysisBridgeError, AnalysisBridgeResult, AnalysisPartitionImport,
};

/// Imports only canonical field storage blocks intersecting the sorted source indices
/// required by one observation partition.
pub(crate) struct RequiredFieldPartitionImport<'a> {
    pub source_kind: AnalysisNumericalObjectKind,
    pub root: &'a ValueRef,
    pub source_logical_identity: runmat_execution::Digest,
    pub source_payload_digest: runmat_execution::Digest,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
    pub entry: &'a RecoveredFieldArtifactEntry,
    pub required_indices: &'a [usize],
}

pub(crate) fn import_required_field_partitions(
    source: &impl CacheImport,
    request: RequiredFieldPartitionImport<'_>,
) -> AnalysisBridgeResult<RecoveredFieldPartitionSet> {
    let RequiredFieldPartitionImport {
        source_kind,
        root,
        source_logical_identity,
        source_payload_digest,
        access,
        limits,
        entry,
        required_indices,
    } = request;
    if !matches!(
        source_kind,
        AnalysisNumericalObjectKind::Field | AnalysisNumericalObjectKind::SolverResult
    ) {
        return Err(AnalysisBridgeError::Invalid(
            "observation source must be a recovered field or solver result".into(),
        ));
    }
    let selection = select_canonical_field_partitions(entry, required_indices)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let partitions = selection
        .descriptors
        .into_iter()
        .map(|descriptor| {
            let imported = import_analysis_partition(
                source,
                AnalysisPartitionImport {
                    root,
                    expected_kind: source_kind,
                    expected_logical_identity: source_logical_identity,
                    expected_payload_digest: source_payload_digest,
                    partition_identity: descriptor.partition.digest(),
                    access,
                    limits,
                },
            )?;
            let partition = RecoveredFieldPartition::from_canonical_bytes(&imported.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            if partition.identity != descriptor.partition
                || partition.source_field != entry.field
                || partition.definition != entry.definition
                || partition.sample != entry.sample
                || partition
                    .source_indices
                    .iter()
                    .copied()
                    .ne(descriptor.first_source_index
                        ..descriptor.first_source_index + descriptor.entity_count)
            {
                return Err(AnalysisBridgeError::Identity(
                    "field storage block differs from its recovered-field manifest",
                ));
            }
            Ok(partition)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let set = RecoveredFieldPartitionSet::new(partitions)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if set.identity != selection.identity {
        return Err(AnalysisBridgeError::Identity(
            "imported field partition set differs from canonical metadata selection",
        ));
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use runmat_analysis_core::{
        AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
    };
    use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, AnalysisPayloadPartition};
    use runmat_analysis_fea::results::field::{
        canonical_field_partition_descriptors, canonical_field_partitions, RecoveredField,
        RecoveredFieldArtifactEntry, RecoveredFieldDefinition, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
        RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
    };
    use runmat_execution::value::ValueRefKind;
    use runmat_execution::Digest;
    use runmat_execution_artifact::cache::CacheImport;
    use runmat_execution_artifact::{ArtifactAccess, ArtifactResult, ObjectNamespace};

    use crate::prepare_partitioned_analysis_objects;

    use super::*;

    struct RecordingStore {
        objects: BTreeMap<Digest, Vec<u8>>,
        reads: RefCell<Vec<Digest>>,
    }

    impl CacheImport for RecordingStore {
        fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
            self.reads.borrow_mut().push(digest);
            Ok(self.objects.get(&digest).cloned())
        }
    }

    #[test]
    fn large_field_import_reads_only_the_intersecting_storage_block() {
        check_large_partition_import(AnalysisNumericalObjectKind::Field);
        check_large_partition_import(AnalysisNumericalObjectKind::SolverResult);
    }

    fn check_large_partition_import(source_kind: AnalysisNumericalObjectKind) {
        check_native_partition_import(
            source_kind,
            ObservationQuantity::Temperature,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
        );
    }

    #[test]
    fn native_cells_and_all_oriented_faces_read_only_intersecting_blocks() {
        for (quantity, location, domain) in [
            (
                ObservationQuantity::FluidPressure,
                FieldLocation::Cell,
                RecoveredFieldEntityDomain::VolumeElements,
            ),
            (
                ObservationQuantity::VolumeFlowRate,
                FieldLocation::Face,
                RecoveredFieldEntityDomain::OrientedFaces,
            ),
        ] {
            check_native_partition_import(
                AnalysisNumericalObjectKind::Field,
                quantity,
                location,
                domain,
            );
        }
    }

    fn check_native_partition_import(
        source_kind: AnalysisNumericalObjectKind,
        quantity: ObservationQuantity,
        location: FieldLocation,
        domain: RecoveredFieldEntityDomain,
    ) {
        let entity_count = RECOVERED_FIELD_PARTITION_ENTITY_COUNT * 2 + 17;
        let definition = RecoveredFieldDefinition::new(
            "large.temperature",
            quantity,
            location,
            domain,
            ObservationValueShape::Scalar,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap();
        let field = RecoveredField::new(
            definition.clone(),
            RecoveredFieldSample::Final,
            AnalysisField::host_f64(
                "large.temperature", definition.layout(),
                vec![entity_count],
                (0..entity_count).map(|index| index as f64).collect(),
            ),
        )
        .unwrap();
        let storage_partitions = canonical_field_partitions(&field).unwrap();
        assert_eq!(storage_partitions.len(), 3);
        let entry = RecoveredFieldArtifactEntry {
            field: field.identity(),
            definition,
            sample: RecoveredFieldSample::Final,
            partitions: canonical_field_partition_descriptors(&field).unwrap(),
        };
        let payloads = storage_partitions
            .iter()
            .map(|partition| {
                (
                    partition.identity.digest(),
                    partition.canonical_bytes().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let payload_views = payloads
            .iter()
            .map(|(identity, payload)| AnalysisPayloadPartition {
                logical_identity: *identity,
                payload,
            })
            .collect::<Vec<_>>();
        let logical_identity = Digest::sha256("large-field-artifact");
        let prepared = prepare_partitioned_analysis_objects(
            source_kind,
            logical_identity,
            &payload_views,
            4096,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let access = ArtifactAccess {
            authorization_scope: "large-field:test".into(),
            encryption_context: Digest::sha256("large-field-encryption"),
        };
        let root = prepared
            .root_reference(&access, ValueRefKind::ResultObject)
            .unwrap();
        let objects = prepared
            .objects
            .iter()
            .map(|object| (object.descriptor.digest, object.bytes.clone()))
            .collect::<BTreeMap<_, _>>();
        let store = RecordingStore {
            objects: objects.clone(),
            reads: RefCell::new(Vec::new()),
        };
        let selected = import_required_field_partitions(
            &store,
            RequiredFieldPartitionImport {
                source_kind,
                root: &root,
                source_logical_identity: prepared.manifest.logical_identity,
                source_payload_digest: prepared.manifest.payload_digest,
                access: &access,
                limits: ObjectInventoryLimits::default(),
                entry: &entry,
                required_indices: &[RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 3],
            },
        )
        .unwrap();
        assert_eq!(selected.partitions.len(), 1);
        assert_eq!(
            selected.partitions[0].identity,
            storage_partitions[1].identity
        );
        assert_eq!(
            selected.value(RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 3),
            Some(&[(RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 3) as f64][..])
        );

        let unread_partitions = [
            &prepared.manifest.partitions[0],
            &prepared.manifest.partitions[2],
        ];
        let reads = store.reads.borrow();
        for partition in unread_partitions {
            let start = partition.first_chunk_ordinal as usize;
            let end = start + partition.chunk_count as usize;
            for chunk in &prepared.manifest.chunks[start..end] {
                assert!(
                    !reads.contains(&chunk.digest),
                    "nonintersecting field storage block was read"
                );
            }
        }
        drop(reads);

        let mut substituted_entry = entry.clone();
        substituted_entry.partitions[1].entity_count += 1;
        assert!(import_required_field_partitions(
            &store,
            RequiredFieldPartitionImport {
                source_kind,
                root: &root,
                source_logical_identity: prepared.manifest.logical_identity,
                source_payload_digest: prepared.manifest.payload_digest,
                access: &access,
                limits: ObjectInventoryLimits::default(),
                entry: &substituted_entry,
                required_indices: &[RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 3],
            },
        )
        .is_err());

        let middle_manifest_partition = &prepared.manifest.partitions[1];
        let missing_chunk =
            prepared.manifest.chunks[middle_manifest_partition.first_chunk_ordinal as usize].digest;
        let mut missing_objects = objects;
        missing_objects.remove(&missing_chunk);
        let missing_store = RecordingStore {
            objects: missing_objects,
            reads: RefCell::new(Vec::new()),
        };
        assert!(import_required_field_partitions(
            &missing_store,
            RequiredFieldPartitionImport {
                source_kind,
                root: &root,
                source_logical_identity: prepared.manifest.logical_identity,
                source_payload_digest: prepared.manifest.payload_digest,
                access: &access,
                limits: ObjectInventoryLimits::default(),
                entry: &entry,
                required_indices: &[RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 3],
            },
        )
        .is_err());
    }
}
