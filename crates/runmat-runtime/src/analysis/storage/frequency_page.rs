mod selection;

use runmat_analysis_fea::execution::{
    AnalysisObjectManifest, FrequencySamplePublicationManifest,
    StructuralFrequencyPublicationManifest,
};
use runmat_analysis_fea::results::field::{
    RecoveredFieldArtifactEntry, RecoveredFieldNumericRepresentation, RecoveredFieldPartition,
    RecoveredFieldPartitionDescriptor,
};
use runmat_execution::Digest;

use crate::analysis::{
    AnalysisArtifactRoot, AnalysisFieldDescriptor, AnalysisFieldPageValues,
    AnalysisFieldRequestOptions, AnalysisFieldSelection, FrequencySampleArtifactRoots,
    StructuralFrequencyArtifactRootIndex,
};

use super::AnalysisFieldPageLookup;
#[cfg(test)]
use selection::select_candidates;
use selection::{
    empty_values, page_result, select_field, select_sample_field, LogicalPage, SelectedField,
};

pub(super) fn load_page(
    run_id: &str,
    artifact_roots: &StructuralFrequencyArtifactRootIndex,
    publication: &StructuralFrequencyPublicationManifest,
    selection: &AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
    read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisFieldPageLookup, String> {
    let selected = match select_field(publication, selection) {
        Ok(selected) => selected,
        Err(lookup) => return Ok(lookup),
    };
    load_selected_page(run_id, selected, artifact_roots.samples(), options, read)
}

pub(in crate::analysis) fn load_sample_page(
    run_id: &str,
    artifact_roots: &[FrequencySampleArtifactRoots],
    publication: &FrequencySamplePublicationManifest,
    selection: &AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
    read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisFieldPageLookup, String> {
    let selected = match select_sample_field(publication, selection) {
        Ok(selected) => selected,
        Err(lookup) => return Ok(lookup),
    };
    load_selected_page(run_id, selected, artifact_roots, options, read)
}

fn load_selected_page(
    run_id: &str,
    selected: SelectedField<'_>,
    artifact_roots: &[FrequencySampleArtifactRoots],
    options: AnalysisFieldRequestOptions,
    mut read: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisFieldPageLookup, String> {
    let descriptor = AnalysisFieldDescriptor::from_recovered_entry(selected.entry)?;
    let page = LogicalPage::new(descriptor.value_count, options)?;
    let representation = selected.entry.definition.numeric_representation();
    if page.count == 0 {
        return Ok(page_result(
            run_id,
            selected.entry,
            descriptor,
            page,
            empty_values(representation),
        ));
    }

    let root = artifact_roots
        .get(selected.sample_index)
        .ok_or_else(|| "frequency sample has no artifact root".to_string())?
        .fields();
    let object_manifest = read_root_manifest(root, &mut read)?;
    let values = read_logical_values(selected.entry, &object_manifest, page, &mut read)?;
    Ok(page_result(
        run_id,
        selected.entry,
        descriptor,
        page,
        match representation {
            RecoveredFieldNumericRepresentation::Real => AnalysisFieldPageValues::Real(values),
            RecoveredFieldNumericRepresentation::ComplexInterleaved => {
                AnalysisFieldPageValues::ComplexInterleaved(values)
            }
        },
    ))
}

fn read_logical_values(
    entry: &RecoveredFieldArtifactEntry,
    object_manifest: &AnalysisObjectManifest,
    page: LogicalPage,
    read: &mut impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<Vec<f64>, String> {
    let components = entry
        .definition
        .value_shape()
        .component_count()
        .ok_or_else(|| "recovered field has an invalid value shape".to_string())?;
    let storage_width = match entry.definition.numeric_representation() {
        RecoveredFieldNumericRepresentation::Real => 1,
        RecoveredFieldNumericRepresentation::ComplexInterleaved => 2,
    };
    let storage_capacity = page
        .count
        .checked_mul(storage_width)
        .ok_or_else(|| "analysis field page storage size overflow".to_string())?;
    let mut values = Vec::with_capacity(storage_capacity);
    for partition_descriptor in &entry.partitions {
        let Some(overlap) = partition_overlap(partition_descriptor, components, page)? else {
            continue;
        };
        let payload = read_payload_partition(
            object_manifest,
            partition_descriptor.partition.digest(),
            read,
        )?;
        let partition = RecoveredFieldPartition::from_canonical_bytes(&payload)
            .map_err(|error| error.to_string())?;
        validate_field_partition(entry, partition_descriptor, &partition)?;
        let partition_values = partition
            .storage
            .as_host_f64()
            .ok_or_else(|| "recovered field partition is not host-resident f64".to_string())?;
        let local_start = overlap
            .start
            .checked_mul(storage_width)
            .ok_or_else(|| "recovered field page offset overflow".to_string())?;
        let local_end = overlap
            .end
            .checked_mul(storage_width)
            .ok_or_else(|| "recovered field page end overflow".to_string())?;
        values.extend_from_slice(partition_values.get(local_start..local_end).ok_or_else(
            || "recovered field partition does not contain its declared logical range".to_string(),
        )?);
    }
    if values.len() != storage_capacity {
        return Err(format!(
            "recovered field page is incomplete: expected {storage_capacity} storage values, found {}",
            values.len()
        ));
    }
    Ok(values)
}

struct PartitionOverlap {
    start: usize,
    end: usize,
}

fn partition_overlap(
    descriptor: &RecoveredFieldPartitionDescriptor,
    components: usize,
    page: LogicalPage,
) -> Result<Option<PartitionOverlap>, String> {
    let partition_start = descriptor
        .first_source_index
        .checked_mul(components)
        .ok_or_else(|| "recovered field partition range overflow".to_string())?;
    let partition_count = descriptor
        .entity_count
        .checked_mul(components)
        .ok_or_else(|| "recovered field partition range overflow".to_string())?;
    let partition_end = partition_start
        .checked_add(partition_count)
        .ok_or_else(|| "recovered field partition range overflow".to_string())?;
    let overlap_start = page.offset.max(partition_start);
    let overlap_end = page.end().min(partition_end);
    if overlap_start >= overlap_end {
        return Ok(None);
    }
    Ok(Some(PartitionOverlap {
        start: overlap_start - partition_start,
        end: overlap_end - partition_start,
    }))
}

fn read_root_manifest(
    root: &AnalysisArtifactRoot,
    read: &mut impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisObjectManifest, String> {
    let reference = root.reference();
    let bytes = read(reference.logical_digest)?
        .ok_or_else(|| "frequency field root is missing".to_string())?;
    if bytes.len() as u64 != reference.encoded_length
        || Digest::sha256(&bytes) != reference.logical_digest
    {
        return Err("frequency field root failed content verification".into());
    }
    let manifest =
        AnalysisObjectManifest::from_canonical_bytes(&bytes).map_err(|error| error.to_string())?;
    if manifest.kind != root.expected_kind()
        || manifest.logical_identity != root.expected_logical_identity()
        || manifest.payload_digest != root.expected_payload_digest()
    {
        return Err("frequency field root differs from its typed expectation".into());
    }
    Ok(manifest)
}

use super::object_payload::read_payload_partition;
fn validate_field_partition(
    entry: &RecoveredFieldArtifactEntry,
    descriptor: &RecoveredFieldPartitionDescriptor,
    partition: &RecoveredFieldPartition,
) -> Result<(), String> {
    let expected_indices = descriptor.first_source_index
        ..descriptor
            .first_source_index
            .checked_add(descriptor.entity_count)
            .ok_or_else(|| "recovered field partition index range overflow".to_string())?;
    if partition.identity != descriptor.partition
        || partition.source_field != entry.field
        || partition.definition != entry.definition
        || partition.sample != entry.sample
        || !partition
            .source_indices
            .iter()
            .copied()
            .eq(expected_indices)
    {
        return Err("recovered field partition differs from its semantic manifest".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use runmat_analysis_core::{
        AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
    };
    use runmat_analysis_fea::execution::{
        chunk_partitioned_numerical_payload, AnalysisNumericalObjectKind, AnalysisPayloadPartition,
    };
    use runmat_analysis_fea::results::field::{
        canonical_field_partition_descriptors, canonical_field_partitions, RecoveredField,
        RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
        RecoveredFieldSample, RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
    };

    use super::*;
    use runmat_analysis_core::{AnalysisFieldLayout, ComplexRepresentation};

    struct PartitionedFieldFixture {
        entry: RecoveredFieldArtifactEntry,
        manifest: AnalysisObjectManifest,
        objects: BTreeMap<Digest, Vec<u8>>,
        partition_chunk_digests: Vec<Vec<Digest>>,
    }

    fn complex_displacement_field(
        sample: RecoveredFieldSample,
        entity_count: usize,
        value_bias: f64,
    ) -> RecoveredField {
        let definition = RecoveredFieldDefinition::new(
            "structural.displacement",
            ObservationQuantity::Displacement,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::ComplexInterleaved,
        )
        .unwrap();
        let logical_count = entity_count * 3;
        let values = (0..logical_count * 2)
            .map(|index| index as f64 + value_bias)
            .collect::<Vec<_>>();
        RecoveredField::new(
            definition.clone(),
            sample,
            AnalysisField::host_f64("structural.displacement", definition.layout(), vec![entity_count, 3, 2], values),
        )
        .unwrap()
    }

    fn partitioned_fixture(field: &RecoveredField) -> PartitionedFieldFixture {
        let partitions = canonical_field_partitions(field).unwrap();
        let payloads = partitions
            .iter()
            .map(|partition| partition.canonical_bytes().unwrap())
            .collect::<Vec<_>>();
        let inputs = partitions
            .iter()
            .zip(&payloads)
            .map(|(partition, payload)| AnalysisPayloadPartition {
                logical_identity: partition.identity.digest(),
                payload,
            })
            .collect::<Vec<_>>();
        let chunked = chunk_partitioned_numerical_payload(
            AnalysisNumericalObjectKind::Field,
            Digest::sha256("frequency-field-root"),
            &inputs,
            4096,
        )
        .unwrap();
        let mut objects = BTreeMap::new();
        for (descriptor, bytes) in chunked.manifest.chunks.iter().zip(&chunked.chunks) {
            objects.insert(descriptor.digest, bytes.clone());
        }
        let partition_chunk_digests = chunked
            .manifest
            .partitions
            .iter()
            .map(|partition| {
                let first = partition.first_chunk_ordinal as usize;
                let end = first + partition.chunk_count as usize;
                chunked.manifest.chunks[first..end]
                    .iter()
                    .map(|chunk| chunk.digest)
                    .collect()
            })
            .collect();
        PartitionedFieldFixture {
            entry: RecoveredFieldArtifactEntry {
                field: field.identity(),
                definition: field.definition().clone(),
                sample: field.sample(),
                partitions: canonical_field_partition_descriptors(field).unwrap(),
            },
            manifest: chunked.manifest,
            objects,
            partition_chunk_digests,
        }
    }

    fn artifact_entry(field: &RecoveredField) -> RecoveredFieldArtifactEntry {
        RecoveredFieldArtifactEntry {
            field: field.identity(),
            definition: field.definition().clone(),
            sample: field.sample(),
            partitions: canonical_field_partition_descriptors(field).unwrap(),
        }
    }

    fn candidate<'a>(
        sample_index: usize,
        entry: &'a RecoveredFieldArtifactEntry,
    ) -> SelectedField<'a> {
        SelectedField {
            sample_index,
            entry,
        }
    }

    #[test]
    fn field_selection_requires_exact_sample_and_rejects_ambiguous_suffixes() {
        let at_ten = artifact_entry(&complex_displacement_field(
            RecoveredFieldSample::FrequencyHertz(10.0),
            1,
            0.0,
        ));
        let at_twenty = artifact_entry(&complex_displacement_field(
            RecoveredFieldSample::FrequencyHertz(20.0),
            1,
            1.0,
        ));
        let candidates = || vec![candidate(0, &at_ten), candidate(1, &at_twenty)];

        let missing_sample = select_candidates(
            candidates(),
            &AnalysisFieldSelection::new("structural.displacement", None),
        );
        let Err(AnalysisFieldPageLookup::SampleSelectionRequired(samples)) = missing_sample else {
            panic!("multiple frequency samples must require an exact typed sample")
        };
        assert_eq!(
            samples,
            vec![
                RecoveredFieldSample::FrequencyHertz(10.0),
                RecoveredFieldSample::FrequencyHertz(20.0)
            ]
        );

        let selected = select_candidates(
            candidates(),
            &AnalysisFieldSelection::new(
                "structural.displacement",
                Some(RecoveredFieldSample::FrequencyHertz(20.0)),
            ),
        );
        let Ok(selected) = selected else {
            panic!("exact typed sample must select its matching field")
        };
        assert_eq!(selected.sample_index, 1);

        let missing = select_candidates(
            candidates(),
            &AnalysisFieldSelection::new(
                "structural.displacement",
                Some(RecoveredFieldSample::FrequencyHertz(30.0)),
            ),
        );
        assert!(matches!(
            missing,
            Err(AnalysisFieldPageLookup::SampleNotFound)
        ));

        let solid = complex_displacement_field(RecoveredFieldSample::FrequencyHertz(10.0), 1, 0.0);
        let beam = RecoveredField::new(
            RecoveredFieldDefinition::new(
                "beam.displacement",
                ObservationQuantity::Displacement,
                FieldLocation::Node,
                RecoveredFieldEntityDomain::Nodes,
                ObservationValueShape::Vector3,
                RecoveredFieldNumericRepresentation::ComplexInterleaved,
            )
            .unwrap(),
            RecoveredFieldSample::FrequencyHertz(10.0),
            AnalysisField::host_f64("beam.displacement", AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node).representation(ComplexRepresentation::Complex), vec![1, 3, 2], vec![0.0; 6]),
        )
        .unwrap();
        let solid = artifact_entry(&solid);
        let beam = artifact_entry(&beam);
        let ambiguous = select_candidates(
            vec![candidate(0, &solid), candidate(0, &beam)],
            &AnalysisFieldSelection::new("displacement", None),
        );
        let Err(AnalysisFieldPageLookup::FieldSelectionAmbiguous(field_ids)) = ambiguous else {
            panic!("ambiguous suffix selection must fail closed")
        };
        assert_eq!(
            field_ids,
            vec![
                "beam.displacement".to_string(),
                "structural.displacement".to_string()
            ]
        );
    }

    #[test]
    fn complex_page_reads_only_intersecting_field_partition_chunks() {
        let entity_count = RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 2;
        let field = complex_displacement_field(
            RecoveredFieldSample::FrequencyHertz(125.0),
            entity_count,
            0.0,
        );
        let fixture = partitioned_fixture(&field);
        assert_eq!(fixture.entry.partitions.len(), 2);
        let offset = RECOVERED_FIELD_PARTITION_ENTITY_COUNT * 3 + 1;
        let page = LogicalPage {
            offset,
            count: 3,
            total_count: entity_count * 3,
        };
        let mut reads = BTreeSet::new();
        let values = read_logical_values(&fixture.entry, &fixture.manifest, page, &mut |digest| {
            reads.insert(digest);
            Ok(fixture.objects.get(&digest).cloned())
        })
        .unwrap();

        assert_eq!(
            values,
            (offset * 2..(offset + 3) * 2)
                .map(|index| index as f64)
                .collect::<Vec<_>>()
        );
        assert!(fixture.partition_chunk_digests[0]
            .iter()
            .all(|digest| !reads.contains(digest)));
        assert_eq!(
            reads,
            fixture.partition_chunk_digests[1].iter().copied().collect()
        );
    }

    #[test]
    fn selected_field_partition_chunk_corruption_fails_closed() {
        let field = complex_displacement_field(
            RecoveredFieldSample::FrequencyHertz(25.0),
            RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 1,
            0.0,
        );
        let mut fixture = partitioned_fixture(&field);
        let digest = fixture.partition_chunk_digests[1][0];
        fixture.objects.get_mut(&digest).unwrap()[0] ^= 0x80;
        let page = LogicalPage {
            offset: RECOVERED_FIELD_PARTITION_ENTITY_COUNT * 3,
            count: 1,
            total_count: (RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 1) * 3,
        };
        let error = read_logical_values(&fixture.entry, &fixture.manifest, page, &mut |digest| {
            Ok(fixture.objects.get(&digest).cloned())
        })
        .unwrap_err();
        assert!(error.contains("differs from its manifest descriptor"));
    }

    #[test]
    fn semantically_substituted_field_partition_fails_closed() {
        let entity_count = RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 1;
        let expected = complex_displacement_field(
            RecoveredFieldSample::FrequencyHertz(50.0),
            entity_count,
            0.0,
        );
        let fixture = partitioned_fixture(&expected);
        let substituted = complex_displacement_field(
            RecoveredFieldSample::FrequencyHertz(75.0),
            entity_count,
            1.0,
        );
        let substituted_partition = canonical_field_partitions(&substituted).unwrap().remove(1);
        let substituted_payload = substituted_partition.canonical_bytes().unwrap();
        let expected_partition_identity = fixture.entry.partitions[1].partition.digest();
        let chunked = chunk_partitioned_numerical_payload(
            AnalysisNumericalObjectKind::Field,
            Digest::sha256("substituted-frequency-field-root"),
            &[AnalysisPayloadPartition {
                logical_identity: expected_partition_identity,
                payload: &substituted_payload,
            }],
            4096,
        )
        .unwrap();
        let objects = chunked
            .manifest
            .chunks
            .iter()
            .zip(&chunked.chunks)
            .map(|(descriptor, bytes)| (descriptor.digest, bytes.clone()))
            .collect::<BTreeMap<_, _>>();
        let page = LogicalPage {
            offset: RECOVERED_FIELD_PARTITION_ENTITY_COUNT * 3,
            count: 1,
            total_count: entity_count * 3,
        };
        let error = read_logical_values(&fixture.entry, &chunked.manifest, page, &mut |digest| {
            Ok(objects.get(&digest).cloned())
        })
        .unwrap_err();
        assert!(error.contains("semantic manifest"));
    }

    #[test]
    fn logical_page_bounds_are_clamped_without_storage_lane_confusion() {
        let page = LogicalPage::new(
            10,
            AnalysisFieldRequestOptions {
                offset: Some(8),
                limit: Some(20),
            },
        )
        .unwrap();
        assert_eq!(page.offset, 8);
        assert_eq!(page.count, 2);
        assert!(page.truncated());

        let exhausted = LogicalPage::new(
            10,
            AnalysisFieldRequestOptions {
                offset: Some(99),
                limit: Some(4),
            },
        )
        .unwrap();
        assert_eq!(exhausted.offset, 10);
        assert_eq!(exhausted.count, 0);
        assert!(exhausted.truncated());
    }
}
