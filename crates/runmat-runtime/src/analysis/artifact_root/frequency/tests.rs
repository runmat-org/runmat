use super::*;
use runmat_analysis_fea::results::field::{
    RecoveredFieldSample, RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};
use std::collections::BTreeSet;

mod fixture;
use fixture::fixture;

#[test]
fn exact_frequency_pages_reopen_original_partitions_without_loading_other_samples() {
    let fixture = fixture();
    let access = fixture.access();
    let cold: FrequencySampleArtifactRootInventory =
        serde_json::from_value(serde_json::to_value(&fixture.inventory).unwrap()).unwrap();
    cold.validate(&access).unwrap();
    let mut admission_reads = BTreeSet::new();
    let admitted = cold
        .admit(access, |digest| {
            admission_reads.insert(digest);
            Ok(fixture.objects.get(&digest).cloned())
        })
        .unwrap();
    assert_eq!(
        admission_reads,
        fixture
            .catalog
            .roots
            .iter()
            .map(|root| root.logical_digest)
            .collect()
    );
    let offset = RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 1;
    let mut reads = BTreeSet::new();
    let page = admitted
        .load_field_page(
            "cold-frequency-contract",
            &AnalysisFieldSelection::new(
                "pressure",
                Some(RecoveredFieldSample::FrequencyHertz(2.0)),
            ),
            AnalysisFieldRequestOptions {
                offset: Some(offset),
                limit: Some(1),
            },
            |digest| {
                reads.insert(digest);
                Ok(fixture.objects.get(&digest).cloned())
            },
        )
        .unwrap();
    let AnalysisFieldPageLookup::Page(page) = page else {
        panic!("expected exact cold field page")
    };
    assert_eq!(page.offset, offset);
    assert_eq!(page.count, 1);
    assert_eq!(page.total_count, RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 2);
    assert_eq!(
        page.values,
        crate::analysis::AnalysisFieldPageValues::ComplexInterleaved(vec![
            200_000.0 + (offset * 2) as f64,
            200_000.0 + (offset * 2 + 1) as f64
        ],)
    );
    let expected = fixture
        .second_partition_chunks
        .iter()
        .copied()
        .chain(std::iter::once(
            cold.samples()[1].fields().reference().logical_digest,
        ))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        reads, expected,
        "only the selected root and intersecting partition chunks are read"
    );
    assert!(!reads.contains(&cold.samples()[0].fields().reference().logical_digest));
    for selection in [
        AnalysisFieldSelection::new("pressure", None),
        AnalysisFieldSelection::new("pressure", Some(RecoveredFieldSample::FrequencyHertz(3.0))),
    ] {
        let lookup = admitted
            .load_field_page(
                "cold-frequency-contract",
                &selection,
                AnalysisFieldRequestOptions {
                    offset: Some(0),
                    limit: Some(1),
                },
                |_| panic!("ambiguous or absent samples must not perform storage reads"),
            )
            .unwrap();
        assert!(matches!(
            lookup,
            AnalysisFieldPageLookup::SampleSelectionRequired(_)
                | AnalysisFieldPageLookup::SampleNotFound
        ));
    }
}

#[test]
fn sample_root_inventory_rejects_missing_swapped_foreign_and_extra_roots() {
    let fixture = fixture();
    let access = fixture.access();
    for mutation in 0..5 {
        let mut inventory = fixture.inventory.clone();
        match mutation {
            0 => {
                inventory.samples.pop();
            }
            1 => inventory.samples.swap(0, 1),
            2 => inventory.samples[1] = inventory.samples[0].clone(),
            3 => {
                inventory.publication.expected_payload_digest =
                    Digest::sha256("foreign publication payload")
            }
            4 => {
                inventory.samples[0].fields.expected_kind = AnalysisNumericalObjectKind::Observation
            }
            _ => unreachable!(),
        }
        assert!(inventory.validate(&access).is_err());
    }
    let foreign = FrequencySamplePublicationAuthority::new(
        fixture.authority.workload(),
        runmat_analysis_fea::execution::SolverResultIdentity::from_step_result(
            fixture.authority.producer(),
            Digest::sha256("foreign aggregate result"),
        )
        .unwrap(),
        fixture.authority.producer(),
        fixture.authority.topology(),
        fixture.authority.plan().clone(),
        fixture.authority.entries().to_vec(),
    )
    .unwrap();
    assert!(fixture
        .inventory
        .validate(&FrequencySamplePublicationAccess {
            authority: &foreign,
            ..access
        })
        .is_err());
    let mut extra = fixture.catalog.clone();
    let child = extra
        .objects
        .iter()
        .find(|object| !extra.roots.contains(object))
        .unwrap()
        .clone();
    extra.roots.push(child);
    extra.roots.sort_by_key(|reference| reference.id);
    extra.validate().unwrap();
    assert!(fixture
        .inventory
        .validate(&FrequencySamplePublicationAccess {
            object_catalog: &extra,
            ..fixture.access()
        })
        .is_err());
}

#[test]
fn complete_manifest_admission_checks_unselected_roots_and_chunk_declarations() {
    let fixture = fixture();
    let unselected = fixture.inventory.samples()[0]
        .fields()
        .reference()
        .logical_digest;
    for corrupt in [false, true] {
        assert!(fixture
            .inventory
            .admit(fixture.access(), |digest| {
                let mut bytes = fixture.objects.get(&digest).cloned();
                if digest == unselected {
                    if corrupt {
                        bytes.as_mut().unwrap()[0] ^= 1;
                    } else {
                        bytes = None;
                    }
                }
                Ok(bytes)
            })
            .is_err());
    }
    let mut missing = fixture.catalog.clone();
    let root_bytes = fixture.objects.get(&unselected).unwrap();
    let manifest =
        runmat_analysis_fea::execution::AnalysisObjectManifest::from_canonical_bytes(root_bytes)
            .unwrap();
    let missing_digest = manifest.chunks[0].digest;
    missing
        .objects
        .retain(|reference| reference.logical_digest != missing_digest);
    missing.validate().unwrap();
    let access = FrequencySamplePublicationAccess {
        object_catalog: &missing,
        ..fixture.access()
    };
    fixture.inventory.validate(&access).unwrap();
    assert!(fixture
        .inventory
        .admit(access, |digest| Ok(fixture.objects.get(&digest).cloned()))
        .is_err());
}

#[test]
fn shared_physical_closure_allows_identical_reuse_but_checks_every_typed_role() {
    let fixture = fixture();
    let mut roots = vec![fixture.inventory.publication()];
    roots.extend(
        fixture
            .inventory
            .samples()
            .iter()
            .map(|sample| sample.fields()),
    );
    roots.push(fixture.inventory.publication());
    super::super::closure::validate_closure(&roots, &fixture.catalog, |digest| {
        Ok(fixture.objects.get(&digest).cloned())
    })
    .unwrap();
    let mut forged = fixture.inventory.publication().clone();
    forged.expected_kind = AnalysisNumericalObjectKind::Field;
    roots.push(&forged);
    assert!(
        super::super::closure::validate_closure(&roots, &fixture.catalog, |digest| Ok(fixture
            .objects
            .get(&digest)
            .cloned()))
        .is_err()
    );
}

#[test]
fn selected_frequency_root_and_chunk_corruption_fail_closed() {
    let fixture = fixture();
    let admitted = fixture
        .inventory
        .admit(fixture.access(), |digest| {
            Ok(fixture.objects.get(&digest).cloned())
        })
        .unwrap();
    let selection =
        AnalysisFieldSelection::new("pressure", Some(RecoveredFieldSample::FrequencyHertz(2.0)));
    let root = fixture.inventory.samples()[1]
        .fields()
        .reference()
        .logical_digest;
    for digest in [root, fixture.second_partition_chunks[0]] {
        for corrupt in [false, true] {
            assert!(admitted
                .load_field_page(
                    "cold-frequency-contract",
                    &selection,
                    AnalysisFieldRequestOptions {
                        offset: Some(RECOVERED_FIELD_PARTITION_ENTITY_COUNT),
                        limit: Some(1)
                    },
                    |requested| {
                        let mut bytes = fixture.objects.get(&requested).cloned();
                        if requested == digest {
                            if corrupt {
                                bytes.as_mut().unwrap()[0] ^= 1;
                            } else {
                                bytes = None;
                            }
                        }
                        Ok(bytes)
                    },
                )
                .is_err());
        }
    }
}
