use super::*;
use runmat_meshing_core::{
    build_chunked_stage_payload, close_stage_payload_with_admission, MeshingChunkMediaType,
    MeshingChunkPolicy, MeshingChunkStream, MeshingManifestDisposition, MeshingManifestMetadata,
    MeshingStageKind, MeshingStageResultKind, NeverCancelled, StableDigest,
};

fn closed(
    mut admit: impl FnMut(u64) -> Result<(), runmat_meshing_core::MeshingContractError>,
) -> runmat_meshing_core::ClosedMeshingStage {
    let payload = build_chunked_stage_payload(
        &[MeshingChunkStream {
            media_type: MeshingChunkMediaType::MeshNodes,
            schema_version: 2,
            records: vec![vec![1; 200], vec![2; 13000]],
        }],
        MeshingChunkPolicy {
            maximum_chunk_bytes: 1024,
            maximum_records_per_chunk: 10,
            maximum_total_encoded_bytes: 8 << 20,
        },
    )
    .unwrap();
    close_stage_payload_with_admission(
        MeshingManifestMetadata {
            stage: MeshingStageKind::Tetrahedralization,
            result_kind: MeshingStageResultKind::WholeStage,
            producer_identity_digest: StableDigest::from_bytes([1; 32]),
            invariant_summary_digest: StableDigest::from_bytes([2; 32]),
            disposition: MeshingManifestDisposition::ValidatedDependency,
        },
        vec![],
        payload,
        &NeverCancelled,
        &mut admit,
    )
    .unwrap()
}
#[test]
fn construction_authority_preserves_canonical_publication_and_cold_import_verifies_it() {
    let mut admitted = 0u64;
    let closed = closed(|bytes| {
        admitted += bytes;
        Ok(())
    });
    let objects = crate::objects::prepare_closed_stage_objects(
        closed,
        ObjectInventoryLimits::default(),
        |bytes| {
            admitted += bytes;
            Ok(())
        },
    )
    .unwrap();
    let publication = crate::publication::prepare_validated_result_publication(
        objects,
        &access("run", 1),
        &mut |bytes| {
            admitted += bytes;
            Ok(())
        },
    )
    .unwrap();
    let expected = prepare_result_publication(
        fixture(vec![vec![1; 200], vec![2; 13000]], 1024),
        access("run", 1),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(publication, expected);
    assert!(
        admitted
            > publication.result_objects().len() as u64
                * std::mem::size_of::<runmat_execution::value::ValueRef>() as u64
    );
    let mut cache = MemoryCache::default();
    cache.insert_all(&publication.stage_objects().objects);
    let runmat_execution::value::ValuePayload::Object(root) = publication.root_output() else {
        panic!("manifest reference required")
    };
    assert_eq!(
        import_result_publication(
            &cache,
            root,
            access("run", 1),
            ObjectInventoryLimits::default()
        )
        .unwrap(),
        publication
    );
    let mut hostile = publication.stage_objects().clone();
    hostile.objects[1].bytes[0] ^= 1;
    assert!(prepare_result_publication(
        hostile,
        access("run", 1),
        ObjectInventoryLimits::default()
    )
    .is_err());
}
#[test]
fn every_object_and_reference_allocation_boundary_can_deny_before_publication() {
    let mut count = 0;
    crate::objects::prepare_closed_stage_objects(
        closed(|_| Ok(())),
        ObjectInventoryLimits::default(),
        |_| {
            count += 1;
            Ok(())
        },
    )
    .unwrap();
    for denied in 0..count {
        let mut observed = 0;
        let result = crate::objects::prepare_closed_stage_objects(
            closed(|_| Ok(())),
            ObjectInventoryLimits::default(),
            |_| {
                let ordinal = observed;
                observed += 1;
                if ordinal == denied {
                    Err(crate::MeshingExecutionError::Invalid(
                        "denied object metadata".into(),
                    ))
                } else {
                    Ok(())
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(observed, denied + 1);
    }
    let mut count = 0;
    let objects = crate::objects::prepare_closed_stage_objects(
        closed(|_| Ok(())),
        ObjectInventoryLimits::default(),
        |_| Ok(()),
    )
    .unwrap();
    crate::publication::prepare_validated_result_publication(
        objects,
        &access("run", 1),
        &mut |_| {
            count += 1;
            Ok(())
        },
    )
    .unwrap();
    for denied in 0..count {
        let objects = crate::objects::prepare_closed_stage_objects(
            closed(|_| Ok(())),
            ObjectInventoryLimits::default(),
            |_| Ok(()),
        )
        .unwrap();
        let mut observed = 0;
        let result = crate::publication::prepare_validated_result_publication(
            objects,
            &access("run", 1),
            &mut |_| {
                let ordinal = observed;
                observed += 1;
                if ordinal == denied {
                    Err(crate::MeshingExecutionError::Invalid(
                        "denied reference metadata".into(),
                    ))
                } else {
                    Ok(())
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(observed, denied + 1);
    }
}

mod cold;
