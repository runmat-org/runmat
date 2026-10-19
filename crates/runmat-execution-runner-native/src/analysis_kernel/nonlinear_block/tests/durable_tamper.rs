//! Rehashed completion changes exercise semantics, not only hash integrity.
use runmat_analysis_execution::{
    import_analysis_objects, import_nonlinear_solid_study_root, prepare_analysis_objects,
    AnalysisOutputPolicy, PreparedNonlinearSolidStudyRoot,
};
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, AnalysisObjectManifest};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::{value::ValueRefKind, Digest};
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport, FilesystemObjectStore},
    ArtifactAccess, ObjectNamespace,
};

pub(super) fn verify_rehashed_metadata(
    source: &FilesystemObjectStore,
    original: &PreparedNonlinearSolidStudyRoot,
    access: &ArtifactAccess,
    policy: AnalysisOutputPolicy,
    maximum_resident_bytes: u64,
) {
    let limits = policy.inventory_limits();
    let manifest = AnalysisObjectManifest::from_canonical_bytes(
        &source
            .read_verified(original.root.logical_digest)
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    let payload = import_analysis_objects(
        source,
        &original.root,
        AnalysisNumericalObjectKind::NonlinearSolidStudyCompletion,
        manifest.logical_identity,
        manifest.payload_digest,
        access,
        limits,
    )
    .unwrap()
    .payload();
    let codec = CanonicalLimits::new(256 * 1024 * 1024, 8_000_000, 65_536, 64);
    let original_wire: serde_json::Value = decode_contract(
        b"runmat-nonlinear-solid-study-completion\0",
        "nonlinear-solid-study-completion",
        &payload,
        codec,
    )
    .unwrap();
    let unrelated = prepare_analysis_objects(
        AnalysisNumericalObjectKind::NumericalReceipt,
        Digest::sha256(b"unrelated-closure"),
        &[7, 8, 9],
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    assert_eq!(
        original_wire["schema_version"], 2,
        "original completion schema"
    );
    for alteration in 0..7 {
        let mut wire = original_wire.clone();
        match alteration {
            0 => {} // Round trip is admitted: JSON value transport itself is not a rejection cause.
            1 => wire["preparation"]["accepted_block_limit"] = 8.into(),
            2 => {
                wire["closures"].as_array_mut().unwrap().remove(1);
            }
            3 => wire["closures"].as_array_mut().unwrap().swap(0, 1),
            4 => wire["schema_version"] = 1.into(),
            5 => {
                let catalog: runmat_execution::value::PublishedObjectCatalog =
                    serde_json::from_value(wire["descendants"].clone()).unwrap();
                let mut objects = catalog.objects;
                objects.extend(
                    unrelated
                        .unique_references(access, ValueRefKind::ResultObject)
                        .unwrap(),
                );
                wire["descendants"] = serde_json::to_value(
                    runmat_execution::value::PublishedObjectCatalog::new(catalog.roots, objects)
                        .unwrap(),
                )
                .unwrap();
            }
            6 => wire["schema_version"] = 3.into(),
            _ => unreachable!(),
        }
        if alteration != 0 {
            assert_ne!(
                wire, original_wire,
                "alteration {alteration} must change original metadata"
            );
        }
        let bytes = encode_contract(
            b"runmat-nonlinear-solid-study-completion\0",
            "nonlinear-solid-study-completion",
            &wire,
            codec,
        )
        .unwrap();
        let objects = prepare_analysis_objects(
            AnalysisNumericalObjectKind::NonlinearSolidStudyCompletion,
            Digest::sha256(&bytes),
            &bytes,
            policy.chunk_bytes(),
            ObjectNamespace::ResultValue,
            limits,
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let mut store =
            FilesystemObjectStore::open(directory.path().join("objects"), limits.max_object_bytes)
                .unwrap();
        for reference in &original.catalog.objects {
            let object = runmat_execution_artifact::LogicalObject::new(
                ObjectNamespace::ResultValue,
                "unchanged-j2-descendant",
                "application/octet-stream",
                source
                    .read_verified(reference.logical_digest)
                    .unwrap()
                    .unwrap(),
            )
            .unwrap();
            store.write_verified(&object).unwrap();
        }
        for object in &objects.objects {
            store.write_verified(object).unwrap();
        }
        for object in &unrelated.objects {
            store.write_verified(object).unwrap();
        }
        let root = objects
            .root_reference(access, ValueRefKind::ResultObject)
            .unwrap();
        let imported = import_nonlinear_solid_study_root(
            &store,
            &root,
            root.logical_digest,
            access,
            limits,
            maximum_resident_bytes,
        );
        assert_eq!(
            imported.is_ok(),
            alteration == 0,
            "fully rehashed alteration {alteration}"
        );
    }
}
