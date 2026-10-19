//! Provider read witnesses: admission happens before oversized original payload import.
use super::*;
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport},
    ArtifactResult, LogicalObject,
};
use std::cell::RefCell;
struct Reads {
    store: MemoryObjects,
    reads: RefCell<Vec<(Digest, u64)>>,
}
impl CacheImport for Reads {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        panic!("reserved input used an unbounded provider read")
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.borrow_mut().push((digest, maximum));
        self.store.read_verified_bounded(digest, maximum)
    }
}
impl CacheExport for Reads {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        self.store.write_verified(object)
    }
}
struct Never;
impl AnalysisKernel for Never {
    fn execute(
        &self,
        _: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        panic!("rejected original input entered kernel")
    }
}
#[test]
fn original_invocation_exhaustion_rejects_before_any_object_read_or_kernel_entry() {
    let (_, program, store) = super::host_transport::fixture();
    let mut source = Reads {
        store,
        reads: RefCell::new(vec![]),
    };
    let resources = runmat_execution::resource::ResourceRequest {
        cpu_millicores: 1000,
        memory_bytes: 1,
        scratch_bytes: 0,
        max_wall_millis: 1,
        max_artifact_bytes: 0,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: vec![],
        required_capabilities: BTreeSet::new(),
    };
    let error = execute_reserved_serial_analysis(
        program.request(),
        &mut source,
        &Never,
        6,
        ObjectInventoryLimits::default(),
        &resources,
    )
    .unwrap_err();
    assert!(matches!(error, AnalysisBridgeError::Resource(
        runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling { maximum: 1, required, .. }
    ) if required > 1));
    assert!(source.reads.borrow().is_empty());
}
fn numerical_fixture(payload: &[u8], chunk_bytes: usize) -> (PreparedAnalysisObjects, Reads) {
    let mut source = Reads {
        store: MemoryObjects::default(),
        reads: RefCell::new(vec![]),
    };
    let prepared = prepare_analysis_objects(
        AnalysisNumericalObjectKind::NumericalReceipt,
        digest("bounded-original"),
        payload,
        chunk_bytes,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    for object in &prepared.objects {
        source.write_verified(object).unwrap();
    }
    (prepared, source)
}
#[test]
fn complete_manifest_payload_demand_rejects_before_the_first_chunk_read() {
    let payload = runmat_canonical_codec::encode_contract(
        b"bounded-payload\0",
        "bounded-payload",
        &"x".repeat(2_000_000),
        runmat_canonical_codec::CanonicalLimits::new(4_000_000, 100, 4_000_000, 8),
    )
    .unwrap();
    let (prepared, source) = numerical_fixture(&payload, 4096);
    let (_, program, _) = super::host_transport::fixture();
    let admission =
        crate::object::AnalysisInputAdmission::new(program.request(), 1_000_000).unwrap();
    let reference = prepared
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let error = crate::object::import_analysis_objects_admitted(
        &source,
        crate::object::AnalysisObjectImport {
            root: &reference,
            expected_kind: prepared.manifest.kind,
            expected_logical_identity: prepared.manifest.logical_identity,
            expected_payload_digest: prepared.manifest.payload_digest,
            access: &access(),
            limits: ObjectInventoryLimits::default(),
        },
        Some(&admission),
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("execution reservation"),
        "{error}"
    );
    assert_eq!(
        &*source.reads.borrow(),
        &[(reference.logical_digest, reference.encoded_length)]
    );
}
#[test]
fn bounded_import_preserves_original_repeated_chunk_occurrences_and_canonical_identity() {
    let payload = runmat_canonical_codec::encode_contract(
        b"bounded-payload\0",
        "bounded-payload",
        &vec![0.0; 32],
        runmat_canonical_codec::CanonicalLimits::new(4096, 100, 256, 8),
    )
    .unwrap();
    let (prepared, source) = numerical_fixture(&payload, 9);
    assert!(
        prepared
            .manifest
            .chunks
            .iter()
            .map(|c| c.digest)
            .collect::<BTreeSet<_>>()
            .len()
            < prepared.manifest.chunks.len()
    );
    let (_, program, _) = super::host_transport::fixture();
    let admission =
        crate::object::AnalysisInputAdmission::new(program.request(), 1_000_000).unwrap();
    let reference = prepared
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let imported = crate::object::import_analysis_objects_admitted(
        &source,
        crate::object::AnalysisObjectImport {
            root: &reference,
            expected_kind: prepared.manifest.kind,
            expected_logical_identity: prepared.manifest.logical_identity,
            expected_payload_digest: prepared.manifest.payload_digest,
            access: &access(),
            limits: ObjectInventoryLimits::default(),
        },
        Some(&admission),
    )
    .unwrap();
    assert_eq!(imported, prepared);
    admission
        .admit_payload_decoding(std::slice::from_ref(&imported))
        .unwrap();
    let reads = source.reads.borrow();
    assert_eq!(reads.len(), prepared.manifest.chunks.len() + 1);
    for ((digest, limit), chunk) in reads[1..].iter().zip(&prepared.manifest.chunks) {
        assert_eq!((*digest, *limit), (chunk.digest, chunk.encoded_length));
    }
}
