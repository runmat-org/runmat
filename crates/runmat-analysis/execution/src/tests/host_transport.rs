//! Transport is explicit immutable invocation data; worker settings are ceilings.
use super::*;

pub(super) fn fixture() -> (
    AnalysisHostWorkload,
    PreparedAnalysisHostInvocation,
    MemoryObjects,
) {
    let policy = AnalysisOutputPolicy::new(6, ObjectInventoryLimits::default()).unwrap();
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        digest("transport-mesh"),
        b"canonical mesh bytes",
        6,
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )
    .unwrap();
    let root = input
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let (study, step, execution) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
        step,
        execution,
        AnalysisWorkKind::Solve,
        vec![AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let host = AnalysisHostWorkload::new(workload, access(), policy).unwrap();
    let invocation = host.prepare_program_request(revision(), &[root]).unwrap();
    let mut store = MemoryObjects::default();
    for object in input.objects {
        runmat_execution_artifact::cache::CacheExport::write_verified(&mut store, &object).unwrap();
    }
    invocation.write_to(&mut store).unwrap();
    (host, invocation, store)
}

#[test]
fn cold_host_policy_is_explicit_and_worker_defaults_cannot_replace_it() {
    let (host, invocation, mut store) = fixture();
    let cold = AnalysisHostWorkload::from_canonical_bytes(
        &host.canonical_bytes().unwrap(),
        host.artifact_access.clone(),
    )
    .unwrap();
    assert_eq!(cold, host);
    #[derive(serde::Serialize)]
    struct Claim<'a> {
        schema_version: u16,
        workload: &'a AnalysisWorkloadRequest,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_policy: Option<AnalysisOutputPolicy>,
    }
    let limits =
        runmat_canonical_codec::CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
    let mut claim = Claim {
        schema_version: 2,
        workload: &host.workload,
        output_policy: Some(host.output_policy),
    };
    let encode = |claim: &Claim<'_>| {
        runmat_canonical_codec::encode_contract(
            b"runmat-analysis-host-invocation/v2\0",
            "host-invocation",
            claim,
            limits,
        )
        .unwrap()
    };
    // Positive control proves the only missing-policy change is the omitted field.
    assert_eq!(
        AnalysisHostWorkload::from_canonical_bytes(&encode(&claim), host.artifact_access.clone())
            .unwrap(),
        host
    );
    claim.output_policy = None;
    assert!(AnalysisHostWorkload::from_canonical_bytes(
        &encode(&claim),
        host.artifact_access.clone()
    )
    .is_err());
    claim.output_policy = Some(host.output_policy);
    claim.schema_version = 1;
    assert!(AnalysisHostWorkload::from_canonical_bytes(
        &encode(&claim),
        host.artifact_access.clone()
    )
    .is_err());
    let completed = execute_serial_analysis(
        invocation.request(),
        &mut store,
        &FieldKernel,
        8192,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    // FieldKernel independently requires six-byte output chunks. Placement offered 8192.
    assert_eq!(
        completed.outputs[0].manifest.kind,
        AnalysisNumericalObjectKind::Field
    );
    let mut changed = host.clone();
    changed.output_policy =
        AnalysisOutputPolicy::new(7, host.output_policy.inventory_limits()).unwrap();
    assert_ne!(
        host.canonical_bytes().unwrap(),
        changed.canonical_bytes().unwrap()
    );
    let mut changed = host;
    changed.output_policy = serde_json::from_value(serde_json::json!({
        "chunk_bytes": 0,
        "inventory_limits": {"max_objects": 100, "max_object_bytes": 1024, "max_total_bytes": 4096}
    }))
    .unwrap();
    assert!(changed.validate().is_err());
}

#[test]
fn exceeding_worker_transport_ceiling_rejects_before_physical_input_reads() {
    let (_, invocation, store) = fixture();
    struct Store(RecordingObjects);
    impl runmat_execution_artifact::cache::CacheImport for Store {
        fn read_verified(
            &self,
            digest: Digest,
        ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
            runmat_execution_artifact::cache::CacheImport::read_verified(&self.0, digest)
        }
    }
    impl runmat_execution_artifact::cache::CacheExport for Store {
        fn write_verified(
            &mut self,
            _: &runmat_execution_artifact::LogicalObject,
        ) -> runmat_execution_artifact::ArtifactResult<()> {
            panic!("rejected transport cannot publish an output");
        }
    }
    let mut source = Store(RecordingObjects {
        objects: store.0,
        reads: RefCell::new(Vec::new()),
    });
    assert!(execute_serial_analysis(
        invocation.request(),
        &mut source,
        &FieldKernel,
        5,
        ObjectInventoryLimits::default()
    )
    .is_err());
    assert_eq!(
        source.0.reads.borrow().as_slice(),
        &[invocation.descriptor().descriptor.digest]
    );
}

#[test]
fn oversized_output_set_root_rejects_before_any_cache_read() {
    let (host, _, _) = fixture();
    let object = runmat_execution_artifact::LogicalObject::new(
        ObjectNamespace::ResultValue,
        "oversized-output-root",
        crate::response::ANALYSIS_OUTPUT_SET_MEDIA_TYPE,
        vec![1],
    )
    .unwrap();
    let mut root = host
        .artifact_access
        .reference(
            &object,
            ValueRefKind::ResultObject,
            crate::response::ANALYSIS_OUTPUT_SET_VALUE_SCHEMA,
        )
        .unwrap();
    root.encoded_length = crate::response::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES + 1;
    let response = runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![runmat_execution::value::ValuePayload::Object(Box::new(
            root.clone(),
        ))],
        result_objects: vec![root],
    };
    let source = RecordingObjects {
        objects: Default::default(),
        reads: RefCell::new(Vec::new()),
    };
    assert!(import_analysis_output_set(&source, response, &host).is_err());
    assert!(source.reads.borrow().is_empty());
}

#[test]
fn host_failure_retains_original_cause_and_reads_no_success_artifacts() {
    let (host, _, _) = fixture();
    let source = RecordingObjects {
        objects: Default::default(),
        reads: RefCell::new(Vec::new()),
    };
    let message = "modal eigensolver rejected its numerical result";
    let error = import_analysis_output_set(
        &source,
        runmat_execution_artifact::ProgramExecutionResponse::Failure {
            message: message.into(),
        },
        &host,
    )
    .unwrap_err();
    assert!(
        matches!(error, AnalysisBridgeError::HostExecution(ref original) if original == message)
    );
    assert!(source.reads.borrow().is_empty());
}
