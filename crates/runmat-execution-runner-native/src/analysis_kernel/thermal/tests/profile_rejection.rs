use super::*;
use runmat_analysis_execution::{
    execute_serial_analysis, import_thermal_result, AnalysisHostResponse,
};
use runmat_analysis_fea::execution::StatePublicationProfiles;

/// Produce a self-consistent transport closure whose size certificate is dishonest.
struct UnderdeclaredProfileKernel;

impl AnalysisKernel for UnderdeclaredProfileKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        let output_policy = invocation.output_policy;
        let mut outputs = NativeAnalysisKernel.execute(invocation)?;
        assert_eq!(outputs.len(), 3);
        let original = outputs.pop().unwrap().into_prepared(
            output_policy.chunk_bytes(),
            output_policy.inventory_limits(),
        )?;
        assert_eq!(
            original.manifest.kind,
            AnalysisNumericalObjectKind::StatePublicationProfiles
        );
        let catalog = StatePublicationProfiles::from_canonical_bytes(&original.payload())?;
        let result = outputs.remove(0).into_prepared(
            output_policy.chunk_bytes(),
            output_policy.inventory_limits(),
        )?;
        let manifest = decode_prepared_thermal_manifest(&result)?;
        let mut entries = catalog.entries().to_vec();
        let entry = entries.first_mut().expect("fixture declares thermal state");
        assert!(entry.encoded_bytes_upper_bound > 1);
        entry.encoded_bytes_upper_bound -= 1;
        let forged = StatePublicationProfiles::new(
            manifest.producer,
            manifest.identity(),
            manifest.topology_digest,
            result.manifest.payload_digest,
            entries,
        )?;
        assert_ne!(forged.identity(), catalog.identity());
        forged.validate_source(
            manifest.producer,
            manifest.identity(),
            manifest.topology_digest,
            result.manifest.payload_digest,
        )?;
        let bytes = forged.canonical_bytes()?;
        // The scalar mutation preserves encoded width and therefore transport admission.
        assert_eq!(bytes.len(), catalog.canonical_bytes()?.len());
        assert_eq!(
            StatePublicationProfiles::from_canonical_bytes(&bytes)?,
            forged
        );
        outputs.insert(0, AnalysisKernelOutput::prepared(result));
        outputs.push(AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::StatePublicationProfiles,
            forged.identity(),
            bytes,
        ));
        Ok(outputs)
    }
}

#[test]
fn thermal_import_rejects_rehashed_underdeclared_publication_profile() {
    let (_, task) = task(ElementOrder::Tet4, 1024);
    let limits = ObjectInventoryLimits::default();
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    task.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &task.program.request,
        &mut store,
        &UnderdeclaredProfileKernel,
        1024,
        limits,
    )
    .expect("rehashed catalog must pass output transport validation and admission");
    let response = AnalysisHostResponse::completed(&task.host, &completed)
        .unwrap()
        .program_response();
    let error = import_thermal_result(&store, response, &task, limits).unwrap_err();
    assert!(
        matches!(
            error,
            AnalysisBridgeError::Identity(
                "thermal publication profiles differ from canonical metadata projection"
            )
        ),
        "expected semantic projection mismatch, got {error:?}"
    );
}
