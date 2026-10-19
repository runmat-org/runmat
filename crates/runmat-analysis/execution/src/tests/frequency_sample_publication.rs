use super::*;
use runmat_analysis_core::IndependentDomain;
use runmat_analysis_fea::{
    execution::{
        AcousticFrequencyResult, AcousticFrequencyWorkload, FrequencySamplePublicationSample,
    },
    results::field::RecoveredFieldSample,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn original_acoustic_frequency_catalog_survives_chunked_transport_and_rejects_substitution() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut template = super::acoustic_accuracy_tests::workload();
    template.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.2, 0.4]);
    let workload = AcousticFrequencyWorkload::new(template).unwrap();
    let entries = workload.plan().entries().unwrap();
    let results = entries
        .iter()
        .map(|entry| workload.entry(*entry).unwrap().execute(&mesh).unwrap())
        .collect::<Vec<_>>();
    let samples = entries
        .iter()
        .zip(&results)
        .map(|(entry, result)| {
            let recovery_request = result.native_field_request_from_result().unwrap();
            let fields = result
                .recover_native_fields_from_result(&recovery_request)
                .unwrap()
                .manifest()
                .unwrap();
            FrequencySamplePublicationSample {
                entry: *entry,
                recovery_request,
                fields,
                observations: None,
            }
        })
        .collect();
    let result = AcousticFrequencyResult::assemble(workload, &mesh, results).unwrap();
    let publication = result.publication_manifest(samples).unwrap();
    let authority = result.publication_authority().unwrap();
    assert!(publication.select_sample(None).is_err());
    assert!(publication
        .select_sample(Some(RecoveredFieldSample::FrequencyHertz(0.3)))
        .is_err());
    for chunk in [1024, 8192] {
        let mut prepared = prepare_frequency_sample_publication_objects(
            &publication,
            chunk,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let access = access();
        let root = prepared
            .root_reference(&access, ValueRefKind::ResultObject)
            .unwrap();
        let mut cold_store = MemoryObjects::default();
        for object in &prepared.objects {
            runmat_execution_artifact::cache::CacheExport::write_verified(&mut cold_store, object)
                .unwrap();
        }
        let cold_objects = import_analysis_objects(
            &cold_store,
            &root,
            AnalysisNumericalObjectKind::FrequencySamplePublication,
            publication.identity().digest(),
            prepared.manifest.payload_digest,
            &access,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let cold = decode_prepared_frequency_sample_publication(&cold_objects, &authority).unwrap();
        cold.validate_for(&authority).unwrap();
        assert_eq!(cold, publication);
        for sample in cold.samples() {
            assert_eq!(
                cold.select_sample(Some(RecoveredFieldSample::FrequencyHertz(
                    sample.entry.frequency_hz()
                )))
                .unwrap(),
                sample
            );
        }
        let foreign_authority =
            runmat_analysis_fea::execution::FrequencySamplePublicationAuthority::new(
                authority.workload(),
                runmat_analysis_fea::execution::SolverResultIdentity::from_step_result(
                    authority.producer(),
                    digest("foreign aggregate result"),
                )
                .unwrap(),
                authority.producer(),
                authority.topology(),
                authority.plan().clone(),
                authority.entries().to_vec(),
            )
            .unwrap();
        assert!(
            decode_prepared_frequency_sample_publication(&cold_objects, &foreign_authority)
                .is_err()
        );
        assert!(import_analysis_objects(
            &cold_store,
            &root,
            AnalysisNumericalObjectKind::FrequencySamplePublication,
            digest("foreign root authority"),
            prepared.manifest.payload_digest,
            &access,
            ObjectInventoryLimits::default()
        )
        .is_err());
        cold_store.0.remove(&prepared.manifest.chunks[0].digest);
        assert!(import_analysis_objects(
            &cold_store,
            &root,
            AnalysisNumericalObjectKind::FrequencySamplePublication,
            publication.identity().digest(),
            prepared.manifest.payload_digest,
            &access,
            ObjectInventoryLimits::default()
        )
        .is_err());
        prepared.manifest.logical_identity = digest("foreign publication");
        assert!(decode_prepared_frequency_sample_publication(&prepared, &authority).is_err());
        prepared.manifest.logical_identity = publication.identity().digest();
        prepared.manifest.kind = AnalysisNumericalObjectKind::SolverResult;
        assert!(decode_prepared_frequency_sample_publication(&prepared, &authority).is_err());
    }
}
