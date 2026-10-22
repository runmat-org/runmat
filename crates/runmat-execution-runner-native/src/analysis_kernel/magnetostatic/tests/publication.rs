use super::*;
use runmat_analysis_execution::{
    execute_serial_analysis, import_magnetostatic_result, import_semantic_state_completion,
    prepare_semantic_state_publication, AnalysisHostResponse, SemanticStatePublicationPreparation,
};
use runmat_analysis_fea::execution::{CanonicalSemanticState, SemanticStatePublicationSource};
use runmat_execution_artifact::{cache::FilesystemObjectStore, object::ObjectInventoryLimits};

#[test]
fn native_magnetic_cold_result_receipt_and_each_state_reopen_without_mesh_or_workload() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut identity = None;
        for chunk in [1024, 8192] {
            let (_, _, task) = fixture::task(order, 2, 1.0, 1.0, chunk);
            let limits = ObjectInventoryLimits::default();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("objects");
            let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            task.program.write_to(&mut store).unwrap();
            let done = execute_serial_analysis(
                &task.program.request,
                &mut store,
                &NativeAnalysisKernel,
                chunk,
                limits,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&task.host, &done)
                .unwrap()
                .program_response();
            drop(store);
            let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
            let imported = import_magnetostatic_result(&store, response, &task, limits).unwrap();
            let requests = imported.result.state_publication_requests().unwrap();
            assert_eq!(requests.len(), 2);
            let mut states = Vec::new();
            for request in requests {
                let publication =
                    prepare_semantic_state_publication(SemanticStatePublicationPreparation {
                        source_host: task.host.clone(),
                        source_result: imported.objects.clone(),
                        source_profiles: imported.source_profiles.clone(),
                        source: SemanticStatePublicationSource::MagnetostaticNativeFields,
                        declaration: request.declaration.clone(),
                        topology_digest: request.topology_digest,
                        program_revision: task.program.request.recipe.program_revision.clone(),
                        chunk_bytes: chunk,
                        inventory_limits: limits,
                    })
                    .unwrap();
                assert_eq!(publication.request, request);
                assert!(publication.inputs.iter().all(|o| !matches!(
                    o.manifest.kind,
                    AnalysisNumericalObjectKind::Mesh
                        | AnalysisNumericalObjectKind::MagnetostaticWorkload
                )));
                publication.program.write_to(&mut store).unwrap();
                let done = execute_serial_analysis(
                    &publication.program.request,
                    &mut store,
                    &NativeAnalysisKernel,
                    chunk,
                    limits,
                )
                .unwrap();
                let response = AnalysisHostResponse::completed(&publication.host, &done)
                    .unwrap()
                    .program_response();
                drop(store);
                store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
                let published =
                    import_semantic_state_completion(&store, response, &publication, limits)
                        .unwrap();
                let state =
                    CanonicalSemanticState::from_canonical_bytes(&published.objects.payload())
                        .unwrap();
                assert_eq!(state, published.state);
                assert_eq!(
                    state.source_result,
                    imported.result.source_result_identity().unwrap()
                );
                assert_eq!(
                    state.payload.magnetostatic().unwrap().result.as_ref(),
                    &imported.result
                );
                assert_eq!(
                    state.payload.magnetostatic().unwrap().fields.fields().len(),
                    6
                );
                states.push(state.identity);
            }
            if let Some(previous) = identity {
                assert_eq!(previous, states);
            }
            identity = Some(states);
        }
    }
}
