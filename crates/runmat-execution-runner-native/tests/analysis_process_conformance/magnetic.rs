//! Actual child-process solve and cold retained-state import; not distributed qualification.
use super::publication_chain::{ProcessDriver, PublicationChainDriver};
use runmat_analysis_execution::{
    import_magnetostatic_result, import_semantic_state_completion, prepare_magnetostatic_task,
    prepare_semantic_state_publication, AnalysisOutputPolicy, MagnetostaticTaskPreparation,
    PreparedMagnetostaticTask, SemanticStatePublicationPreparation,
};
use runmat_analysis_fea::execution::SemanticStatePublicationSource;
use runmat_execution::Digest;
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport, FilesystemObjectStore},
    object::ObjectInventoryLimits,
    LogicalObject, ObjectNamespace, ProgramExecutionResponse,
};
use runmat_execution_runner_native::{NativeExecutionConfig, NativeProgramSession};
use runmat_meshing_core::ElementOrder;

#[path = "../../src/analysis_kernel/magnetostatic/tests/fixture.rs"]
mod fixture;

pub(super) async fn run() {
    let limits = ObjectInventoryLimits::default();
    let chunk = runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (_, _, task) = fixture::task(order, 2, 1.0, 1.0, chunk);
        let directory = tempfile::tempdir().unwrap();
        let object_root = directory.path().join("retained");
        let mut retained =
            FilesystemObjectStore::open(&object_root, limits.max_object_bytes).unwrap();
        let root = directory.path().join("session");
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.executable = std::env::current_exe().unwrap();
        config.max_workers = 1;
        config.store_root = root;
        config.worker_capabilities = super::capabilities();
        let session = NativeProgramSession::new(config).unwrap();
        let mut store = session.object_store();
        let response = ProcessDriver {
            session: &session,
            store: &mut store,
        }
        .execute(&task.host, &task.program, Some(&task.resource_plan))
        .await;
        response.resources.validate().unwrap();
        let mut expected = task.resource_plan.request.clone();
        expected.required_capabilities = super::capabilities();
        assert_eq!(response.resources.resources.allocation, expected);
        let imported =
            import_magnetostatic_result(&store, response.response, &task, limits).unwrap();
        assert!(
            imported.result.response().magnetic_energy_j
                > runmat_types::physical::Quantity::new(0.0)
        );
        let mut publications = Vec::new();
        for request in imported.result.state_publication_requests().unwrap() {
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
            let response = ProcessDriver {
                session: &session,
                store: &mut store,
            }
            .execute(
                &publication.host,
                &publication.program,
                Some(&publication.resource_plan),
            )
            .await;
            response.resources.validate().unwrap();
            let mut expected = publication.resource_plan.request.clone();
            expected.required_capabilities = super::capabilities();
            assert_eq!(response.resources.resources.allocation, expected);
            let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } =
                &response.response
            else {
                panic!("publication must externalize its complete closure");
            };
            for reference in result_objects {
                let bytes = store
                    .read_verified(reference.logical_digest)
                    .unwrap()
                    .unwrap();
                assert_eq!(bytes.len() as u64, reference.encoded_length);
                let object = LogicalObject::new(
                    ObjectNamespace::ResultValue,
                    "retained",
                    "application/octet-stream",
                    bytes,
                )
                .unwrap();
                retained.write_verified(&object).unwrap();
            }
            publications.push((publication, response.response));
        }
        // All producers are stopped before any terminal artifacts are reopened.
        drop(retained);
        drop(store);
        drop(session);
        let store = FilesystemObjectStore::open(object_root, limits.max_object_bytes).unwrap();
        for (publication, response) in publications {
            let state =
                import_semantic_state_completion(&store, response, &publication, limits).unwrap();
            assert_eq!(
                state.state.payload.magnetostatic().unwrap().result.as_ref(),
                &imported.result
            );
        }
    }
}
