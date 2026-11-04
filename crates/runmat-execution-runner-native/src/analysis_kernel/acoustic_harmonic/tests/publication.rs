//! A separately admitted shared publication consumes the committed acoustic result, not a solve.
use super::*;
use runmat_analysis_core::AnalysisFieldValues;
use runmat_analysis_execution::{
    execute_serial_analysis, import_acoustic_harmonic_result, import_semantic_state_completion,
    prepare_analysis_objects, prepare_semantic_state_publication, AnalysisHostResponse,
    SemanticStatePublicationPreparation,
};
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, SemanticStatePublicationSource, StatePublicationProfiles,
};
use runmat_execution_artifact::cache::FilesystemObjectStore;

#[test]
fn acoustic_result_to_shared_publication_reopens_cold_fields_for_both_orders() {
    for radiation in [false, true] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mut expected_identity = None;
            for chunk in [1024, 8192] {
                let (_, _, task) = task_with_boundary(order, chunk, "pressure", radiation);
                let limits = ObjectInventoryLimits::default();
                let directory = tempfile::tempdir().unwrap();
                let object_path = directory.path().join("objects");
                let mut store =
                    FilesystemObjectStore::open(&object_path, limits.max_object_bytes).unwrap();
                task.program.write_to(&mut store).unwrap();
                let completed = execute_serial_analysis(
                    &task.program.request,
                    &mut store,
                    &NativeAnalysisKernel,
                    chunk,
                    limits,
                )
                .unwrap();
                let response = AnalysisHostResponse::completed(&task.host, &completed)
                    .unwrap()
                    .program_response();
                // Reopen durable objects before importing the numerical completion. This is cold
                // artifact recovery, not process replacement or a remote scheduler claim.
                drop(store);
                let mut store =
                    FilesystemObjectStore::open(&object_path, limits.max_object_bytes).unwrap();
                let imported =
                    import_acoustic_harmonic_result(&store, response, &task, limits).unwrap();
                let requests = imported.result.state_publication_requests().unwrap();
                assert_eq!(requests.len(), 1);
                let request = &requests[0];
                let publication =
                    prepare_semantic_state_publication(SemanticStatePublicationPreparation {
                        source_host: task.host.clone(),
                        source_result: imported.objects.clone(),
                        source_profiles: imported.source_profiles,
                        source: SemanticStatePublicationSource::AcousticNativeFields,
                        declaration: request.declaration.clone(),
                        topology_digest: request.topology_digest,
                        program_revision: task.program.request.recipe.program_revision.clone(),
                        chunk_bytes: chunk,
                        inventory_limits: limits,
                    })
                    .unwrap();
                assert_eq!(
                    publication.host.workload.kind,
                    AnalysisWorkKind::PublishState
                );
                assert_eq!(publication.request, *request);
                assert!(publication.inputs.iter().all(|input| !matches!(
                    input.manifest.kind,
                    AnalysisNumericalObjectKind::Mesh
                        | AnalysisNumericalObjectKind::AcousticHarmonicWorkload
                )));
                publication.program.write_to(&mut store).unwrap();
                let completed = execute_serial_analysis(
                    &publication.program.request,
                    &mut store,
                    &NativeAnalysisKernel,
                    chunk,
                    limits,
                )
                .unwrap();
                let response = AnalysisHostResponse::completed(&publication.host, &completed)
                    .unwrap()
                    .program_response();
                drop(store);
                let store =
                    FilesystemObjectStore::open(&object_path, limits.max_object_bytes).unwrap();
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
                assert_eq!(state.producer, task.host.workload.step);
                let payload = state.payload.acoustic().unwrap();
                assert_eq!(payload.ordered_node_ids, imported.result.ordered_node_ids());
                assert_eq!(
                    payload.ordered_element_ids,
                    imported.result.ordered_element_ids()
                );
                // Pressure, particle velocity, active intensity and plane-wave port intensity.
                assert_eq!(payload.fields.fields().len(), 4);
                assert_eq!(payload.fields.entity_counts.volume_integration_points, 0);
                if let Some(expected) = expected_identity {
                    assert_eq!(state.identity, expected);
                } else {
                    expected_identity = Some(state.identity);
                }
                if order == ElementOrder::Tet4 {
                    assert_exact_element_pressure(&state);
                }
            }
        }
    }
}

fn assert_exact_element_pressure(state: &CanonicalSemanticState) {
    // Independent unit-right-tetrahedron Helmholtz row: rho=3,c=2,Z=6,f=.2,
    // prescribed p1=p2=p3=1. No RunMat assembly or result is used as the reference.
    let omega = 2.0 * std::f64::consts::PI * 0.2;
    let ar = 1.0 / 18.0 - omega.powi(2) / 720.0;
    let ai = omega * 3.0_f64.sqrt() / 72.0;
    let br = 1.0 / 18.0 + omega.powi(2) / 480.0;
    let bi = -ai;
    let denominator = ar * ar + ai * ai;
    let expected = [
        (br * ar + bi * ai) / denominator,
        (bi * ar - br * ai) / denominator,
    ];
    let payload = state.payload.acoustic().unwrap();
    let node = payload
        .ordered_node_ids
        .iter()
        .position(|id| *id == 4)
        .unwrap();
    let AnalysisFieldValues::HostF64(pressure) = &payload.fields.fields()[0].storage().values
    else {
        panic!("portable pressure must be host-backed")
    };
    for component in 0..2 {
        assert!((pressure[2 * node + component] - expected[component]).abs() < 1e-10);
    }
}

#[test]
fn acoustic_publication_rejects_foreign_source_and_understated_profile() {
    let (workload, mesh, task) = task(ElementOrder::Tet10, 1024);
    let limits = ObjectInventoryLimits::default();
    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveAcousticHarmonic,
            host: &task.host,
            inputs: &task.inputs,
            output_policy: AnalysisOutputPolicy::new(1024, limits).unwrap(),
        })
        .unwrap();
    let mut objects = outputs
        .into_iter()
        .map(|output| output.into_prepared(1024, limits).unwrap());
    let result_objects = objects.next().unwrap();
    let profile_objects = objects
        .find(|objects| {
            objects.manifest.kind == AnalysisNumericalObjectKind::StatePublicationProfiles
        })
        .unwrap();
    let result =
        AcousticHarmonicSolverResult::from_canonical_bytes(&result_objects.payload()).unwrap();
    let request = result.state_publication_requests().unwrap().remove(0);
    let preparation = |source_result, source_profiles| SemanticStatePublicationPreparation {
        source_host: task.host.clone(),
        source_result,
        source_profiles,
        source: SemanticStatePublicationSource::AcousticNativeFields,
        declaration: request.declaration.clone(),
        topology_digest: request.topology_digest,
        program_revision: task.program.request.recipe.program_revision.clone(),
        chunk_bytes: 1024,
        inventory_limits: limits,
    };
    let mut foreign = workload;
    foreign.solver.relative_tolerance *= 0.5;
    let foreign = foreign.execute(&mesh).unwrap();
    let foreign_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        foreign.source_result_identity().unwrap().digest(),
        &foreign.canonical_bytes().unwrap(),
        1024,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    assert!(prepare_semantic_state_publication(preparation(
        foreign_objects,
        profile_objects.clone()
    ))
    .is_err());

    let catalog =
        StatePublicationProfiles::from_canonical_bytes(&profile_objects.payload()).unwrap();
    let mut entries = catalog.entries().to_vec();
    entries[0].encoded_bytes_upper_bound -= 1;
    let understated = StatePublicationProfiles::new(
        request.producer,
        request.source_result,
        request.topology_digest,
        result_objects.manifest.payload_digest,
        entries,
    )
    .unwrap();
    let understated = prepare_analysis_objects(
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        understated.identity(),
        &understated.canonical_bytes().unwrap(),
        1024,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    assert!(prepare_semantic_state_publication(preparation(result_objects, understated)).is_err());
}
