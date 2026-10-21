use super::*;
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::CanonicalMeshingContract;

pub(crate) fn frequency_setup(order: ElementOrder) -> ThermalStudyPreparation {
    super::fixture::frequency_setup(super::super::preparation(order, false))
}

#[test]
fn replayed_named_thermal_publication_drives_nonzero_complex_frequency_worker() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let directory = tempfile::tempdir().unwrap();
        let limits = NativeAnalysisHostLimits::default();
        let mut store =
            FilesystemObjectStore::open(directory.path(), limits.inventory.max_object_bytes)
                .unwrap();
        let completed = solve_heat(&mut store, frequency_setup(order));
        let source = seal(&store, &completed);
        let dependency = source
            .state_publication_dependency(&StateId("second-temperature".into()))
            .unwrap();
        assert!(
            StructuralPrestressedFrequencyStudyCoordinator::new(structural_preparation!(
                StructuralPrestressedFrequencyStudyPreparation,
                completed
            ))
            .is_err()
        );
        let mut study = StructuralPrestressedFrequencyStudyCoordinator::new_with_thermal_source(
            structural_preparation!(StructuralPrestressedFrequencyStudyPreparation, completed),
            source,
        )
        .unwrap();
        preload!(study, &mut store, dependency);
        let bound = study.bind_frequency().unwrap();
        assert_eq!(bound.state_task, study.state_task_identity().unwrap());
        execute_frequency(&mut store, &completed, bound);
    }
}

fn execute_frequency(
    store: &mut FilesystemObjectStore,
    completed: &CompletedThermalStudy,
    bound: PreparedStructuralPrestressedFrequency,
) {
    let limits = NativeAnalysisHostLimits::default();
    let heat = completed.preparation();
    let workload = prepare_structural_frequency_workload_objects(
        &bound.workload,
        limits.chunk_bytes,
        ObjectNamespace::InputValue,
        limits.inventory,
    )
    .unwrap();
    let mesh = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*heat.mesh.canonical_digest.bytes()),
        &heat.mesh.canonical_encode().unwrap(),
        limits.chunk_bytes,
        ObjectNamespace::InputValue,
        limits.inventory,
    )
    .unwrap();
    let prestress_state = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SemanticState,
        bound.prestress_state.identity.digest(),
        &bound.prestress_state.canonical_bytes().unwrap(),
        limits.chunk_bytes,
        ObjectNamespace::InputValue,
        limits.inventory,
    )
    .unwrap();
    let modal_state = bound.modal_state.as_ref().map(|state| {
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::SemanticState,
            state.identity.digest(),
            &state.canonical_bytes().unwrap(),
            limits.chunk_bytes,
            ObjectNamespace::InputValue,
            limits.inventory,
        )
        .unwrap()
    });
    let tasks = prepare_structural_frequency_fanout(StructuralFrequencyFanoutPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        workload,
        mesh,
        prestress_state: Some(prestress_state),
        modal_state,
        artifact_access: heat.artifact_access,
        program_revision: heat.program_revision,
        chunk_bytes: limits.chunk_bytes,
        inventory_limits: limits.inventory,
    })
    .unwrap();
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    let (response, _) = execute(store, &task.host, &task.program, &task.resource_plan, 7);
    let result =
        import_structural_frequency_entry(store, response, task, limits.inventory).unwrap();
    assert_eq!(result.result.request(), &task.request);
    let displacement = result.result.sample().displacement();
    assert!(displacement
        .real()
        .iter()
        .any(|value| value.abs() > 1.0e-15));
    assert!(displacement
        .imaginary()
        .iter()
        .any(|value| value.abs() > 1.0e-18));
}

pub(crate) mod reduced;
