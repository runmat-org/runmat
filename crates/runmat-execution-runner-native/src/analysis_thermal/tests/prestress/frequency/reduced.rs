use super::*;

pub(crate) fn reduced_setup(order: ElementOrder) -> ThermalStudyPreparation {
    super::super::fixture::reduced_setup(super::super::super::preparation(order, false))
}

#[test]
fn replayed_thermal_source_survives_basis_publication_and_modal_reduced_worker() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let directory = tempfile::tempdir().unwrap();
        let limits = NativeAnalysisHostLimits::default();
        let mut store =
            FilesystemObjectStore::open(directory.path(), limits.inventory.max_object_bytes)
                .unwrap();
        let completed = solve_heat(&mut store, reduced_setup(order));
        let source = seal(&store, &completed);
        let dependency = source
            .state_publication_dependency(&StateId("second-temperature".into()))
            .unwrap();
        assert!(
            StructuralPrestressedModalFrequencyStudyCoordinator::new(structural_preparation!(
                StructuralPrestressedFrequencyStudyPreparation,
                completed
            ))
            .is_err()
        );
        let mut study =
            StructuralPrestressedModalFrequencyStudyCoordinator::new_with_thermal_source(
                structural_preparation!(StructuralPrestressedFrequencyStudyPreparation, completed),
                source,
            )
            .unwrap();
        assert!(study.bind_frequency().is_err());
        let task = study.prepare_preload().unwrap();
        let (response, _) = execute(
            &mut store,
            &task.host,
            &task.program,
            &task.resource_plan,
            5,
        );
        study.complete_preload(&store, response, &task).unwrap();
        assert_eq!(
            study.preload_task_identity(),
            Some(AnalysisTaskIdentity::new(task.host.workload.identity, vec![dependency]).unwrap())
        );
        let source = study
            .preload_result()
            .unwrap()
            .result
            .linearization
            .as_ref()
            .unwrap()
            .thermal_source()
            .unwrap();
        assert_eq!(
            source.transition().source_state,
            completed.states[1].state.identity
        );
        let task = study.prepare_prestress_state().unwrap();
        let (response, _) = execute(
            &mut store,
            &task.host,
            &task.program,
            &task.resource_plan,
            6,
        );
        study
            .complete_prestress_state(&store, response, &task)
            .unwrap();
        let task = study.prepare_modal().unwrap();
        let (response, _) = execute(
            &mut store,
            &task.host,
            &task.program,
            &task.resource_plan,
            7,
        );
        study.complete_modal(&store, response, &task).unwrap();
        assert!(study.bind_frequency().is_err());
        let task = study.prepare_basis_state().unwrap();
        let (response, _) = execute(
            &mut store,
            &task.host,
            &task.program,
            &task.resource_plan,
            8,
        );
        study.complete_basis_state(&store, response, &task).unwrap();
        let bound = study.bind_frequency().unwrap();
        assert_eq!(bound.state_task, study.basis_state_task_identity().unwrap());
        assert_eq!(
            bound.modal_state.as_ref().unwrap().identity,
            study.basis_state().unwrap().state.identity
        );
        assert_eq!(
            bound.prestress_state.identity,
            study.prestress_state().unwrap().state.identity
        );
        assert_eq!(
            study.basis_state().unwrap().state.source_result,
            study.modal_result().unwrap().identity
        );
        execute_frequency(&mut store, &completed, bound);
    }
}
