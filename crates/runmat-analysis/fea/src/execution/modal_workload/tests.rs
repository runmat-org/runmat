use super::*;

fn workload() -> StructuralModalWorkload {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let truncation = ModalTruncationPolicy::requested(3);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: truncation.requested_mode_count(),
    };
    model.sequence.steps = vec![step];
    StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model,
        step_id: ProcedureStepId("modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation,
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap()
}

#[test]
fn modal_workload_round_trips_with_convergent_identity() {
    let workload = workload();
    let restored =
        StructuralModalWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(restored, workload);
    assert_eq!(
        workload.modal_basis_output().unwrap(),
        &workload.procedure_step().unwrap().state_outputs[0]
    );
}

#[test]
fn residual_vector_directions_are_canonical_workload_inputs() {
    let base = workload();
    let direction = runmat_analysis_core::SpectrumDirection {
        frame: runmat_analysis_core::CoordinateFrame::Global,
        vector: [1.0, 0.5, 0.25],
    };
    let enriched = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: base.model().clone(),
        step_id: base.step_id().clone(),
        backend: base.backend(),
        observation_execution: base.observation_execution(),
        truncation: base.truncation(),
        convergence: base.convergence(),
        residual_vector_directions: vec![direction.clone()],
        prestress_transition: base.prestress_transition().cloned(),
    })
    .unwrap();
    assert_ne!(enriched.identity(), base.identity());
    assert_eq!(
        enriched.residual_vector_directions(),
        std::slice::from_ref(&direction)
    );
    assert_eq!(
        StructuralModalWorkload::from_canonical_bytes(&enriched.canonical_bytes().unwrap())
            .unwrap(),
        enriched
    );

    assert!(StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: base.model().clone(),
        step_id: base.step_id().clone(),
        backend: base.backend(),
        observation_execution: base.observation_execution(),
        truncation: base.truncation(),
        convergence: base.convergence(),
        residual_vector_directions: vec![
            direction.clone(),
            runmat_analysis_core::SpectrumDirection {
                frame: runmat_analysis_core::CoordinateFrame::Global,
                vector: [2.0, 1.0, 0.5],
            },
        ],
        prestress_transition: base.prestress_transition().cloned(),
    })
    .is_err());
}

#[test]
fn modal_workload_rejects_an_unpublishable_basis_declaration() {
    let valid = workload();
    let invalid_models = [
        {
            let mut model = valid.model().clone();
            model.sequence.steps[0].state_outputs.clear();
            model
        },
        {
            let mut model = valid.model().clone();
            let duplicate = model.sequence.steps[0].state_outputs[0].clone();
            model.sequence.steps[0].state_outputs.push(duplicate);
            model
        },
        {
            let mut model = valid.model().clone();
            model.sequence.steps[0].state_outputs[0].kind = StateKind::StructuralPrimary;
            model
        },
        {
            let mut model = valid.model().clone();
            model.sequence.steps[0].state_outputs[0].completeness = StateCompleteness::Partial;
            model
        },
        {
            let mut model = valid.model().clone();
            model.sequence.steps[0].state_outputs[0].persistence = StatePersistence::CheckpointOnly;
            model
        },
    ];

    for model in invalid_models {
        assert!(StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model,
            step_id: valid.step_id().clone(),
            backend: valid.backend(),
            observation_execution: valid.observation_execution(),
            truncation: valid.truncation(),
            convergence: valid.convergence(),
            residual_vector_directions: valid.residual_vector_directions().to_vec(),
            prestress_transition: valid.prestress_transition().cloned(),
        })
        .is_err());
    }
}

#[test]
fn modal_workload_accepts_a_basis_retained_until_dependents_complete() {
    let valid = workload();
    let mut model = valid.model().clone();
    model.sequence.steps[0].state_outputs[0].persistence =
        StatePersistence::UntilDependentsComplete;

    let workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model,
        step_id: valid.step_id().clone(),
        backend: valid.backend(),
        observation_execution: valid.observation_execution(),
        truncation: valid.truncation(),
        convergence: valid.convergence(),
        residual_vector_directions: valid.residual_vector_directions().to_vec(),
        prestress_transition: valid.prestress_transition().cloned(),
    })
    .unwrap();
    assert_eq!(
        workload.modal_basis_output().unwrap().persistence,
        StatePersistence::UntilDependentsComplete
    );
}

#[test]
fn modal_workload_rejects_a_mismatched_requested_eigenspace() {
    let workload = workload();
    let mut model = workload.model().clone();
    model.sequence.steps[0].independent_domain =
        IndependentDomain::Eigenvalue { requested_count: 2 };
    assert!(StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model,
        step_id: workload.step_id().clone(),
        backend: workload.backend(),
        observation_execution: workload.observation_execution(),
        truncation: workload.truncation(),
        convergence: workload.convergence(),
        residual_vector_directions: workload.residual_vector_directions().to_vec(),
        prestress_transition: workload.prestress_transition().cloned(),
    })
    .is_err());
}

#[test]
fn model_rebinding_regenerates_the_workload_identity() {
    let mut workload = workload();
    let identity = workload.identity();
    let mut model = workload.model().clone();
    model.geometry_revision += 1;
    workload.replace_model(model).unwrap();
    assert_ne!(workload.identity(), identity);
}
