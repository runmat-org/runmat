use super::*;

#[test]
fn solver_identity_views_keep_original_canonical_bytes() {
    // Captured from the pre-change copied-vector implementation in legacy job49378.
    let expected = [
        [
            "sha256:8b89568d3841c1ce441c78e479e5adf2213566efc0f2886ec0a6f29d4ba2c961",
            "sha256:0676503e55e3f8a763a4def7460a29d3a17a1434c8de6a7b36f7e50e67904b35",
            "sha256:d5ae2437c81fe4537d9c33465801af6d0ca3c926c3ac2225e333430a2383edb6",
        ],
        [
            "sha256:7270d4cc04253625b90b8e85fc75e6a142f445b6248a357b6a89221c0d56fef1",
            "sha256:5bdbb13137dc2ddcfae602095d70aba8237a6fbde9d5995a25dda7f0a446f435",
            "sha256:88d6d35a5c8efc3ea43077b8218f479e9dc4a168adca2b395e4f8d671f328b69",
        ],
    ];
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    for (case, restart) in [
        runmat_analysis_core::RestartPolicy::Fresh,
        runmat_analysis_core::RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "identity-serialization-fixture".into(),
            required_state_digest: "ab".repeat(32),
        },
    ]
    .into_iter()
    .enumerate()
    {
        // Checkpoint variant is a serialization fixture, not an executable restart witness.
        model.sequence.steps[0].restart = restart;
        let (_, live, peak) = crate::topology::measure_test_allocations(|| {
            (
                model.solver_identity_contract(),
                model.sequence.solver_identity_contract(),
                model.sequence.steps[0].solver_identity_contract(),
            )
        });
        assert_eq!((live, peak), (0, 0));
        for (position, (label, bytes)) in [
            (
                "model",
                encode_contract(
                    b"solver-view-reference\0",
                    "model",
                    &model.solver_identity_contract(),
                    LIMITS,
                )
                .unwrap(),
            ),
            (
                "sequence",
                encode_contract(
                    b"solver-view-reference\0",
                    "sequence",
                    &model.sequence.solver_identity_contract(),
                    LIMITS,
                )
                .unwrap(),
            ),
            (
                "step",
                encode_contract(
                    b"solver-view-reference\0",
                    "step",
                    &model.sequence.steps[0].solver_identity_contract(),
                    LIMITS,
                )
                .unwrap(),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                Digest::sha256(&bytes).to_string(),
                expected[case][position],
                "{case}/{label}"
            );
        }
    }
}

#[test]
fn original_source_identity_binding_fits_codec_reservation_and_rejects_short_cap() {
    use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps(
        "identity_heat",
        vec![ProcedureStep::thermal_transient("heat")],
    );
    let workload = ThermalWorkload {
        model,
        step_id: ProcedureStepId("heat".into()),
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
        initial_fields: vec![],
    };
    let budget = workload
        .source_identity_resident_bytes_upper_bound()
        .unwrap();
    let (bound, _, peak) = crate::topology::measure_test_allocations(|| {
        workload.bind_source_identities_admitted(budget).unwrap()
    });
    assert!(peak as u64 <= budget);
    assert_eq!(bound.workload, workload.identity().unwrap());
    assert_eq!(
        bound.numerical_intent,
        workload.numerical_intent_identity().unwrap()
    );
    assert_eq!(
        bound.producer,
        crate::execution::StepIdentity::from_solver_contract(
            crate::execution::SequenceIdentity::from_solver_contract(&workload.model.sequence)
                .unwrap(),
            bound.step
        )
        .unwrap()
    );
    assert!(matches!(
        workload.bind_source_identities_admitted(budget - 1),
        Err(AnalysisExecutionError::Limit(
            "thermal source identity workspace"
        ))
    ));
}
