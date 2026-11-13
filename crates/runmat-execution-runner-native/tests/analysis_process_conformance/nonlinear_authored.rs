//! Original authored CAD/native/cold J2 lifecycle; independent homogeneous hardening reference.
use runmat_execution::value::PublishedObjectSource;
use runmat_execution_artifact::{
    cache::{CacheExport, FilesystemObjectStore},
    object::ObjectInventoryLimits,
    LogicalObject, ObjectNamespace,
};
use runmat_execution_runner_native::{
    execute_prepared_fea_study_locally, nonlinear_solid_study_completion_from_source,
    prepare_checked_fea_study, CompletedFeaStudyResult, LocalFeaExecutionOptions,
    StudyRootPlanning,
};
use runmat_runtime::analysis::FeaRootSelector;

const SOURCE: &str = include_str!("../../../../docs/simulation/examples/small-strain-j2.fea");

pub(super) async fn run() {
    let mut original_history_failures = Vec::new();
    for quadratic in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        std::env::set_var(
            "RUNMAT_EXECUTION_STATE_DIR",
            directory.path().join("native"),
        );
        std::env::set_var("RUNMAT_FEA_ARTIFACT_ROOT", directory.path().join("runs"));
        std::env::set_var(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            directory.path().join("studies"),
        );
        runmat_runtime::analysis::storage::configure_artifact_store_from_env().unwrap();
        std::fs::create_dir(directory.path().join("geometry")).unwrap();
        std::fs::write(
            directory.path().join("geometry/box.brep"),
            include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep"),
        )
        .unwrap();
        let path = directory.path().join("plastic.fea");
        std::fs::write(
            &path,
            SOURCE.replace(
                "order = linear",
                if quadratic {
                    "order = quadratic"
                } else {
                    "order = linear"
                },
            ),
        )
        .unwrap();
        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        let checked = session.load_and_check_fea_source_path(&path).await.unwrap();
        assert_eq!(
            checked.checked().check_result().status,
            runmat_runtime::analysis::FeaCheckStatus::Valid,
            "{:?}",
            checked.checked().check_result().diagnostics
        );
        let options = LocalFeaExecutionOptions {
            root: Some(FeaRootSelector::Study {
                id: "plastic".into(),
            }),
            workers: Some(1),
            ..Default::default()
        };
        let runmat_runtime::analysis::ResolvedFeaRoot::Study(resolved) =
            checked.resolve_root(options.root.as_ref()).unwrap()
        else {
            panic!("expected authored J2 study");
        };
        let prepared =
            prepare_checked_fea_study(&checked, resolved, &[], "original-authored-j2", &options)
                .await
                .unwrap();
        let StudyRootPlanning::NonlinearSolid(root) = &prepared.root.planning else {
            panic!("J2 was sent to a linear root");
        };
        assert_eq!(root.controls.increment_count, 12);
        assert_eq!(root.controls.max_accepted_increments, 4096);
        assert_eq!(root.accepted_block_limit, 8);
        eprintln!("authored J2 starting original CAD/native lifecycle quadratic={quadratic}");
        let case = execute_prepared_fea_study_locally(prepared, &options)
            .await
            .unwrap();
        let CompletedFeaStudyResult::Nonlinear(result) = case.result else {
            panic!("J2 domain substituted");
        };
        assert!(result.completion.block_count >= 2);
        assert_eq!(result.completion.states.len(), 3);
        let expected = result
            .completion
            .terminal_block
            .result
            .accepted_state
            .clone();
        let mut maximum_alpha_error = 0.0_f64;
        let mut maximum_plastic_errors = [0.0_f64; 6];
        let reference = [0.02, -0.01, -0.01, 0.0, 0.0, 0.0];
        for (_, _, point) in expected.point_history() {
            maximum_alpha_error =
                maximum_alpha_error.max((point.equivalent_plastic_strain - 0.02).abs());
            for (component, error) in maximum_plastic_errors.iter_mut().enumerate() {
                *error = error.max(
                    (point.engineering_plastic_strain[component] - reference[component]).abs(),
                );
            }
        }
        println!("original authored J2 history quadratic={quadratic}, alpha_error={maximum_alpha_error:e}, plastic_errors={maximum_plastic_errors:?}, residual_max={:e}, increment_max={:e}",
            result.completion.terminal_block.checkpoint.maximum_relative_residual(),
            result.completion.terminal_block.checkpoint.maximum_relative_increment());
        if maximum_alpha_error >= 2e-9 || maximum_plastic_errors.iter().any(|e| *e >= 2e-9) {
            original_history_failures.push(format!("quadratic={quadratic}, alpha_error={maximum_alpha_error:e}, plastic_errors={maximum_plastic_errors:?}"));
        }
        assert!(result.completion.observation_catalog.is_some());
        let mesh = result.mesh.clone();
        let evidence = result.meshing_evidence.clone();
        evidence.validate(&mesh).unwrap();
        let limits = ObjectInventoryLimits::default();
        let mut retained =
            FilesystemObjectStore::open(directory.path().join("cold"), limits.max_object_bytes)
                .unwrap();
        for reference in &result.completion.catalog.objects {
            let bytes = result.source.read_verified(reference).unwrap();
            retained
                .write_verified(
                    &LogicalObject::new(
                        ObjectNamespace::ResultValue,
                        "original-authored-j2",
                        "application/octet-stream",
                        bytes,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        let root = result.completion.root.clone();
        let access =
            runmat_execution_runner_native::checked_fea_artifact_access("original-authored-j2");
        let memory = result.completion.workload.model.sequence.steps[0]
            .accuracy
            .resources
            .resident_byte_budget();
        let publication = case.publication;
        drop(result);
        let (cold, source) = nonlinear_solid_study_completion_from_source(
            retained,
            root.clone(),
            root.logical_digest,
            access,
            limits,
            memory,
        )
        .unwrap();
        source.authenticate(&cold, memory).unwrap();
        assert_eq!(cold.terminal_block.result.accepted_state, expected);
        let final_block = source
            .materialize_block(&cold, cold.block_count - 1, 0, memory)
            .unwrap();
        assert_eq!(final_block.accepted_state, expected);
        let stored =
            runmat_runtime::analysis::storage::load_run_result(&publication.steps[0].run_id)
                .unwrap()
                .unwrap();
        assert_eq!(stored.run.observations.len(), 2);
        assert_eq!(stored.run.observations, final_block.run.observations);
        let stress = stored
            .run
            .observations
            .iter()
            .find(|observation| {
                observation.quantity == runmat_analysis_core::ObservationQuantity::Stress
            })
            .unwrap();
        let mut maximum_stress_errors = [0.0_f64; 6];
        for value in stress.field.as_host_f64().unwrap().chunks_exact(6) {
            for (component, error) in maximum_stress_errors.iter_mut().enumerate() {
                let reference = if component == 0 { 5.6 } else { 0.0 };
                *error = error.max((value[component] - reference).abs());
            }
        }
        println!(
            "original authored J2 stress quadratic={quadratic}, errors={maximum_stress_errors:?}"
        );
        if maximum_stress_errors.iter().any(|error| *error >= 2e-7) {
            original_history_failures.push(format!(
                "quadratic={quadratic}, stress_errors={maximum_stress_errors:?}"
            ));
        }
        let history = stored.nonlinear_results.unwrap();
        assert_eq!(history.load_factors.last(), Some(&1.0));
        assert_eq!(
            history.load_factors.len(),
            cold.terminal_block.checkpoint.work().accepted_increments()
        );
        println!("authored CAD/native/cold J2 transport checks passed quadratic={quadratic}, nodes={}, elements={}, accepted={}",
            mesh.topology.nodes.len(), mesh.topology.volume_elements.len(), history.load_factors.len());
    }
    assert!(
        original_history_failures.is_empty(),
        "unchanged authored J2 history/stress gates failed: {original_history_failures:?}"
    );
}
