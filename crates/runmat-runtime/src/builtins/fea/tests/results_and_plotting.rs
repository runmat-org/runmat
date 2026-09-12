use super::*;

#[test]
fn fea_results_field_exposes_values_metadata_and_plot_context() {
    let (run_value, _study) = synthetic_plot_run_value();

    let results = block_on(fea_results_builtin(vec![run_value])).expect("results should load");
    let Value::Object(results_object) = results.clone() else {
        panic!("expected results object");
    };
    assert!(results_object.class_name.is(FEA_RESULTS_CLASS));
    assert_eq!(
        results_object.properties.get("run_id"),
        Some(&Value::String("synthetic_plot_run".to_string()))
    );
    assert!(results_object
        .properties
        .contains_key(FEA_STUDY_CONTEXT_JSON_PROPERTY));

    let field = block_on(fea_field_builtin(vec![
        results,
        Value::String("von_mises".to_string()),
    ]))
    .expect("field should resolve by unique suffix");
    let Value::Object(field_object) = field else {
        panic!("expected field object");
    };
    assert!(field_object.class_name.is(FEA_FIELD_CLASS));
    assert_eq!(
        field_object.properties.get("field_id"),
        Some(&Value::String("structural.von_mises".to_string()))
    );
    assert_eq!(
        field_object.properties.get("unit"),
        Some(&Value::String("Pa".to_string()))
    );
    assert_eq!(
        field_object.properties.get("location"),
        Some(&Value::String("element".to_string()))
    );
    assert_eq!(
        field_object.properties.get("topology_id"),
        Some(&Value::String("analysis_mesh".to_string()))
    );
    assert_eq!(
        field_object.properties.get("element_kind"),
        Some(&Value::String("tetrahedron4".to_string()))
    );
    assert_eq!(
        field_object.properties.get("entity_count"),
        Some(&Value::Int(runmat_value::IntValue::U64(1)))
    );
    assert_eq!(
        field_object.properties.get("value_count"),
        Some(&Value::Int(runmat_value::IntValue::U64(1)))
    );
    assert_eq!(
        field_object.properties.get("element_count"),
        Some(&Value::Int(runmat_value::IntValue::U64(1)))
    );
    let Some(Value::Tensor(values)) = field_object.properties.get("values") else {
        panic!("expected values tensor");
    };
    assert_eq!(values.shape, vec![1]);
    assert_eq!(values.materialize_f64(), vec![42.0]);
    assert!(field_object
        .properties
        .contains_key(FEA_STUDY_CONTEXT_JSON_PROPERTY));
    assert_eq!(
        field_object.properties.get(FEA_RUN_ID_CONTEXT_PROPERTY),
        Some(&Value::String("synthetic_plot_run".to_string()))
    );
}

#[cfg(feature = "plot-core")]
#[test]
fn fea_plot_returns_figure_handle_for_contextual_run_results_and_fields() {
    let (run_value, _study) = synthetic_plot_run_value();

    let run_handle = block_on(fea_plot_builtin(vec![
        run_value.clone(),
        Value::String("von_mises".to_string()),
    ]))
    .expect("run plot should create a figure");
    assert!(matches!(run_handle, Value::Num(handle) if handle >= 1.0));

    let results = block_on(fea_results_builtin(vec![run_value])).expect("results should load");
    let field = block_on(fea_field_builtin(vec![
        results,
        Value::String("structural.von_mises".to_string()),
    ]))
    .expect("field should resolve");
    let field_handle =
        block_on(fea_plot_builtin(vec![field])).expect("field plot should create a figure");
    assert!(matches!(field_handle, Value::Num(handle) if handle >= 1.0));
}

#[test]
fn fea_plot_request_accepts_solver_mesh_edge_option() {
    let (run_value, _study) = synthetic_plot_run_value();

    let request = plot_request_from_args(&[
        run_value,
        Value::String("von_mises".to_string()),
        Value::String("mesh".to_string()),
        Value::String("solver".to_string()),
        Value::String("deformed".to_string()),
        Value::Bool(false),
        Value::String("overlay".to_string()),
        Value::String("cad".to_string()),
    ])
    .expect("plot request should parse mesh, deformation, and overlay options");

    assert_eq!(request.field_id.as_deref(), Some("von_mises"));
    assert!(request.options.show_solver_mesh_edges);
    assert!(!request.options.apply_deformation_overlay);
    assert_eq!(
        request.options.mesh_source,
        crate::analysis::AnalysisFigureMeshSource::CadReference
    );
}

#[cfg(feature = "plot-core")]
#[test]
fn fea_plot_default_prefers_von_mises_scalar_figure() {
    let mut figures = vec![
        generated_test_figure("deformation", vec!["structural.displacement"]),
        generated_test_figure("stress", vec!["structural.von_mises"]),
        generated_test_figure("residual", vec!["structural.residual_norm"]),
    ];

    let selected =
        select_generated_figure(&mut figures, None).expect("default figure should select");

    assert_eq!(selected.title, "stress");
}

#[cfg(feature = "plot-core")]
#[test]
fn fea_plot_default_selects_representative_non_structural_figures() {
    let cases = [
        (
            vec![
                generated_test_figure("thermal residual", vec!["thermal.residual_norm"]),
                generated_test_figure("temperature", vec!["thermal.temperature.0"]),
            ],
            "temperature",
        ),
        (
            vec![
                generated_test_figure("flow residual", vec!["cfd.residual_momentum"]),
                generated_test_figure("velocity", vec!["fluid.velocity"]),
            ],
            "velocity",
        ),
        (
            vec![
                generated_test_figure("acoustic phase", vec!["acoustic.phase"]),
                generated_test_figure("pressure", vec!["acoustic.pressure"]),
            ],
            "pressure",
        ),
        (
            vec![
                generated_test_figure(
                    "coupling residual",
                    vec!["thermo_mechanical.coupling_residual.0"],
                ),
                generated_test_figure("thermal stress", vec!["thermo_mechanical.thermal_stress.0"]),
            ],
            "thermal stress",
        ),
    ];

    for (mut figures, expected_title) in cases {
        let selected =
            select_generated_figure(&mut figures, None).expect("default figure should select");
        assert_eq!(selected.title, expected_title);
    }
}

#[cfg(feature = "plot-core")]
fn generated_test_figure(
    title: &str,
    field_ids: Vec<&str>,
) -> crate::analysis::AnalysisGeneratedFigure {
    crate::analysis::AnalysisGeneratedFigure {
        kind: crate::analysis::AnalysisGeneratedFigureKind::MeshResult,
        title: title.to_string(),
        field_ids: field_ids.into_iter().map(str::to_string).collect(),
        topology_ids: Vec::new(),
        warnings: Vec::new(),
        figure: runmat_plot::plots::Figure::new(),
    }
}

#[test]
fn fea_usize_parser_reads_typed_integer_storage_exactly_and_rejects_float_boundary() {
    let wide = if usize::BITS == 64 {
        9_007_199_254_740_993
    } else {
        u32::MAX as u64
    };
    let typed = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![wide]), vec![1, 1])
        .expect("typed integer");

    assert_eq!(
        usize_from_value(STUDY_NAME, &Value::Tensor(typed)).expect("typed integer"),
        wide as usize
    );

    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    assert!(usize_from_value(STUDY_NAME, &Value::Num(boundary)).is_err());
}

fn synthetic_plot_run_value() -> (Value, Value) {
    crate::analysis::storage::configure_artifact_store(
        crate::analysis::storage::AnalysisArtifactStoreConfig::InMemory,
    )
    .expect("artifact store should configure");

    let tmp = tempfile::tempdir().expect("tempdir should be created");
    std::fs::write(tmp.path().join("part.stl"), TRIANGLE_STL)
        .expect("geometry fixture should write");
    let fea_path = tmp.path().join("plot.fea");
    std::fs::write(
        &fea_path,
        r#"
version: 1
kind: study
id: synthetic_plot
geometry:
  path: part.stl
  units: meter
model:
  profile: linear_static_structural
run:
  backend: cpu
"#,
    )
    .expect("FEA fixture should write");
    let study = block_on(fea_load_builtin(fea_path.to_string_lossy().to_string()))
        .expect("study should load");
    let Value::Object(study_object) = &study else {
        panic!("expected study object");
    };
    let study_json = match study_object.properties.get(FEA_STUDY_SPEC_JSON_PROPERTY) {
        Some(Value::String(json)) => json.clone(),
        _ => panic!("expected study spec payload"),
    };

    let run = crate::analysis::AnalysisRunResult {
        run_id: "synthetic_plot_run".to_string(),
        run: runmat_analysis_fea::FeaRunResult {
            backend: ComputeBackend::Cpu,
            solver_backend: "synthetic".to_string(),
            solver_device_apply_k_ratio: 0.0,
            solver_method: "synthetic".to_string(),
            preconditioner: "none".to_string(),
            solver_host_sync_count: 0,
            diagnostics: Vec::new(),
            fields: vec![AnalysisField::host_f64(
                "structural.von_mises",
                vec![1],
                vec![42.0],
            )],
        },
        render_topology: Some(crate::analysis::AnalysisRenderTopology {
            schema_version: "analysis_render_topology/v1".to_string(),
            source: crate::analysis::AnalysisRenderTopologySource::AnalysisMesh,
            meshes: vec![crate::analysis::AnalysisRenderMesh {
                mesh_id: "synthetic_plot_boundary".to_string(),
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                triangles: vec![[0, 1, 2]],
                regions: Vec::new(),
                vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
                triangle_volume_element_indices: vec![Some(0)],
            }],
        }),
        modal_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        electromagnetic_results: None,
        model_validity: crate::analysis::QualityGate::Pass,
        solver_convergence: crate::analysis::QualityGate::Pass,
        result_quality: crate::analysis::QualityGate::Pass,
        run_status: crate::analysis::RunStatus::Publishable,
        publishable: true,
        quality_reasons: Vec::new(),
        provenance: crate::analysis::RunProvenance {
            backend: ComputeBackend::Cpu,
            solver_backend: "synthetic".to_string(),
            solver_device_apply_k_ratio: 0.0,
            solver_host_sync_count: 0,
            precision_mode: "fp64".to_string(),
            deterministic_mode: true,
            solver_method: "synthetic".to_string(),
            preconditioner: "none".to_string(),
            quality_policy: "balanced".to_string(),
            fallback_events: Vec::new(),
        },
    };
    crate::analysis::storage::persist_run_result(&run).expect("run should persist");

    let mut object = ObjectInstance::new(FEA_RUN_RESULT_CLASS.to_string());
    object.properties.insert(
        "run_id".to_string(),
        Value::String("synthetic_plot_run".to_string()),
    );
    object.properties.insert(
        FEA_RUN_ID_CONTEXT_PROPERTY.to_string(),
        Value::String("synthetic_plot_run".to_string()),
    );
    object.properties.insert(
        FEA_STUDY_CONTEXT_JSON_PROPERTY.to_string(),
        Value::String(study_json),
    );
    (Value::Object(object), study)
}

fn persist_synthetic_indexed_results() -> String {
    let indexed_fields = |prefix: &str| {
        vec![
            AnalysisField::host_f64(format!("{prefix}.0"), vec![1], vec![10.0]),
            AnalysisField::host_f64(format!("{prefix}.1"), vec![1], vec![20.0]),
        ]
    };
    let run_id = "synthetic_indexed_results".to_string();
    let run = crate::analysis::AnalysisRunResult {
        run_id: run_id.clone(),
        run: runmat_analysis_fea::FeaRunResult {
            backend: ComputeBackend::Cpu,
            solver_backend: "synthetic".to_string(),
            solver_device_apply_k_ratio: 0.0,
            solver_method: "synthetic".to_string(),
            preconditioner: "none".to_string(),
            solver_host_sync_count: 0,
            diagnostics: Vec::new(),
            fields: Vec::new(),
        },
        render_topology: None,
        modal_results: Some(crate::analysis::ModalResultsData {
            modal_payload_version: "modal_results/v1".to_string(),
            eigenvalues_hz: vec![10.0, 20.0],
            mode_shapes: indexed_fields("mode_shape"),
            residual_norms: vec![0.1, 0.2],
            mode_units: crate::analysis::ModalFrequencyUnits::Hz,
            frequency_basis: crate::analysis::ModalFrequencyBasis::NativeEigenSolve,
        }),
        thermal_results: None,
        transient_results: Some(crate::analysis::TransientResultsData {
            transient_payload_version: "transient_results/v1".to_string(),
            time_points_s: vec![0.0, 1.0],
            displacement_snapshots: indexed_fields("displacement"),
            rotation_snapshots: Vec::new(),
            velocity_snapshots: indexed_fields("velocity"),
            angular_velocity_snapshots: Vec::new(),
            acceleration_snapshots: indexed_fields("acceleration"),
            angular_acceleration_snapshots: Vec::new(),
            von_mises_snapshots: indexed_fields("von_mises"),
            kinetic_energy_snapshots: indexed_fields("kinetic_energy"),
            strain_energy_snapshots: indexed_fields("strain_energy"),
            residual_norm_snapshots: indexed_fields("residual_norm"),
            thermo_mechanical_temperature_snapshots: Vec::new(),
            thermo_mechanical_thermal_strain_snapshots: Vec::new(),
            thermo_mechanical_thermal_stress_snapshots: Vec::new(),
            thermo_mechanical_displacement_snapshots: Vec::new(),
            thermo_mechanical_von_mises_snapshots: Vec::new(),
            thermo_mechanical_coupling_residual_snapshots: Vec::new(),
            electro_thermal_temperature_snapshots: Vec::new(),
            electro_thermal_thermal_residual_snapshots: Vec::new(),
            residual_norms: vec![0.25],
            integration_method: crate::analysis::TransientIntegrationMethod::ImplicitEuler,
        }),
        nonlinear_results: None,
        electromagnetic_results: None,
        model_validity: crate::analysis::QualityGate::Pass,
        solver_convergence: crate::analysis::QualityGate::Pass,
        result_quality: crate::analysis::QualityGate::Pass,
        run_status: crate::analysis::RunStatus::Publishable,
        publishable: true,
        quality_reasons: Vec::new(),
        provenance: crate::analysis::RunProvenance {
            backend: ComputeBackend::Cpu,
            solver_backend: "synthetic".to_string(),
            solver_device_apply_k_ratio: 0.0,
            solver_host_sync_count: 0,
            precision_mode: "fp64".to_string(),
            deterministic_mode: true,
            solver_method: "synthetic".to_string(),
            preconditioner: "none".to_string(),
            quality_policy: "balanced".to_string(),
            fallback_events: Vec::new(),
        },
    };
    crate::analysis::storage::persist_run_result(&run).expect("indexed run should persist");
    run_id
}

#[test]
fn fea_results_translates_successful_selectors_and_public_indices_once() {
    let run_id = persist_synthetic_indexed_results();
    let selected = block_on(fea_results_builtin(vec![
        Value::String(run_id.clone()),
        Value::String("ModeIndices".to_string()),
        Value::Int(IntValue::U8(2)),
        Value::String("TransientSnapshotIndices".to_string()),
        Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).expect("double selector")),
    ]))
    .expect("one-based selectors should resolve the second stored entries");
    let Value::Object(selected) = selected else {
        panic!("results object");
    };
    let Some(Value::Struct(modal)) = selected.properties.get("modal_results") else {
        panic!("modal results");
    };
    let Some(Value::Tensor(eigenvalues)) = modal.fields.get("eigenvalues_hz") else {
        panic!("modal eigenvalues");
    };
    assert_eq!(eigenvalues.materialize_f64(), vec![20.0]);
    let Some(Value::Struct(transient)) = selected.properties.get("transient_results") else {
        panic!("transient results");
    };
    let Some(Value::Tensor(time_points)) = transient.fields.get("time_points_s") else {
        panic!("transient time points");
    };
    assert_eq!(time_points.materialize_f64(), vec![1.0]);

    let full =
        block_on(fea_results_builtin(vec![Value::String(run_id)])).expect("full indexed results");
    let Value::Object(full) = full else {
        panic!("results object");
    };
    let Some(Value::Struct(summary)) = full.properties.get("summary") else {
        panic!("results summary");
    };
    let Some(Value::Tensor(indices)) = summary.fields.get("available_mode_indices") else {
        panic!("available mode indices");
    };
    assert_eq!(
        indices
            .integer_storage()
            .expect("exact public indices")
            .exact_values(),
        vec![IntValue::U64(1), IntValue::U64(2)]
    );
}
