//! Authored exact-CAD process chain and cold completion replay, not restart qualification.
use runmat_execution::value::PublishedObjectSource;
use runmat_execution_artifact::{
    cache::{CacheExport, FilesystemObjectStore},
    object::ObjectInventoryLimits,
    ArtifactAccess, LogicalObject, ObjectNamespace,
};
use runmat_execution_runner_native::{
    execute_checked_fea_locally, magnetic_study_completion_from_source, CompletedFeaStudyResult,
    LocalFeaExecution, LocalFeaExecutionOptions,
};
use runmat_meshing_core::ElementOrder;
use runmat_runtime::analysis::FeaRootSelector;

const SOURCE: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region bulk : solid { select result; expect exactly_one }
    region exterior : face { select result; expect one_or_more }
}
mesh volume on geometry.part {
    tetrahedra { order = ORDER
        target_size = 0.5 m
        maximum_deviation = 0.02 m }
}
material vacuum {
    electrical {
        reference_temperature = 293.15 K
        conductivity = 0 A/V/m
        resistive_heating_coefficient = 0 K^-1
        relative_permittivity = 1
        relative_permeability = 1
    }
}
model coil {
    electromagnetic { equation = magnetostatics }
    assign material.vacuum to geometry.part.region.bulk
    constraint ground {
        magnetic.vector_potential_ground { target = geometry.part.region.exterior }
    }
    constraint gauge {
        magnetic.coulomb_gauge { target = geometry.part.region.bulk }
    }
    load current {
        magnetic_volume_current_density on geometry.part.region.bulk = [0, 0, 1] A/m^2
    }
    load_case drive { include load.current }
}
study response {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.coil
    analysis potential {
        electromagnetic static {}
        accuracy { algebraic_relative_tolerance = 1e-10 }
    }
    output flux = analysis.potential.field.integration_point.magnetic_flux_density
    output intensity = analysis.potential.field.integration_point.magnetic_field_intensity
}
"#;

pub(super) async fn run() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    let limits = ObjectInventoryLimits::default();
    for (order, name) in [
        (ElementOrder::Tet4, "linear"),
        (ElementOrder::Tet10, "quadratic"),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("magnetic.fea");
        std::fs::write(&path, SOURCE.replace("ORDER", name)).unwrap();
        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        let checked = session.load_and_check_fea_source_path(&path).await.unwrap();
        assert_eq!(
            checked.checked().check_result().status,
            runmat_runtime::analysis::FeaCheckStatus::Valid,
            "{:?}",
            checked.checked().check_result().diagnostics
        );
        let effects = Arc::new(AtomicUsize::new(0));
        let progress_effects = Arc::clone(&effects);
        let mesh_effects = Arc::clone(&effects);
        let options = LocalFeaExecutionOptions {
            root: Some(FeaRootSelector::Study {
                id: "response".into(),
            }),
            workers: Some(1),
            cpu_threads_per_worker: Some(
                runmat_analysis_fea::planning::resource::LocalThreadCount::new(1).unwrap(),
            ),
            progress_observer: Some(Arc::new(move |_| {
                progress_effects.fetch_add(1, Ordering::SeqCst);
            })),
            mesh_observer: Some(Arc::new(move |_| {
                mesh_effects.fetch_add(1, Ordering::SeqCst);
            })),
            ..Default::default()
        };
        // Run the real local mesh/solve root without any billing host or credential.
        let completed = execute_checked_fea_locally(&checked, options)
            .await
            .unwrap();
        assert!(
            effects.load(Ordering::SeqCst) > 0,
            "successful root must exercise the observers"
        );
        let LocalFeaExecution::Study(case) = completed else {
            panic!("expected standalone study");
        };
        let CompletedFeaStudyResult::Magnetic(result) = case.result else {
            panic!("magnetic domain was substituted");
        };
        assert!(result
            .mesh
            .topology
            .volume_elements
            .iter()
            .all(|element| element.order == order));
        let definitions =
            runmat_analysis_fea::execution::magnetic_native_field_definition_inventory().unwrap();
        assert_eq!(result.completion.fields.fields.len(), definitions.len());
        for definition in definitions {
            assert!(result
                .completion
                .fields
                .fields
                .iter()
                .any(|field| field.definition == definition));
        }
        assert!(!result.completion.observation_results.is_empty());
        assert!(result.completion.observation_catalog.is_some());
        assert!(!result.completion.states.is_empty());
        let response = result.completion.result.response();
        assert!(response.magnetic_energy_j > runmat_types::physical::Quantity::new(0.0));
        assert!(response.relative_physical_residual <= 1e-10);
        assert!(response.relative_gauge_residual <= 1e-10);
        assert!(response.relative_energy_work_error <= 1e-9);
        let exact_energy = reference_energy(255);
        let mu = 4e-7 * std::f64::consts::PI;
        let energy_tail = 8.0 * mu / (3.0 * std::f64::consts::PI.powi(4) * 255.0_f64.powi(3));
        assert!((exact_energy - reference_energy(511)).abs() <= energy_tail);
        assert!(energy_tail / exact_energy < 1e-6);
        let energy_error =
            (response.magnetic_energy_j.canonical_value() / exact_energy - 1.0).abs();
        assert!(
            energy_error < 0.9,
            "independent square-current energy error={energy_error}"
        );
        let mut squared_error = 0.0;
        let mut squared_reference = 0.0;
        let projection =
            runmat_analysis_fea::topology::AnalysisProjection::derive(&result.mesh).unwrap();
        for sample in &response.samples(&projection).unwrap() {
            let expected = reference_flux(sample.position_m[0], sample.position_m[1], 1023);
            for (actual, expected) in sample
                .flux_density_t
                .canonical_values()
                .iter()
                .zip(expected)
            {
                squared_error += sample.volume_weight_m3 * (actual - expected).powi(2);
                squared_reference += sample.volume_weight_m3 * expected.powi(2);
            }
        }
        let field_error = (squared_error / squared_reference).sqrt();
        assert!(
            field_error < 0.9,
            "independent square-current field error={field_error}"
        );
        eprintln!("authored magnetic {order:?}: energy_error={energy_error:e}, B_relative_L2={field_error:e}");
        let expected = result.completion.clone();
        let root = expected.root.clone();
        let access = ArtifactAccess {
            authorization_scope: root.authorization_scope.clone(),
            encryption_context: root.encryption_context,
        };
        let object_root = directory.path().join("retained");
        let mut retained =
            FilesystemObjectStore::open(&object_root, limits.max_object_bytes).unwrap();
        for reference in &expected.catalog.objects {
            let bytes = result.source.read_verified(reference).unwrap();
            let object = LogicalObject::new(
                ObjectNamespace::ResultValue,
                "magnetic-cold",
                "application/octet-stream",
                bytes,
            )
            .unwrap();
            assert_eq!(object.descriptor.digest, reference.logical_digest);
            retained.write_verified(&object).unwrap();
        }
        drop(retained);
        drop(result);
        drop(case.publication);
        drop(checked);
        drop(session);
        // Executing geometry/numerical pools and the original runtime source are gone.
        // Reopening authenticates the retained closure; it never calls a numerical solve.
        let cold = FilesystemObjectStore::open(&object_root, limits.max_object_bytes).unwrap();
        let reopened = runmat_analysis_execution::import_magnetic_study_root(
            &cold,
            &root,
            root.logical_digest,
            &access,
            limits,
        )
        .unwrap();
        let (actual, _) = magnetic_study_completion_from_source(reopened, cold).unwrap();
        assert_eq!(actual, expected);
        let mut wrong = expected.compiled.clone();
        wrong.model.sequence.steps[0].restart =
            runmat_analysis_core::RestartPolicy::ResumeCheckpoint {
                checkpoint_id: "unqualified".into(),
                required_state_digest: "unqualified".into(),
            };
        let policy = runmat_analysis_fea::study::MagneticStudyPlanningPolicy {};
        assert!(runmat_analysis_fea::study::plan_magnetic_study(&wrong, policy).is_err());
        for equation in [
            runmat_analysis_core::StaticElectricalEquation::Electrostatics,
            runmat_analysis_core::StaticElectricalEquation::SteadyConduction,
        ] {
            let mut wrong = expected.compiled.clone();
            wrong.model.electromagnetic =
                Some(runmat_analysis_core::ElectromagneticDomain::Static { equation });
            assert!(runmat_analysis_fea::study::plan_magnetic_study(&wrong, policy).is_err());
        }
    }
}

// Independent separated Poisson solution: -ΔAz=μ0 on the unit square, Az=0
// on its perimeter, constant along z. B=curl(A), not another RunMat solve.
// Uniform omitted-gradient bound <= 4μ0/(π²N); N=1023 makes reference
// truncation small relative to this deliberately coarse, non-qualification check.
fn reference_flux(x: f64, y: f64, maximum: usize) -> [f64; 3] {
    let mut dx = 0.0;
    let mut dy = 0.0;
    for n in (1..=maximum).step_by(2) {
        let k = n as f64 * std::f64::consts::PI;
        let left = (-k * x).exp();
        let right = (-k * (1.0 - x)).exp();
        let denominator = 1.0 + (-k).exp();
        dx -= 4.0 / k.powi(2) * (right - left) / denominator * (k * y).sin();
        dy += 4.0 / k.powi(2) * (1.0 - (left + right) / denominator) * (k * y).cos();
    }
    let mu = 4e-7 * std::f64::consts::PI;
    [mu * dy, -mu * dx, 0.0]
}

fn reference_energy(maximum: usize) -> f64 {
    let mut sum = 0.0;
    for m in (1..=maximum).step_by(2) {
        for n in (1..=maximum).step_by(2) {
            let mm = (m as f64).powi(2);
            let nn = (n as f64).powi(2);
            sum += 1.0 / (mm * nn * (mm + nn));
        }
    }
    32.0 * (4e-7 * std::f64::consts::PI) / std::f64::consts::PI.powi(6) * sum
}
