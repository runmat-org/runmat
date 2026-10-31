//! Authored CAD, actual native completed closure and cold replay.
//! Independent separated Fourier reference, not another RunMat solve.
#[path = "maxwell_authored/replay.rs"]
mod replay;
use runmat_execution::value::PublishedObjectSource;
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport, FilesystemObjectStore},
    object::ObjectInventoryLimits,
    ArtifactAccess, LogicalObject, ObjectNamespace,
};
use runmat_execution_runner_native::{
    execute_checked_fea_locally, maxwell_study_completion_from_source, CompletedFeaStudyResult,
    LocalFeaExecution, LocalFeaExecutionOptions,
};
use runmat_meshing_core::ElementOrder;
use runmat_runtime::analysis::FeaRootSelector;

const FREQUENCY: f64 = 159154.94309189534;
const SIGMA: f64 = 3.0;
const MU: f64 = 4e-7 * std::f64::consts::PI;
const EPSILON: f64 = 4.0 * 8.8541878128e-12;
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
material conductor {
    electrical {
        reference_temperature = 293.15 K
        conductivity = 3 A/V/m
        resistive_heating_coefficient = 0 K^-1
        relative_permittivity = 4
        relative_permeability = 1
    }
}
model field {
    electromagnetic { equation = harmonic_maxwell
        reference_frequency = 159154.94309189534 Hz }
    assign material.conductor to geometry.part.region.bulk
    constraint wall {
        electrical.perfect_conductor { target = geometry.part.region.exterior }
    }
    signal current : electric_current_density over frequency {
        samples = [{ at = 159154.94309189534 Hz,
            value = { real = [0 A/m^2, 0.6 A/m^2, 0 A/m^2],
                imaginary = [0 A/m^2, 0.8 A/m^2, 0 A/m^2] } }]
        interpolation = linear
        extrapolation = reject
        frame = global
    }
    excitation drive {
        magnetic_volume_current_density { target = geometry.part.region.bulk; signal = signal.current }
    }
    load_case driven { include excitation.drive }
}
study response {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.field
    analysis phasor {
        electromagnetic harmonic { frequencies = [159154.94309189534 Hz] }
        accuracy { algebraic_relative_tolerance = 1e-10 }
    }
    output electric = analysis.phasor.field.integration_point.electric_field
    output flux = analysis.phasor.field.integration_point.magnetic_flux_density
}
"#;

pub(super) async fn run() {
    let limits = ObjectInventoryLimits::default();
    let exact_power = reference_power(255);
    let truncation = (exact_power - reference_power(511)).abs();
    assert!(
        truncation / exact_power < 1e-7,
        "Fourier power truncation not resolved"
    );
    for (order, name) in [
        (ElementOrder::Tet4, "linear"),
        (ElementOrder::Tet10, "quadratic"),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("maxwell.fea");
        std::fs::write(&path, SOURCE.replace("ORDER", name)).unwrap();
        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        let checked = session.load_and_check_fea_source_path(&path).await.unwrap();
        assert_eq!(
            checked.checked().check_result().status,
            runmat_runtime::analysis::FeaCheckStatus::Valid,
            "{:?}",
            checked.checked().check_result().diagnostics
        );
        let completed = execute_checked_fea_locally(
            &checked,
            LocalFeaExecutionOptions {
                root: Some(FeaRootSelector::Study {
                    id: "response".into(),
                }),
                workers: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let LocalFeaExecution::Study(case) = completed else {
            panic!("expected standalone study")
        };
        let CompletedFeaStudyResult::Maxwell(result) = case.result else {
            panic!("Maxwell domain was substituted")
        };
        assert!(result
            .mesh
            .topology
            .volume_elements
            .iter()
            .all(|element| element.order == order));
        assert_eq!(
            result.completion.fields.fields.len(),
            runmat_analysis_fea::execution::MAXWELL_NATIVE_FIELD_COUNT
        );
        assert!(result.completion.observation_catalog.is_some());
        assert!(!result.completion.observation_results.is_empty());
        assert!(!result.completion.states.is_empty());
        let response = result.completion.result.response();
        assert_eq!(
            response.angular_frequency_rad_per_s,
            runmat_types::physical::Quantity::new(FREQUENCY * std::f64::consts::TAU)
        );
        assert!(response.relative_physical_residual <= 1e-10);
        assert!(response.relative_complex_power_error <= 1e-9);
        assert!(response.relative_charge_continuity_residual <= 1e-9);
        let power_error = (response.ohmic_power_w.canonical_value() / exact_power - 1.0).abs();
        assert!(
            power_error < 0.9,
            "independent Fourier ohmic-power error={power_error}"
        );
        let mut error = [0.0; 2];
        let mut norm = [0.0; 2];
        let mut tail = [0.0; 2];
        for sample in &response.samples {
            let exact = reference_fields(sample.position_m[0], sample.position_m[2], 255);
            let coarse = reference_fields(sample.position_m[0], sample.position_m[2], 127);
            for (field, actual) in [
                sample.electric_field_v_per_m.canonical_value(),
                sample.flux_density_t.canonical_value(),
            ]
            .iter()
            .enumerate()
            {
                for (axis, value) in actual.iter().enumerate() {
                    let actual = [value.real, value.imaginary];
                    for (phase, value) in actual.iter().enumerate() {
                        let expected = exact[field][axis][phase];
                        error[field] += sample.volume_weight_m3 * (value - expected).powi(2);
                        norm[field] += sample.volume_weight_m3 * expected.powi(2);
                        tail[field] += sample.volume_weight_m3
                            * (coarse[field][axis][phase] - expected).powi(2);
                    }
                }
            }
        }
        for field in 0..2 {
            let relative = (error[field] / norm[field]).sqrt();
            let truncation = (tail[field] / norm[field]).sqrt();
            assert!(
                truncation < 0.01,
                "Fourier field {field} truncation={truncation}"
            );
            assert!(
                relative < 0.95,
                "independent Fourier field {field} relative L2={relative}"
            );
            eprintln!("authored Maxwell {order:?}: field={field}, relative L2={relative:e}, truncation={truncation:e}, power_error={power_error:e}");
        }
        let expected = result.completion.clone();
        let mesh = result.mesh.clone();
        let meshing_evidence = result.meshing_evidence.clone();
        let root = expected.root.clone();
        let access = ArtifactAccess {
            authorization_scope: root.authorization_scope.clone(),
            encryption_context: root.encryption_context,
        };
        let object_root = directory.path().join("retained");
        let mut retained =
            FilesystemObjectStore::open(&object_root, limits.max_object_bytes).unwrap();
        for reference in &expected.catalog.objects {
            let object = LogicalObject::new(
                ObjectNamespace::ResultValue,
                "maxwell-cold",
                "application/octet-stream",
                result.source.read_verified(reference).unwrap(),
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
        // Every original executing pool/source is gone. Cold import authenticates original
        // complex phases and full closure; it does not re-solve or reconstruct missing fields.
        let cold = FilesystemObjectStore::open(&object_root, limits.max_object_bytes).unwrap();
        assert!(runmat_analysis_execution::import_maxwell_study_root(
            &cold,
            &root,
            runmat_execution::Digest::sha256(b"foreign-trusted-root"),
            &access,
            limits,
        )
        .is_err());
        let missing = expected
            .catalog
            .objects
            .iter()
            .find(|object| object.logical_digest != root.logical_digest)
            .unwrap();
        let filtered = MissingObject {
            source: &cold,
            missing: missing.logical_digest,
        };
        assert!(runmat_analysis_execution::import_maxwell_study_root(
            &filtered,
            &root,
            root.logical_digest,
            &access,
            limits,
        )
        .is_err());
        let reopened = runmat_analysis_execution::import_maxwell_study_root(
            &cold,
            &root,
            root.logical_digest,
            &access,
            limits,
        )
        .unwrap();
        replay::check(&reopened, &cold);
        let (actual, source) = maxwell_study_completion_from_source(reopened, cold).unwrap();
        assert_eq!(actual, expected);
        let options = runmat_runtime::analysis::AnalysisMaxwellRunOptions {
            precision_mode: runmat_runtime::analysis::PrecisionMode::Fp64,
            residual_target: actual.workload.relative_residual_target,
        };
        runmat_runtime::analysis::publish_maxwell_study_result(
            &actual,
            &source,
            &mesh,
            &meshing_evidence,
            &options,
            actual.compiled.definition_digest().unwrap(),
            runmat_runtime::operations::OperationContext::new(None, None),
            runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                "cold-maxwell:{}",
                actual.root.logical_digest
            ))
            .unwrap(),
        )
        .unwrap();
        let policy = runmat_analysis_fea::study::MaxwellStudyPlanningPolicy {};
        let mut wrong = actual.compiled.clone();
        wrong.model.sequence.steps[0].restart =
            runmat_analysis_core::RestartPolicy::ResumeCheckpoint {
                checkpoint_id: "unqualified".into(),
                required_state_digest: "unqualified".into(),
            };
        assert!(runmat_analysis_fea::study::plan_maxwell_study(&wrong, policy).is_err());
        let mut sweep = actual.compiled.clone();
        sweep.model.sequence.steps[0].independent_domain =
            runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![
                FREQUENCY,
                2.0 * FREQUENCY,
            ]);
        let planned = runmat_analysis_fea::study::plan_maxwell_study(&sweep, policy).unwrap();
        assert_eq!(planned.model, sweep.model);
        assert_eq!(
            planned.frequency,
            runmat_analysis_core::Frequency::new(FREQUENCY)
        );
        for invalid_frequency in [0.0, -FREQUENCY, FREQUENCY, f64::INFINITY, f64::NAN] {
            let mut wrong = sweep.clone();
            wrong.model.sequence.steps[0].independent_domain =
                runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![
                    FREQUENCY,
                    invalid_frequency,
                ]);
            assert!(runmat_analysis_fea::study::plan_maxwell_study(&wrong, policy).is_err());
        }
    }
}

struct MissingObject<'a> {
    source: &'a FilesystemObjectStore,
    missing: runmat_execution::Digest,
}
impl CacheImport for MissingObject<'_> {
    fn read_verified(
        &self,
        digest: runmat_execution::Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        if digest == self.missing {
            Ok(None)
        } else {
            self.source.read_verified(digest)
        }
    }
}

// Peak exp(+iωt): E=(0,Ey(x,z),0). PEC gives Ey=0 on x/z walls;
// Ey is normal (unconstrained) on y walls. Uniform Jy=(0.6+0.8i) A/m².
// Coefficients solve [π²(m²+n²)/μ - ω²ε + iωσ] c_mn = -iω 16Jy/(mnπ²).
// Scalar pair arithmetic is local and independent of production phasor/assembly code.
fn coefficient(m: usize, n: usize) -> [f64; 2] {
    let omega = FREQUENCY * std::f64::consts::TAU;
    let pi = std::f64::consts::PI;
    let scale = 16.0 * omega / (m as f64 * n as f64 * pi.powi(2));
    let a = pi.powi(2) * (m * m + n * n) as f64 / MU - omega.powi(2) * EPSILON;
    let b = omega * SIGMA;
    let denominator = a * a + b * b;
    [
        (0.8 * scale * a - 0.6 * scale * b) / denominator,
        (-0.6 * scale * a - 0.8 * scale * b) / denominator,
    ]
}

fn reference_fields(x: f64, z: f64, maximum: usize) -> [[[f64; 2]; 3]; 2] {
    let mut out = [[[0.0; 2]; 3]; 2];
    let omega = FREQUENCY * std::f64::consts::TAU;
    let pi = std::f64::consts::PI;
    for m in (1..=maximum).step_by(2) {
        let mx = m as f64 * pi;
        for n in (1..=maximum).step_by(2) {
            let nz = n as f64 * pi;
            let c = coefficient(m, n);
            let e = (mx * x).sin() * (nz * z).sin();
            let dx = mx * (mx * x).cos() * (nz * z).sin() / omega;
            let dz = nz * (mx * x).sin() * (nz * z).cos() / omega;
            for (phase, value) in c.iter().enumerate() {
                out[0][1][phase] += value * e;
            }
            // B=i curlE/ω=(-i ∂zEy,0,+i ∂xEy)/ω.
            out[1][0][0] += c[1] * dz;
            out[1][0][1] -= c[0] * dz;
            out[1][2][0] -= c[1] * dx;
            out[1][2][1] += c[0] * dx;
        }
    }
    out
}

fn reference_power(maximum: usize) -> f64 {
    let mut sum = 0.0;
    for m in (1..=maximum).step_by(2) {
        for n in (1..=maximum).step_by(2) {
            let c = coefficient(m, n);
            sum += c[0] * c[0] + c[1] * c[1];
        }
    }
    // Orthogonality ∫sin²(mx)sin²(nz)=1/4 and average peak Joule heat σ|E|²/2.
    SIGMA * sum / 8.0
}
