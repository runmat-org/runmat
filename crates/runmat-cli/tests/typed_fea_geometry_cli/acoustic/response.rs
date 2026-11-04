//! Cavity modes and pressure transients run through the authored CLI and publish fields.
use super::super::*;

fn project(study: &str) -> TempDir {
    let project = TempDir::new().unwrap();
    fs::create_dir(project.path().join("src")).unwrap();
    fs::write(project.path().join("runmat.toml"),
        "[package]\nname = \"typed-acoustic-response\"\n[sources]\nroots = [\"src\"]\n[runtime.fea]\nartifact_root = \"artifacts\"\nstudy_artifact_root = \"study-artifacts\"\n").unwrap();
    fs::write(
        project.path().join("src/makePart.m"),
        "function part = makePart()\npart = geometry.finish(geometry.box(1, 1, 1));\nend\n",
    )
    .unwrap();
    fs::write(project.path().join("src/main.fea"), study).unwrap();
    project
}

fn run(project: &TempDir, study: &str) -> (Value, AnalysisRunResultView) {
    let output = runmat(
        project,
        &["run", "src/main.fea", "--study", study, "--json"],
    );
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    let run_id = summary["step_run_ids"][0].as_str().unwrap().to_string();
    let store = FilesystemAnalysisArtifactStore::new(project.path().join("artifacts"));
    let run = store
        .load_run(&run_id)
        .unwrap()
        .expect("standard run record");
    (summary, AnalysisRunResultView(run))
}

struct AnalysisRunResultView(runmat_runtime::analysis::AnalysisRunResult);
impl AnalysisRunResultView {
    fn field(&self, id: &str) -> Vec<&runmat_analysis_core::AnalysisField> {
        self.0
            .run
            .fields
            .iter()
            .filter(|f| f.field_id == id)
            .collect()
    }

    fn unique_field(&self, id: &str) -> &runmat_analysis_core::AnalysisField {
        let fields = self.field(id);
        assert_eq!(fields.len(), 1, "exactly one public field {id}");
        fields[0]
    }

    fn assert_inventory(&self, expected: impl IntoIterator<Item = String>) {
        let mut expected: Vec<_> = expected.into_iter().collect();
        let mut actual: Vec<_> = self
            .0
            .run
            .fields
            .iter()
            .map(|f| f.field_id.clone())
            .collect();
        expected.sort();
        actual.sort();
        assert_eq!(actual, expected, "complete public field inventory");
    }
}

// witness: FEA-ACOUSTIC-MODAL-001/rigid_box_modes @ acoustic_modal/tet10/public_study; coverage:
//   partial;
//   boundary: A rigid unit cube on a quadratic mesh run through the CLI returns a zero-frequency
//   uniform mode and three axial modes within 2 percent of c/2L = 171.5 Hz; the first ten modes
//   within 0.5 percent are not asserted.
// witness: FEA-ACOUSTIC-MODAL-001/modal_publication @ acoustic_modal/tet10/public_study; coverage:
//   partial;
//   boundary: The modal run publishes modal.frequency_hz and five acoustic.pressure mode fields and
//   is numerically acceptable; residuals and normalization are not asserted; a cabin-sized model is
//   not run.
#[test]
fn rigid_cube_cavity_modes_match_rectangular_room_frequencies() {
    let project = project(&study("acoustic modal { modes = 5 }", "").replace(
        "    constraint drive {\n        acoustic.pressure { target = geometry.part.region.surface; value = signal.pulse }\n    }\n",
        "    load walls {\n        acoustic.rigid_wall { target = geometry.part.region.surface }\n    }\n",
    ));
    let (summary, run) = run(&project, "response");
    assert_eq!(summary["procedure"], "acoustic_modal");
    let frequencies = run.field("modal.frequency_hz");
    let values = frequencies[0].as_host_f64().unwrap();
    // c / 2L = 171.5 Hz for the three axial modes of the unit cube; 0 Hz uniform mode.
    assert!(values[0].abs() < 1e-3, "{values:?}");
    for value in &values[1..4] {
        assert!((value - 171.5).abs() / 171.5 < 0.02, "{values:?}");
    }
    run.assert_inventory(
        (1..=5)
            .map(|mode| format!("acoustic.mode_shape.{mode}"))
            .chain(["modal.frequency_hz".into(), "modal.eigenvalue".into()]),
    );
    for mode in 1..=5 {
        let pressure = run.unique_field(&format!("acoustic.mode_shape.{mode}"));
        assert!(!pressure.as_host_f64().unwrap().is_empty());
        assert!(pressure
            .as_host_f64()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }
    assert!(run.0.numerically_acceptable);
}

// witness: FEA-ACOUSTIC-TRANSIENT-001/pressure_time_history_publication @
//   acoustic_transient/tet10/public_study; coverage: partial;
//   boundary: A quadratic-mesh pulse study run through the CLI publishes 21 native pressure
//   snapshots and the complete authored history on a 21-point time axis ending at 2 ms, all
//   finite, with the prescribed surface carrying the pulse peak at 0.5 ms. The run is reopened
//   from public storage before these assertions.
#[test]
fn pressure_transient_publishes_its_complete_time_history() {
    let project = project(&study(
        "acoustic transient { start_time = 0 s; end_time = 2 ms; time_step = 0.1 ms }",
        r#"
    output pressure_history {
        value = sample(
            analysis.sound.field.node.acoustic_pressure,
            over = time,
            at = [0 ms, 0.1 ms, 0.2 ms, 0.3 ms, 0.4 ms, 0.5 ms, 0.6 ms,
                  0.7 ms, 0.8 ms, 0.9 ms, 1 ms, 1.1 ms, 1.2 ms, 1.3 ms,
                  1.4 ms, 1.5 ms, 1.6 ms, 1.7 ms, 1.8 ms, 1.9 ms, 2 ms],
        )
        retention = explicit_samples
        maximum_samples = 21
    }
"#,
    ));
    let (summary, run) = run(&project, "response");
    assert_eq!(summary["procedure"], "acoustic_transient");
    run.assert_inventory(
        (1..=21)
            .map(|index| format!("acoustic.pressure.time.{index}"))
            .chain([
                "acoustic.time_s".into(),
                "acoustic.energy_j".into(),
                "acoustic.boundary_absorbed_energy_j".into(),
            ]),
    );
    let snapshots: Vec<_> = (1..=21)
        .map(|index| run.unique_field(&format!("acoustic.pressure.time.{index}")))
        .collect();
    assert_eq!(snapshots.len(), 21);
    let times = run.field("acoustic.time_s")[0].as_host_f64().unwrap();
    assert_eq!(times.len(), 21);
    assert!((times[20] - 2e-3).abs() < 1e-12);
    // The prescribed surface carries the exact pulse peak at 0.5 ms; the interior responds.
    let at_peak = snapshots[5].as_host_f64().unwrap();
    let peak = at_peak.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!((peak - 1.0).abs() < 1e-9 || peak > 1.0, "{peak}");
    assert!(snapshots.iter().all(|field| field
        .as_host_f64()
        .unwrap()
        .iter()
        .all(|v| v.is_finite())));
    assert!(at_peak.iter().any(|v| (v - 1.0).abs() < 1e-9));
    let [history] = run.0.run.observation_histories.as_slice() else {
        panic!("one cold-authored acoustic pressure history");
    };
    assert_eq!(history.manifest().observation_id.0, "pressure_history");
    assert_eq!(history.manifest().samples.len(), 21);
    assert_eq!(
        history.manifest().samples.first().unwrap().coordinate,
        runmat_analysis_fea::results::field::RecoveredFieldSample::TimeSeconds(0.0)
    );
    assert_eq!(
        history.manifest().samples.last().unwrap().coordinate,
        runmat_analysis_fea::results::field::RecoveredFieldSample::TimeSeconds(2e-3)
    );
}

fn study(analysis: &str, outputs: &str) -> String {
    format!(
        "{MODEL}\nstudy response {{\n    geometry = geometry.part\n    mesh = mesh.volume\n    model = model.cavity\n    analysis sound {{ {analysis} }}\n{outputs}\n}}\n"
    )
}

const MODEL: &str = r#"
import function makePart from "./makePart.m"
geometry part {
    generated by function.makePart { arguments {} }
    region body : solid {
        select result
        expect exactly_one
    }
    region surface : face {
        select result
        expect one_or_more
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = quadratic
        target_size = 0.25 m
        maximum_deviation = 0.02 m
    }
}
material fluid {
    mass_density = 1.2 kg/m^3
    acoustic {
        speed_of_sound = 343 m/s
        damping_ratio = 0.0
    }
}
model cavity {
    assign material.fluid to geometry.part.region.body
    signal pulse : pressure over time {
        samples = [
            { at = 0 s, value = 0 Pa },
            { at = 0.5 ms, value = 1 Pa },
            { at = 1 ms, value = 0 Pa },
            { at = 10 ms, value = 0 Pa },
        ]
        interpolation = linear
        extrapolation = reject
    }
    constraint drive {
        acoustic.pressure { target = geometry.part.region.surface; value = signal.pulse }
    }
}
"#;
