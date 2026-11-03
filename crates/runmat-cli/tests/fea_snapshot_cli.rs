#![cfg(feature = "occt-native")]
//! `runmat snapshot` per physics family and per analysis step. Every study runs on the
//! documentation box (`docs/simulation/examples/geometry/box.brep`, two named faces) with a
//! coarse mesh, through the public command, and the captured views are checked against the
//! fields the run published: their typed layout (location, components), the step captured,
//! and the field's true extrema.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use tempfile::TempDir;

const GEOMETRY: &str = r#"
geometry part {
    file = "./box.brep"
    units = m
    region body : solid { select id "persistent-entity:sha256:eb1929ea502ba35824eceb7a1aa23719ab7fa023c89662b8de06cc458560a3f0"; expect exactly_one }
    region support : face { select id "persistent-entity:sha256:8af3f3e08304762c1332b213d0ecd818feb256a2d233fd3e99e8978e41ae7a23"; expect exactly_one }
    region hot : face { select id "persistent-entity:sha256:db5e0a2817b1cfb20a910953eda8e66c0ed46aa95c61e200de19fec0ab781129"; expect exactly_one }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 10 m
        maximum_deviation = 0.1 m
    }
}
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
    thermal {
        reference_temperature = 293.15 K
        modulus_temperature_coefficient = 0 K^-1
        conductivity = 45 W/m/K
        specific_heat = 500 J/kg/K
        expansion_coefficient = 0.000012 K^-1
    }
}
material air {
    mass_density = 1.2 kg/m^3
    acoustic {
        speed_of_sound = 343 m/s
        damping_ratio = 0.0
    }
}
"#;

const STRUCTURAL_STATIC: &str = r#"
model loaded {
    assign material.steel to geometry.part.region.body
    constraint base { structural.fixed { target = geometry.part.region.support } }
    load push { traction on geometry.part.region.hot = [0, 0, -100000] Pa }
    load_case service { include load.push }
}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.loaded
    analysis static { structural static {} }
}
"#;

const THERMOMECHANICAL_STEADY: &str = r#"
model heated {
    thermal { reference_temperature = 293.15 K }
    thermomechanical { reference_temperature = 293.15 K }
    assign material.steel to geometry.part.region.body
    constraint base { structural.fixed { target = geometry.part.region.support } }
    load operating_temperature {
        temperature_field on geometry.part.region.body = 373.15 K
    }
    load_case operating { include load.operating_temperature }
}
study heated_static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.heated
    analysis coupled { thermomechanical steady {} }
}
"#;

const ACOUSTIC_MODAL: &str = r#"
model cavity {
    assign material.air to geometry.part.region.body
    load walls { acoustic.rigid_wall { target = geometry.part.region.support } }
}
study modes {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.cavity
    analysis cavity { acoustic modal { modes = 3 } }
}
"#;

const STRUCTURAL_MODAL: &str = r#"
model supported {
    assign material.steel to geometry.part.region.body
    constraint base { structural.fixed { target = geometry.part.region.support } }
}
study modes {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.supported
    analysis modes { structural modal { modes = 2 } retain basis }
}
"#;

/// Three analysis steps: thermal steady, thermomechanical preload, prestressed modal.
const HEATED_MODES: &str = r#"
model structure {
    thermal { reference_temperature = 293.15 K }
    thermomechanical { reference_temperature = 293.15 K }
    assign material.steel to geometry.part.region.body
    constraint base { structural.fixed { target = geometry.part.region.support } }
    constraint hot {
        thermal.temperature { target = geometry.part.region.hot; value = 303.15 K }
    }
}
study heated_modes {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis heat { thermal steady {} retain heated }
    analysis preload after analysis.heat {
        thermomechanical preload { temperature = analysis.heat.state.heated }
    }
    analysis modes after analysis.preload {
        structural prestressed_modal {
            prestress = analysis.preload.state.geometric_stiffness
            modes = 1
        }
    }
}
"#;

fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/simulation/examples")
}

fn project(study: &str) -> TempDir {
    let project = TempDir::new().unwrap();
    fs::copy(
        examples().join("geometry/box.brep"),
        project.path().join("box.brep"),
    )
    .unwrap();
    fs::write(
        project.path().join("main.fea"),
        format!("{GEOMETRY}{study}"),
    )
    .unwrap();
    project
}

/// Run `runmat snapshot main.fea` with `extra` arguments and return the snapshot.
fn snapshot(project: &TempDir, extra: &[&str]) -> Value {
    snapshot_to(project, "out", extra)
}

/// The same, writing the snapshot into the project directory `directory`.
fn snapshot_to(project: &TempDir, directory: &str, extra: &[&str]) -> Value {
    let out = project.path().join(directory).join("snapshot.json");
    let mut args = vec!["snapshot", "main.fea", "--out"];
    let out_arg = out.to_string_lossy().to_string();
    args.push(&out_arg);
    args.extend_from_slice(extra);
    let output = Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(project.path())
        .args(&args)
        .env_remove("RUNMAT_CONFIG")
        .env(
            "RUNMAT_PACKAGE_CACHE_DIR",
            project.path().join("package-cache"),
        )
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            project.path().join("execution-state"),
        )
        .env("RUNMAT_FEA_ARTIFACT_ROOT", project.path().join("artifacts"))
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            project.path().join("study-artifacts"),
        )
        .output()
        .expect("run snapshot");
    assert!(
        output.status.success(),
        "snapshot {extra:?} failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&fs::read(&out).unwrap()).unwrap()
}

fn view<'a>(snapshot: &'a Value, id: &str) -> &'a Value {
    snapshot["views"]
        .as_array()
        .unwrap()
        .iter()
        .find(|view| view["id"] == id)
        .unwrap_or_else(|| panic!("no view {id}: {}", view_ids(snapshot).join(", ")))
}

fn view_ids(snapshot: &Value) -> Vec<String> {
    snapshot["views"]
        .as_array()
        .unwrap()
        .iter()
        .map(|view| view["id"].as_str().unwrap().to_string())
        .collect()
}

/// The published descriptor of `field_id` in the snapshot's primary run.
fn descriptor<'a>(snapshot: &'a Value, field_id: &str) -> &'a Value {
    snapshot["results"]["field_descriptors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|descriptor| descriptor["field_id"] == field_id)
        .unwrap_or_else(|| panic!("no descriptor {field_id}"))
}

/// Each view's scene exists beside the snapshot, inside the 16 MiB scene budget, and its
/// legend range is the field's true extrema.
fn check_scenes(project: &TempDir, snapshot: &Value) {
    for view in snapshot["views"].as_array().unwrap() {
        let scene = project
            .path()
            .join("out")
            .join(view["scene"].as_str().unwrap());
        let bytes = fs::metadata(&scene)
            .unwrap_or_else(|_| panic!("missing scene {}", scene.display()))
            .len();
        assert!(
            bytes > 0 && bytes < 16 * 1024 * 1024,
            "{scene:?}: {bytes} bytes"
        );
        if let Some(extrema) = view.get("extrema") {
            assert_eq!(
                view["minimum"], extrema["minimum"]["value"],
                "{}",
                view["id"]
            );
            assert_eq!(
                view["maximum"], extrema["maximum"]["value"],
                "{}",
                view["id"]
            );
        }
    }
}

#[test]
fn structural_static_snapshot_shows_displacement_and_stress() {
    let project = project(STRUCTURAL_STATIC);
    let snapshot = snapshot(&project, &[]);
    let displacement = view(&snapshot, "displacement");
    let stress = view(&snapshot, "stress");
    assert_eq!(displacement["field"], "structural.displacement");
    let layout = &descriptor(&snapshot, "structural.displacement")["layout"];
    assert_eq!(layout["location"], "node");
    assert_eq!(layout["value_shape"], "vector3");
    assert!(stress["maximum"].as_f64().unwrap() > 0.0);
    check_scenes(&project, &snapshot);
}

#[test]
fn thermomechanical_steady_snapshot_draws_element_temperature() {
    let project = project(THERMOMECHANICAL_STEADY);
    let snapshot = snapshot(&project, &[]);
    let temperature = view(&snapshot, "temperature");
    let field = temperature["field"].as_str().unwrap();
    let published = descriptor(&snapshot, field);
    assert_eq!(published["layout"]["quantity"], "temperature");
    assert_eq!(published["layout"]["location"], "element");
    assert_eq!(published["location"], "element");
    assert_eq!(published["component_count"], 1);
    for bound in ["minimum", "maximum"] {
        let value = temperature[bound].as_f64().unwrap();
        assert!((value - 373.15).abs() < 1e-9, "{bound} {value}");
    }
    view(&snapshot, "stress");
    check_scenes(&project, &snapshot);
}

#[test]
fn acoustic_modal_snapshot_draws_scalar_mode_shapes() {
    let project = project(ACOUSTIC_MODAL);
    let snapshot = snapshot(&project, &[]);
    let mode = view(&snapshot, "mode-2");
    let field = mode["field"].as_str().unwrap();
    let published = descriptor(&snapshot, field);
    assert_eq!(published["layout"]["quantity"], "acoustic_pressure");
    assert_eq!(published["layout"]["value_shape"], "scalar");
    assert_eq!(published["layout"]["ordinal"]["kind"], "mode");
    assert_eq!(published["component_count"], 1);
    assert_eq!(mode["unit"], "relative");
    assert!(mode.get("deformation_field").is_none());
    assert!(mode["label"].as_str().unwrap().contains("Hz"));
    check_scenes(&project, &snapshot);
}

#[test]
fn multi_step_snapshot_captures_the_final_step_or_the_named_step() {
    let project = project(HEATED_MODES);
    let final_step = snapshot(&project, &[]);
    assert_eq!(final_step["step"], "modes");
    view(&final_step, "mode-1");
    check_scenes(&project, &final_step);

    let heat = snapshot(&project, &["--step", "heat"]);
    assert_eq!(heat["step"], "heat");
    let temperature = view(&heat, "temperature");
    assert_ne!(heat["run_id"], final_step["run_id"]);
    let maximum = temperature["maximum"].as_f64().unwrap();
    assert!((maximum - 303.15).abs() < 1e-6, "{maximum}");
    // The interior: a section colored by the nodal temperature evaluated with the linear
    // element basis, so the drawn colors are exact (zero display error).
    let section = view(&heat, "temperature-section");
    assert_eq!(section["volume"]["kind"], "section");
    let error = &section["volume"]["error"];
    assert!(error["count"].as_u64().unwrap() > 0, "{error}");
    assert!(error["display_error"].as_f64().unwrap() < 1e-9, "{error}");
    assert_eq!(error["geometric_error_m"], 0.0);
    assert_eq!(section["maximum"], section["extrema"]["maximum"]["value"]);
    check_scenes(&project, &heat);

    let missing = Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(project.path())
        .args([
            "snapshot",
            "main.fea",
            "--out",
            "out/missing.json",
            "--step",
            "nope",
        ])
        .env("RUNMAT_FEA_ARTIFACT_ROOT", project.path().join("artifacts"))
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            project.path().join("study-artifacts"),
        )
        .output()
        .unwrap();
    assert!(!missing.status.success());
}

/// A modal step shows its geometry and one view per mode; the mass-normalized shapes never
/// appear as physical displacement or stress. The metadata carries descriptors, not values,
/// so it stays inside the project file limit and verifies cold; `--view` keeps named views.
#[test]
fn structural_modal_snapshot_shows_only_mode_shapes_and_verifies() {
    let project = project(STRUCTURAL_MODAL);
    let snapshot = snapshot(&project, &[]);
    assert_eq!(view_ids(&snapshot), ["geometry", "mode-1", "mode-2"]);
    for id in ["mode-1", "mode-2"] {
        let mode = view(&snapshot, id);
        assert_eq!(mode["unit"], "relative");
        assert!(mode["label"].as_str().unwrap().contains("Hz"));
        assert!(mode.get("extrema").is_none());
    }
    assert_eq!(snapshot["results"]["fields"], serde_json::json!([]));
    assert!(snapshot["results"]["modal_results"].is_null());
    let metadata = fs::metadata(project.path().join("out/snapshot.json"))
        .unwrap()
        .len();
    assert!(metadata < 1024 * 1024, "{metadata} bytes of metadata");
    check_scenes(&project, &snapshot);

    let verified = Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(project.path())
        .args([
            "fea",
            "verify-snapshot",
            "out/snapshot.json",
            "--out",
            "out/verify.json",
        ])
        .env_remove("RUNMAT_CONFIG")
        .env("RUNMAT_FEA_ARTIFACT_ROOT", project.path().join("artifacts"))
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            project.path().join("study-artifacts"),
        )
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "verify-snapshot failed: {}",
        String::from_utf8_lossy(&verified.stderr)
    );

    let selected = snapshot_to(&project, "out-selected", &["--view", "mode-2"]);
    assert_eq!(view_ids(&selected), ["mode-2"]);
    let refused = Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(project.path())
        .args([
            "snapshot",
            "main.fea",
            "--out",
            "out/refused.json",
            "--view",
            "stress",
        ])
        .env_remove("RUNMAT_CONFIG")
        .env("RUNMAT_FEA_ARTIFACT_ROOT", project.path().join("artifacts"))
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            project.path().join("study-artifacts"),
        )
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("no view 'stress'"));
}
