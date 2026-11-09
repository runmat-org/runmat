use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use tempfile::TempDir;

use runmat_analysis_core::{AnalysisModel, ExcitationTarget, PropertyRegionScope};
use runmat_meshing_core::{CanonicalMeshingContract, MeshingEvidence, SolverMeshArtifact};

#[path = "support/local_offline.rs"]
mod local_offline;
#[path = "support/fea_admission.rs"]
mod fea_admission;

fn runmat(temp: &TempDir, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    local_offline::configure(&mut command, temp.path());
    command.current_dir(temp.path()).args(args);
    command
        .env_remove("RUNMAT_CONFIG")
        .env(
            "RUNMAT_PACKAGE_CACHE_DIR",
            temp.path().join("package-cache"),
        )
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            temp.path().join("execution-state"),
        );
    let _admission = fea_admission::AdmissionServer::for_command(&mut command, temp.path());
    command.output().expect("run runmat check")
}

fn runmat_with_env(temp: &TempDir, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    local_offline::configure(&mut command, temp.path());
    command.current_dir(temp.path()).args(args);
    command
        .env_remove("RUNMAT_CONFIG")
        .env_remove("NO_COLOR")
        .env_remove("CLICOLOR")
        .env_remove("CLICOLOR_FORCE")
        .env_remove("FORCE_COLOR")
        .env_remove("TERM");
    for (key, value) in env {
        command.env(key, value);
    }
    let _admission = fea_admission::AdmissionServer::for_command(&mut command, temp.path());
    command.output().expect("run runmat check")
}

fn repository_fixture(temp: &TempDir) -> PathBuf {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = temp.path().join("repository");
    let destination = fixture.join("docs/simulation/examples");
    if !destination.exists() {
        fn copy_sources(source: &std::path::Path, destination: &std::path::Path) {
            fs::create_dir_all(destination).unwrap();
            for entry in fs::read_dir(source).unwrap() {
                let entry = entry.unwrap();
                // Saved selections/caches belong to their current operator/CAD revision,
                // not this independent documentation fixture. Never mutate the originals.
                if entry.file_name().to_string_lossy().starts_with('.') {
                    continue;
                }
                if entry.file_type().unwrap().is_dir() {
                    copy_sources(&entry.path(), &destination.join(entry.file_name()));
                } else if entry.file_type().unwrap().is_file() {
                    fs::copy(entry.path(), destination.join(entry.file_name())).unwrap();
                }
            }
        }
        copy_sources(&repository.join("docs/simulation/examples"), &destination);
    }
    fixture
}

fn runmat_from_repository(temp: &TempDir, args: &[&str]) -> Output {
    runmat_from_repository_through(temp, args, None).0
}

/// Run from the repository against the test Server, optionally closing the Server's port
/// mid-run. Returns the output, each lease route and status, the receipts and the outage.
fn runmat_from_repository_through(
    temp: &TempDir,
    args: &[&str],
    outage: Option<fea_admission::Outage>,
) -> (
    Output,
    Vec<(String, &'static str)>,
    Vec<Value>,
    Option<std::time::Duration>,
) {
    let repository = repository_fixture(temp);
    let config = temp.path().join("runmat.toml");
    let artifact_root =
        toml::Value::String(temp.path().join("artifacts").to_string_lossy().into_owned());
    let study_artifact_root = toml::Value::String(
        temp.path()
            .join("study-artifacts")
            .to_string_lossy()
            .into_owned(),
    );
    fs::write(
        &config,
        format!(
            "[runtime.accelerate]\nenabled = false\nprovider = \"inprocess\"\n\n[runtime.fea]\nartifact_store = \"filesystem\"\nartifact_root = {artifact_root}\nstudy_artifact_root = {study_artifact_root}\n"
        ),
    )
    .unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    local_offline::configure(&mut command, temp.path());
    command
        .current_dir(repository)
        .args(args)
        .env("RUNMAT_CONFIG", config)
        .env(
            "RUNMAT_PACKAGE_CACHE_DIR",
            temp.path().join("package-cache"),
        )
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            temp.path().join("execution-state"),
        );
    let admission = match outage {
        Some(_) => Some(fea_admission::AdmissionServer::configure_with(
            &mut command,
            temp.path(),
            outage,
        )),
        None => fea_admission::AdmissionServer::for_command(&mut command, temp.path()),
    };
    let output = command.output().expect("run repository FEA example");
    let Some(admission) = admission else {
        return (output, Vec::new(), Vec::new(), None);
    };
    let journal = admission.journal();
    (
        output,
        journal.requests.clone(),
        journal.receipts.clone(),
        journal.closed_for,
    )
}

fn assert_fea_checks(temp: &TempDir, path: &str) {
    let output = runmat_from_repository(temp, &["check", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed check:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["kind"], "fea", "{path}");
    assert_eq!(payload["status"], "valid", "{path}");
    assert_eq!(payload["diagnostics"], serde_json::json!([]), "{path}");
}

fn fea_plan(temp: &TempDir, path: &str) -> Value {
    let output = runmat_from_repository(temp, &["plan", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed plan:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

/// A rerun is a new attempt of the same definition: identical definition, mesh and inventory,
/// distinct attempt run ids, the same definition run ids, and the definition run id resolves to
/// the latest attempt.
fn assert_repeated_result_identity(temp: &TempDir, path: &str, first: &Value) {
    let repeated = runmat_from_repository(temp, &["run", path, "--json"]);
    assert!(
        repeated.status.success(),
        "{path} failed repeated run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&repeated.stdout),
        String::from_utf8_lossy(&repeated.stderr)
    );
    let repeated: Value = serde_json::from_slice(&repeated.stdout).unwrap();
    for field in [
        "study_definition_digest",
        "mesh_digest",
        "nodes",
        "elements",
        "boundary_faces",
        "definition_run_ids",
    ] {
        assert_eq!(
            repeated[field], first[field],
            "{path} changed verified result field `{field}` on repeat"
        );
    }
    assert_ne!(
        repeated["step_run_ids"], first["step_run_ids"],
        "{path} reused an earlier attempt instead of publishing a fresh one"
    );
    let latest = repeated["step_run_ids"][0]
        .as_str()
        .expect("latest attempt id");
    let definition = repeated["definition_run_ids"][0]
        .as_str()
        .expect("definition run id");
    let script_path = temp.path().join("latest-attempt.m");
    fs::write(
        &script_path,
        format!("r = fea.results(\"{definition}\");\ndisp(r.run_id);\n"),
    )
    .unwrap();
    let output = runmat_from_repository(
        temp,
        &["run", script_path.to_str().expect("UTF-8 script path")],
    );
    assert!(
        output.status.success(),
        "latest-attempt query failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains(latest),
        "definition run id did not resolve to the latest attempt `{latest}`"
    );
}

#[test]
fn canonical_fea_documentation_examples_check_cleanly() {
    let temp = TempDir::new().unwrap();
    for name in [
        "structural-static.fea",
        "structural-transient.fea",
        "modal-response-spectrum.fea",
        "modal-random-response.fea",
        "acoustic-cavity-harmonic.fea",
        "acoustic-duct-modes.fea",
        "acoustic-duct-pulse.fea",
    ] {
        let path = format!("docs/simulation/examples/{name}");
        assert_fea_checks(&temp, &path);
    }
}

// The trace-impedance example is written in its final form. Its `cross_section` geometries,
// triangle meshes and model-less studies lower; `runmat check` rejects it until the
// line-parameter outputs land (ledger rows electromagnetic.line_parameters.*). When it starts
// checking, move it into `canonical_fea_documentation_examples_check_cleanly`.
#[test]
fn trace_impedance_example_waits_for_the_line_parameters_procedure() {
    let temp = TempDir::new().unwrap();
    let path = "docs/simulation/examples/trace-impedance.fea";
    let output = runmat_from_repository(&temp, &["check", path, "--json"]);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "{path} now checks: list it with the canonical examples and land the ledger rows\n{text}"
    );
    assert!(!text.trim().is_empty(), "{path}: check failed without a diagnostic");
    // Each rejection names the ledger row that owns it.
    assert!(
        !text.contains("cross_section_geometry") && !text.contains("fea.semantic.mesh"),
        "{path}: the cross-section geometry should lower\n{text}"
    );
    for id in [
        "pending:electromagnetic.line_parameters.impedance",
        "pending:electromagnetic.line_parameters.differential_pair",
    ] {
        assert!(
            text.contains(id),
            "{path}: diagnostics do not name {id}\n{text}"
        );
    }
}

#[test]
fn representative_documentation_plans_expose_the_complete_resolved_contract() {
    struct Expected<'a> {
        file: &'a str,
        root: &'a str,
        model: &'a str,
        steps: &'a [(&'a str, &'a str)],
        outputs: &'a [&'a str],
        material_assignments: usize,
        response_spectra: usize,
        random_excitations: usize,
    }

    let cases = [
        Expected {
            file: "structural-static.fea",
            root: "structural_static",
            model: "loaded_bracket",
            steps: &[("static", "static")],
            outputs: &["displacement", "von_mises"],
            material_assignments: 2,
            response_spectra: 0,
            random_excitations: 0,
        },
        Expected {
            file: "structural-transient.fea",
            root: "structural_transient",
            model: "impulsive_bracket",
            steps: &[("response", "transient_response")],
            outputs: &["displacement"],
            material_assignments: 1,
            response_spectra: 0,
            random_excitations: 0,
        },
        Expected {
            file: "modal-response-spectrum.fea",
            root: "modal_response_spectrum",
            model: "launch_bracket",
            steps: &[
                ("modes", "modal_extraction"),
                ("response_spectrum", "modal_response_spectrum"),
            ],
            outputs: &["displacement"],
            material_assignments: 1,
            response_spectra: 1,
            random_excitations: 0,
        },
        Expected {
            file: "modal-random-response.fea",
            root: "modal_random_response",
            model: "random_environment",
            steps: &[
                ("modes", "modal_extraction"),
                ("random_response", "random_response"),
            ],
            outputs: &["displacement"],
            material_assignments: 1,
            response_spectra: 0,
            random_excitations: 1,
        },
        Expected {
            file: "acoustic-cavity-harmonic.fea",
            root: "acoustic_cavity_harmonic",
            model: "cavity",
            steps: &[("harmonic", "harmonic_response")],
            outputs: &["pressure"],
            material_assignments: 1,
            response_spectra: 0,
            random_excitations: 0,
        },
    ];
    let temp = TempDir::new().unwrap();
    for expected in cases {
        let path = format!("docs/simulation/examples/{}", expected.file);
        let plan = fea_plan(&temp, &path);
        assert_eq!(plan["schema_version"], 1, "{path}");
        assert_eq!(plan["root_kind"], "study", "{path}");
        assert_eq!(plan["root_id"], expected.root, "{path}");
        assert!(plan["geometry_dependencies"].as_array().is_some(), "{path}");

        let models = plan["model_closure"].as_array().expect("model closure");
        assert_eq!(models.len(), 1, "{path}");
        let model = &models[0];
        assert_eq!(model["model_id"], expected.model, "{path}");
        assert_eq!(
            model["response_spectra"].as_array().unwrap().len(),
            expected.response_spectra,
            "{path}"
        );
        assert_eq!(
            model["random_excitations"].as_array().unwrap().len(),
            expected.random_excitations,
            "{path}"
        );
        assert_eq!(
            plan["material_catalog"].as_array().unwrap().len(),
            1,
            "{path}"
        );
        assert_eq!(
            model["material_assignments"].as_array().unwrap().len(),
            expected.material_assignments,
            "{path}"
        );

        let steps = plan["study"]["sequence"]["steps"]
            .as_array()
            .expect("procedure steps");
        assert_eq!(steps.len(), expected.steps.len(), "{path}");
        for (step, (step_id, procedure)) in steps.iter().zip(expected.steps) {
            assert_eq!(step["step_id"], *step_id, "{path}");
            assert_eq!(step["procedure"], *procedure, "{path}");
        }
        let outputs = plan["study"]["outputs"]
            .as_array()
            .expect("authored outputs")
            .iter()
            .map(|output| output["output_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(outputs, expected.outputs, "{path}");
        assert_eq!(
            plan["mesh"]["geometry"], plan["study"]["geometry"]["definition"],
            "{path}"
        );
    }
}

// witness: FEA-PUBLIC-STUDIES-001/docs_examples_run @
//   shared_infrastructure/canonical_mixed_topology/public_study; coverage: partial;
//   boundary: The structural static STEP documentation example runs from the CLI and publishes
//   fields; the other examples are separate cases.
#[test]
fn structural_static_step_documentation_example_runs_and_publishes_fields() {
    let temp = TempDir::new().unwrap();
    let path = "docs/simulation/examples/structural-static.fea";
    let output = runmat_from_repository(&temp, &["run", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["root_id"], "structural_static");
    assert_eq!(payload["procedure"], "structural_linear_sequence");
    let manifest_path = payload["result_manifest_artifact_path"]
        .as_str()
        .expect("STEP study manifest path");
    let manifest: Value = serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["public_run"]["run_status"], "completed");
    assert_eq!(manifest["public_run"]["numerically_acceptable"], true);
    assert_eq!(
        manifest["public_run"]["steps"][0]["quality_reasons"],
        serde_json::json!([])
    );
    let mesh_path = temp.path().join("standalone-two-box.mesh.cbor");
    let evidence_path = temp.path().join("standalone-two-box.evidence.cbor");
    let standalone = runmat_from_repository(
        &temp,
        &[
            "mesh",
            "docs/simulation/examples/geometry/two-box-assembly.step",
            "--source-units",
            "meter",
            "--target-size",
            "1.5",
            "--deviation",
            "0.05",
            "--max-elements",
            "10000000",
            "--seed",
            "0",
            "--output",
            mesh_path.to_str().expect("UTF-8 mesh path"),
            "--evidence",
            evidence_path.to_str().expect("UTF-8 evidence path"),
            "--json",
        ],
    );
    assert!(
        standalone.status.success(),
        "standalone STEP mesh failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&standalone.stdout),
        String::from_utf8_lossy(&standalone.stderr)
    );
    let mesh = SolverMeshArtifact::canonical_decode(&fs::read(&mesh_path).unwrap()).unwrap();
    let evidence = MeshingEvidence::canonical_decode(&fs::read(&evidence_path).unwrap()).unwrap();
    evidence.validate(&mesh).expect("standalone mesh evidence");
    let manifest_digest = manifest["solver_mesh_digest"]
        .as_array()
        .expect("study solver mesh digest")
        .iter()
        .map(|byte| u8::try_from(byte.as_u64().expect("digest byte")).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(manifest_digest, mesh.canonical_digest.bytes().to_vec());
    assert_eq!(
        manifest["meshing_evidence"]["artifact_digest"],
        manifest["solver_mesh_digest"]
    );
    assert_eq!(evidence.artifact_digest, mesh.canonical_digest);
    // Golden counts for the authored independent-definition STEP fixture,
    // qualified by two byte-identical standalone meshes with seed zero.
    // Face gaps are sized to at least one element layer across (thin-layer sizing): over a
    // whole face when all its edges face the gap, otherwise from the edges that do.
    assert_eq!(
        (
            mesh.topology.nodes.len(),
            mesh.topology.volume_elements.len(),
            mesh.topology.boundary_faces.len()
        ),
        (584, 2330, 776)
    );

    let model: AnalysisModel =
        serde_json::from_value(manifest["compiled"]["model"].clone()).unwrap();
    let resolved_solids = model
        .material_assignments
        .iter()
        .map(|assignment| {
            let PropertyRegionScope::Region(selection) = assignment.target.scope() else {
                panic!("two-box material must select one resolved solid");
            };
            let [solid] = selection.resolved_entities() else {
                panic!("two-box material must resolve one solid occurrence");
            };
            assert!(mesh
                .topology
                .volume_elements
                .iter()
                .any(|element| element.provenance.contains(solid)));
            solid.clone()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(resolved_solids.len(), 2);
    let boundary_targets = model
        .constraints
        .iter()
        .filter_map(|constraint| constraint.target())
        .chain(model.excitations.iter().filter_map(|action| {
            let ExcitationTarget::Region(target) = &action.excitation().target else {
                return None;
            };
            Some(target)
        }))
        .collect::<Vec<_>>();
    assert_eq!(boundary_targets.len(), 3);
    for target in boundary_targets {
        let [face] = target.resolved_entities() else {
            panic!("two-box boundary target must resolve one face occurrence");
        };
        assert!(mesh
            .topology
            .boundary_faces
            .iter()
            .any(|boundary| boundary.provenance.contains(face)));
    }
    let step_run_id = payload["step_run_ids"]
        .as_array()
        .and_then(|steps| steps.first())
        .and_then(Value::as_str)
        .expect("static step run id");
    assert_repeated_result_identity(&temp, path, &payload);

    let script_path = temp.path().join("static-fields.m");
    fs::write(
        &script_path,
        format!(
            "results = fea.results(\"{step_run_id}\");\ndisplacement = fea.field(results, fea.OutputField.named(\"displacement\"));\ndisplacementPage = fea.fieldPage(displacement, 1, 3);\nstress = fea.field(results, fea.OutputField.named(\"von_mises\"));\nstressPage = fea.fieldPage(stress, 1, 3);\ndisp(displacement.field_id);\ndisp(displacementPage.count);\ndisp(stress.field_id);\ndisp(stressPage.count);\n"
        ),
    )
    .unwrap();
    let fields = runmat_from_repository(
        &temp,
        &["run", script_path.to_str().expect("UTF-8 script path")],
    );
    assert!(
        fields.status.success(),
        "STEP-backed result field smoke failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&fields.stdout),
        String::from_utf8_lossy(&fields.stderr)
    );
    let stdout = String::from_utf8_lossy(&fields.stdout);
    assert!(stdout.contains("displacement"));
    assert!(stdout.contains("von_mises"));
}

// witness: FEA-PUBLIC-STUDIES-001/docs_examples_run @
//   shared_infrastructure/canonical_mixed_topology/public_study; coverage: partial;
//   boundary: The response-spectrum documentation example runs its two-step study.
#[test]
fn response_spectrum_documentation_example_runs_its_two_step_study() {
    let temp = TempDir::new().unwrap();
    let path = "docs/simulation/examples/modal-response-spectrum.fea";
    let output = runmat_from_repository(&temp, &["run", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["root_id"], "modal_response_spectrum");
    assert_eq!(payload["procedure"], "structural_modal_response_spectrum");
    let step_run_ids = payload["step_run_ids"]
        .as_array()
        .expect("study step run ids");
    assert_eq!(step_run_ids.len(), 2);
    let response_run_id = step_run_ids[1]
        .as_str()
        .expect("response-spectrum step run id");

    let script_path = temp.path().join("response-spectrum-field.m");
    fs::write(
        &script_path,
        format!(
            "response = fea.results(\"{response_run_id}\");\nfield = fea.field(response, fea.ResponseSpectrumField.pseudoAcceleration());\npage = fea.fieldPage(field, 1, 3);\ndisp(field.field_id);\ndisp(field.statistic);\ndisp(page.count);\n"
        ),
    )
    .unwrap();
    let output = runmat_from_repository(
        &temp,
        &["run", script_path.to_str().expect("UTF-8 script path")],
    );
    assert!(
        output.status.success(),
        "response-spectrum language field smoke failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("structural.response_spectrum.nodal.pseudo_acceleration"));
    assert!(stdout.contains("component_magnitude_envelope"));
    assert!(stdout.contains("3"));
}

#[test]
fn named_response_spectrum_generator_materializes_before_the_solve() {
    let temp = TempDir::new().unwrap();
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"fea-spectrum-generator\"\n\n[sources]\nroots = [\".\"]\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("design_input.m"),
        "function y = design_input(frequency, scale)\narguments\n    frequency (:,1) double\n    scale (1,1) double\nend\ny = scale .* (1 + 0 .* frequency);\nend\n",
    )
    .unwrap();
    let source = repository.join("docs/simulation/examples/modal-response-spectrum.fea");
    fs::create_dir_all(temp.path().join("geometry")).unwrap();
    fs::copy(
        repository.join("crates/runmat-geometry/io/tests/fixtures/box.brep"),
        temp.path().join("geometry/box.brep"),
    )
    .unwrap();
    let document = format!(
        "import function design_input from \"./design_input.m\"\n\n{}",
        fs::read_to_string(source).unwrap().replace(
            "        ordinates = [1.2, 1.2, 2.5, 1.8, 0.7] g",
            "        generated by function.design_input { arguments { scale = 1.5 } }",
        )
    );
    fs::write(temp.path().join("study.fea"), document).unwrap();

    let checked = runmat(&temp, &["check", "study.fea", "--json"]);
    assert!(
        checked.status.success(),
        "generated response spectrum failed check:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&checked.stdout),
        String::from_utf8_lossy(&checked.stderr)
    );

    let run = runmat(&temp, &["run", "study.fea", "--json"]);
    assert!(
        run.status.success(),
        "generated response spectrum failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&run.stdout).unwrap()["root_id"],
        "modal_response_spectrum"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&run.stdout).unwrap()["procedure"],
        "structural_modal_response_spectrum"
    );
}

// witness: FEA-PUBLIC-STUDIES-001/docs_examples_run @
//   shared_infrastructure/canonical_mixed_topology/public_study; coverage: partial;
//   boundary: The random-response documentation example runs and exposes paged statistical fields.
#[test]
fn random_response_documentation_example_exposes_paged_statistical_fields() {
    let temp = TempDir::new().unwrap();
    let path = "docs/simulation/examples/modal-random-response.fea";
    let run = runmat_from_repository(&temp, &["run", path, "--json"]);
    assert!(
        run.status.success(),
        "{path} failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let payload: Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(payload["procedure"], "structural_random_response");
    let response_run_id = payload["step_run_ids"]
        .as_array()
        .and_then(|steps| steps.get(1))
        .and_then(Value::as_str)
        .expect("random-response step run id");
    let script_path = temp.path().join("random-response-fields.m");
    fs::write(
        &script_path,
        format!(
            "response = fea.results(\"{response_run_id}\");\nrms = fea.field(response, fea.RandomResponseField.rms(fea.StructuralField.displacement()));\nrmsPage = fea.fieldPage(rms, 1, 3);\nvariance = fea.field(response, fea.RandomResponseField.variance(fea.StructuralField.stress()));\nvariancePage = fea.fieldPage(variance, 1, 3);\ndisp(rms.field_id);\ndisp(rms.statistic);\ndisp(rmsPage.count);\ndisp(variance.field_id);\ndisp(variance.statistic);\ndisp(variancePage.count);\n"
        ),
    )
    .unwrap();
    let output = runmat_from_repository(
        &temp,
        &["run", script_path.to_str().expect("UTF-8 script path")],
    );
    assert!(
        output.status.success(),
        "random-response language field smoke failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("structural.random_response.rms.displacement"));
    assert!(stdout.contains("root_mean_square"));
    assert!(stdout.contains("structural.random_response.variance.stress"));
    assert!(stdout.contains("variance"));
}

#[test]
fn fea_check_and_run_materialize_a_named_project_signal_generator() {
    let temp = TempDir::new().unwrap();
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"fea-callable-check\"\n\n[sources]\nroots = [\".\"]\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("pulse.m"),
        "function y = pulse(t, magnitude)\narguments\n    t (:,1) double\n    magnitude (1,1) double\nend\ny = magnitude .* (1 + 0 .* t);\nend\n",
    )
    .unwrap();
    let fixture = repository.join("docs/simulation/examples/structural-transient.fea");
    fs::create_dir_all(temp.path().join("geometry")).unwrap();
    fs::copy(
        repository.join("crates/runmat-geometry/io/tests/fixtures/box.brep"),
        temp.path().join("geometry/box.brep"),
    )
    .unwrap();
    let document = format!(
        "import function pulse from \"./pulse.m\"\n\n{}",
        fs::read_to_string(fixture)
            .unwrap()
        .replace("            end_time = 0.1 s", "            end_time = 0.003 s")
        .replace(
            "        samples = [\n            { at = 0 s, value = 0 N },\n            { at = 0.01 s, value = 1 kN },\n            { at = 0.02 s, value = 0 N },\n            { at = 0.1 s, value = 0 N },\n        ]\n        interpolation = linear\n        extrapolation = reject",
            "        generated by function.pulse {\n            arguments { magnitude = 1000 }\n            sample from 0 s to 0.003 s at 1 kHz\n            interpolation = linear\n            extrapolation = reject\n            tolerance = 1e-6 N\n        }",
        )
    );
    fs::write(temp.path().join("study.fea"), document).unwrap();

    let output = runmat(&temp, &["check", "study.fea", "--json"]);
    assert!(
        output.status.success(),
        "callable-backed study failed check:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["status"], "valid");

    let output = runmat(&temp, &["run", "study.fea", "--json"]);
    assert!(
        output.status.success(),
        "callable-backed study failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["root_id"], "structural_transient");
    assert_eq!(payload["procedure"], "structural_transient");
    assert!(payload["study_definition_digest"].as_str().is_some());
}

// witness: FEA-PUBLIC-STUDIES-001/docs_examples_run @
//   shared_infrastructure/canonical_mixed_topology/public_study; coverage: partial;
//   boundary: The acoustic cavity harmonic example runs its frequency sweep to a completed,
//   numerically acceptable run.
#[test]
fn acoustic_cavity_example_runs_its_certified_frequency_sweep() {
    let temp = TempDir::new().unwrap();
    let path = "docs/simulation/examples/acoustic-cavity-harmonic.fea";
    let output = runmat_from_repository(&temp, &["run", path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["root_id"], "acoustic_cavity_harmonic");
    let manifest_path = payload["result_manifest_artifact_path"]
        .as_str()
        .expect("acoustic study manifest path");
    let manifest: Value = serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["public_run"]["run_status"], "completed");
    assert_eq!(manifest["public_run"]["numerically_acceptable"], true);
}

/// A Server restart during a run (its port closed for 30 s) ends neither the lease nor the
/// run, and the receipt carries exactly the operations of an uninterrupted run.
#[test]
fn a_server_restart_mid_run_keeps_the_lease_and_meters_the_same_operations() {
    let path = "docs/simulation/examples/acoustic-cavity-harmonic.fea";
    let steady_temp = TempDir::new().unwrap();
    let (steady, _, steady_receipts, _) = runmat_from_repository_through(
        &steady_temp,
        &["run", path, "--json"],
        Some(fea_admission::Outage {
            after: usize::MAX,
            down: std::time::Duration::ZERO,
        }),
    );
    assert!(
        steady.status.success(),
        "{}",
        String::from_utf8_lossy(&steady.stderr)
    );
    let temp = TempDir::new().unwrap();
    let (output, requests, receipts, closed_for) = runmat_from_repository_through(
        &temp,
        &["run", path, "--json"],
        Some(fea_admission::Outage {
            after: 1,
            down: std::time::Duration::from_secs(30),
        }),
    );
    assert!(
        output.status.success(),
        "{path} failed through a Server restart:\nrequests={requests:?}\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let closed_for = closed_for.expect("the Server's port was closed mid-run");
    assert!(closed_for >= std::time::Duration::from_secs(30));
    eprintln!("lease requests through a {closed_for:?} outage: {requests:?}");
    assert!(requests[0].0.ends_with("/simulation/leases"));
    assert!(requests.iter().all(|(_, status)| status.starts_with("200")));
    let metered = |receipts: &[Value]| {
        receipts
            .iter()
            .map(|r| (r["outcome"].clone(), r["usedOps"].clone(), r["totalOps"].clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(receipts.len(), 1);
    assert_eq!(metered(&receipts), metered(&steady_receipts));
}

#[test]
fn unresolved_function_is_a_warning_by_default_and_an_error_when_denied() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let default = runmat(&temp, &["check", "main.m"]);
    assert!(default.status.success(), "{default:?}");
    let stdout = String::from_utf8(default.stdout).unwrap();
    assert!(stdout.contains("warning[RM-RES0001]: cannot find function `definitely_missing`"));
    assert!(stdout.contains("checked main.m: 0 error(s), 1 warning(s)"));

    let denied = runmat(&temp, &["check", "main.m", "-D", "warnings"]);
    assert!(!denied.status.success(), "{denied:?}");
    let stdout = String::from_utf8(denied.stdout).unwrap();
    assert!(stdout.contains("warning[RM-RES0001]"));
}

#[test]
fn json_output_is_stable_and_contains_source_coordinates() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"check-json\"\n\n[sources]\nroots = [\".\"]\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let output = runmat(&temp, &["check", "main.m", "--json"]);
    assert!(output.status.success(), "{output:?}");
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["outcome"], "warnings");
    assert_eq!(payload["summary"]["warnings"], 1);
    assert_eq!(payload["diagnostics"][0]["code"], "RM-RES0001");
    assert_eq!(payload["diagnostics"][0]["severity"], "warning");
    assert_eq!(payload["diagnostics"][0]["primary"]["line"], 1);
    assert_eq!(payload["diagnostics"][0]["primary"]["column"], 9);
    assert_eq!(payload["analysis"]["name_resolution"], "complete");
    assert!(payload["project_revision"]["graph_digest"]
        .as_str()
        .is_some_and(|digest| digest.starts_with("sha256:")));
    assert!(payload["project_revision"]["source_revision"]
        .as_str()
        .is_some_and(|digest| digest.starts_with("sha256:")));
}

#[test]
fn human_diagnostics_support_forced_and_plain_color_modes() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let styled = runmat_with_env(&temp, &["--color=always", "check", "main.m"], &[]);
    assert!(styled.status.success(), "{styled:?}");
    assert!(
        styled.stdout.windows(2).any(|bytes| bytes == b"\x1b["),
        "forced human output did not contain ANSI styling: {}",
        String::from_utf8_lossy(&styled.stdout)
    );

    let plain = runmat_with_env(
        &temp,
        &["check", "main.m", "--color=never"],
        &[("FORCE_COLOR", "3")],
    );
    assert!(plain.status.success(), "{plain:?}");
    assert!(!plain.stdout.windows(2).any(|bytes| bytes == b"\x1b["));
    assert_eq!(strip_ansi(&styled.stdout), plain.stdout);
}

#[test]
fn no_color_is_nonempty_and_wins_over_environment_force() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let disabled = runmat_with_env(
        &temp,
        &["check", "main.m"],
        &[("NO_COLOR", "0"), ("FORCE_COLOR", "3")],
    );
    assert!(disabled.status.success(), "{disabled:?}");
    assert!(!disabled.stdout.windows(2).any(|bytes| bytes == b"\x1b["));

    let empty_is_unset = runmat_with_env(
        &temp,
        &["check", "main.m"],
        &[("NO_COLOR", ""), ("FORCE_COLOR", "1")],
    );
    assert!(empty_is_unset.status.success(), "{empty_is_unset:?}");
    assert!(empty_is_unset
        .stdout
        .windows(2)
        .any(|bytes| bytes == b"\x1b["));
}

#[test]
fn json_remains_plain_even_when_human_color_is_forced() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "value = definitely_missing(1);\n",
    )
    .unwrap();

    let output = runmat_with_env(&temp, &["--color=always", "check", "main.m", "--json"], &[]);
    assert!(output.status.success(), "{output:?}");
    assert!(!output.stdout.windows(2).any(|bytes| bytes == b"\x1b["));
    assert!(!output.stderr.windows(2).any(|bytes| bytes == b"\x1b["));
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["schema_version"], 1);
}

#[test]
fn fea_human_path_uses_the_shared_color_policy() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("invalid.fea"), "{}\n").unwrap();

    let styled = runmat_with_env(&temp, &["--color=always", "check", "invalid.fea"], &[]);
    assert!(!styled.status.success(), "{styled:?}");
    assert!(
        styled.stdout.windows(2).any(|bytes| bytes == b"\x1b["),
        "FEA human output did not contain ANSI styling: {}",
        String::from_utf8_lossy(&styled.stdout)
    );
    assert!(String::from_utf8_lossy(&styled.stdout).contains("checked"));

    let plain = runmat_with_env(&temp, &["--color=never", "check", "invalid.fea"], &[]);
    assert!(!plain.status.success(), "{plain:?}");
    assert!(!plain.stdout.windows(2).any(|bytes| bytes == b"\x1b["));
    assert_eq!(strip_ansi(&styled.stdout), plain.stdout);
}

#[test]
fn loose_sibling_and_explicit_search_path_functions_resolve() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("main.m"), "value = helper(1);\n").unwrap();
    fs::write(
        temp.path().join("helper.m"),
        "function y = helper(x)\ny = x;\nend\n",
    )
    .unwrap();
    let sibling = runmat(&temp, &["check", "main.m", "--json"]);
    assert!(sibling.status.success(), "{sibling:?}");
    let payload: Value = serde_json::from_slice(&sibling.stdout).unwrap();
    assert_eq!(payload["outcome"], "clean");
    assert_eq!(payload["resolution"][0]["state"], "resolved");
    assert_eq!(payload["resolution"][0]["name"], "helper");
    assert!(payload["resolution"][0]["definition"]["source_path"]
        .as_str()
        .expect("source path")
        .ends_with("helper.m"));

    fs::remove_file(temp.path().join("helper.m")).unwrap();
    fs::create_dir(temp.path().join("toolbox")).unwrap();
    fs::write(
        temp.path().join("toolbox/helper.m"),
        "function y = helper(x)\ny = x;\nend\n",
    )
    .unwrap();
    let without_path = runmat(&temp, &["check", "main.m", "--json"]);
    let payload: Value = serde_json::from_slice(&without_path.stdout).unwrap();
    assert_eq!(payload["outcome"], "warnings");

    let with_path = runmat(&temp, &["check", "main.m", "--path", "toolbox", "--json"]);
    assert!(with_path.status.success(), "{with_path:?}");
    let payload: Value = serde_json::from_slice(&with_path.stdout).unwrap();
    assert_eq!(payload["outcome"], "clean");
}

#[test]
fn runtime_path_mutation_reports_the_causal_site() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("main.m"),
        "addpath('plugins');\nvalue = selected_later(1);\n",
    )
    .unwrap();

    let output = runmat(&temp, &["check", "main.m", "--json"]);
    assert!(output.status.success(), "{output:?}");
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    let diagnostic = payload["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|diagnostic| diagnostic["code"] == "RM-RES0002")
        .expect("runtime-dependent resolution diagnostic");
    assert_eq!(diagnostic["secondary"][0]["line"], 1);
    assert_eq!(diagnostic["primary"]["line"], 2);
    assert_eq!(payload["analysis"]["name_resolution"], "runtime_dependent");
}

#[test]
fn syntax_and_source_catalog_failures_return_structured_json() {
    let syntax = TempDir::new().unwrap();
    fs::write(syntax.path().join("main.m"), "value = ;\n").unwrap();
    let output = runmat(&syntax, &["check", "main.m", "--json"]);
    assert!(!output.status.success(), "{output:?}");
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["outcome"], "failed");
    assert_eq!(payload["diagnostics"][0]["code"], "RunMat:ParseError");
    assert_eq!(payload["analysis"]["name_resolution"], "unavailable");

    let catalog = TempDir::new().unwrap();
    fs::write(catalog.path().join("main.m"), "value = 1;\n").unwrap();
    let output = runmat(
        &catalog,
        &["check", "main.m", "--path", "missing-source-root", "--json"],
    );
    assert!(!output.status.success(), "{output:?}");
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(payload["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .any(|diagnostic| diagnostic["code"] == "RM-CAT0001"));
}

fn strip_ansi(bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == 0x1b && bytes.get(index + 1) == Some(&b'[') {
            index += 2;
            while index < bytes.len() {
                let byte = bytes[index];
                index += 1;
                if byte.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }
    output
}

/// The homepage "Physics simulation as code" figure: `runmat check cantilever.fea` on the
/// cantilever-static study with the traction authored in N and no `load_case` line.
// witness: FEA-PUBLIC-LANGUAGE-001/located_diagnostics @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Checking a study with a force given as a traction prints error[fea.semantic.model],
//   the file:line:col arrow, the source line, a caret underline labeled with the expected unit (Pa)
//   and the checked summary, with no cascade error; leading-indent trimming and the JSON form are
//   not asserted here.
// witness: FEA-PUBLIC-LANGUAGE-001/default_load_case @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: The same file with the unit corrected and no load_case checks cleanly, so the load
//   joins the default case; multiple loads with an explicit case set are not compared.
#[test]
fn fea_check_prints_located_diagnostics_like_m_files() {
    let temp = TempDir::new().unwrap();
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::create_dir_all(temp.path().join("geometry")).unwrap();
    fs::copy(
        repository.join("docs/simulation/examples/geometry/cantilever.step"),
        temp.path().join("geometry/cantilever.step"),
    )
    .unwrap();
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/home_study_check/cantilever.fea"),
    )
    .unwrap();
    fs::write(temp.path().join("cantilever.fea"), &source).unwrap();

    let output = runmat_with_env(
        &temp,
        &["check", "cantilever.fea"],
        &[
            ("NO_COLOR", "1"),
            (
                "RUNMAT_PACKAGE_CACHE_DIR",
                &temp.path().join("cache").to_string_lossy(),
            ),
        ],
    );
    assert!(!output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&strip_ansi(&output.stdout)),
        concat!(
            "error[fea.semantic.model]: traction must be force per area, found N\n",
            " --> cantilever.fea:28:48\n",
            "   |\n",
            "28 | traction on geometry.beam.region.tip = [0, 0, -100000] N\n",
            "   |                                        ^^^^^^^^^^^^^^^^^ expected Pa\n",
            "checked cantilever.fea: 1 error(s), 0 warning(s)\n",
        ),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    // With the unit corrected, the same file checks cleanly: the load joins the default case.
    fs::write(
        temp.path().join("cantilever.fea"),
        source.replace("-100000] N", "-100000] Pa"),
    )
    .unwrap();
    let output = runmat_with_env(
        &temp,
        &["check", "cantilever.fea"],
        &[
            ("NO_COLOR", "1"),
            (
                "RUNMAT_PACKAGE_CACHE_DIR",
                &temp.path().join("cache").to_string_lossy(),
            ),
        ],
    );
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&strip_ansi(&output.stdout)),
        "checked cantilever.fea: 0 error(s), 0 warning(s)\n"
    );
}

// witness: FEA-SHARED-EXECUTION-001/logical_study_identity @
//   shared_infrastructure/canonical_mixed_topology/native_process; coverage: partial;
//   boundary: Two executions of one random-response study from a script keep one definition run
//   id, publish distinct attempts, and the definition id resolves to the latest attempt. Remote
//   placement and retry attempts inside one execution are not exercised.
#[test]
fn a_script_running_the_same_study_twice_publishes_two_attempts() {
    let temp = TempDir::new().unwrap();
    let repository = repository_fixture(&temp);
    let study = repository.join("docs/simulation/examples/modal-random-response.fea");
    let script_path = temp.path().join("twice.m");
    fs::write(
        &script_path,
        format!(
            "first = fea.stepResult(fea.run(\"{0}\"), \"random_response\");\nsecond = fea.stepResult(fea.run(\"{0}\"), \"random_response\");\ndisp(first.run_id);\ndisp(second.run_id);\n",
            study.display()
        ),
    )
    .unwrap();
    let output = runmat_from_repository(
        &temp,
        &["run", script_path.to_str().expect("UTF-8 script path")],
    );
    assert!(
        output.status.success(),
        "running a study twice failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let ids = stdout
        .split_whitespace()
        .filter(|token| token.starts_with("run_remote_"))
        .collect::<Vec<_>>();
    let [first, second] = ids.as_slice() else {
        panic!("expected two attempt run ids, got {stdout}");
    };
    assert_ne!(first, second, "the second run reused the first attempt");
    let (definition, _) = second.rsplit_once('_').expect("attempt run id");
    assert_eq!(
        first.rsplit_once('_').map(|(definition, _)| definition),
        Some(definition)
    );
    let query = temp.path().join("latest.m");
    fs::write(
        &query,
        format!("r = fea.results(\"{definition}\");\ndisp(r.run_id);\n"),
    )
    .unwrap();
    let output = runmat_from_repository(&temp, &["run", query.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "latest-attempt query failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains(second),
        "latest attempt of {definition} is not {second}:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Runs a documentation example with `extra` study members (outputs and assessments) appended,
/// asserts a completed, numerically acceptable public run, and returns the run payload.
fn run_example_witness(temp: &TempDir, name: &str, extra: &str) -> Value {
    let examples = repository_fixture(temp).join("docs/simulation/examples");
    let path = if extra.is_empty() {
        format!("docs/simulation/examples/{name}.fea")
    } else {
        let source = fs::read_to_string(examples.join(format!("{name}.fea"))).unwrap();
        let body = source.trim_end().strip_suffix('}').expect("study block closes the file");
        fs::write(
            examples.join(format!("{name}.witness.fea")),
            format!("{body}{extra}}}\n"),
        )
        .unwrap();
        format!("docs/simulation/examples/{name}.witness.fea")
    };
    let output = runmat_from_repository(temp, &["run", &path, "--json"]);
    assert!(
        output.status.success(),
        "{path} failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).unwrap();
    let manifest_path = payload["result_manifest_artifact_path"]
        .as_str()
        .unwrap_or_else(|| panic!("{path}: no manifest path in {payload}"));
    let manifest: Value = serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["public_run"]["run_status"], "completed", "{path}");
    assert_eq!(
        manifest["public_run"]["numerically_acceptable"], true,
        "{path}"
    );
    payload
}

/// The measured value of a passed assessment.
fn passed_assessment(payload: &Value, id: &str) -> f64 {
    let assessment = payload["assessments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["result"]["assessment_id"] == id)
        .unwrap_or_else(|| panic!("no assessment {id} in {}", payload["assessments"]));
    let status = assessment["result"]["status"].as_str().unwrap_or_default();
    assert!(status.eq_ignore_ascii_case("passed"), "{id}: {assessment}");
    assessment["evidence"]["value"].as_f64().unwrap()
}

// witness: FEA-FLUID-OUTPUTS-001/pressure_drop_and_flow_rate @
//   incompressible_laminar_steady/tet4/public_study;
//   coverage: partial; check: plane-poiseuille; boundary: pressure_drop on the public plane Poiseuille example
//   (L/h = 30, Re = 1) lies within 5% of the fully developed 12 mu U L / h^2 = 36 Pa; the
//   uniform inlet's entrance loss (about 0.9 h of equivalent length, +3%) is inside the band.
#[test]
fn plane_poiseuille_example_publishes_the_analytic_pressure_drop() {
    let temp = TempDir::new().unwrap();
    let payload = run_example_witness(
        &temp,
        "fluid-plane-poiseuille",
        "    assessment poiseuille_drop { value = output.pressure_drop; minimum = 34.2 Pa; maximum = 37.8 Pa; evidence = authored_samples }\n",
    );
    let drop = passed_assessment(&payload, "poiseuille_drop");
    eprintln!("plane Poiseuille pressure_drop = {drop} Pa (developed 36 Pa)");
}

// witness: FEA-ACOUSTIC-OUTPUTS-001/transmission_and_insertion_loss @
//   acoustic_harmonic/tet10/public_study; coverage: partial; check: two-fluid-interface; boundary: transmission_loss on the
//   public two-fluid interface example (m = 4) lies within 0.1 dB of 1.938 dB at 500, 1000 and
//   2000 Hz; no muffler or one-third-octave sweep.
#[test]
fn two_fluid_interface_example_publishes_the_plane_wave_transmission_loss() {
    let temp = TempDir::new().unwrap();
    let mut extra = String::new();
    for f in [500, 1000, 2000] {
        extra.push_str(&format!(
            "    output tl_at_{f} = max(sample(output.tl, over = frequency, at = [{f} Hz]), over = frequency)\n    assessment tl_{f} {{ value = output.tl_at_{f}; minimum = 1.84; maximum = 2.04; evidence = authored_samples }}\n"
        ));
    }
    let payload = run_example_witness(&temp, "acoustic-interface-transmission-loss", &extra);
    for f in [500, 1000, 2000] {
        let tl = passed_assessment(&payload, &format!("tl_{f}"));
        eprintln!("transmission loss at {f} Hz = {tl} dB (analytic 1.938 dB)");
    }
}

// witness: FEA-COUPLING-CHT-001/interface @
//   conjugate_heat_transfer/canonical_mixed_topology/public_study; coverage: partial;
//   boundary: The CHT plate-channel example runs two-way from `runmat run` to a completed,
//   numerically acceptable run; interface closure is the runtime gate, not asserted here.
// witness: FEA-FLUID-HEAT-001/two_way_cht @ conjugate_heat_transfer/tet4/public_study;
//   coverage: partial; boundary: One public two-way CHT example on a linear mesh converges.
// witness: FEA-PUBLIC-STUDIES-001/chained_and_coupled_studies_run @
//   conjugate_heat_transfer/canonical_tetrahedra/public_study; coverage: partial;
//   boundary: Local public CHT run only; remote execution is a separate case.
#[test]
fn cht_plate_channel_example_runs_as_a_coupled_study() {
    let temp = TempDir::new().unwrap();
    let payload = run_example_witness(
        &temp,
        "cht-plate-channel",
        "    assessment plate_warms { value = output.peak_temperature; minimum = 293 K; maximum = 1000 K; evidence = authored_samples }\n",
    );
    passed_assessment(&payload, "plate_warms");
}

// witness: FEA-COUPLING-FSI-001/interface @
//   fluid_structure_interaction/canonical_mixed_topology/public_study; coverage: partial;
//   boundary: The compliant-tube example runs steady two-way FSI from `runmat run` to a
//   completed, numerically acceptable run.
// witness: FEA-COUPLING-FSI-002/compliant_channel @ fluid_structure_interaction/tet4/public_study;
//   coverage: partial; boundary: Steady compliant tube only; no transient ALE run.
// witness: FEA-PUBLIC-STUDIES-001/chained_and_coupled_studies_run @
//   fluid_structure_interaction/canonical_tetrahedra/public_study; coverage: partial;
//   boundary: Local public FSI run only; remote execution is a separate case.
#[test]
fn fsi_compliant_tube_example_runs_as_a_coupled_study() {
    let temp = TempDir::new().unwrap();
    let payload = run_example_witness(
        &temp,
        "fsi-compliant-tube",
        "    assessment pressurized { value = output.inlet_pressure; minimum = 0 Pa; evidence = authored_samples }\n",
    );
    passed_assessment(&payload, "pressurized");
}

/// `runmat check` applies the run's fluid–solid face rule to the CAD's shared faces: the
/// compliant tube's bore with its declared interface checks clean, the bore with a fluid wall
/// condition and no interface is a rigid wall, and the bore with neither is the same
/// `fea.admission.fluid_solid_face` error the run reports.
#[test]
fn fsi_check_classifies_the_cad_faces_that_join_fluid_and_solid() {
    let temp = TempDir::new().unwrap();
    let examples = repository_fixture(&temp).join("docs/simulation/examples");
    let source = fs::read_to_string(examples.join("fsi-compliant-tube.fea")).unwrap();
    assert_fea_checks(&temp, "docs/simulation/examples/fsi-compliant-tube.fea");
    let start = source.find("    interface wetted {").unwrap();
    let end = start + source[start..].find("\n    }\n").unwrap() + "\n    }\n".len();
    let check = |name: &str, replacement: &str| -> Value {
        fs::write(
            examples.join(name),
            format!("{}{replacement}{}", &source[..start], &source[end..]),
        )
        .unwrap();
        let output = runmat_from_repository(
            &temp,
            &["check", &format!("docs/simulation/examples/{name}"), "--json"],
        );
        serde_json::from_slice(&output.stdout).unwrap_or_else(|_| {
            panic!(
                "{name}: stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
    };
    let undeclared = check("fsi-undeclared-bore.fea", "");
    assert_eq!(undeclared["status"], "invalid", "{undeclared}");
    let diagnostics = undeclared["diagnostics"].to_string();
    assert!(
        diagnostics.contains("fea.admission.fluid_solid_face") && diagnostics.contains("Bore"),
        "{diagnostics}"
    );
    let rigid = check(
        "fsi-rigid-bore.fea",
        "    constraint bore_wall { fluid.no_slip_wall { target = geometry.tube.region.bore } }\n",
    );
    assert!(
        !rigid["diagnostics"]
            .to_string()
            .contains("fea.admission.fluid_solid_face"),
        "{rigid}"
    );
}

// witness: FEA-COUPLING-CHAIN-001/thermal_to_prestress_to_modal @ thermoelastic/tet4/public_study;
//   coverage: partial; boundary: The thermal-prestress example runs thermal steady, then
//   thermomechanical preload, then prestressed modal from `runmat run`; no reference values.
// witness: FEA-PUBLIC-STUDIES-001/chained_and_coupled_studies_run @
//   thermoelastic/canonical_tetrahedra/public_study; coverage: partial;
//   boundary: Local public chained run only; remote execution is a separate case.
#[test]
fn thermal_prestress_example_runs_its_three_step_chain() {
    let temp = TempDir::new().unwrap();
    let payload = run_example_witness(&temp, "thermal-prestress", "");
    assert_eq!(payload["step_run_ids"].as_array().unwrap().len(), 3, "{payload}");
}

// witness: FEA-EM-FREQUENCY-001/multi_frequency_study @ harmonic_maxwell/edge_elements/public_study;
//   coverage: partial; boundary: The induction-gear example runs three authored frequencies with
//   a current-driven coil terminal from `runmat run`, one case per frequency, to a completed,
//   numerically acceptable run; no skin-depth reference is asserted here.
#[test]
fn induction_gear_example_runs_its_three_frequency_harmonic_study() {
    let temp = TempDir::new().unwrap();
    let payload = run_example_witness(&temp, "em-induction-gear-harmonic", "");
    assert_eq!(payload["step_run_ids"].as_array().unwrap().len(), 3, "{payload}");
}
