//! Actual authored CAD/process execution and post-exit shared artifact reopening.
use super::*;

#[path = "acoustic/duct.rs"]
mod duct;

#[test]
fn authored_acoustic_orders_publish_cold_native_fields_and_exact_state_catalog() {
    for quadratic in [false, true] {
        let project = TempDir::new().unwrap();
        fs::create_dir(project.path().join("src")).unwrap();
        fs::write(project.path().join("runmat.toml"),
            "[package]\nname = \"typed-acoustic\"\n[sources]\nroots = [\"src\"]\n[runtime.fea]\nartifact_root = \"artifacts\"\nstudy_artifact_root = \"study-artifacts\"\n").unwrap();
        fs::write(
            project.path().join("src/makePart.m"),
            "function part = makePart()\npart = geometry.finish(geometry.box(1, 1, 1));\nend\n",
        )
        .unwrap();
        let source = if quadratic {
            STUDY.replace("order = linear", "order = quadratic")
        } else {
            STUDY.to_owned()
        };
        fs::write(project.path().join("src/main.fea"), source).unwrap();
        let output = runmat(
            &project,
            &[
                "fea",
                "run",
                "src/main.fea",
                "--study",
                "response",
                "--json",
            ],
        );
        assert!(
            output.status.success(),
            "quadratic={quadratic}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(summary["procedure"], "acoustic_harmonic");
        let run_ids = summary["step_run_ids"].as_array().unwrap();
        assert_eq!(run_ids.len(), 1);
        let manifest: Value = serde_json::from_slice(
            &fs::read(
                project
                    .path()
                    .join(summary["result_manifest_artifact_path"].as_str().unwrap()),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(manifest["numerical_evidence"]["root"].is_object());
        let meshes = fs::read_dir(project.path().join("artifacts/meshes"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".solver-mesh.cbor"))
            .collect::<Vec<_>>();
        assert_eq!(meshes.len(), 1);
        let mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
            &fs::read(&meshes[0]).unwrap(),
        )
        .unwrap();
        mesh.validate().unwrap();
        let order = if quadratic {
            runmat_meshing_core::ElementOrder::Tet10
        } else {
            runmat_meshing_core::ElementOrder::Tet4
        };
        assert!(mesh
            .topology
            .volume_elements
            .iter()
            .all(|e| e.order == order));
        // A new store after producer process exit: no live solver storage remains.
        let store = FilesystemAnalysisArtifactStore::new(project.path().join("artifacts"));
        let retained = store
            .load_run(run_ids[0].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            retained.run.identity.family(),
            runmat_analysis_core::PhysicsFamily::Acoustic
        );
        assert!(!retained.production_publishable);
        assert!(retained.thermal_results.is_none() && retained.modal_results.is_none());
        let pressure = retained
            .run
            .fields
            .iter()
            .find(|f| f.field_id == "acoustic.pressure")
            .unwrap();
        let values = pressure.as_host_f64().unwrap();
        assert_eq!(values.len(), 2 * mesh.topology.nodes.len());
        // Small-k limit in a unit cavity with p=1 on every wall; this is an execution
        // witness, not the independent duct/cavity convergence qualification gate.
        for p in values.chunks_exact(2) {
            assert!((p[0] - 1.0).abs() < 1e-5 && p[1].abs() < 1e-10, "{p:?}");
        }
        let velocity = retained
            .run
            .fields
            .iter()
            .find(|f| f.field_id == "acoustic.particle_velocity")
            .unwrap();
        assert_eq!(
            velocity.as_host_f64().unwrap().len(),
            6 * mesh.topology.volume_elements.len()
        );
        assert!(velocity
            .as_host_f64()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
        let catalog = &manifest["numerical_evidence"]["catalog"];
        assert!(catalog["objects"].as_array().unwrap().len() > 3);
        let object_paths = manifest["numerical_evidence"]["object_artifact_paths"]
            .as_array()
            .unwrap();
        assert_eq!(
            object_paths.len(),
            catalog["objects"].as_array().unwrap().len()
        );
        let mut cold = runmat_execution_artifact::cache::FilesystemObjectStore::open(
            project.path().join("cold-closure"),
            256 * 1024 * 1024,
        )
        .unwrap();
        for path in object_paths {
            let object: Value = serde_json::from_slice(
                &fs::read(project.path().join(path.as_str().unwrap())).unwrap(),
            )
            .unwrap();
            let bytes: Vec<u8> = serde_json::from_value(object["bytes"].clone()).unwrap();
            assert_eq!(
                runmat_execution::Digest::sha256(&bytes).to_string(),
                object["reference"]["logical_digest"].as_str().unwrap()
            );
            let reference: runmat_execution::value::ValueRef =
                serde_json::from_value(object["reference"].clone()).unwrap();
            let logical = runmat_execution_artifact::LogicalObject::new(
                runmat_execution_artifact::ObjectNamespace::ResultValue,
                format!("cold/{}", reference.logical_digest),
                reference.media_type,
                bytes,
            )
            .unwrap();
            runmat_execution_artifact::cache::CacheExport::write_verified(&mut cold, &logical)
                .unwrap();
        }
        let states = manifest["public_run"]["states"].as_array().unwrap();
        assert_eq!(states.len(), 1);
        let state = runmat_analysis_fea::execution::CanonicalSemanticState::from_canonical_bytes(
            &fs::read(
                project
                    .path()
                    .join(states[0]["artifact_path"].as_str().unwrap()),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(state.kind, runmat_analysis_core::StateKind::AcousticComplex);
        let acoustic = state.payload.acoustic().unwrap();
        assert_eq!(acoustic.frequency.hertz, 0.2);
        assert_eq!(acoustic.fields.fields()[0].storage(), pressure);
        assert_eq!(acoustic.fields.fields()[1].storage(), velocity);
        let root: runmat_execution::value::ValueRef =
            serde_json::from_value(manifest["numerical_evidence"]["root"].clone()).unwrap();
        let access = runmat_execution_artifact::ArtifactAccess {
            authorization_scope: root.authorization_scope.clone(),
            encryption_context: root.encryption_context,
        };
        let limits = runmat_execution_artifact::object::ObjectInventoryLimits::default();
        let completed = runmat_analysis_execution::import_acoustic_study_root(
            &cold,
            &root,
            root.logical_digest,
            &access,
            limits,
        )
        .unwrap();
        assert_eq!(completed.states[0].state, state);
        assert_eq!(completed.result.result.mesh_digest(), mesh.canonical_digest);
        assert!(runmat_analysis_execution::import_acoustic_study_root(
            &cold,
            &root,
            runmat_execution::Digest::sha256("foreign-root"),
            &access,
            limits,
        )
        .is_err());
        let (completion, _source) =
            runmat_execution_runner_native::acoustic_study_completion_from_source(completed, cold)
                .unwrap();
        assert_eq!(completion.states, vec![state]);
    }
}

const STUDY: &str = r#"
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
        order = linear
        target_size = 0.5 m
        maximum_deviation = 0.02 m
    }
}
material fluid {
    mass_density = 1.225 kg/m^3
    acoustic {
        speed_of_sound = 343 m/s
        damping_ratio = 0.0
    }
}
model cavity {
    assign material.fluid to geometry.part.region.body
    constraint pressure {
        acoustic.pressure {
            target = geometry.part.region.surface
            value = 1 Pa
        }
    }
}
study response {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.cavity
    formulation = experimental
    analysis sound { acoustic harmonic { frequencies = [0.2] Hz } }
}
"#;
