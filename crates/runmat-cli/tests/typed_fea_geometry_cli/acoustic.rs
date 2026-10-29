//! Actual authored CAD/process execution and post-exit shared artifact reopening.
use super::*;

#[path = "acoustic/duct.rs"]
mod duct;
#[path = "acoustic/response.rs"]
mod response;

#[test]
fn authored_acoustic_orders_publish_cold_native_fields_and_exact_state_catalog() {
    for (quadratic, frequencies) in [
        (false, vec![0.2]),
        (true, vec![0.2]),
        (false, vec![0.2, 0.3]),
        (true, vec![0.2, 0.3]),
    ] {
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
        let source = source.replace(
            "frequencies = [0.2]",
            if frequencies.len() == 1 {
                "frequencies = [0.2]"
            } else {
                "frequencies = [0.2, 0.3]"
            },
        );
        fs::write(project.path().join("src/main.fea"), source).unwrap();
        let output = runmat(
            &project,
            &[
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
        let run_id = run_ids[0].as_str().unwrap();
        assert!(store.load_run(run_id).unwrap().is_none());
        let retained = store.load_acoustic_frequency(run_id).unwrap().unwrap();
        assert_eq!(
            retained.identity.family(),
            runmat_analysis_core::PhysicsFamily::Acoustic
        );
        let runmat_runtime::analysis::AnalysisStudyStepNumericalResultData::AcousticFrequency(
            numerical,
        ) = &retained.numerical_result
        else {
            panic!("expected exact frequency result");
        };
        assert_eq!(numerical.entries.len(), frequencies.len());
        assert_eq!(numerical.publication.samples().len(), frequencies.len());
        let omitted = store
            .load_field_page(
                run_id,
                &runmat_runtime::analysis::AnalysisFieldSelection::new("acoustic.pressure", None),
                runmat_runtime::analysis::AnalysisFieldRequestOptions::default(),
            )
            .unwrap();
        if frequencies.len() > 1 {
            assert!(matches!(
                omitted,
                runmat_runtime::analysis::storage::AnalysisFieldPageLookup::SampleSelectionRequired(
                    _
                )
            ));
        } else {
            assert!(matches!(
                omitted,
                runmat_runtime::analysis::storage::AnalysisFieldPageLookup::Page(_)
            ));
        }
        assert!(
            matches!(store.load_field_page(run_id,
            &runmat_runtime::analysis::AnalysisFieldSelection::new("acoustic.pressure", Some(
                runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(999.0))),
            runmat_runtime::analysis::AnalysisFieldRequestOptions::default()).unwrap(),
            runmat_runtime::analysis::storage::AnalysisFieldPageLookup::SampleNotFound)
        );
        for frequency in &frequencies {
            let pressure = field_values(&store, run_id, "acoustic.pressure", *frequency);
            let velocity = field_values(&store, run_id, "acoustic.particle_velocity", *frequency);
            assert_eq!(pressure.len(), 2 * mesh.topology.nodes.len());
            assert_eq!(velocity.len(), 6 * mesh.topology.volume_elements.len());
            for p in pressure.chunks_exact(2) {
                assert!((p[0] - 1.0).abs() < 1e-5 && p[1].abs() < 1e-10, "{p:?}");
            }
            assert!(velocity.iter().all(|v| v.is_finite()));
        }
        let cold = crate::closure::reopen_catalog_objects(&project, &manifest);
        assert!(manifest["public_run"]["states"]
            .as_array()
            .unwrap()
            .is_empty());
        let states = numerical
            .entries
            .iter()
            .map(|entry| {
                assert_eq!(entry.states.len(), 1);
                // The run record keeps a reference; the state is persisted once as binary.
                let reference = &entry.states[0];
                reference.verify_identity().unwrap();
                let state =
                    runmat_analysis_fea::execution::CanonicalSemanticState::from_canonical_bytes(
                        &fs::read(project.path().join(&reference.artifact_path)).unwrap(),
                    )
                    .unwrap();
                assert_eq!(state.identity, reference.identity);
                assert_eq!(state.kind, runmat_analysis_core::StateKind::AcousticComplex);
                assert_eq!(
                    state
                        .payload
                        .acoustic()
                        .unwrap()
                        .frequency
                        .canonical_value(),
                    entry.entry.frequency_hz()
                );
                let payload = state.payload.acoustic().unwrap();
                assert_eq!(
                    payload.fields.fields()[0].storage().as_host_f64().unwrap(),
                    field_values(
                        &store,
                        run_id,
                        "acoustic.pressure",
                        entry.entry.frequency_hz()
                    )
                );
                assert_eq!(
                    payload.fields.fields()[1].storage().as_host_f64().unwrap(),
                    field_values(
                        &store,
                        run_id,
                        "acoustic.particle_velocity",
                        entry.entry.frequency_hz()
                    )
                );
                state
            })
            .collect::<Vec<_>>();
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
        assert_eq!(completed.entries.len(), frequencies.len());
        for ((entry, state), frequency) in completed.entries.iter().zip(&states).zip(&frequencies) {
            assert_eq!(entry.states[0].state, *state);
            assert_eq!(entry.result.result.mesh_digest(), mesh.canonical_digest);
            assert_eq!(entry.entry.frequency_hz(), *frequency);
        }
        // Replay consumes the exact saved entry/phase inventory; it never reruns a solve.
        let mut omitted = completed.entry_closures();
        omitted.pop();
        assert!(runmat_analysis_execution::import_acoustic_study_completion(
            &cold,
            completed.preparation().unwrap(),
            omitted,
            completed.assembly_closure.clone(),
            completed.wave_wall_millis,
        )
        .is_err());
        if frequencies.len() > 1 {
            let mut reversed = completed.entry_closures();
            reversed.reverse();
            assert!(runmat_analysis_execution::import_acoustic_study_completion(
                &cold,
                completed.preparation().unwrap(),
                reversed,
                completed.assembly_closure.clone(),
                completed.wave_wall_millis,
            )
            .is_err());
        }
        let mut missing_state = completed.entry_closures();
        let ordinal = missing_state[0]
            .closures
            .iter()
            .position(|closure| {
                matches!(
                    closure.phase,
                    runmat_analysis_execution::AcousticStudyResultPhase::State { .. }
                )
            })
            .unwrap();
        missing_state[0].closures.remove(ordinal);
        assert!(runmat_analysis_execution::import_acoustic_study_completion(
            &cold,
            completed.preparation().unwrap(),
            missing_state,
            completed.assembly_closure.clone(),
            completed.wave_wall_millis,
        )
        .is_err());
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
        assert_eq!(
            completion
                .entries
                .iter()
                .flat_map(|entry| entry.states.clone())
                .collect::<Vec<_>>(),
            states
        );
        assert_eq!(completion.publication.samples().len(), frequencies.len());
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
    analysis sound { acoustic harmonic { frequencies = [0.2] Hz } }
}
"#;

fn field_values(
    store: &FilesystemAnalysisArtifactStore,
    run_id: &str,
    field_id: &str,
    frequency: f64,
) -> Vec<f64> {
    let selection = runmat_runtime::analysis::AnalysisFieldSelection::new(
        field_id,
        Some(runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(frequency)),
    );
    let mut values = Vec::new();
    let mut offset = 0;
    let total = loop {
        let runmat_runtime::analysis::storage::AnalysisFieldPageLookup::Page(page) = store
            .load_field_page(
                run_id,
                &selection,
                runmat_runtime::analysis::AnalysisFieldRequestOptions {
                    offset: Some(offset),
                    limit: None,
                },
            )
            .unwrap()
        else {
            panic!("exact acoustic sample page missing");
        };
        assert_eq!(page.offset, offset);
        assert_eq!(
            page.count,
            runmat_runtime::analysis::ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT
                .min(page.total_count - offset)
        );
        assert_eq!(
            page.truncated,
            offset > 0 || offset + page.count < page.total_count
        );
        let runmat_runtime::analysis::AnalysisFieldPageValues::ComplexInterleaved(chunk) =
            page.values
        else {
            panic!("acoustic fields must preserve complex values");
        };
        assert_eq!(chunk.len(), 2 * page.count);
        values.extend(chunk);
        offset += page.count;
        if offset == page.total_count {
            break page.total_count;
        }
        assert!(page.count > 0);
    };
    let runmat_runtime::analysis::storage::AnalysisFieldPageLookup::Page(partial) = store
        .load_field_page(
            run_id,
            &selection,
            runmat_runtime::analysis::AnalysisFieldRequestOptions {
                offset: Some(1),
                limit: Some(7),
            },
        )
        .unwrap()
    else {
        panic!("exact acoustic partial page missing");
    };
    assert_eq!(
        (partial.offset, partial.count, partial.total_count),
        (1, 7, total)
    );
    assert!(partial.truncated);
    assert_eq!(
        partial.values,
        runmat_runtime::analysis::AnalysisFieldPageValues::ComplexInterleaved(
            values[2..16].to_vec()
        )
    );
    values
}
