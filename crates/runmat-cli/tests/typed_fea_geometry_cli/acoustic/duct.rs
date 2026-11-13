//! Authored matched duct through real CAD, native processes and cold public queries.
use super::*;
use runmat_runtime::analysis::storage::AnalysisFieldPageLookup;
use runmat_runtime::analysis::{
    AnalysisFieldPageValues, AnalysisFieldRequestOptions, AnalysisFieldSelection,
};

#[path = "duct/closure.rs"]
mod closure;
#[path = "duct/fixture.rs"]
mod fixture;
#[path = "duct/reference.rs"]
mod reference;

#[derive(Clone, Copy, Debug)]
enum Drive {
    NormalVelocity,
    IncidentPressure,
    VolumeVelocity,
}

struct Evidence {
    project: Option<TempDir>,
    pressure: Vec<f64>,
    velocity: Vec<f64>,
    mesh_digest: runmat_meshing_core::StableDigest,
    pressure_error: f64,
    velocity_error: f64,
    nodes: usize,
}

impl Drop for Evidence {
    fn drop(&mut self) {
        if std::thread::panicking() {
            if let Some(project) = self.project.take() {
                eprintln!(
                    "failed duct comparison/refinement retained_project={}",
                    project.keep().display()
                );
            }
        }
    }
}

#[test]
fn authored_matched_duct_refines_and_reopens_full_partial_complex_components() {
    qualify(Drive::NormalVelocity);
}

#[test]
fn authored_incident_duct_refines_and_reopens_full_partial_complex_components() {
    qualify(Drive::IncidentPressure);
}

#[test]
fn authored_volume_velocity_duct_refines_and_reopens_full_partial_complex_components() {
    qualify(Drive::VolumeVelocity);
}

fn qualify(drive: Drive) {
    // Eight actual processes: two orders, two independently remeshed target sizes,
    // generated/explicit complex drives. Not a nested mesh rate claim.
    // Exercise the previously rejecting order first; do not spend four producer runs
    // rediscovering that Tet4 alone cannot establish the all-order closure gate.
    for quadratic in [true, false] {
        let mut refinement = Vec::new();
        for size in [0.5, 0.25] {
            let generated = execute(drive, quadratic, size, true);
            let explicit = execute(drive, quadratic, size, false);
            assert_eq!(generated.mesh_digest, explicit.mesh_digest);
            for (left, right) in [
                (&generated.pressure, &explicit.pressure),
                (&generated.velocity, &explicit.velocity),
            ] {
                assert_eq!(left.len(), right.len());
                for (a, b) in left.iter().zip(right) {
                    assert!((a - b).abs() < 1e-11, "generated={a} explicit={b}");
                }
            }
            refinement.push(generated);
        }
        assert!(refinement[1].nodes > refinement[0].nodes);
        assert!(refinement[1].pressure_error < refinement[0].pressure_error);
        assert!(refinement[1].velocity_error < refinement[0].velocity_error);
        // Existing independent matched-wave finest bounds, fixed before any run.
        // Irregular OCCT meshes do not justify structured-slab rate predicates.
        assert!(refinement[1].pressure_error < 0.004);
        assert!(refinement[1].velocity_error < 0.01);
    }
}

fn execute(drive: Drive, quadratic: bool, size: f64, generated: bool) -> Evidence {
    let project = TempDir::new().unwrap();
    let evidence = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let imported = fixture::prepare(&project, drive, quadratic, size, generated);
        let output = runmat(
            &project,
            &["run", "src/main.fea", "--study", "response", "--json"],
        );
        assert!(
            output.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(summary["procedure"], "acoustic_harmonic");
        let ids = summary["step_run_ids"].as_array().unwrap();
        assert_eq!(ids.len(), 1);
        let run_id = ids[0].as_str().unwrap();
        let manifest: Value = serde_json::from_slice(
            &fs::read(
                project
                    .path()
                    .join(summary["result_manifest_artifact_path"].as_str().unwrap()),
            )
            .unwrap(),
        )
        .unwrap();
        let paths = fs::read_dir(project.path().join("artifacts/meshes"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.to_string_lossy().ends_with(".solver-mesh.cbor"))
            .collect::<Vec<_>>();
        assert_eq!(paths.len(), 1);
        let mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
            &fs::read(&paths[0]).unwrap(),
        )
        .unwrap();
        mesh.validate_canonical().unwrap();
        let order = if quadratic {
            runmat_meshing_core::ElementOrder::Tet10
        } else {
            runmat_meshing_core::ElementOrder::Tet4
        };
        assert!(!mesh.topology.volume_elements.is_empty());
        assert!(mesh
            .topology
            .volume_elements
            .iter()
            .all(|e| e.order == order));
        fixture::verify_faces(&mesh, &imported);
        let store = FilesystemAnalysisArtifactStore::new(project.path().join("artifacts"));
        let retained = store.load_acoustic_frequency(run_id).unwrap().unwrap();
        assert!(
            retained.numerically_acceptable,
            "{:?}",
            retained.quality_reasons
        );
        assert_eq!(
            retained.provenance.solver_method,
            runmat_analysis_fea::contracts::SolverMethod::AcousticCanonicalHelmholtzCertifiedLu
        );
        assert_eq!(
            retained.provenance.preconditioner,
            runmat_analysis_fea::contracts::SolverPreconditioner::None
        );
        let pressure = full_and_partial(&store, run_id, "acoustic.pressure");
        let velocity = full_and_partial(&store, run_id, "acoustic.particle_velocity");
        assert_eq!(pressure.len(), 2 * mesh.topology.nodes.len());
        assert_eq!(velocity.len(), 6 * mesh.topology.volume_elements.len());
        let completed = closure::reopen(&project, &manifest, &retained, &mesh);
        let entry = &completed.entries[0];
        assert_eq!(
            entry.workload.solver,
            runmat_analysis_fea::physics::acoustic::AcousticHarmonicPolicy {
                relative_tolerance: 1e-9,
                ..Default::default()
            }
        );
        fixture::verify_medium(&entry.workload.model, &mesh);
        fixture::verify_drive(&entry.workload.model, drive, generated);
        let result = &entry.result.result;
        let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
        assert!(result
            .ordered_node_ids()
            .iter()
            .copied()
            .eq(projection.nodes.iter().map(|n| n.node_id)));
        assert!(result
            .ordered_element_ids()
            .iter()
            .copied()
            .eq(projection.elements.iter().map(|e| e.element_id)));
        assert_eq!(result.frequency().canonical_value(), 0.2);
        assert_eq!(result.source_evidence().len(), 1);
        assert!(result.source_evidence()[0].accepted);
        assert!(result.relative_residual() <= 1e-8);
        assert!(result.power_relative_error() < 1e-8);
        // These are the retained numerical result's advertised signed boundary powers,
        // independently compared with peak-phasor continuum flux |A|²/(2 rho c).
        let data = serde_json::to_value(result).unwrap();
        let exact_power = 1.16 / 12.0;
        for key in ["mean_input_power_w", "mean_impedance_loss_w"] {
            let power = data[key].as_f64().unwrap();
            assert!(
                power > 0.0 && (power - exact_power).abs() < 0.005,
                "{key}={power}"
            );
        }
        let (pressure_error, velocity_error, intensity_error) = reference::errors(
            &projection,
            &pressure,
            &velocity,
            result.intensity_w_per_m2(),
        );
        assert!(pressure_error.is_finite() && velocity_error.is_finite());
        if size == 0.25 {
            assert!(intensity_error < 0.005);
        }
        eprintln!("authored matched duct drive={drive:?} quadratic={quadratic} size={size} generated={generated} nodes={} elements={} pressure={pressure_error} velocity={velocity_error} intensity={intensity_error}", projection.nodes.len(), projection.elements.len());
        Evidence {
            project: None,
            pressure,
            velocity,
            mesh_digest: mesh.canonical_digest,
            pressure_error,
            velocity_error,
            nodes: projection.nodes.len(),
        }
    }));
    match evidence {
        Ok(mut evidence) => {
            evidence.project = Some(project);
            evidence
        }
        Err(error) => {
            let path = project.keep();
            eprintln!("failed authored matched duct retained_project={} quadratic={quadratic} size={size} generated={generated}", path.display());
            std::panic::resume_unwind(error)
        }
    }
}

fn full_and_partial(
    store: &FilesystemAnalysisArtifactStore,
    run_id: &str,
    field_id: &str,
) -> Vec<f64> {
    let selection = AnalysisFieldSelection::new(field_id, None);
    let mut values = Vec::new();
    let mut offset = 0;
    let total = loop {
        let AnalysisFieldPageLookup::Page(page) = store
            .load_field_page(
                run_id,
                &selection,
                AnalysisFieldRequestOptions {
                    offset: Some(offset),
                    limit: None,
                },
            )
            .unwrap()
        else {
            panic!("missing cold packed field {field_id}");
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
        let AnalysisFieldPageValues::ComplexInterleaved(chunk) = page.values else {
            panic!("native acoustic public contract preserves complex entity values");
        };
        assert_eq!(chunk.len(), 2 * page.count);
        values.extend(chunk);
        offset += page.count;
        if offset == page.total_count {
            break page.total_count;
        }
        assert!(page.count > 0);
    };
    let AnalysisFieldPageLookup::Page(partial) = store
        .load_field_page(
            run_id,
            &selection,
            AnalysisFieldRequestOptions {
                offset: Some(1),
                limit: Some(7),
            },
        )
        .unwrap()
    else {
        panic!("missing partial field");
    };
    assert_eq!(
        (partial.offset, partial.count, partial.total_count),
        (1, 7, total)
    );
    assert!(partial.truncated);
    // Starts at an exact logical component and reads both phasor components.
    assert_eq!(
        partial.values,
        AnalysisFieldPageValues::ComplexInterleaved(values[2..16].to_vec())
    );
    values
}
