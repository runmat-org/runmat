//! Actual authored static electrical CAD/process/cold-result chain, not kernel transport fixtures.
use super::*;
use runmat_runtime::analysis::storage::AnalysisFieldPageLookup;
use runmat_runtime::analysis::{
    AnalysisFieldPageValues, AnalysisFieldRequestOptions, AnalysisFieldSelection,
};

#[path = "electrical/charge_reference.rs"]
mod charge_reference;
#[path = "electrical/closure.rs"]
mod closure;
#[path = "electrical/fixture.rs"]
mod fixture;
#[path = "electrical/reference.rs"]
mod reference;

#[test]
fn authored_charged_poisson_tet10_signed_gauge_and_cold_publication() {
    for sign in [-1.0, 1.0] {
        execute_with_charge(true, "electrostatics", 1000.0, 0.0, Some(sign));
    }
}

#[test]
fn authored_electrical_prism_both_equations_orders_sign_and_gauge_reopen_cold() {
    // Eight actual producer processes. Identical CAD/mesh policy in the signed/gauged
    // pair makes comparisons attributable to potential data, not mesh changes.
    for quadratic in [true, false] {
        for equation in ["electrostatics", "steady_conduction"] {
            let positive = execute(quadratic, equation, 0.0, 2.0);
            let negative_gauged = execute(quadratic, equation, 5.0, -2.0);
            assert_eq!(positive.0, negative_gauged.0);
            for (a, b) in positive.1[0].iter().zip(&negative_gauged.1[0]) {
                assert!((a + b - 5.0).abs() <= 2e-7);
            }
            for index in [1, 2] {
                let scale = positive.1[index]
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max);
                assert!(scale > 0.0);
                for (a, b) in positive.1[index].iter().zip(&negative_gauged.1[index]) {
                    assert!((a + b).abs() / scale <= 1e-7);
                }
            }
            for (a, b) in positive.1[3].iter().zip(&negative_gauged.1[3]) {
                assert!((a - b).abs() / a.abs() <= 1e-7);
            }
        }
    }
}

fn execute(
    quadratic: bool,
    equation: &str,
    gauge: f64,
    voltage: f64,
) -> (runmat_meshing_core::StableDigest, [Vec<f64>; 4]) {
    execute_with_charge(quadratic, equation, gauge, voltage, None)
}

fn execute_with_charge(
    quadratic: bool,
    equation: &str,
    gauge: f64,
    voltage: f64,
    charge_sign: Option<f64>,
) -> (runmat_meshing_core::StableDigest, [Vec<f64>; 4]) {
    let project = TempDir::new().unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let faces = fixture::prepare(&project, quadratic, equation, gauge, voltage);
        if let Some(sign) = charge_sign {
            let path = project.path().join("src/main.fea");
            let source = fs::read_to_string(&path).unwrap();
            let rho = sign * 8.0 * 8.854_187_818_8e-12;
            let charge = format!("model electric {{\nload bulk_charge {{ electric_charge_density on geometry.part.region.body = {rho:.17e} C/m^3 }}\nload_case charge {{ include load.bulk_charge }}");
            fs::write(path, source.replace("model electric {", &charge)).unwrap();
        }
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
            "stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(summary["procedure"], equation);
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
        fixture::verify_faces(&mesh, &faces);
        // New filesystem store; no producer in-memory arrays serve these reads.
        let store = FilesystemAnalysisArtifactStore::new(project.path().join("artifacts"));
        let retained = store.load_run(run_id).unwrap().unwrap();
        assert!(
            retained.numerically_acceptable,
            "{:?}",
            retained.quality_reasons
        );
        let (flux, density) = if equation == "electrostatics" {
            (
                "electrical.electric_displacement",
                "electrical.electric_energy_density",
            )
        } else {
            (
                "electrical.current_density",
                "electrical.joule_power_density",
            )
        };
        let names = [
            "electrical.potential",
            "electrical.electric_field",
            flux,
            density,
        ];
        let fields = names.map(|id| full_and_partial(&store, run_id, id));
        let shapes = [
            vec![mesh.topology.nodes.len()],
            vec![mesh.topology.volume_elements.len(), 3],
            vec![mesh.topology.volume_elements.len(), 3],
            vec![mesh.topology.volume_elements.len()],
        ];
        assert_eq!(retained.run.fields.len(), 4);
        for index in 0..4 {
            let field = retained
                .run
                .fields
                .iter()
                .find(|field| field.field_id == names[index])
                .unwrap();
            assert_eq!(field.shape, shapes[index]);
            assert_eq!(field.as_host_f64().unwrap(), &fields[index]);
        }
        let completion = closure::reopen(&project, &manifest, &retained, &mesh);
        if let Some(sign) = charge_sign {
            charge_reference::verify(&completion, &mesh, gauge, sign, &fields);
        } else {
            reference::verify(&completion, &mesh, gauge, voltage, &fields);
        }
        eprintln!("authored electrical equation={equation} quadratic={quadratic} gauge={gauge} voltage={voltage} nodes={} elements={} residual={} flux={} energy={}",
            mesh.topology.nodes.len(), mesh.topology.volume_elements.len(),
            completion.result.response().linear_relative_residual,
            completion.result.response().relative_flux_imbalance,
            completion.result.response().relative_energy_identity_error);
        (mesh.canonical_digest, fields)
    }));
    match result {
        Ok(evidence) => evidence,
        Err(error) => {
            eprintln!("failed electrical CAD qualification retained_project={} equation={equation} quadratic={quadratic} gauge={gauge} voltage={voltage}", project.keep().display());
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
    loop {
        let offset = values.len();
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
            panic!("missing cold native field {field_id}");
        };
        assert_eq!(page.offset, offset);
        assert_eq!(
            page.count,
            runmat_runtime::analysis::ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT
                .min(page.total_count - offset)
        );
        let AnalysisFieldPageValues::Real(chunk) = page.values else {
            panic!("electrical field must be real");
        };
        assert_eq!(chunk.len(), page.count);
        assert_eq!(
            page.truncated,
            offset > 0 || offset + page.count < page.total_count
        );
        values.extend(chunk);
        if values.len() == page.total_count {
            break;
        }
        assert!(page.count > 0);
    }
    assert!(values.len() >= 8);
    let AnalysisFieldPageLookup::Page(page) = store
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
        panic!("missing partial native field");
    };
    assert_eq!(
        (page.offset, page.count, page.total_count),
        (1, 7, values.len())
    );
    assert!(page.truncated);
    assert_eq!(
        page.values,
        AnalysisFieldPageValues::Real(values[1..8].to_vec())
    );
    values
}
