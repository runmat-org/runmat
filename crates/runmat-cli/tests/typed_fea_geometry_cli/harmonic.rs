//! Authored complex sources through real CAD/process execution and cold sampled fields.
//! Reference wave equation/stress law: UT Austin, equations 5.11–5.16:
//! https://farside.ph.utexas.edu/teaching/315/Waves/node34.html
//! Imposing u(0)=0 and E*u'(1)=T gives u=T*sin(k*x)/(E*k*cos(k)).
use super::*;
use runmat_geometry_ops::{
    GeometryFeatureProvider, GeometryFeatureProviderSelection, ResolvedGeometryFeatureOperation,
};
use runmat_runtime::analysis::storage::AnalysisFieldPageLookup;
use runmat_runtime::analysis::{
    AnalysisFieldPageValues, AnalysisFieldRequestOptions, AnalysisFieldSelection,
    AnalysisStudyStepNumericalResultData,
};

#[test]
fn authored_harmonic_generated_and_explicit_sources_preserve_rod_response_and_phase() {
    for quadratic in [true, false] {
        let generated = execute(quadratic, true);
        let explicit = execute(quadratic, false);
        assert_eq!(explicit.len(), generated.len());
        for (left, right) in explicit.iter().zip(&generated) {
            assert_eq!(left.len(), right.len());
            for (a, b) in left.iter().zip(right) {
                assert!((a - b).abs() < 1e-12, "explicit={a} generated={b}");
            }
        }
    }
}

fn execute(quadratic: bool, generated: bool) -> Vec<Vec<f64>> {
    let project = TempDir::new().unwrap();
    fs::create_dir(project.path().join("src")).unwrap();
    fs::write(project.path().join("runmat.toml"),
        "[package]\nname = \"harmonic-rod\"\n[sources]\nroots = [\"src\"]\n[runtime.fea]\nartifact_root = \"artifacts\"\nstudy_artifact_root = \"study-artifacts\"\n").unwrap();
    let provider = runmat_geometry_io::feature::OcctExactFeatureProvider::new(
        runmat_geometry_io::exact::ExactCadImportOptions::default(),
        runmat_geometry_io::GeometryImportContext::new(),
    );
    let unit_box = provider
        .execute(
            "unit-box",
            &ResolvedGeometryFeatureOperation::Box {
                width_m: runmat_types::physical::Length::new(1.0),
                depth_m: runmat_types::physical::Length::new(1.0),
                height_m: runmat_types::physical::Length::new(1.0),
            },
            &GeometryFeatureProviderSelection::None,
            &[],
            None,
        )
        .unwrap()
        .geometry;
    fs::write(
        project.path().join("src/unit-box.brep"),
        &unit_box.representation,
    )
    .unwrap();
    let imported = runmat_geometry_io::exact::import_exact_cad(
        "unit-box.brep",
        &unit_box.representation,
        runmat_geometry_io::GeometryFormat::Brep,
        &runmat_geometry_io::exact::ExactCadImportOptions {
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            ..Default::default()
        },
        &runmat_geometry_io::GeometryImportContext::new(),
    )
    .unwrap();
    fs::write(
        project.path().join("src/drive.m"),
        "function value = drive(frequency)\narguments\nfrequency (:,1) double\nend\nvalue = (1 + 0.4i) * frequency;\nend\n",
    )
    .unwrap();
    let source = STUDY.replace("SOURCE", if generated { GENERATED } else { EXPLICIT });
    let source = if quadratic {
        source.replace("order = linear", "order = quadratic")
    } else {
        source
    };
    let source = source
        .replace(
            "FIRST_FACE_ID",
            &imported.source_face_ids[&0]
                .analysis_selection_id()
                .unwrap(),
        )
        .replace(
            "SECOND_FACE_ID",
            &imported.source_face_ids[&1]
                .analysis_selection_id()
                .unwrap(),
        );
    fs::write(project.path().join("src/main.fea"), source).unwrap();
    let output = runmat(
        &project,
        &["run", "src/main.fea", "--study", "response", "--json"],
    );
    if !output.status.success() {
        let evidence_path = project.keep();
        panic!(
            "quadratic={quadratic} generated={generated}: retained_project={} stdout={} stderr={}",
            evidence_path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["procedure"], "structural_frequency");
    let runs = summary["step_run_ids"].as_array().unwrap();
    assert_eq!(runs.len(), 1);
    let run_id = runs[0].as_str().unwrap();
    let manifest_path = summary["result_manifest_artifact_path"].as_str().unwrap();
    let manifest: Value =
        serde_json::from_slice(&fs::read(project.path().join(manifest_path)).unwrap()).unwrap();
    assert!(manifest.is_object(), "persisted study manifest must reopen");
    let mesh_paths = fs::read_dir(project.path().join("artifacts/meshes"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.to_string_lossy().ends_with(".solver-mesh.cbor"))
        .collect::<Vec<_>>();
    assert_eq!(mesh_paths.len(), 1);
    let mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
        &fs::read(&mesh_paths[0]).unwrap(),
    )
    .unwrap();
    mesh.validate_canonical().unwrap();
    let expected_order = if quadratic {
        runmat_meshing_core::ElementOrder::Tet10
    } else {
        runmat_meshing_core::ElementOrder::Tet4
    };
    assert!(!mesh.topology.volume_elements.is_empty());
    assert!(mesh
        .topology
        .volume_elements
        .iter()
        .all(|element| element.order == expected_order));
    // These are the exact imported source ordinals used by face_000001/face_000002,
    // not merely two end planes that happen to exist somewhere in the mesh.
    // Raw triangle winding is not an outward-normal contract: the analysis projection
    // orients normals using the owning volume, as pressure/normal-velocity operators do.
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
    for (ordinal, x, normal_x) in [(0, 0.0, -1.0), (1, 1.0, 1.0)] {
        let identity = &imported.source_face_ids[&ordinal];
        let mut area = 0.0;
        let mut count = 0;
        for face in &mesh.topology.boundary_faces {
            if !face.provenance.contains(identity) {
                continue;
            }
            count += 1;
            let points = face
                .node_ids
                .iter()
                .map(|id| {
                    let node = mesh
                        .topology
                        .nodes
                        .iter()
                        .find(|node| node.node_id == *id)
                        .unwrap();
                    assert!(
                        (node.coordinates_m.canonical_value()[0] - x).abs() < 1e-12,
                        "source ordinal {ordinal}: {:?}",
                        node.coordinates_m.canonical_value()
                    );
                    node.coordinates_m.canonical_value()
                })
                .collect::<Vec<_>>();
            let a = points[0];
            let b = points[1];
            let c = points[2];
            let y1 = b[1] - a[1];
            let z1 = b[2] - a[2];
            let y2 = c[1] - a[1];
            let z2 = c[2] - a[2];
            let signed_area = 0.5 * (y1 * z2 - z1 * y2);
            let projected = projection
                .boundary_faces
                .iter()
                .find(|projected| projected.face_id == face.face_id)
                .unwrap();
            assert!(
                projected.oriented_area_vector_m2[0] * normal_x > 0.0,
                "source ordinal {ordinal}: wrong projected outward normal"
            );
            area += signed_area.abs();
        }
        assert!(
            count > 0,
            "source ordinal {ordinal} absent from persisted mesh provenance"
        );
        assert!(
            (area - 1.0).abs() < 1e-12,
            "source ordinal {ordinal}: area={area}"
        );
    }
    // Mesh provenance uses persistent CAD identities, not authored source-alias strings.
    // Check both physical end planes; the full nodal reference below also verifies
    // the applied clamp/load orientation rather than accepting an arbitrary alias order.
    for x in [0.0, 1.0] {
        let faces = mesh
            .topology
            .boundary_faces
            .iter()
            .filter(|face| {
                face.node_ids.iter().all(|id| {
                    let node = mesh
                        .topology
                        .nodes
                        .iter()
                        .find(|node| node.node_id == *id)
                        .unwrap();
                    (node.coordinates_m.canonical_value()[0] - x).abs() < 1e-12
                })
            })
            .collect::<Vec<_>>();
        assert!(!faces.is_empty(), "missing end plane x={x}");
        for face in faces {
            for id in &face.node_ids {
                let node = mesh
                    .topology
                    .nodes
                    .iter()
                    .find(|node| node.node_id == *id)
                    .unwrap();
                assert!(
                    (node.coordinates_m.canonical_value()[0] - x).abs() < 1e-12,
                    "end plane x={x}: {:?}",
                    node.coordinates_m.canonical_value()
                );
            }
        }
    }
    // This store is constructed only after the producer process has exited.
    let store = FilesystemAnalysisArtifactStore::new(project.path().join("artifacts"));
    let retained = store.load_structural_frequency(run_id).unwrap().unwrap();
    assert!(
        retained.numerically_acceptable,
        "{:?}",
        retained.quality_reasons
    );
    let AnalysisStudyStepNumericalResultData::StructuralFrequency(result) =
        retained.numerical_result
    else {
        panic!("expected retained harmonic publication");
    };
    assert_eq!(result.publication.samples().len(), 2);
    let mut fields = Vec::new();
    for sample in result.publication.samples() {
        let field = sample
            .fields
            .fields
            .iter()
            .find(|field| {
                field.definition.quantity()
                    == runmat_analysis_core::ObservationQuantity::Displacement
            })
            .unwrap();
        let selection =
            AnalysisFieldSelection::new(field.definition.field_id(), Some(sample.coordinate));
        assert!(matches!(
            store
                .load_field_page(
                    run_id,
                    &AnalysisFieldSelection::new(field.definition.field_id(), None),
                    AnalysisFieldRequestOptions::default(),
                )
                .unwrap(),
            AnalysisFieldPageLookup::SampleSelectionRequired(_)
        ));
        // Default materialization is deliberately bounded. Reopen every logical page
        // rather than assuming a cold field fits in one page or bypassing that contract.
        let total_count = 3 * mesh.topology.nodes.len();
        let mut values = Vec::with_capacity(2 * total_count);
        let mut offset = 0;
        while offset < total_count {
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
                panic!("missing cold displacement page at {offset}");
            };
            assert_eq!(page.offset, offset);
            assert_eq!(page.total_count, total_count);
            assert_eq!(
                page.count,
                runmat_runtime::analysis::ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT
                    .min(total_count - offset)
            );
            assert!(page.count > 0);
            assert_eq!(
                page.truncated,
                offset > 0 || offset + page.count < total_count
            );
            let AnalysisFieldPageValues::ComplexInterleaved(page_values) = page.values else {
                panic!("harmonic displacement must preserve complex values");
            };
            assert_eq!(page_values.len(), 2 * page.count);
            values.extend(page_values);
            offset += page.count;
        }
        assert_eq!(values.len(), 2 * total_count);
        let AnalysisFieldPageLookup::Page(partial) = store
            .load_field_page(
                run_id,
                &selection,
                AnalysisFieldRequestOptions {
                    offset: Some(2),
                    limit: Some(7),
                },
            )
            .unwrap()
        else {
            panic!("missing partial sampled field");
        };
        assert_eq!(partial.offset, 2);
        assert_eq!(partial.count, 7);
        assert_eq!(partial.total_count, 3 * mesh.topology.nodes.len());
        assert!(partial.truncated);
        assert_eq!(
            partial.values,
            AnalysisFieldPageValues::ComplexInterleaved(values[4..18].to_vec())
        );
        let frequency = sample.result.plan_entry().frequency_hz();
        let k = std::f64::consts::TAU * frequency / 2.0;
        // Independent nu=0 three-dimensional rod: E=4, rho=A=L=1,
        // fixed x=0, traction T=f*(1+0.4i) at x=1, u=T*sin(k*x)/(E*k*cos(k)).
        // Bounds fixed before execution; this is not the high-frequency convergence gate.
        let amplitude = frequency / (4.0 * k * k.cos());
        let scale = amplitude * k.sin();
        let mut squared_error = 0.0;
        let mut squared_reference = 0.0;
        for (node, u) in mesh.topology.nodes.iter().zip(values.chunks_exact(6)) {
            let reference = amplitude * (k * node.coordinates_m.canonical_value()[0]).sin();
            squared_error += (u[0] - reference).powi(2)
                + (u[1] - 0.4 * reference).powi(2)
                + u[2..].iter().map(|v| v * v).sum::<f64>();
            squared_reference += 1.16 * reference.powi(2);
            assert!(
                (u[1] - 0.4 * u[0]).abs() < 1e-10 * scale,
                "complex source phase: {u:?}"
            );
        }
        let relative_error = (squared_error / squared_reference).sqrt();
        eprintln!(
            "harmonic rod: quadratic={quadratic} generated={generated} nodes={} f={frequency} relative_error={relative_error}",
            mesh.topology.nodes.len()
        );
        assert!(
            relative_error < if quadratic { 0.005 } else { 0.02 },
            "quadratic={quadratic} generated={generated} f={frequency}: error={relative_error}"
        );
        fields.push(values);
    }
    fields
}

const EXPLICIT: &str = r#"
samples = [
    { at = 0.07957747154594767 Hz, value = { real = 0.07957747154594767 N, imaginary = 0.03183098861837907 N } },
    { at = 0.15915494309189535 Hz, value = { real = 0.15915494309189535 N, imaginary = 0.06366197723675814 N } }
]
interpolation = linear
extrapolation = reject
"#;

const GENERATED: &str = r#"
generated by function.drive {
    arguments {}
    sample_points = [0.07957747154594767 Hz, 0.15915494309189535 Hz]
    precision = complex128
    interpolation = linear
    extrapolation = reject
}
"#;

const STUDY: &str = r#"
import function drive from "./drive.m"
geometry part {
    file = "./unit-box.brep"
    units = m
    region body : solid {
        select result
        expect exactly_one
    }
    region support : face {
        select id "FIRST_FACE_ID"
        expect exactly_one
    }
    region loaded_face : face {
        select id "SECOND_FACE_ID"
        expect exactly_one
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.25 m
        maximum_deviation = 0.02 m
    }
}
material rod {
    mass_density = 1 kg/m^3
    elastic {
        youngs_modulus = 4 Pa
        poisson_ratio = 0
    }
}
model structure {
    assign material.rod to geometry.part.region.body
    constraint clamp { structural.fixed { target = geometry.part.region.support } }
    signal phasor : force over frequency { SOURCE }
    excitation end_force {
        force {
            target = geometry.part.region.loaded_face
            signal = signal.phasor
            direction = global([1, 0, 0])
            kinematics = fixed_frame
        }
    }
    load_case driven { include excitation.end_force }
}
study response {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis harmonic {
        structural harmonic { frequencies = [0.07957747154594767 Hz, 0.15915494309189535 Hz] }
    }
    output displacement = analysis.harmonic.field.node.displacement
}
"#;
