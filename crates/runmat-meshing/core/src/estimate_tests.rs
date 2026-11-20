use super::*;
use runmat_geometry_core::{PreviewMeasure, PreviewSurfacePatch};

#[test]
fn one_tetrahedron_counts_its_edges_faces_nodes_and_node_pairs() {
    let linear = SolidMeshCounts {
        vertices: 4,
        tetrahedra: 1,
        boundary_facets: 4,
        order: ElementOrder::Tet4,
    };
    assert_eq!((linear.edges(), linear.faces()), (6, 4));
    assert_eq!((linear.nodes(), linear.node_pairs()), (4, 6));
    let quadratic = SolidMeshCounts {
        order: ElementOrder::Tet10,
        ..linear
    };
    assert_eq!((quadratic.nodes(), quadratic.node_pairs()), (10, 45));
}

/// A 100 x 100 x 3 mm plate: two flat faces with 3 mm of material behind them.
fn plate() -> PreviewMeasure {
    let face = 0.1 * 0.1;
    PreviewMeasure {
        volume_m3: face * 0.003,
        area_m2: 2.0 * face,
        patches: vec![PreviewSurfacePatch {
            area_m2: 2.0 * face,
            radius_m: None,
            thickness_m: Some(0.003),
            gap_m: None,
        }],
    }
}

fn settings(target: f64, deviation: f64) -> MeshingRequestSettings {
    MeshingRequestSettings {
        target_edge_length_m: runmat_types::physical::Length::new(target),
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(deviation),
        ..MeshingRequestSettings::default()
    }
}

#[test]
fn thin_walls_bound_the_size_and_refinement_grows_the_count() {
    let estimate = |target| {
        estimate_solid_mesh_size(&plate(), &settings(target, 1.0e-4))
            .unwrap()
            .counts
            .tetrahedra
    };
    // Above the wall thickness the wall sets the size: a 10 mm and a 30 mm target agree.
    assert_eq!(estimate(0.010), estimate(0.030));
    // Below it the count grows with the inverse cube of the size.
    let ratio = estimate(0.0005) as f64 / estimate(0.001) as f64;
    assert!((6.0..=8.5).contains(&ratio), "{ratio}");
}

#[test]
fn curved_faces_refine_to_the_chordal_size() {
    let mut curved = plate();
    curved.patches[0].radius_m = Some(0.001);
    let count = |measure: &PreviewMeasure, deviation| {
        estimate_solid_mesh_size(measure, &settings(0.002, deviation))
            .unwrap()
            .counts
    };
    let flat = count(&plate(), 1.0e-5);
    let coarse = count(&curved, 1.0e-4);
    let fine = count(&curved, 1.0e-5);
    assert!(coarse.boundary_facets > flat.boundary_facets);
    assert!(fine.tetrahedra > coarse.tetrahedra);
    assert!(estimate_solid_mesh_size(&plate(), &settings(0.0, 1.0e-4)).is_none());
}

/// Production meshes of the bracket and bar fixtures (`runmat mesh`, release CLI of b6dfbf5cd7,
/// tetrahedra / vertices / boundary facets) against the estimate from its preview measure
/// (`runmat-geometry-io` test `preview_measure` writes the fixture).
#[test]
fn estimates_are_within_about_a_tenth_of_production_meshes() {
    let bracket: PreviewMeasure = serde_json::from_str(include_str!(
        "../tests/fixtures/camera_bracket.preview-measure.json"
    ))
    .unwrap();
    let bar: PreviewMeasure =
        serde_json::from_str(include_str!("../tests/fixtures/bar.preview-measure.json")).unwrap();
    let measured = [
        (&bracket, 10.0, 0.1, 29_843, 6_549, 7_238),
        (&bracket, 3.0, 0.1, 29_558, 6_500, 7_202),
        (&bracket, 2.0, 0.05, 47_043, 9_872, 9_692),
        (&bracket, 1.5, 0.05, 56_872, 11_898, 11_456),
        (&bracket, 1.0, 0.05, 124_318, 24_263, 18_722),
        (&bracket, 1.0, 0.01, 229_624, 43_264, 28_620),
        (&bracket, 0.7, 0.05, 318_636, 58_897, 35_208),
        (&bar, 3.0, 0.1, 23_996, 4_841, 4_044),
        (&bar, 1.5, 0.05, 188_695, 33_679, 16_092),
    ];
    for (measure, target_mm, deviation_mm, tetrahedra, vertices, facets) in measured {
        let mut settings = settings(target_mm * 1.0e-3, deviation_mm * 1.0e-3);
        settings.element_order = ElementOrder::Tet10;
        let estimate = estimate_solid_mesh_size(measure, &settings).unwrap().counts;
        let actual = SolidMeshCounts {
            vertices,
            tetrahedra,
            boundary_facets: facets,
            order: ElementOrder::Tet10,
        };
        let error = |estimated: u64, actual: u64| estimated as f64 / actual as f64 - 1.0;
        let elements = error(estimate.tetrahedra, tetrahedra);
        let nodes = error(estimate.nodes(), actual.nodes());
        eprintln!(
            "{target_mm} mm / {deviation_mm} mm: tetrahedra {} vs {tetrahedra} ({:+.1}%), tet10 nodes {} vs {} ({:+.1}%)",
            estimate.tetrahedra,
            100.0 * elements,
            estimate.nodes(),
            actual.nodes(),
            100.0 * nodes
        );
        assert!(
            (-0.09..=0.11).contains(&elements),
            "{target_mm} mm: {elements}"
        );
        assert!((-0.11..=0.12).contains(&nodes), "{target_mm} mm: {nodes}");
    }
    // A coarse mesh of a chunky part is overestimated, never underestimated: the bar at its
    // 10 mm default has 1,138 tetrahedra.
    let coarse = estimate_solid_mesh_size(&bar, &settings(0.010, 1.0e-4))
        .unwrap()
        .counts;
    assert!((1_138..=4_500).contains(&coarse.tetrahedra), "{coarse:?}");
    // The 1 mm quadratic mesh had 182,210 nodes; each of the six through-bores adds one edge
    // to Euler's count for a ball.
    assert_eq!(
        SolidMeshCounts {
            vertices: 24_263,
            tetrahedra: 124_318,
            boundary_facets: 18_722,
            order: ElementOrder::Tet10
        }
        .nodes()
            + 6,
        182_210
    );
}
