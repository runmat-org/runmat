use super::{super::AnalysisProjection, geometry, *};
use runmat_meshing_core::{fixtures, ElementOrder, StableDigest};

fn policy() -> FiniteVolumeProjectionPolicy {
    FiniteVolumeProjectionPolicy {
        max_cells: Some(10000),
        max_face_incidences: Some(40000),
        max_boundary_faces: Some(40000),
        max_scratch_bytes: Some(100_000_000),
        max_traversal_operations: Some(100_000_000),
    }
}

fn projected(order: ElementOrder) -> AnalysisProjection {
    AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order)).unwrap()
}

fn two_cells() -> AnalysisProjection {
    // Two unit right tetrahedra share z=0; exact volumes are 1/6 each.
    let mut source = projected(ElementOrder::Tet4);
    let mut node = source.nodes[3].clone();
    node.coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([0.0, 0.0, -1.0]);
    node.node_id = 5;
    node.stable_identity = StableDigest::from_bytes([231; 32]);
    source.nodes.push(node);
    let mut second = source.elements[0].clone();
    second.node_indices = vec![0, 2, 1, 4];
    second.element_id = 2;
    second.stable_identity = StableDigest::from_bytes([232; 32]);
    second.centroid_m = [0.25, 0.25, -0.25];
    source.elements.push(second);
    let template = source.boundary_faces[0].clone();
    source.boundary_faces.clear();
    let mut candidates =
        std::collections::BTreeMap::<[StableDigest; 3], (usize, [usize; 3], usize)>::new();
    for (i, element) in source.elements.iter().enumerate() {
        for local in geometry::FACE_CORNERS {
            let nodes = local.map(|j| element.node_indices[j]);
            let key = geometry::key(&source, &nodes).unwrap();
            candidates
                .entry(key)
                .and_modify(|v| v.2 += 1)
                .or_insert((i, nodes, 1));
        }
    }
    for (key, (cell, nodes, count)) in candidates {
        if count == 2 {
            continue;
        }
        let mut face = template.clone();
        face.node_indices = nodes.to_vec();
        face.adjacent_element_indices = vec![cell];
        face.face_id = source.boundary_faces.len() as u64 + 100;
        face.stable_identity = StableDigest::from_bytes(
            runmat_execution::Digest::sha256_parts(key.map(|k| k.0))
                .bytes()
                .to_owned(),
        );
        let (centroid, area, magnitude) = geometry::face_geometry(
            nodes.map(|n| source.nodes[n].coordinates_m.canonical_value()),
            source.elements[cell].centroid_m,
        )
        .unwrap();
        face.centroid_m = centroid;
        face.oriented_area_vector_m2 = area;
        face.area_m2 = runmat_types::physical::Area::new(magnitude);
        source.boundary_faces.push(face);
    }
    source
}

// witness: FEA-FLUID-TOPOLOGY-001/inventory @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Two affine Tet4 cells; oriented surface closure, volume and original face mapping. No
//   skew/curved mesh envelope proof.
#[test]
fn exact_two_cell_geometry_is_conservative_and_retains_original_boundary_mapping() {
    let source = two_cells();
    let fv = FiniteVolumeTopology::derive(&source, policy()).unwrap();
    assert_eq!(fv.cells.len(), 2);
    assert_eq!(fv.faces.len(), 7);
    let internal = fv.faces.iter().find(|f| f.neighbor.is_some()).unwrap();
    assert_eq!(internal.area_m2.canonical_value(), 0.5);
    assert_eq!(internal.centroid_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    assert!((fv.cells.iter().map(|c| c.volume_m3).sum::<f64>() - 1.0 / 3.0).abs() < 1e-15);
    for (i, cell) in fv.cells.iter().enumerate() {
        assert!((cell.volume_m3 - 1.0 / 6.0).abs() < 1e-15);
        let mut sum = [0.0; 3];
        for incidence in cell.faces {
            let face = &fv.faces[incidence.face_index];
            assert_eq!(incidence.sign, if face.owner == i { 1 } else { -1 });
            for (a, b) in sum.iter_mut().zip(face.owner_area_vector_m2) {
                *a += f64::from(incidence.sign) * b;
            }
        }
        assert_eq!(sum, [0.0; 3]);
    }
    for face in fv.faces.iter().filter(|f| f.neighbor.is_none()) {
        let original =
            &source.boundary_faces[face.boundary_source.unwrap().projection_boundary_index];
        assert_eq!(original.provenance, source.boundary_faces[0].provenance);
        assert_eq!(
            original.stable_identity,
            face.boundary_source.unwrap().face_identity
        );
    }
    fv.validate_for(&source).unwrap();
}

#[test]
fn refined_cube_exact_volume_and_both_affine_orders_have_closed_faces() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for refinement in [1, 2, 3] {
            let mesh = if order == ElementOrder::Tet4 {
                fixtures::slab_mesh(refinement)
            } else {
                fixtures::quadratic_slab_mesh(refinement)
            };
            let source = AnalysisProjection::derive(&mesh).unwrap();
            let fv = FiniteVolumeTopology::derive(&source, policy()).unwrap();
            assert!((fv.cells.iter().map(|c| c.volume_m3).sum::<f64>() - 1.0).abs() < 1e-13);
            let outer = fv
                .faces
                .iter()
                .filter(|f| f.neighbor.is_none())
                .map(|f| f.area_m2.canonical_value())
                .sum::<f64>();
            assert!((outer - 6.0).abs() < 1e-13);
        }
    }
}

// witness: FEA-FLUID-TOPOLOGY-001/geometry @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Two-cell source access reordering, not execution placement equivalence.
#[test]
fn stable_cell_face_physics_is_invariant_to_reordered_source_access_indices() {
    let source = two_cells();
    let expected = FiniteVolumeTopology::derive(&source, policy()).unwrap();
    let mut shuffled = source.clone();
    shuffled.elements.reverse();
    shuffled.boundary_faces.reverse();
    for face in &mut shuffled.boundary_faces {
        for cell in &mut face.adjacent_element_indices {
            *cell = 1 - *cell;
        }
    }
    // Also reorder nodes, adjusting every original access index.
    shuffled.nodes.reverse();
    let n = shuffled.nodes.len();
    for element in &mut shuffled.elements {
        for node in &mut element.node_indices {
            *node = n - 1 - *node;
        }
    }
    for face in &mut shuffled.boundary_faces {
        for node in &mut face.node_indices {
            *node = n - 1 - *node;
        }
    }
    let actual = FiniteVolumeTopology::derive(&shuffled, policy()).unwrap();
    assert_eq!(actual.geometry_digest, expected.geometry_digest);
    for (a, b) in actual.cells.iter().zip(&expected.cells) {
        assert_eq!(a.element_identity, b.element_identity);
        assert_eq!(a.faces, b.faces);
        assert_eq!(a.centroid_m, b.centroid_m);
    }
    for (a, b) in actual.faces.iter().zip(&expected.faces) {
        assert_eq!(a.corner_identities, b.corner_identities);
        assert_eq!(a.owner, b.owner);
        assert_eq!(a.neighbor, b.neighbor);
        assert_eq!(a.owner_area_vector_m2, b.owner_area_vector_m2);
    }
}

#[test]
fn rigid_rotation_tiny_scale_and_translation_preserve_vector_geometry() {
    let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for node in &mut mesh.topology.nodes {
        let [x, y, z] = node.coordinates_m.canonical_value();
        node.coordinates_m = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new([-y * 1e-6 + 2e-6, x * 1e-6 - 3e-6, z * 1e-6 + 4e-6]);
    }
    mesh.seal_canonical_digest().unwrap();
    let source = AnalysisProjection::derive(&mesh).unwrap();
    let fv = FiniteVolumeTopology::derive(&source, policy()).unwrap();
    assert!((fv.cells[0].volume_m3 / (1e-18 / 6.0) - 1.0).abs() < 1e-13);
    let mut sum = [0.0; 3];
    for face in fv.faces {
        for (a, b) in sum.iter_mut().zip(face.owner_area_vector_m2) {
            *a += b;
        }
    }
    assert!(geometry::length(sum) < 1e-25);
}

/// Cells far from the origin relative to their size (0.2 mm cells 0.2 m away, as at the wall
/// of a 200 mm tube) derive their finite-volume geometry: the source centroid of a linear
/// tetrahedron is its corner mean, so the centroid check never sees quadrature rounding.
#[test]
fn small_cells_far_from_the_origin_keep_their_source_centroids() {
    for offset in [[0.2, 0.15, -0.1], [0.05, -0.2, 0.18], [-0.19, 0.01, 0.2]] {
        let mut mesh = fixtures::slab_mesh(3);
        for node in &mut mesh.topology.nodes {
            let [x, y, z] = node.coordinates_m.canonical_value();
            node.coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                offset[0] + 6e-4 * x + 1e-4 * y,
                offset[1] + 6e-4 * y,
                offset[2] + 6e-4 * z - 1e-4 * x,
            ]);
        }
        mesh.seal_canonical_digest().unwrap();
        let source = AnalysisProjection::derive(&mesh).unwrap();
        for element in &source.elements {
            let corners = &element.node_indices[..4];
            let mean: [f64; 3] = std::array::from_fn(|axis| {
                corners
                    .iter()
                    .map(|&n| source.nodes[n].coordinates_m.canonical_value()[axis])
                    .sum::<f64>()
                    / 4.0
            });
            assert_eq!(element.centroid_m, mean);
        }
        FiniteVolumeTopology::derive(&source, policy()).unwrap();
    }
}

// witness: FEA-FLUID-TOPOLOGY-001/quality @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Selected malformed source cases only, not complete quality-envelope convergence.
#[test]
fn rejects_nonmanifold_missing_boundary_inversion_curvature_and_mutated_record() {
    let mut source = two_cells();
    let mut third = source.elements[1].clone();
    third.element_id = 3;
    third.stable_identity = StableDigest::from_bytes([233; 32]);
    source.elements.push(third);
    assert!(FiniteVolumeTopology::derive(&source, policy()).is_err());
    let mut source = two_cells();
    source.boundary_faces.pop();
    assert_eq!(
        FiniteVolumeTopology::derive(&source, policy()).unwrap_err(),
        FiniteVolumeProjectionError::Invalid("missing original exterior boundary")
    );
    let mut source = projected(ElementOrder::Tet4);
    source.elements[0].node_indices.swap(1, 2);
    assert!(FiniteVolumeTopology::derive(&source, policy()).is_err());
    let mut source = projected(ElementOrder::Tet10);
    let mid = source.elements[0].node_indices[4];
    {
        let quantity_change = 0.01;
        let quantity_index = 2;
        let quantity_slot = &mut source.nodes[mid].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!(matches!(
        FiniteVolumeTopology::derive(&source, policy()),
        Err(FiniteVolumeProjectionError::CurvedTet10(_))
    ));
    let source = two_cells();
    let mut fv = FiniteVolumeTopology::derive(&source, policy()).unwrap();
    fv.cells[0].volume_m3 *= 2.0;
    assert_eq!(
        fv.validate_for(&source).unwrap_err(),
        FiniteVolumeProjectionError::SourceMismatch
    );
}

// witness: FEA-FLUID-TOPOLOGY-001/budget @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Exact finite-volume construction counts and one-below limits; requested bytes are not
//   RSS.
#[test]
fn exact_admission_bounds_succeed_and_each_one_below_rejects() {
    let source = two_cells();
    let r = FiniteVolumeTopology::reserve(
        &source,
        policy(),
        &crate::topology::FiniteVolumeCellSelection::AllElements,
    )
    .unwrap();
    let exact = FiniteVolumeProjectionPolicy {
        max_cells: Some(r.cell_count),
        max_face_incidences: Some(r.face_incidence_count),
        max_boundary_faces: Some(r.boundary_face_count),
        max_scratch_bytes: Some(r.scratch_bytes),
        max_traversal_operations: Some(r.traversal_operations),
    };
    FiniteVolumeTopology::derive(&source, exact).unwrap();
    for field in 0..5 {
        let mut p = exact;
        match field {
            0 => p.max_cells = p.max_cells.map(|v| v - 1),
            1 => p.max_face_incidences = p.max_face_incidences.map(|v| v - 1),
            2 => p.max_boundary_faces = p.max_boundary_faces.map(|v| v - 1),
            3 => p.max_scratch_bytes = p.max_scratch_bytes.map(|v| v - 1),
            _ => p.max_traversal_operations = p.max_traversal_operations.map(|v| v - 1),
        }
        assert!(matches!(
            FiniteVolumeTopology::derive(&source, p),
            Err(FiniteVolumeProjectionError::Limit(_))
        ));
    }
}

#[test]
fn count_reservations_reject_checked_overflow_without_allocating() {
    assert_eq!(
        super::derive::checked_counts(usize::MAX, 0).unwrap_err(),
        FiniteVolumeProjectionError::Overflow
    );
    assert_eq!(
        super::derive::checked_counts(1, usize::MAX).unwrap_err(),
        FiniteVolumeProjectionError::Overflow
    );
}

/// Diagnostic over persisted solver meshes: `RUNMAT_FV_DIAG_MESHES=a.cbor:b.cbor`.
#[test]
#[ignore]
fn diagnose_persisted_mesh_closure() {
    use super::super::element::{dot, subtract, tetrahedron_geometry};
    use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
    let paths = std::env::var("RUNMAT_FV_DIAG_MESHES").unwrap_or_default();
    for path in paths.split(':').filter(|p| !p.is_empty()) {
        let bytes = std::fs::read(path).unwrap();
        let mesh = match SolverMeshArtifact::canonical_decode(&bytes) {
            Ok(mesh) => mesh,
            Err(error) => {
                println!("== {path}: undecodable ({error})");
                continue;
            }
        };
        let source = AnalysisProjection::derive(&mesh).unwrap();
        let mut regions = source
            .elements
            .iter()
            .map(|e| e.region_id.clone())
            .collect::<Vec<_>>();
        regions.sort();
        regions.dedup();
        println!(
            "== {path}\n nodes {} cells {} boundary {} bounds {:?} order {:?} regions {:?}",
            source.nodes.len(),
            source.elements.len(),
            source.boundary_faces.len(),
            source.bounds_m,
            source.elements[0].order,
            regions
                .iter()
                .map(|r| &r.source_topology_id)
                .collect::<Vec<_>>()
        );
        let mut rows = Vec::new();
        for (index, element) in source.elements.iter().enumerate() {
            let cell = geometry::corners(&source, index).unwrap();
            let (volume, centroid, _) = tetrahedron_geometry(cell.points).unwrap();
            let mut area_sum = [0.0; 3];
            let mut area_scale = 0.0;
            let mut divergence = 0.0;
            for local in geometry::FACE_CORNERS {
                let (c, a, m) =
                    geometry::face_geometry(local.map(|i| cell.points[i]), centroid).unwrap();
                for axis in 0..3 {
                    area_sum[axis] += a[axis];
                }
                area_scale += m;
                divergence += dot(subtract(c, centroid), a) / 3.0;
            }
            let edge = runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS
                .iter()
                .map(|[a, b]| geometry::length(subtract(cell.points[*a], cell.points[*b])))
                .fold(0.0_f64, f64::max);
            let x = cell
                .points
                .iter()
                .flatten()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            let area_ratio = geometry::length(area_sum) / area_scale / f64::EPSILON;
            let volume_ratio = (divergence - volume).abs() / volume / f64::EPSILON;
            rows.push((
                volume_ratio,
                area_ratio,
                volume / edge.powi(3),
                x / edge,
                element.element_id,
                (element.signed_volume_m3 - volume).abs() / volume / f64::EPSILON,
            ));
        }
        let limit = 4096.0;
        println!(
            " volume>tol {} area>tol {} minimum V/L^3 {:e}",
            rows.iter().filter(|r| r.0 > limit).count(),
            rows.iter().filter(|r| r.1 > limit).count(),
            rows.iter().map(|r| r.2).fold(f64::INFINITY, f64::min)
        );
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        for r in rows.iter().take(6) {
            println!(
                "  vol/eps {:.0} area/eps {:.0} V/L^3 {:.3e} X/L {:.0} element {} srcvol/eps {:.0}",
                r.0, r.1, r.2, r.3, r.4, r.5
            );
        }
        let all = FiniteVolumeTopology::derive(&source, FiniteVolumeProjectionPolicy::default());
        println!(
            " derive all: {:?}",
            all.as_ref()
                .map(|t| t.cells.len())
                .map_err(|e| e.to_string())
        );
        for region in &regions {
            let selected = FiniteVolumeTopology::derive_selected(
                &source,
                FiniteVolumeProjectionPolicy::default(),
                FiniteVolumeCellSelection::regions(vec![region.clone()]).unwrap(),
            );
            println!(
                " derive {}: {:?}",
                region.source_topology_id,
                selected.map(|t| t.cells.len()).map_err(|e| e.to_string())
            );
        }
    }
}

/// The flattest cells of the cht-plate-channel, fsi-compliant-tube and fluid-cavity-re100 meshes
/// (element 299299, 116309 and 152370 of their default meshes): valid, closed, positive cells
/// with V/L³ of 5e-4, 6e-4 and 2e-4 at 250, 160 and 40 edge lengths from the origin. Their
/// divergence-theorem volume differs from the corner determinant by 2e4, 2e4 and 5e3 ε·V from
/// rounding alone, which a tolerance relative to V had rejected as "cell closure or
/// conservative volume".
fn flat_study_cells() -> [[[f64; 3]; 4]; 3] {
    [
        [
            [
                0.2703240944345575,
                0.014837542820181687,
                0.0029361822594769246,
            ],
            [
                0.27074992383287905,
                0.01436665392339019,
                0.0031065135267128387,
            ],
            [
                0.2706811327174461,
                0.013838443675568954,
                0.0027213622749574897,
            ],
            [
                0.2699491569348651,
                0.014584541772600445,
                0.0023765873696541596,
            ],
        ],
        [
            [
                0.2971650907484167,
                -0.007463709771425251,
                -0.003483843067396201,
            ],
            [
                0.296438346632368,
                -0.006850637605723358,
                -0.005066920317002789,
            ],
            [
                0.2963943473871062,
                -0.006425615263212107,
                -0.003791464681184605,
            ],
            [
                0.29718784259388153,
                -0.007819866439350185,
                -0.004709278240842538,
            ],
        ],
        [
            [0.060546875, 0.576171875, 0.05],
            [
                0.05089771259382207,
                0.5681421237191052,
                0.039730286038129455,
            ],
            [0.05710946929066528, 0.5765492555751789, 0.03807009134139609],
            [0.052978515625, 0.566650390625, 0.05],
        ],
    ]
}

// witness: FEA-FLUID-TOPOLOGY-001/quality @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel; coverage: partial;
//   boundary: The flattest cells of three study meshes at their study offsets; the closure check
//   admits them and still resolves a reversed face (V/2) by more than four orders of magnitude.
#[test]
fn flat_study_cells_close_within_the_rounding_bound_and_keep_face_errors_detectable() {
    use super::super::element::{dot, subtract, tetrahedron_geometry};
    for corners in flat_study_cells() {
        let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let element = &mesh.topology.volume_elements[0];
        let order = element.node_ids.clone();
        // Keep the canonical element's corner order; orient positively.
        let mut points = corners;
        if tetrahedron_geometry(points).is_none() {
            points.swap(1, 2);
        }
        for (local, node_id) in order.iter().enumerate() {
            let node = mesh
                .topology
                .nodes
                .iter_mut()
                .find(|n| n.node_id == *node_id)
                .unwrap();
            node.coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(points[local]);
        }
        mesh.seal_canonical_digest().unwrap();
        let source = AnalysisProjection::derive(&mesh).unwrap();
        let fv = FiniteVolumeTopology::derive(&source, policy()).unwrap();
        let cell = &fv.cells[0];
        let (volume, centroid, _) = tetrahedron_geometry(points).unwrap();
        let edge = runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS
            .iter()
            .map(|[a, b]| geometry::length(subtract(points[*a], points[*b])))
            .fold(0.0_f64, f64::max);
        assert!(volume / edge.powi(3) < 1e-3, "the fixture is a flat cell");
        assert_eq!(cell.volume_m3, volume);
        let mut divergence = 0.0;
        let mut scale = 0.0;
        for face in &fv.faces {
            divergence += dot(
                subtract(face.centroid_m, centroid),
                face.owner_area_vector_m2,
            ) / 3.0;
            scale += geometry::divergence_term_scale(
                face.centroid_m,
                centroid,
                face.area_m2.canonical_value(),
            );
        }
        let rounding = (divergence - volume).abs();
        assert!(
            rounding > 1e3 * f64::EPSILON * volume,
            "rounding is relative to the terms"
        );
        assert!(rounding <= geometry::RELATIVE_TOLERANCE * scale);
        // One reversed face moves the sum by 2·V/4; the admitted band stays far below it.
        assert!(geometry::RELATIVE_TOLERANCE * scale < 1e-4 * volume / 2.0);
    }
}
