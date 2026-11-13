use super::*;
use runmat_meshing_core::NeverCancelled;

fn run(coordinates: Vec<[f64; 2]>, ceiling: u64) -> (Vec<[u32; 3]>, Vec<[u32; 3]>) {
    let (_, pslg) = super::super::super::tests::fixture();
    let options = ExactFaceDelaunayOptions {
        maximum_predicate_evaluations: ceiling,
        ..Default::default()
    };
    let mut points = coordinates
        .into_iter()
        .enumerate()
        .map(|(i, coordinates)| {
            let mut bytes = [0x50; 32];
            bytes[..4].copy_from_slice(&(i as u32).to_be_bytes());
            PlanarPredicatePoint {
                coordinates,
                identity: StableDigest::from_bytes(bytes),
            }
        })
        .collect::<Vec<_>>();
    let n = points.len();
    points.extend(super_triangle(&points, &pslg.source_face_id).unwrap());
    let mut budget = PredicateBudget::new(&pslg, &NeverCancelled, options);
    let mut local = triangulate(&points, n, &mut budget, options).unwrap();
    let local_work = options.maximum_predicate_evaluations - budget.remaining;
    let mut budget = PredicateBudget::new(&pslg, &NeverCancelled, Default::default());
    let mut exhaustive =
        super::super::exhaustive(&points, n, &mut budget, Default::default()).unwrap();
    let exhaustive_work =
        ExactFaceDelaunayOptions::default().maximum_predicate_evaluations - budget.remaining;
    if n > 1000 {
        eprintln!("points={n} local_work={local_work} exhaustive_work={exhaustive_work}");
        assert!(
            exhaustive_work > ceiling,
            "bound must reject the exhaustive scan"
        );
    }
    for triangles in [&mut local, &mut exhaustive] {
        for t in triangles.iter_mut() {
            *t = canonical_triangle(*t);
        }
        triangles.sort_unstable();
    }
    (local, exhaustive)
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Connected cavity traversal preserves the exhaustive builder’s canonical topology on
//   grids, reversed insertion order, transformed coordinates and cocircular inputs. This finite
//   component comparison does not establish arbitrary CAD coverage or native throughput.
#[test]
fn connected_cavities_match_exhaustive_predicates_including_degeneracies() {
    for side in [3, 7, 17] {
        let coordinates = (0..side)
            .flat_map(|x| (0..side).map(move |y| [x as f64, y as f64]))
            .collect::<Vec<_>>();
        for reverse in [false, true] {
            for scale in [0.125, 1.0, 1024.0] {
                let mut points = coordinates
                    .iter()
                    .map(|[x, y]| [(x - y) * scale + 4.0, (x + y) * scale - 2.0])
                    .collect::<Vec<_>>();
                if reverse {
                    points.reverse();
                }
                let (local, oracle) = run(points, 1_000_000);
                assert_eq!(local, oracle);
            }
        }
    }
    let circle = vec![
        [5., 0.],
        [4., 3.],
        [3., 4.],
        [0., 5.],
        [-3., 4.],
        [-4., 3.],
        [-5., 0.],
        [-4., -3.],
        [-3., -4.],
        [0., -5.],
        [3., -4.],
        [4., -3.],
    ];
    let (local, oracle) = run(circle, 10_000);
    assert_eq!(local, oracle);
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A 2025-point planar inventory compares exact canonical output against the exhaustive
//   builder while applying a predicate ceiling that the exhaustive scan exceeds. This component
//   work bound does not establish whole-task memory, constrained refinement convergence or native
//   study performance.
#[test]
fn bounded_local_work_handles_a_large_planar_inventory() {
    let coordinates = (0..45)
        .flat_map(|x| (0..45).map(move |y| [x as f64, y as f64]))
        .collect();
    // The exhaustive builder needs over four million incircle evaluations alone.
    let (local, oracle) = run(coordinates, 1_000_000);
    assert_eq!(local, oracle);
}
