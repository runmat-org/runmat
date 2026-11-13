use super::*;
use crate::testutil::poisson3d;

// witness: FEA-SCALE-PARALLEL-001/shared_parallel_kernels @
//   shared_infrastructure/canonical_mixed_topology/parallel_cpu; coverage: partial;
//   boundary: Nested-dissection ordering of a 20x20x20 Poisson problem is a valid permutation,
//   identical across runs and at 4 threads.
// witness: FEA-SHARED-EXECUTION-001/deterministic_parallel_boundaries @
//   shared_infrastructure/canonical_mixed_topology/parallel_cpu; coverage: partial;
//   boundary: Fill-reducing ordering is deterministic across thread counts; assembly and solve
//   boundaries are covered by the sibling tests.
#[test]
fn nested_dissection_is_a_valid_deterministic_permutation() {
    let a = poisson3d::<f64>(20, 20, 20);
    let first = order(a.as_ref(), &Ordering::NestedDissection).unwrap();
    let second = order(a.as_ref(), &Ordering::NestedDissection).unwrap();
    assert_eq!(first, second);
    let mut seen = vec![false; a.nrows()];
    for &v in &first.perm {
        assert!(!seen[v]);
        seen[v] = true;
    }
    let threaded = crate::par::with_threads(4, || {
        order(a.as_ref(), &Ordering::NestedDissection).unwrap()
    });
    assert_eq!(first, threaded);
}

#[test]
fn supervariables_compress_vector_dofs() {
    // Expand a scalar Laplacian to 3 coupled dofs per node.
    let s = poisson3d::<f64>(5, 5, 5);
    let n = s.nrows();
    let mut t = crate::csr::TripletBuilder::new(3 * n, 3 * n);
    for row in 0..n {
        for &col in s.as_ref().row(row).0 {
            for a in 0..3 {
                for b in 0..3 {
                    t.push(3 * row + a, 3 * col + b, 1.0);
                }
            }
        }
    }
    let a = t.build();
    let graph = Graph::from_matrix(a.as_ref());
    let (quotient, members) = graph.compress_supervariables();
    assert_eq!(quotient.vertex_count(), n);
    assert!(members.iter().all(|m| m.len() == 3));
}
