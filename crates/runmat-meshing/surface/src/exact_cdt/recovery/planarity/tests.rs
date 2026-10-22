use super::*;
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    PersistentEntityId, PersistentEntityKind,
};
use runmat_meshing_core::{NeverCancelled, StableDigest};
use std::sync::atomic::{AtomicU64, Ordering};
struct Control {
    probes: AtomicU64,
    held: AtomicU64,
    peak: AtomicU64,
    maximum: u64,
    cancel_at: u64,
}
impl MeshingCancellationSignal for Control {
    fn is_cancelled(&self) -> bool {
        self.probes.fetch_add(1, Ordering::Relaxed) + 1 >= self.cancel_at
    }
}
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        let next = self.held.load(Ordering::Relaxed) + bytes;
        if next > self.maximum {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                "planarity index denied",
            ));
        }
        self.held.store(next, Ordering::Relaxed);
        self.peak.fetch_max(next, Ordering::Relaxed);
        Ok(())
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        assert!(self.held.fetch_sub(bytes, Ordering::Relaxed) >= bytes);
    }
}
fn fixture(
    points: Vec<[f64; 2]>,
    pairs: Vec<[u32; 2]>,
) -> (ExactFacePslg, BTreeMap<[u32; 2], Vec<EdgeUse>>) {
    let face = PersistentEntityId::new(
        PersistentEntityKind::Face,
        "planarity-fixture".into(),
        vec![],
    );
    let pslg = ExactFacePslg {
        source_face_id: face,
        vertices: points
            .into_iter()
            .enumerate()
            .map(|(index, uv)| crate::ExactFacePslgVertex {
                node_id: StableDigest::from_bytes([(index % 255 + 1) as u8; 32]),
                seam_image: None,
                uv,
            })
            .collect(),
        segments: vec![],
        loops: vec![],
    };
    (pslg, pairs.into_iter().map(|edge| (edge, vec![])).collect())
}
fn exhaustive(
    edges: &BTreeMap<[u32; 2], Vec<EdgeUse>>,
    pslg: &ExactFacePslg,
) -> Result<u64, ExactFaceDelaunayError> {
    let mut predicates = 0;
    for left in edges.keys() {
        for right in edges.keys().filter(|right| *right > left) {
            if !bounds_overlap(left, right, pslg) {
                continue;
            }
            predicates += 4;
            if segments_intersect(*left, *right, pslg)? {
                predicates += 2;
                if !intersection_is_shared_endpoint(*left, *right, pslg)? {
                    return Err(error(
                        pslg,
                        ExactFaceDelaunayErrorKind::InvalidTopology,
                        "exhaustive crossing",
                    ));
                }
            }
        }
    }
    Ok(predicates)
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Indexed edge-planarity decisions and successful predicate counts match exhaustive
//   closed-bounds pair checks across the tested scales, translations, crossings, endpoint contacts
//   and collinear overlaps. This finite witness does not establish every CAD topology or PDE
//   qualification.
#[test]
fn indexed_planarity_matches_exhaustive_crossings_contacts_and_collinear_overlap() {
    for scale in [1e-9, 1.0, 1e9] {
        for offset in [0.0, 100.0] {
            for pairs in [
                vec![[0, 1], [1, 2], [0, 2]],
                vec![[0, 1], [2, 3]],
                vec![[0, 1], [0, 4]],
                vec![[0, 1], [4, 5]],
            ] {
                let points = [
                    [0.0, 0.0],
                    [2.0, 2.0],
                    [0.0, 2.0],
                    [2.0, 0.0],
                    [1.0, 1.0],
                    [3.0, 3.0],
                ]
                .map(|p| [offset + p[0] * scale, offset + p[1] * scale])
                .to_vec();
                let (pslg, edges) = fixture(points, pairs);
                let expected = exhaustive(&edges, &pslg);
                let actual = validate_planar_edges(
                    &edges,
                    &pslg,
                    &NeverCancelled,
                    ExactFaceDelaunayOptions::default(),
                    None,
                );
                match (expected, actual) {
                    (Ok(left), Ok(right)) => assert_eq!(left, right),
                    (Err(left), Err(right)) => assert_eq!(left.kind, right.kind),
                    values => panic!("planarity mismatch: {values:?}"),
                }
            }
        }
    }
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: For 8,192 disjoint long edges sharing the same x interval, spatial traversal has
//   bounded local work and preserves zero intersection predicates. Exact requested packed-index
//   construction storage accepts, one byte less rejects, and temporary storage releases completely.
//   The requested-buffer inventory excludes allocator overhead and process RSS.
#[test]
fn long_parallel_edges_have_local_work_and_exact_scratch_admission() {
    let count = 8192;
    let points = (0..count)
        .flat_map(|index| [[0.0, index as f64], [1000.0, index as f64]])
        .collect();
    let pairs = (0..count)
        .map(|index| [(index * 2) as u32, (index * 2 + 1) as u32])
        .collect();
    let (pslg, edges) = fixture(points, pairs);
    let profile=runmat_meshing_core::quality::spatial_index::PackedBoundsSpatialIndex::<[u32;2]>::storage_profile(edges.len()).unwrap();
    let exact = profile.retained_bytes
        + profile.construction_scratch_bytes
        + (edges.len()
            * std::mem::size_of::<
                runmat_meshing_core::quality::spatial_index::SpatialEntry<[u32; 2]>,
            >()) as u64;
    for maximum in [exact - 1, exact] {
        let control = Control {
            probes: AtomicU64::new(0),
            held: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            maximum,
            cancel_at: u64::MAX,
        };
        let result = validate_planar_edges(
            &edges,
            &pslg,
            &control,
            ExactFaceDelaunayOptions::default(),
            Some(&control),
        );
        if maximum < exact {
            assert_eq!(
                result.unwrap_err().kind,
                ExactFaceDelaunayErrorKind::GeometryEvaluation(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded
                )
            );
        } else {
            assert_eq!(result.unwrap(), 0);
            assert_eq!(control.peak.load(Ordering::Relaxed), exact);
            assert!(control.probes.load(Ordering::Relaxed) < count as u64 * 200);
        }
        assert_eq!(control.held.load(Ordering::Relaxed), 0);
    }
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An interruption during planarity index construction preserves cancellation and
//   releases previously admitted scratch. This finite component fixture is not whole-task memory
//   qualification.
#[test]
fn interrupted_planarity_index_releases_admitted_storage() {
    let (pslg, edges) = fixture(
        (0..512)
            .flat_map(|index| [[0.0, index as f64], [1000.0, index as f64]])
            .collect(),
        (0..512).map(|index| [index * 2, index * 2 + 1]).collect(),
    );
    let control = Control {
        probes: AtomicU64::new(0),
        held: AtomicU64::new(0),
        peak: AtomicU64::new(0),
        maximum: 1_000_000,
        cancel_at: 20,
    };
    let error = validate_planar_edges(
        &edges,
        &pslg,
        &control,
        ExactFaceDelaunayOptions::default(),
        Some(&control),
    )
    .unwrap_err();
    assert_eq!(error.kind, ExactFaceDelaunayErrorKind::Cancelled);
    assert!(control.peak.load(Ordering::Relaxed) > 0);
    assert_eq!(control.held.load(Ordering::Relaxed), 0);
}
