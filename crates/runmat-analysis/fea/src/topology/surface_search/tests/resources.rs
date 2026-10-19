//! Reconcile requested layouts independently of the search's allocation implementation.
use super::*;
#[test]
fn successful_exhausted_and_cancelled_queries_respect_the_frontier_envelope() {
    let surface = curved(0.4);
    for controls in [
        policy(),
        BoundarySearchPolicy {
            maximum_patches: 1,
            ..policy()
        },
    ] {
        let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
            surface.closest_point([0.2, 0.2, 0.2], controls)
        });
        assert_eq!(retained, 0);
        assert!(peak as u64 <= controls.resident_bytes().unwrap());
        if controls.maximum_patches == 1 {
            assert_eq!(result, Err(BoundarySearchError::Exhausted));
        } else {
            assert!(result.unwrap().visited_patches <= controls.maximum_patches);
        }
    }
    let cancellation = crate::progress::FeaCancellationToken::new();
    cancellation.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation,
            ..Default::default()
        });
    let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
        surface.closest_point([0.2, 0.2, 0.2], policy())
    });
    assert_eq!(result, Err(BoundarySearchError::Cancelled));
    assert_eq!((retained, peak), (0, 0));
}
