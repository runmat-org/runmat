use super::super::test_fixtures::pair;
use super::*;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An exact three-to-two bipyramid replacement retains three removed and two created
//   connectivity keys with zero or 64 unrelated cells. The delta reconstructs the full ordered
//   candidate inventory, distinguishes unchanged connectivity and preserves independent full-key
//   ordering. No sublinear comparison-work, arbitrary cavity or whole-task resource claim.
#[test]
fn exact_local_delta_reconstructs_connectivity_and_ignores_unrelated_cells() {
    let mut reference = None;
    for unrelated in [0, 64] {
        let (base, candidate) = pair(unrelated);
        let delta = connectivity_delta(&base, &candidate);
        assert_eq!(delta.0.len(), 3);
        assert_eq!(delta.1.len(), 2);
        if let Some(reference) = &reference {
            assert_eq!(&delta, reference);
        }
        reference = Some(delta.clone());
        let mut reconstructed = keys(&base)
            .filter(|key| !delta.0.contains(key))
            .collect::<Vec<_>>();
        reconstructed.extend(delta.1);
        reconstructed.sort_unstable();
        assert_eq!(reconstructed, keys(&candidate).collect::<Vec<_>>());
        assert_eq!(
            compare_connectivity(&base, &candidate),
            keys(&base)
                .collect::<Vec<_>>()
                .cmp(&keys(&candidate).collect::<Vec<_>>())
        );
        assert_eq!(connectivity_delta(&base, &base), (vec![], vec![]));
        assert_ne!(
            connectivity_delta(&base, &base),
            connectivity_delta(&base, &candidate)
        );
    }
}
