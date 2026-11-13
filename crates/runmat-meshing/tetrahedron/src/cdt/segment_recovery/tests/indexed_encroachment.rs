use super::*;
use crate::cdt::mutation::TopologyMutation;

#[test]
fn indexed_sphere_candidates_preserve_exhaustive_exact_witness_and_ranking() {
    let constraints = octahedron_constraints();
    let canonical = octahedron_topology(&constraints);
    let indexed = TopologyMutation::from_canonical(canonical.clone());
    for left in &canonical.nodes {
        for right in &canonical.nodes {
            if left.identity == right.identity {
                continue;
            }
            let mut exhaustive_work =
                RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
            let mut indexed_work =
                RecoveryWork::new(DelaunaySegmentRecoveryOptions::default(), &NeverCancelled);
            let expected = encroachment::encroaching_point(
                &canonical,
                left.identity,
                right.identity,
                0,
                &mut exhaustive_work,
            )
            .unwrap();
            let actual = encroachment::encroaching_point(
                &indexed,
                left.identity,
                right.identity,
                0,
                &mut indexed_work,
            )
            .unwrap();
            assert_eq!(
                actual.map(|point| (point.vertex, point.relation)),
                expected.map(|point| (point.vertex, point.relation))
            );
        }
    }
}
