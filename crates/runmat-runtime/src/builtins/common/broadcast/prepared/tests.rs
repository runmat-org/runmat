use super::*;
use crate::builtins::common::broadcast::BroadcastPlan;

#[test]
fn exact_metadata_matches_existing_broadcast_indexing() {
    for (left, right) in [
        (vec![2, 3], vec![2, 3]),
        (vec![1, 3], vec![2, 1]),
        (vec![1, 1], vec![2, 3]),
        (vec![3], vec![2, 1]),
        (vec![2, 1, 4], vec![1, 3, 1]),
        (vec![0, 3], vec![1, 3]),
    ] {
        let ordinary = BroadcastPlan::new(&left, &right).unwrap();
        let shape = ordinary.output_shape();
        let rank = shape.len();
        assert_eq!(
            PreparedBroadcastPlan::quoted_for_shapes(&left, &right, shape).unwrap(),
            PreparedBroadcastPlan::quoted_bytes(rank).unwrap()
        );
        let mut metadata = vec![0; 4 * rank];
        let len = initialize(&mut metadata, &left, &right, shape).unwrap();
        for _ in 0..2 {
            let (fixed, coords) = metadata.split_at_mut(3 * rank);
            coords.fill(0);
            let (shape, strides) = fixed.split_at(rank);
            let (advance_a, advance_b) = strides.split_at(rank);
            let prepared = PreparedBroadcastIter {
                shape,
                advance_a,
                advance_b,
                coords,
                len,
                offset: 0,
                index_a: 0,
                index_b: 0,
            };
            assert_eq!(
                prepared.collect::<Vec<_>>(),
                ordinary.iter().collect::<Vec<_>>()
            );
        }
    }
    assert!(initialize(&mut [0; 8], &[2, 3], &[4, 1], &[4, 3]).is_err());
    assert!(initialize(&mut [0; 8], &[2, 3], &[2, 3], &[3, 2]).is_err());
    assert!(PreparedBroadcastPlan::quoted_for_shapes(&[2, 3], &[4, 1], &[4, 3]).is_err());
}
