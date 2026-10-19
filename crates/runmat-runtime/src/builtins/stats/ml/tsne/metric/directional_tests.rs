use super::{correlation_distance, cosine_distance, tied_ranks};

#[test]
fn scale_changes_do_not_change_finite_angular_distances() {
    for scale in [1.0e200, 1.0e-200] {
        let left = [scale, 2.0 * scale];
        let right = [2.0 * scale, scale];
        let cosine = futures::executor::block_on(cosine_distance(
            &left,
            &right,
            &mut super::super::cooperation::TsneCooperation::new(None),
        ))
        .unwrap();
        let correlation = futures::executor::block_on(correlation_distance(
            &left,
            &right,
            &mut super::super::cooperation::TsneCooperation::new(None),
        ))
        .unwrap();
        assert!((cosine - 0.2).abs() < 1.0e-13);
        assert!((correlation - 2.0).abs() < 1.0e-13);
    }
}

#[test]
fn spearman_rank_sort_keeps_exact_ties_and_original_columns() {
    let values = [4.0, 1.0, 4.0, -0.0, 0.0, 2.0];
    let mut cooperation = super::super::cooperation::TsneCooperation::new(None);
    let ranks = futures::executor::block_on(tied_ranks(&values, &mut cooperation)).unwrap();
    assert_eq!(ranks, vec![5.5, 3.0, 5.5, 1.5, 1.5, 4.0]);
}
