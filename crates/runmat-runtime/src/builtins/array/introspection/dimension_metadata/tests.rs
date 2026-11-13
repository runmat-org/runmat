use super::*;

#[test]
fn normalizes_scalar_and_vector_shapes() {
    assert_eq!(normalize_dimensions(&[]), vec![1, 1]);
    assert_eq!(normalize_dimensions(&[1]), vec![1, 1]);
    assert_eq!(normalize_dimensions(&[3]), vec![1, 3]);
}

#[test]
fn effective_rank_ignores_only_trailing_singletons() {
    assert_eq!(effective_rank(&[2, 3, 1, 1]), 2);
    assert_eq!(effective_rank(&[1, 1, 3, 1]), 3);
}

#[test]
fn visible_dimensions_expose_structural_queries() {
    let dimensions = VisibleDimensions(vec![3, 4, 5]);
    assert_eq!(dimensions.extent(1), 3);
    assert_eq!(dimensions.extent(2), 4);
    assert_eq!(dimensions.extent(8), 1);
    assert_eq!(dimensions.largest_extent(), 5);
    assert_eq!(dimensions.rank(), 3);
    assert_eq!(dimensions.product(), Some(60));
    assert_eq!(dimensions.selected_product(&[1, 3, 8]), Some(15));
}

#[test]
fn collapses_remaining_dimensions_into_final_output() {
    let dimensions = VisibleDimensions(vec![3, 4, 5]);
    assert_eq!(dimensions.collapsed_outputs(2), Some(vec![3, 20]));
    assert_eq!(dimensions.collapsed_outputs(4), Some(vec![3, 4, 5, 1]));
}

#[test]
fn structural_products_are_checked() {
    assert_eq!(VisibleDimensions(vec![u64::MAX, 2]).product(), None);
}

#[test]
fn reported_size_omits_trailing_singletons_beyond_rank_two() {
    assert_eq!(VisibleDimensions(vec![2, 3, 1, 1]).reported_size(), &[2, 3]);
    assert_eq!(VisibleDimensions(vec![1, 1, 1]).reported_size(), &[1, 1]);
}
