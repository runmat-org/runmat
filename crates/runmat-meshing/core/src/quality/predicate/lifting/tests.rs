use super::*;

#[test]
fn exact_lifting_selects_the_empty_sphere_reconnection_across_scales_and_orders() {
    let points = [
        [0., 0., -2.],
        [0., 0., 2.],
        [1., 0., 0.],
        [-0.5, 1., 0.],
        [-0.5, -1., 0.],
    ];
    for exponent in [-1000, -200, 0, 200, 1000] {
        let points = points.map(|point| point.map(|value| value * 2f64.powi(exponent)));
        let removed =
            [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]].map(|tet| tet.map(|index| points[index]));
        let created = [[0, 2, 3, 4], [1, 2, 4, 3]].map(|tet| tet.map(|index| points[index]));
        assert_eq!(
            tetrahedral_lifting_change(&removed, &created).unwrap(),
            PredicateSign::Negative
        );
        assert_eq!(
            tetrahedral_lifting_change(&created, &removed).unwrap(),
            PredicateSign::Positive
        );
        let reverse = removed.map(|mut tet| {
            tet.swap(0, 1);
            tet
        });
        assert_eq!(
            tetrahedral_lifting_change(&reverse, &created).unwrap(),
            PredicateSign::Negative
        );
    }
}

#[test]
fn cospherical_lifting_is_exactly_equal_and_nonfinite_input_is_rejected() {
    let points = [
        [0., 0., -5.],
        [0., 0., 5.],
        [3., 4., 0.],
        [-3., 4., 0.],
        [0., -5., 0.],
    ];
    let removed =
        [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]].map(|tet| tet.map(|index| points[index]));
    let created = [[0, 2, 3, 4], [1, 2, 4, 3]].map(|tet| tet.map(|index| points[index]));
    assert_eq!(
        tetrahedral_lifting_change(&removed, &created).unwrap(),
        PredicateSign::Zero
    );
    let mut invalid = created;
    invalid[0][0][1] = f64::NAN;
    assert!(matches!(
        tetrahedral_lifting_change(&removed, &invalid),
        Err(SpatialPredicateError::NonFiniteCoordinate { .. })
    ));
}
