use super::*;

#[test]
fn literal_nonsquare_matlab_tensor_preserves_real_and_complex_coordinates() {
    // [samples=2, rows=2, columns=3], MATLAB sample/row/column order.
    let matlab = vec![
        0., 100., 10., 110., 1., 101., 11., 111., 2., 102., 12., 112.,
    ];
    let canonical = vec![
        0., 1., 2., 10., 11., 12., 100., 101., 102., 110., 111., 112.,
    ];
    let NumericValues::Real(actual) =
        canonical_sample_layout(NumericValues::Real(matlab.clone()), 2, &[2, 3]).unwrap()
    else {
        unreachable!()
    };
    assert_eq!(actual, canonical);
    let NumericValues::Complex(actual) = canonical_sample_layout(
        NumericValues::Complex(matlab.into_iter().map(|v| (v, -v - 0.5)).collect()),
        2,
        &[2, 3],
    )
    .unwrap() else {
        unreachable!()
    };
    assert_eq!(
        actual,
        canonical
            .into_iter()
            .map(|v| (v, -v - 0.5))
            .collect::<Vec<_>>()
    );
}

#[test]
fn rank_three_tail_and_singleton_axes_use_row_major_coordinates() {
    let mut matlab = Vec::new();
    for c in 0..4 {
        for b in 0..3 {
            for a in 0..2 {
                for s in 0..5 {
                    matlab.push((1000 * s + 100 * a + 10 * b + c) as f64);
                }
            }
        }
    }
    for extents in [&[2, 3, 4][..], &[1, 2, 3, 4, 1][..]] {
        let NumericValues::Real(actual) =
            canonical_sample_layout(NumericValues::Real(matlab.clone()), 5, extents).unwrap()
        else {
            unreachable!()
        };
        let mut expected = Vec::new();
        for s in 0..5 {
            for a in 0..2 {
                for b in 0..3 {
                    for c in 0..4 {
                        expected.push((1000 * s + 100 * a + 10 * b + c) as f64);
                    }
                }
            }
        }
        assert_eq!(actual, expected);
    }
}

#[test]
fn invalid_layout_cardinality_fails_without_allocating_an_order_map() {
    for (samples, extents) in [
        (0, vec![]),
        (1, vec![0]),
        (2, vec![usize::MAX]),
        (1, vec![usize::MAX, 2]),
        (2, vec![2, 3]),
    ] {
        assert!(canonical_sample_layout(NumericValues::Real(vec![1.]), samples, &extents).is_err());
    }
}
