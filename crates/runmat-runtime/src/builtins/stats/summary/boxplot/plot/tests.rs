use super::*;

#[test]
fn jitter_zero_keeps_outliers_centered_and_datalim_clips() {
    let options = BoxplotOptions {
        data_lim: Some((0.0, 10.0)),
        jitter: 0.0,
        ..Default::default()
    };
    let stats = BoxStats {
        label: "A".to_string(),
        group_key: Vec::new(),
        position: 3.0,
        width: 0.5,
        color: Vec4::ONE,
        q1: 1.0,
        median: 2.0,
        q3: 3.0,
        lower_whisker: 1.0,
        upper_whisker: 3.0,
        outliers: vec![-5.0, 12.0],
        observation_count: 4,
    };
    let (x, y) = outlier_points(&stats, &options);
    assert_eq!(x, vec![3.0, 3.0]);
    assert_eq!(y, vec![0.0, 10.0]);
}
