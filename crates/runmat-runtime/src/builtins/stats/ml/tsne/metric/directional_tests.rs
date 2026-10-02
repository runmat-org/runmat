use super::{correlation_distance, cosine_distance};

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
