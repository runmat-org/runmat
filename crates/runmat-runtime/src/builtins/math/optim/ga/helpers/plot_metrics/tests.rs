use super::*;
#[test]
fn plot_reductions_preserve_wide_values_undefined_rows_and_constant_bins() {
    futures::executor::block_on(async {
        let mut cooperation = ScalarCooperation::for_builtin(super::super::super::IDENTITY, None);
        let summary = summarize(&[Some(f64::MAX), Some(-f64::MAX), None], &mut cooperation)
            .await
            .unwrap();
        assert_eq!(summary.mean, 0.);
        assert_eq!(summary.best, Some(1));
        assert_eq!(summary.count, 2);
        assert_eq!(
            distance(&[2, 2], &[0., 3., 0., 4.], &mut cooperation)
                .await
                .unwrap(),
            5.
        );
        let (_, bins) = histogram(&[Some(2.), None, Some(2.)], &mut cooperation)
            .await
            .unwrap();
        assert_eq!(bins, [2.]);
        let (labels, bins) = histogram(&[None, None], &mut cooperation).await.unwrap();
        assert!(labels.is_empty() && bins.is_empty());
    });
}
