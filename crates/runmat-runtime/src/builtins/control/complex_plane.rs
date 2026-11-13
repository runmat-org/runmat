/// Frames the current axes on the points that matter in the complex plane (poles, zeros, axis
/// crossings), as MATLAB's `rlocus` and `pzmap` do, instead of every plotted point: the origin is
/// always inside, the imaginary range is symmetric, and `pad` (a fraction of the larger span) keeps
/// markers off the frame.
pub(super) async fn frame_complex_plane(points: &[num_complex::Complex64], pad: f64) {
    let finite = points
        .iter()
        .filter(|p| p.re.is_finite() && p.im.is_finite());
    let (mut xmin, mut xmax, mut ymax) = (0.0_f64, 0.0_f64, 0.0_f64);
    for p in finite {
        xmin = xmin.min(p.re);
        xmax = xmax.max(p.re);
        ymax = ymax.max(p.im.abs());
    }
    let mut span = (xmax - xmin).max(2.0 * ymax);
    if span <= f64::EPSILON {
        span = 1.0;
    }
    let margin = pad * span;
    let limits = |lo: f64, hi: f64| {
        runmat_value::Tensor::new(vec![lo, hi], vec![1, 2]).map(runmat_value::Value::Tensor)
    };
    if let (Ok(x), Ok(y)) = (
        limits(xmin - margin, xmax + margin),
        limits(-(ymax + margin), ymax + margin),
    ) {
        let _ = crate::call_builtin_async("xlim", &[x]).await;
        let _ = crate::call_builtin_async("ylim", &[y]).await;
    }
}
