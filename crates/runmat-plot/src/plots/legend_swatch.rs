//! Legend swatches: the dash pattern a line series' swatch repeats.
/// The pieces of a legend swatch from `x0` to `x1` for `style`, in screen px (`scale` per px):
/// one piece for a solid line, dashes of 4 with gaps of 2.5, dots of 1.2 with gaps of 2, and
/// dash-dots alternating the two. A dotted piece is short enough to read as a dot at 2 px wide.
pub fn legend_dashes(
    style: crate::plots::line::LineStyle,
    x0: f32,
    x1: f32,
    scale: f32,
) -> Vec<(f32, f32)> {
    use crate::plots::line::LineStyle;
    let pattern: &[(f32, f32)] = match style {
        LineStyle::Solid | LineStyle::None => return vec![(x0, x1)],
        LineStyle::Dashed => &[(4.0, 2.5)],
        LineStyle::Dotted => &[(1.2, 2.0)],
        LineStyle::DashDot => &[(4.0, 2.0), (1.2, 2.0)],
    };
    let mut out = Vec::new();
    let mut x = x0;
    let mut k = 0;
    while x < x1 {
        let (on, off) = pattern[k % pattern.len()];
        out.push((x, (x + on * scale).min(x1)));
        x += (on + off) * scale;
        k += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::legend_dashes;
    use crate::plots::line::LineStyle;

    #[test]
    fn a_legend_swatch_repeats_the_series_dash_pattern() {
        assert_eq!(
            legend_dashes(LineStyle::Solid, 0.0, 14.0, 1.0),
            vec![(0.0, 14.0)]
        );
        // 14 px of 4-on, 2.5-off dashes: three dashes, the last cut at the swatch's end.
        let dashed = legend_dashes(LineStyle::Dashed, 0.0, 14.0, 1.0);
        assert_eq!(dashed, vec![(0.0, 4.0), (6.5, 10.5), (13.0, 14.0)]);
        // Dots are short pieces, more of them than dashes.
        let dotted = legend_dashes(LineStyle::Dotted, 0.0, 14.0, 1.0);
        assert!(dotted.len() > dashed.len() && dotted.iter().all(|(a, b)| b - a <= 1.2 + 1e-6));
        // Dash-dot alternates a dash and a dot.
        let dd = legend_dashes(LineStyle::DashDot, 0.0, 14.0, 1.0);
        assert_eq!(dd[0], (0.0, 4.0));
        assert!((dd[1].1 - dd[1].0 - 1.2).abs() < 1e-6);
        // At 2x the pattern scales with the swatch.
        assert_eq!(
            legend_dashes(LineStyle::Dashed, 0.0, 28.0, 2.0)[1],
            (13.0, 21.0)
        );
    }
}
