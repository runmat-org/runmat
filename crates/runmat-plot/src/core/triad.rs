//! Where the corner orientation triad sits: inset from the plot's bottom-left corner far enough
//! that every arm, arrowhead and label stays inside the plot, whichever way the camera turns it.

/// The triad's origin in screen pixels (y down) for a plot spanning `min`..`max`.
///
/// `dirs` are the arms' on-screen directions (unit length, y down), `reach` is how far an arm
/// and its label extend from the origin, and `pad` is the least inset from each edge. An arm
/// pointing left or down pushes the origin in by its reach along that side.
pub fn triad_origin(
    min: [f32; 2],
    max: [f32; 2],
    pad: f32,
    reach: f32,
    dirs: &[[f32; 2]],
) -> [f32; 2] {
    let left = dirs.iter().map(|d| -d[0]).fold(0.0f32, f32::max) * reach;
    let down = dirs.iter().map(|d| d[1]).fold(0.0f32, f32::max) * reach;
    [min[0] + pad.max(left), max[1] - pad.max(down)]
}

#[cfg(test)]
mod tests {
    use super::triad_origin;

    #[test]
    fn an_arm_pointing_left_or_down_stays_inside_the_plot() {
        let (pad, reach) = (30.0, 90.0);
        let dirs = [[-0.8, 0.6], [0.95, 0.31], [0.0, -1.0]];
        let o = triad_origin([0.0, 0.0], [800.0, 600.0], pad, reach, &dirs);
        for d in dirs {
            let tip = [o[0] + d[0] * reach, o[1] + d[1] * reach];
            assert!(
                tip[0] >= 0.0 && tip[0] <= 800.0 && tip[1] >= 0.0 && tip[1] <= 600.0,
                "{tip:?}"
            );
        }
        assert!(
            (o[0] - 72.0).abs() < 1e-3 && (o[1] - (600.0 - 54.0)).abs() < 1e-3,
            "{o:?}"
        );
    }

    #[test]
    fn arms_pointing_up_and_right_keep_the_least_inset() {
        let o = triad_origin(
            [10.0, 0.0],
            [800.0, 600.0],
            30.0,
            90.0,
            &[[1.0, 0.0], [0.0, -1.0]],
        );
        assert_eq!(o, [40.0, 570.0]);
    }
}
