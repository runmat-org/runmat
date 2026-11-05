//! Where a legend box sits inside its axes, from MATLAB's `Location` names.
//!
//! The eight compass locations put the box against that side or corner of the plot area.
//! `best` puts it where it covers the fewest data points, trying the corners, the side
//! midpoints and a grid of positions between them. Ties go to the earlier candidate,
//! starting with `northeast`, so a plot with free space at its upper right keeps MATLAB's
//! default place. Unknown names and the `*outside` locations fall back to `northeast` inside
//! the axes.

use egui::{Pos2, Rect, Vec2};

/// The box's top-left corner for a legend of `size` in `plot`, `margin` in from the sides.
/// `points` are the data positions on screen; only `best` reads them.
pub fn legend_origin(plot: Rect, size: Vec2, margin: f32, location: Option<&str>, points: &[Pos2]) -> Pos2 {
    let location = location.map(|s| s.trim().to_ascii_lowercase());
    let at = |fx: f32, fy: f32| -> Pos2 {
        let free_x = (plot.width() - size.x - 2.0 * margin).max(0.0);
        let free_y = (plot.height() - size.y - 2.0 * margin).max(0.0);
        Pos2::new(plot.min.x + margin + fx * free_x, plot.min.y + margin + fy * free_y)
    };
    match location.as_deref() {
        Some("northwest") | Some("northwestoutside") => at(0.0, 0.0),
        Some("north") | Some("northoutside") => at(0.5, 0.0),
        Some("southeast") | Some("southeastoutside") => at(1.0, 1.0),
        Some("southwest") | Some("southwestoutside") => at(0.0, 1.0),
        Some("south") | Some("southoutside") => at(0.5, 1.0),
        Some("east") | Some("eastoutside") => at(1.0, 0.5),
        Some("west") | Some("westoutside") => at(0.0, 0.5),
        Some("best") | Some("bestoutside") => best(points, size, &at),
        _ => at(1.0, 0.0),
    }
}

/// Candidates in preference order: the four corners (MATLAB's default first), the side
/// midpoints, then a 5 x 5 grid between them.
fn best(points: &[Pos2], size: Vec2, at: &dyn Fn(f32, f32) -> Pos2) -> Pos2 {
    let mut candidates = vec![(1.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.0), (0.5, 1.0), (1.0, 0.5), (0.0, 0.5)];
    for i in 0..5 {
        for j in 0..5 {
            candidates.push((i as f32 / 4.0, j as f32 / 4.0));
        }
    }
    let mut chosen = at(1.0, 0.0);
    let mut fewest = usize::MAX;
    for (fx, fy) in candidates {
        let origin = at(fx, fy);
        let rect = Rect::from_min_size(origin, size).expand(2.0);
        let covered = points.iter().filter(|p| rect.contains(**p)).count();
        if covered < fewest {
            fewest = covered;
            chosen = origin;
            if covered == 0 {
                break;
            }
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plot() -> Rect {
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(400.0, 300.0))
    }

    #[test]
    fn compass_locations_sit_against_their_sides() {
        let size = Vec2::new(100.0, 40.0);
        assert_eq!(legend_origin(plot(), size, 8.0, None, &[]), Pos2::new(292.0, 8.0));
        assert_eq!(legend_origin(plot(), size, 8.0, Some("southwest"), &[]), Pos2::new(8.0, 252.0));
        assert_eq!(legend_origin(plot(), size, 8.0, Some("North"), &[]), Pos2::new(150.0, 8.0));
    }

    #[test]
    fn best_keeps_northeast_when_it_is_free_and_avoids_the_data_otherwise() {
        let size = Vec2::new(100.0, 40.0);
        // A falling line along the bottom: the upper right is free.
        let low: Vec<Pos2> = (0..100).map(|k| Pos2::new(k as f32 * 4.0, 250.0 + k as f32 * 0.4)).collect();
        assert_eq!(legend_origin(plot(), size, 8.0, Some("best"), &low), Pos2::new(292.0, 8.0));
        // A response flat across the top, as a magnitude plot's passband: the box drops below it.
        let high: Vec<Pos2> = (0..100).map(|k| Pos2::new(k as f32 * 4.0, 20.0 + if k > 70 { (k - 70) as f32 * 8.0 } else { 0.0 })).collect();
        let origin = legend_origin(plot(), size, 8.0, Some("best"), &high);
        let rect = Rect::from_min_size(origin, size);
        assert!(high.iter().all(|p| !rect.contains(*p)), "the legend at {origin:?} covers data");
    }
}
