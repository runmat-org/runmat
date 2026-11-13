//! The 3-D axis frame (axis lines, tick steps and ticks) shared by the renderer and the
//! overlay, the grid plane's extent, and the axis scale of a fixed data aspect.

use super::*;

/// The floor rectangle and line spacing of a 3-D grid plane, in data units.
pub(super) struct GridPlaneExtent {
    pub(super) min: (f32, f32),
    pub(super) max: (f32, f32),
    pub(super) major: [f32; 2],
    pub(super) minor: [f32; 2],
    /// Distances from the view target where the grid starts and finishes fading out.
    pub(super) fade: [f32; 2],
}

/// One axes' 3-D axis lines: the shared origin, each line's length in data units, and its tick
/// step. Ticks sit at multiples of the step that lie on the line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisFrame3d {
    pub origin: Vec3,
    /// A CAD world-origin triad, whose shared origin carries no tick.
    pub world_origin: bool,
    /// Where the z-axis stands: the origin for a CAD triad, otherwise a bottom corner of the
    /// axis limits chosen for the view.
    pub z_base: Vec3,
    pub lens: [f32; 3],
    pub steps: [f64; 3],
}

impl AxisFrame3d {
    const TARGET_TICKS: f64 = 4.0;
    const MAX_TICKS: usize = 12;

    pub(super) fn origin_triad(steps: [f64; 3], lens: [f32; 3]) -> Self {
        Self {
            origin: Vec3::ZERO,
            world_origin: true,
            z_base: Vec3::ZERO,
            lens,
            steps,
        }
    }

    pub(super) fn from_bounds(min: Vec3, max: Vec3, world_origin: bool) -> Option<Self> {
        if !(min.is_finite() && max.is_finite()) || min.cmpgt(max).any() {
            return None;
        }
        if world_origin {
            let reach = min.abs().max(max.abs());
            let reach = [reach.x, reach.y, reach.z];
            let steps = reach.map(|r| nice_tick_step((r as f64 / Self::TARGET_TICKS).max(1e-9)));
            let lens = [0, 1, 2].map(|k| reach[k].max(steps[k] as f32));
            return Some(Self::origin_triad(steps, lens));
        }
        let span = max - min;
        let largest = span.max_element().max(1.0e-9);
        let span = [span.x, span.y, span.z].map(|extent| extent.max(largest * 1.0e-6));
        let steps = span.map(|extent| nice_tick_step(extent as f64 / Self::TARGET_TICKS));
        Some(Self {
            origin: min,
            world_origin: false,
            z_base: min,
            lens: span,
            steps,
        })
    }

    /// Where one axis line starts.
    pub fn axis_start(&self, axis: usize) -> Vec3 {
        if axis == 2 {
            self.z_base
        } else {
            self.origin
        }
    }

    /// The ticks on one axis as (offset from the axis start along the axis, data value).
    pub fn ticks(&self, axis: usize) -> Vec<(f32, f64)> {
        let step = self.steps[axis];
        let start = self.origin[axis] as f64;
        let len = self.lens[axis] as f64;
        if !(step.is_finite() && step > 0.0 && len.is_finite() && len > 0.0) {
            return Vec::new();
        }
        let first = (start / step).ceil() as i64;
        let mut ticks = Vec::new();
        let mut index = first;
        while ticks.len() < Self::MAX_TICKS {
            let value = index as f64 * step;
            let offset = value - start;
            if offset > len * (1.0 + 1.0e-6) {
                break;
            }
            // The world-origin triad leaves its own origin unlabelled.
            if !(self.world_origin && index == 0) {
                ticks.push((offset as f32, value));
            }
            index += 1;
        }
        ticks
    }
}

/// The axis scale of a fixed data aspect: one data unit along axis k is `ratio[k]` long relative
/// to the others, and the largest relative extent of `size` fills the unit box.
pub(super) fn data_aspect_axis_scale(size: Vec3, ratio: [f32; 3]) -> Vec3 {
    let units = (size.x / ratio[0])
        .max(size.y / ratio[1])
        .max(size.z / ratio[2])
        .max(f32::MIN_POSITIVE);
    Vec3::new(
        1.0 / (units * ratio[0]),
        1.0 / (units * ratio[1]),
        1.0 / (units * ratio[2]),
    )
}

/// A 1-2-5 step at or above `raw`.
pub(crate) fn nice_tick_step(raw: f64) -> f64 {
    if !raw.is_finite() || raw <= 0.0 {
        return 1.0;
    }
    let pow10 = 10.0_f64.powf(raw.log10().floor());
    let norm = raw / pow10;
    let mult = if norm <= 1.0 {
        1.0
    } else if norm <= 2.0 {
        2.0
    } else if norm <= 5.0 {
        5.0
    } else {
        10.0
    };
    mult * pow10
}

#[cfg(test)]
mod axis_frame_tests {
    use super::*;

    #[test]
    fn plot_axes_tick_their_own_limits_from_the_lower_corner() {
        let frame = AxisFrame3d::from_bounds(
            Vec3::new(60.0, 0.05, 0.0),
            Vec3::new(140.0, 2.0, 48.0),
            false,
        )
        .unwrap();
        assert_eq!(frame.origin, Vec3::new(60.0, 0.05, 0.0));
        let x: Vec<f64> = frame.ticks(0).into_iter().map(|(_, v)| v).collect();
        assert_eq!(x, vec![60.0, 80.0, 100.0, 120.0, 140.0]);
        let y: Vec<f64> = frame.ticks(1).into_iter().map(|(_, v)| v).collect();
        assert_eq!(y, vec![0.5, 1.0, 1.5, 2.0]);
        let z = frame.ticks(2);
        assert_eq!(z.first(), Some(&(0.0, 0.0)));
        assert!(z.iter().all(|(offset, _)| *offset <= 48.0 + 1.0e-3));
    }

    #[test]
    fn cad_axes_keep_a_world_origin_triad_reaching_the_model() {
        let frame = AxisFrame3d::from_bounds(
            Vec3::new(-1160.0, -660.0, -40.0),
            Vec3::new(1160.0, 660.0, 180.0),
            true,
        )
        .unwrap();
        assert_eq!(frame.origin, Vec3::ZERO);
        assert_eq!(frame.lens[0], 1160.0);
        let x: Vec<f64> = frame.ticks(0).into_iter().map(|(_, v)| v).collect();
        assert_eq!(x, vec![500.0, 1000.0]);
        assert!(AxisFrame3d::from_bounds(Vec3::ONE, Vec3::ZERO, false).is_none());
    }
}
