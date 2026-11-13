//! Pixel-space stroke geometry shared by 2D line-like plots.
//!
//! Line widths and cap sizes are in pixels. Geometry is built in viewport pixels and mapped
//! back to data values, so it keeps its pixel size on axes of any aspect and on log axes.

use crate::core::vertex_utils;
use crate::core::{BoundingBox, PipelineType, Vertex};
use glam::{Vec3, Vec4};

/// The visible axes bounds when known, else the plot's own data bounds.
pub(crate) fn stroke_bounds(
    data_bounds: BoundingBox,
    view_bounds: Option<(f64, f64, f64, f64)>,
) -> BoundingBox {
    let Some((left, right, bottom, top)) = view_bounds else {
        return data_bounds;
    };
    if !(left.is_finite() && right.is_finite() && bottom.is_finite() && top.is_finite()) {
        return data_bounds;
    }
    let (min_x, max_x) = if left <= right {
        (left as f32, right as f32)
    } else {
        (right as f32, left as f32)
    };
    let (min_y, max_y) = if bottom <= top {
        (bottom as f32, top as f32)
    } else {
        (top as f32, bottom as f32)
    };
    if !(min_x.is_finite() && max_x.is_finite() && min_y.is_finite() && max_y.is_finite())
        || (max_x - min_x).abs() < 1e-12
        || (max_y - min_y).abs() < 1e-12
    {
        return data_bounds;
    }
    BoundingBox {
        min: Vec3::new(min_x, min_y, data_bounds.min.z),
        max: Vec3::new(max_x, max_y, data_bounds.max.z),
    }
}

/// Maps one axis between data values and viewport pixels for stroke extrusion.
///
/// Line widths are in pixels, so strokes are extruded in pixel space and mapped back to
/// data values. On a log axis the mapping goes through log10, matching the draw shader;
/// mapping linearly there would skew the width and send offsets to non-positive values.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StrokeAxis {
    pub log: bool,
    /// Axis minimum in mapped units (log10 of the value on a log axis).
    pub origin: f64,
    /// Axis extent in mapped units.
    pub span: f64,
    pub px_per_unit: f64,
}

impl StrokeAxis {
    pub fn new(min: f32, max: f32, extent_px: f32, log: bool, data: &[f64]) -> Self {
        let (lo, hi) = if log {
            let positive = |v: f64| v.is_finite() && v > 0.0;
            let data_min = data
                .iter()
                .copied()
                .filter(|v| positive(*v))
                .fold(f64::INFINITY, f64::min);
            let data_max = data
                .iter()
                .copied()
                .filter(|v| positive(*v))
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = if positive(min as f64) {
                min as f64
            } else {
                data_min
            };
            let hi = if positive(max as f64) {
                max as f64
            } else {
                data_max
            };
            if positive(lo) && positive(hi) {
                (lo.log10(), hi.log10())
            } else {
                (0.0, 1.0)
            }
        } else {
            (min as f64, max as f64)
        };
        let span = (hi - lo).abs().max(1e-12);
        Self {
            log,
            origin: lo,
            span,
            px_per_unit: extent_px.max(1.0) as f64 / span,
        }
    }

    /// Pixel offset of `value` from the axis minimum; NaN when the value can't be drawn.
    pub fn to_px(self, value: f64) -> f64 {
        let mapped = match (self.log, value > 0.0) {
            (true, true) => value.log10(),
            (true, false) => f64::NAN,
            (false, _) => value,
        };
        (mapped - self.origin) * self.px_per_unit
    }

    pub fn data_at_px(self, px: f32) -> f32 {
        let mapped = self.origin + px as f64 / self.px_per_unit;
        if self.log {
            10f64.powf(mapped) as f32
        } else {
            mapped as f32
        }
    }
}

/// Maps points between data values and viewport pixels on both axes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PixelMap {
    pub x: StrokeAxis,
    pub y: StrokeAxis,
}

impl PixelMap {
    pub fn new(
        bounds: BoundingBox,
        viewport_px: (u32, u32),
        log_axes: (bool, bool),
        x_data: &[f64],
        y_data: &[f64],
    ) -> Self {
        Self {
            x: StrokeAxis::new(
                bounds.min.x,
                bounds.max.x,
                viewport_px.0 as f32,
                log_axes.0,
                x_data,
            ),
            y: StrokeAxis::new(
                bounds.min.y,
                bounds.max.y,
                viewport_px.1 as f32,
                log_axes.1,
                y_data,
            ),
        }
    }

    /// Pixel position of a data point; `None` when it can't be drawn.
    pub fn to_px(self, x: f64, y: f64) -> Option<[f64; 2]> {
        let p = [self.x.to_px(x), self.y.to_px(y)];
        (p[0].is_finite() && p[1].is_finite()).then_some(p)
    }

    /// Like [`Self::to_px`], but a non-positive value on a log axis lands on that axis's lower
    /// edge. Stems from a zero baseline then start at the bottom of a log axes.
    pub fn to_px_clamped(self, x: f64, y: f64) -> Option<[f64; 2]> {
        let clamp = |axis: &StrokeAxis, value: f64| {
            if axis.log && value <= 0.0 {
                0.0
            } else {
                axis.to_px(value)
            }
        };
        let p = [clamp(&self.x, x), clamp(&self.y, y)];
        (p[0].is_finite() && p[1].is_finite()).then_some(p)
    }

    pub fn data_at_px(self, p: [f32; 3]) -> [f32; 3] {
        [self.x.data_at_px(p[0]), self.y.data_at_px(p[1]), p[2]]
    }
}

/// A straight segment in viewport pixels.
pub(crate) type PixelSegment = ([f64; 2], [f64; 2], Vec4);

/// Geometry for independent pixel-space segments, mapped back to data values: triangles
/// `width_px` wide, or 1 px lines when `width_px` is 1 or less.
pub(crate) fn segment_geometry(
    map: &PixelMap,
    segments: &[PixelSegment],
    width_px: f32,
) -> (Vec<Vertex>, PipelineType) {
    let mut vertices = Vec::new();
    let thick = width_px > 1.0;
    for (a, b, color) in segments {
        if thick {
            vertices.extend(vertex_utils::create_thick_polyline(
                &[a[0], b[0]],
                &[a[1], b[1]],
                *color,
                width_px,
            ));
        } else {
            for p in [a, b] {
                vertices.push(Vertex::new(
                    Vec3::new(p[0] as f32, p[1] as f32, 0.0),
                    *color,
                ));
            }
        }
    }
    for vertex in &mut vertices {
        vertex.position = map.data_at_px(vertex.position);
    }
    let pipeline = if thick {
        PipelineType::Triangles
    } else {
        PipelineType::Lines
    };
    (vertices, pipeline)
}

/// Pixel-space segments for data-space vertex pairs (as plots generate for line pipelines).
pub(crate) fn pixel_segments_from_pairs(
    map: &PixelMap,
    pairs: &[Vertex],
    clamp_log: bool,
) -> Vec<PixelSegment> {
    pairs
        .chunks_exact(2)
        .filter_map(|pair| {
            let to_px = |v: &Vertex| {
                let (x, y) = (v.position[0] as f64, v.position[1] as f64);
                if clamp_log {
                    map.to_px_clamped(x, y)
                } else {
                    map.to_px(x, y)
                }
            };
            Some((
                to_px(&pair[0])?,
                to_px(&pair[1])?,
                Vec4::from_array(pair[0].color),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_map() -> PixelMap {
        // 400 px over x in 1..1000 (three decades) and y in 0.01..100 (four decades).
        PixelMap::new(
            BoundingBox::new(Vec3::new(1.0, 0.01, 0.0), Vec3::new(1000.0, 100.0, 0.0)),
            (300, 400),
            (true, true),
            &[],
            &[],
        )
    }

    #[test]
    fn thick_segments_keep_pixel_width_on_log_axes() {
        let map = log_map();
        let a = map.to_px(10.0, 0.1).unwrap();
        let b = map.to_px(10.0, 10.0).unwrap();
        let (vertices, pipeline) = segment_geometry(&map, &[(a, b, Vec4::ONE)], 4.0);
        assert_eq!(pipeline, PipelineType::Triangles);
        // The vertical segment's two start corners sit 4 px apart horizontally.
        let x0 = map.x.to_px(vertices[0].position[0] as f64);
        let x1 = map.x.to_px(vertices[1].position[0] as f64);
        assert!(((x0 - x1).abs() - 4.0).abs() < 0.01, "{x0} {x1}");
        assert!(vertices
            .iter()
            .all(|v| v.position[0] > 0.0 && v.position[1] > 0.0));
    }

    #[test]
    fn zero_on_log_axis_is_dropped_or_clamped_to_the_edge() {
        let map = log_map();
        assert!(map.to_px(10.0, 0.0).is_none());
        assert_eq!(map.to_px_clamped(10.0, 0.0).unwrap()[1], 0.0);
    }
}
