//! One world-space camera for every saved view of an example, including deformation.
use runmat_plot::core::{camera::ProjectionType, BoundingBox, Camera};
use runmat_runtime::builtins::plotting::PlotCameraProjection;
use runmat_runtime::builtins::plotting::{PlotCameraState, PlotSurfaceCameraState};

/// The standard poster direction, from the target towards the camera: about MATLAB
/// `view(19, 23)`.
const POSTER_DIRECTION: [f32; 3] = [0.35, -1.0, 0.45];

/// The direction from the target towards the camera for MATLAB `view(az, el)` in degrees:
/// azimuth from the -y axis towards +x, elevation above the xy plane.
pub(super) fn view_direction(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let (az, el) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
    [az.sin() * el.cos(), -az.cos() * el.cos(), el.sin()]
}

fn union(bounds: impl Iterator<Item = BoundingBox>) -> BoundingBox {
    bounds
        .reduce(|left, right| BoundingBox {
            min: left.min.min(right.min),
            max: left.max.max(right.max),
        })
        .expect("snapshot contains at least one view")
}

/// Clear space the poster keeps on every side of the framed bounds, as a share of the frame's
/// width (left and right) and height (top and bottom).
pub(super) const FRAME_MARGIN: f32 = 0.08;

/// Where the framed bounds go in the poster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Layout {
    /// Clear space on every side, as a share of the frame's width and height.
    pub margin: f32,
    /// The scalar legend's footprint from the frame's top-right corner, as shares of the
    /// frame's width and height; the bounds go beside or below it when there is room.
    pub legend: Option<(f32, f32)>,
    /// Frame the bounds to this share of the frame width instead of filling the free area
    /// (they still fit inside it).
    pub width_share: Option<f32>,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            margin: FRAME_MARGIN,
            legend: None,
            width_share: None,
        }
    }
}

/// A free area of the frame in normalized device coordinates: x and y run from -1 at the left
/// and bottom edges to 1 at the right and top edges.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Area {
    x: (f64, f64),
    y: (f64, f64),
}

impl Area {
    fn width(&self) -> f64 {
        self.x.1 - self.x.0
    }
    fn height(&self) -> f64 {
        self.y.1 - self.y.0
    }
}

/// A fitted bounding box gives up at most this much of its size to clear the legend; past it the
/// legend is drawn over the frame's corner instead.
const LEGEND_ROOM: f64 = 0.8;

/// The frame's free area after the margin, and the areas beside and below the legend.
fn areas(layout: &Layout) -> (Area, Vec<Area>) {
    let m = 2.0 * layout.margin.clamp(0.0, 0.4) as f64;
    let full = Area {
        x: (-1.0 + m, 1.0 - m),
        y: (-1.0 + m, 1.0 - m),
    };
    let Some((width, height)) = layout.legend else {
        return (full, Vec::new());
    };
    // The bounds keep half a margin of clear space from the legend's background.
    let gap = 0.5 * m;
    let beside = Area {
        x: (full.x.0, 1.0 - 2.0 * width as f64 - gap),
        ..full
    };
    let below = Area {
        y: (full.y.0, 1.0 - 2.0 * height as f64 - gap),
        ..full
    };
    let usable = |area: &Area| area.width() > 0.2 && area.height() > 0.2;
    (full, [beside, below].into_iter().filter(usable).collect())
}

/// A point or direction in double precision, for the fit's iterations.
#[derive(Debug, Clone, Copy, PartialEq)]
struct V3([f64; 3]);

impl V3 {
    const ZERO: Self = Self([0.0; 3]);
    fn from_f32(value: [f32; 3]) -> Self {
        Self(value.map(f64::from))
    }
    fn to_f32(self) -> [f32; 3] {
        self.0.map(|value| value as f32)
    }
    fn dot(self, other: Self) -> f64 {
        (0..3).map(|i| self.0[i] * other.0[i]).sum()
    }
    fn cross(self, other: Self) -> Self {
        let ([a0, a1, a2], [b0, b1, b2]) = (self.0, other.0);
        Self([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])
    }
    fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
    fn normalize(self) -> Self {
        self * (1.0 / self.length())
    }
}

impl std::ops::Add for V3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + other.0[i]))
    }
}

impl std::ops::Sub for V3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] - other.0[i]))
    }
}

impl std::ops::Mul<f64> for V3 {
    type Output = Self;
    fn mul(self, scale: f64) -> Self {
        Self(self.0.map(|value| value * scale))
    }
}

/// An orthonormal view basis for a camera looking back along `direction`.
#[derive(Debug, Clone, Copy)]
struct Basis {
    /// From the target towards the camera.
    back: V3,
    right: V3,
    up: V3,
    /// The world up the camera keeps (+z unless it looks straight along z).
    world_up: V3,
    tan_x: f64,
    tan_y: f64,
}

impl Basis {
    fn new(direction: [f32; 3], fov: f32, aspect_ratio: f32) -> Self {
        let back = V3::from_f32(direction).normalize();
        let hint = if back.cross(V3([0.0, 0.0, 1.0])).length() > 1e-4 {
            V3([0.0, 0.0, 1.0])
        } else {
            V3([0.0, 1.0, 0.0])
        };
        let right = hint.cross(back).normalize();
        let up = back.cross(right).normalize();
        let tan_y = (f64::from(fov) * 0.5).tan();
        Self {
            back,
            right,
            up,
            world_up: hint,
            tan_x: tan_y * f64::from(aspect_ratio),
            tan_y,
        }
    }

    /// The normalized device rectangle of `points` seen from `position`, or `None` when a
    /// point is not in front of the camera.
    fn projected(&self, position: V3, points: &[V3]) -> Option<Area> {
        let mut area = Area {
            x: (f64::INFINITY, f64::NEG_INFINITY),
            y: (f64::INFINITY, f64::NEG_INFINITY),
        };
        for point in points {
            let offset = *point - position;
            let depth = -offset.dot(self.back);
            if depth <= 1e-9 {
                return None;
            }
            let x = offset.dot(self.right) / (depth * self.tan_x);
            let y = offset.dot(self.up) / (depth * self.tan_y);
            area.x = (area.x.0.min(x), area.x.1.max(x));
            area.y = (area.y.0.min(y), area.y.1.max(y));
        }
        Some(area)
    }
}

/// The camera position that shows `points` centered in `area`, as large as fits (or at
/// `width_share` of the frame width when given, if that fits), looking back along the basis.
fn fit(basis: &Basis, points: &[V3], area: Area, width_share: Option<f64>) -> V3 {
    let center =
        points.iter().fold(V3::ZERO, |sum, point| sum + *point) * (1.0 / points.len() as f64);
    let radius = points
        .iter()
        .map(|point| (*point - center).length())
        .fold(0.0, f64::max)
        .max(1e-9);
    let mut distance = 2.0 * radius / basis.tan_y.min(basis.tan_x);
    // Offset of the camera across the view, along `right` and `up`.
    let mut lateral = (0.0, 0.0);
    let position = |distance: f64, lateral: (f64, f64)| {
        center + basis.back * distance + basis.right * lateral.0 + basis.up * lateral.1
    };
    // Perspective makes size and center depend on the distance; a few rounds of rescaling and
    // recentering converge to the fit.
    for _ in 0..64 {
        let Some(seen) = basis.projected(position(distance, lateral), points) else {
            distance *= 2.0;
            continue;
        };
        let (width, height) = (seen.width().max(1e-12), seen.height().max(1e-12));
        let fill = (area.width() / width).min(area.height() / height);
        let scale = match width_share {
            Some(share) => (2.0 * share / width).min(fill),
            None => fill,
        };
        distance /= scale;
        let Some(seen) = basis.projected(position(distance, lateral), points) else {
            continue;
        };
        let shift = (
            (seen.x.0 + seen.x.1 - area.x.0 - area.x.1) * 0.5 * basis.tan_x,
            (seen.y.0 + seen.y.1 - area.y.0 - area.y.1) * 0.5 * basis.tan_y,
        );
        lateral = (
            lateral.0 + shift.0 * distance,
            lateral.1 + shift.1 * distance,
        );
        if (scale - 1.0).abs() < 1e-7 && shift.0.abs().max(shift.1.abs()) < 1e-7 {
            break;
        }
    }
    position(distance, lateral)
}

/// The eight corners of `bounds`.
fn corners(bounds: BoundingBox) -> [V3; 8] {
    std::array::from_fn(|corner| {
        let pick = |axis: usize| {
            f64::from(if corner & (1 << axis) == 0 {
                bounds.min[axis]
            } else {
                bounds.max[axis]
            })
        };
        V3([pick(0), pick(1), pick(2)])
    })
}

/// The poster camera, also saved as every view's initial camera. It frames the union of
/// `framed`, grown `padding` times about its center, inside the frame less `layout.margin` on
/// every side, at the poster's `aspect_ratio` (width over height): beside or below the legend
/// when that costs the bounds at most a fifth of their size, centered otherwise. Its clip planes
/// reach every corner of `drawn`, so context geometry past the frame is drawn rather than
/// clipped. `direction` points from the target towards the camera (default: the standard
/// poster view).
pub(super) fn shared_camera(
    framed: impl Iterator<Item = BoundingBox>,
    drawn: impl Iterator<Item = BoundingBox>,
    padding: f32,
    direction: Option<[f32; 3]>,
    aspect_ratio: f32,
    layout: &Layout,
) -> PlotSurfaceCameraState {
    let framed = union(framed);
    let drawn = union(drawn);
    let center = (framed.min + framed.max) * 0.5;
    let half = (framed.max - framed.min) * 0.5 * padding;
    let points = corners(BoundingBox {
        min: center - half,
        max: center + half,
    });
    let mut camera = Camera::new();
    camera.update_aspect_ratio(aspect_ratio);
    let ProjectionType::Perspective { fov, .. } = camera.projection else {
        unreachable!("3D camera must be perspective")
    };
    let basis = Basis::new(direction.unwrap_or(POSTER_DIRECTION), fov, aspect_ratio);
    let width_share = layout.width_share.map(f64::from);
    let (full, beside_legend) = areas(layout);
    let placed = |area: Area| {
        let position = fit(&basis, &points, area, width_share);
        let seen = basis
            .projected(position, &points)
            .expect("a fitted camera sees every framed point");
        (position, seen.width())
    };
    let (mut position, size) = placed(full);
    // The arrangement beside or below the legend that keeps the bounds largest (beside on a
    // tie), when it keeps at least LEGEND_ROOM of their size.
    if let Some((best, best_size)) =
        beside_legend
            .into_iter()
            .map(placed)
            .fold(None, |best: Option<(V3, f64)>, next| match best {
                Some(best) if next.1 <= best.1 * (1.0 + 1e-6) => Some(best),
                _ => Some(next),
            })
    {
        if best_size >= LEGEND_ROOM * size - 1e-9 {
            position = best;
        }
    }
    // The target stays on the line of sight at the framed center's distance, so orbiting the
    // live view turns about the part.
    let distance = (position - V3::from_f32(center.to_array())).dot(basis.back);
    camera.position = position.to_f32().into();
    camera.target = (position - basis.back * distance).to_f32().into();
    camera.up = basis.world_up.to_f32().into();
    set_clip_planes(&mut camera, [framed, drawn]);
    let ProjectionType::Perspective { fov, near, far } = camera.projection else {
        unreachable!("3D camera must be perspective")
    };
    PlotSurfaceCameraState {
        active_axes: 0,
        axes: vec![PlotCameraState {
            position: camera.position.to_array(),
            target: camera.target.to_array(),
            up: camera.up.to_array(),
            zoom: camera.zoom,
            aspect_ratio: camera.aspect_ratio,
            projection: PlotCameraProjection::Perspective { fov, near, far },
        }],
    }
}

/// Place the near and far planes around every corner of `boxes`. A near plane stops at 1/5000
/// of the far one, which keeps depth precision when the camera sits inside context geometry.
fn set_clip_planes(camera: &mut Camera, boxes: [BoundingBox; 2]) {
    let forward = V3::from_f32((camera.target - camera.position).normalize().to_array());
    let (mut closest, mut farthest) = (f64::INFINITY, 0.0_f64);
    for bounds in boxes {
        for point in corners(bounds) {
            let depth = (point - V3::from_f32(camera.position.to_array())).dot(forward);
            closest = closest.min(depth);
            farthest = farthest.max(depth);
        }
    }
    if let ProjectionType::Perspective { near, far, .. } = &mut camera.projection {
        *far = (farthest.max(1e-6) * 1.02) as f32;
        *near = ((closest * 0.98) as f32).max(*far / 5000.0);
    }
}

/// Whether every corner of `bounds` lies in front of the saved camera and inside its frame, at
/// least `margin` (a share of the frame's width and height) from each edge.
pub(super) fn inside_frame(
    state: &PlotSurfaceCameraState,
    bounds: BoundingBox,
    margin: f32,
) -> bool {
    let Some(axis) = state.axes.get(state.active_axes) else {
        return false;
    };
    let PlotCameraProjection::Perspective { fov, near, .. } = axis.projection else {
        return false;
    };
    let position = V3::from_f32(axis.position);
    let back = (position - V3::from_f32(axis.target)).normalize();
    let basis = Basis::new(back.to_f32(), fov, axis.aspect_ratio);
    let edge = 1.0 - 2.0 * f64::from(margin);
    let points = corners(bounds);
    let in_front = points
        .iter()
        .all(|point| -(*point - position).dot(basis.back) > f64::from(near));
    in_front
        && basis.projected(position, &points).is_some_and(|seen| {
            seen.x.0 >= -edge && seen.x.1 <= edge && seen.y.0 >= -edge && seen.y.1 <= edge
        })
}

/// Move the near and far planes of a saved camera out, never in, until every corner of
/// `drawn` lies between them.
pub(super) fn widen_saved_clip_planes(state: &mut PlotSurfaceCameraState, drawn: BoundingBox) {
    for axis in &mut state.axes {
        let position = V3::from_f32(axis.position);
        let forward = (V3::from_f32(axis.target) - position).normalize();
        let (mut closest, mut farthest) = (f64::INFINITY, 0.0_f64);
        for point in corners(drawn) {
            let depth = (point - position).dot(forward);
            closest = closest.min(depth);
            farthest = farthest.max(depth);
        }
        if let PlotCameraProjection::Perspective { near, far, .. } = &mut axis.projection {
            *far = far.max((farthest * 1.02) as f32);
            *near = near.min(((closest * 0.98) as f32).max(*far / 5000.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn unit(a: [f32; 3]) -> [f32; 3] {
        let length = dot(a, a).sqrt();
        a.map(|value| value / length)
    }

    fn bounds(min: [f32; 3], max: [f32; 3]) -> BoundingBox {
        BoundingBox {
            min: min.into(),
            max: max.into(),
        }
    }

    /// Where a saved camera draws each corner of `boxes`, in normalized device coordinates
    /// (-1 to 1 across the frame), as the renderer projects it.
    fn projected(state: &PlotSurfaceCameraState, boxes: &[BoundingBox]) -> Vec<(f32, f32)> {
        let camera = &state.axes[0];
        let PlotCameraProjection::Perspective { fov, .. } = camera.projection else {
            panic!("snapshot cameras are perspective");
        };
        let forward = unit(sub(camera.target, camera.position));
        let right = unit(cross(forward, camera.up));
        let up = cross(right, forward);
        let tan_y = (fov * 0.5).tan();
        let tan_x = tan_y * camera.aspect_ratio;
        boxes
            .iter()
            .flat_map(|view| corners(*view))
            .map(|point| {
                let offset = sub(point.to_f32(), camera.position);
                let depth = dot(offset, forward);
                assert!(depth > 0.0, "corner behind the camera");
                (
                    dot(offset, right) / (depth * tan_x),
                    dot(offset, up) / (depth * tan_y),
                )
            })
            .collect()
    }

    /// The projected rectangle: (left, right, bottom, top).
    fn extent(points: &[(f32, f32)]) -> (f32, f32, f32, f32) {
        points.iter().fold(
            (
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ),
            |(l, r, b, t), &(x, y)| (l.min(x), r.max(x), b.min(y), t.max(y)),
        )
    }

    const EDGE: f32 = 1.0 - 2.0 * FRAME_MARGIN;

    /// A long part with a tab far from its centre (a busbar) and a small deformed copy: the
    /// union of both views' bounds is centered in the poster with an 8% margin on every side,
    /// and touches that margin on its limiting side.
    #[test]
    fn posters_fit_the_bounds_inside_the_margin() {
        let views = [
            bounds([0.0, 0.0, 0.0], [0.40, 0.03, 0.006]),
            bounds([0.38, -0.02, 0.0], [0.45, 0.03, 0.03]),
        ];
        let state = shared_camera(
            views.iter().copied(),
            views.iter().copied(),
            1.0,
            None,
            8.0 / 5.0,
            &Layout::default(),
        );
        let union = bounds([0.0, -0.02, 0.0], [0.45, 0.03, 0.03]);
        let (left, right, bottom, top) = extent(&projected(&state, &[union]));
        for side in [-left, right, -bottom, top] {
            assert!(
                side <= EDGE + 1e-4,
                "a corner reaches into the margin: {side}"
            );
        }
        assert!(
            (left + right).abs() < 1e-4 && (bottom + top).abs() < 1e-4,
            "not centered"
        );
        assert!(
            right >= EDGE - 1e-4 || top >= EDGE - 1e-4,
            "the fit is loose"
        );
    }

    /// The cantilever of the docs example seen from the standard poster direction, with the
    /// legend in the top-right corner: the beam goes below the legend and stays inside the
    /// margin, and the saved camera targets a point on its line of sight through the beam.
    #[test]
    fn a_long_part_clears_the_legend_when_there_is_room() {
        let beam = bounds([0.0, 0.0, -0.025], [1.0, 0.1, 0.025]);
        let legend = (0.36, 0.17);
        let layout = Layout {
            legend: Some(legend),
            ..Layout::default()
        };
        let state = shared_camera(
            std::iter::once(beam),
            std::iter::once(beam),
            1.0,
            None,
            1.6,
            &layout,
        );
        let points = projected(&state, &[beam]);
        let (left, right, bottom, top) = extent(&points);
        for side in [-left, right, -bottom, top] {
            assert!(side <= EDGE + 1e-4);
        }
        let (legend_left, legend_bottom) = (1.0 - 2.0 * legend.0, 1.0 - 2.0 * legend.1);
        assert!(
            right <= legend_left + 1e-4 || top <= legend_bottom + 1e-4,
            "the bounds reach under the legend"
        );
        // Clearing the legend costs this beam less than a fifth of its size.
        let centered = shared_camera(
            std::iter::once(beam),
            std::iter::once(beam),
            1.0,
            None,
            1.6,
            &Layout::default(),
        );
        let (cl, cr, _, _) = extent(&projected(&centered, &[beam]));
        assert!(right - left >= 0.8 * (cr - cl) - 1e-4);
    }

    /// A part as tall as it is wide gives up too much of its size beside or below the legend,
    /// so it stays centered.
    #[test]
    fn a_compact_part_stays_centered_when_the_legend_leaves_no_room() {
        let cube = bounds([0.0, 0.0, 0.0], [0.1, 0.1, 0.1]);
        let layout = Layout {
            legend: Some((0.6, 0.45)),
            ..Layout::default()
        };
        let state = shared_camera(
            std::iter::once(cube),
            std::iter::once(cube),
            1.0,
            None,
            1.6,
            &layout,
        );
        let (left, right, bottom, top) = extent(&projected(&state, &[cube]));
        assert!((left + right).abs() < 1e-4 && (bottom + top).abs() < 1e-4);
        assert!(top >= EDGE - 1e-4, "a compact part fills the frame height");
    }

    /// A framing target at a share of the width: the module spans that share of the frame,
    /// beside the legend.
    #[test]
    fn a_target_spans_its_share_of_the_width_beside_the_legend() {
        let module = bounds([-1.03, 0.07, 0.03], [-0.80, 0.49, 0.13]);
        let pack = bounds([-1.2, -0.6, 0.0], [1.2, 0.6, 0.25]);
        let legend = (0.2, 0.12);
        let layout = Layout {
            legend: Some(legend),
            width_share: Some(0.6),
            ..Layout::default()
        };
        let state = shared_camera(
            std::iter::once(module),
            std::iter::once(pack),
            1.0,
            Some(view_direction(-55.0, 18.0)),
            1.6,
            &layout,
        );
        let (left, right, bottom, top) = extent(&projected(&state, &[module]));
        assert!(
            ((right - left) - 1.2).abs() < 1e-3,
            "width {}",
            right - left
        );
        assert!(right <= 1.0 - 2.0 * legend.0 + 1e-4, "beside the legend");
        assert!(-left <= EDGE + 1e-4 && -bottom <= EDGE + 1e-4 && top <= EDGE + 1e-4);
        // The clip planes still reach every corner of the pack.
        let camera = &state.axes[0];
        let PlotCameraProjection::Perspective { near, far, .. } = camera.projection else {
            panic!("snapshot cameras are perspective");
        };
        let forward = unit(sub(camera.target, camera.position));
        for point in corners(pack) {
            let depth = dot(sub(point.to_f32(), camera.position), forward);
            assert!(depth < far, "pack corner beyond the far plane");
        }
        assert!(near > 0.0 && near >= far / 5000.0 * 0.999);
    }

    /// A millimeter part (the fuse link, 10 x 1 x 0.25 mm) a few millimeters from the camera:
    /// the clip planes come from the fit, so every corner lies between them and the part is
    /// drawn whole.
    #[test]
    fn clip_planes_follow_the_fit_for_a_millimeter_part() {
        let link = bounds([0.0, 0.0, 0.0], [0.010, 0.001, 0.00025]);
        let state = shared_camera(
            std::iter::once(link),
            std::iter::once(link),
            1.0,
            None,
            1.6,
            &Layout {
                legend: Some((0.36, 0.17)),
                ..Layout::default()
            },
        );
        let camera = &state.axes[0];
        let PlotCameraProjection::Perspective { near, far, .. } = camera.projection else {
            panic!("snapshot cameras are perspective");
        };
        let forward = unit(sub(camera.target, camera.position));
        let distance = dot(sub(camera.target, camera.position), forward);
        assert!(
            distance < 0.05,
            "the camera sits close to a small part: {distance} m"
        );
        for point in corners(link) {
            let depth = dot(sub(point.to_f32(), camera.position), forward);
            assert!(near < depth && depth < far, "{near} < {depth} < {far}");
        }
        assert!(near > 0.0 && far < 1.0);
        let (left, right, bottom, top) = extent(&projected(&state, &[link]));
        for side in [-left, right, -bottom, top] {
            assert!(side <= EDGE + 1e-4);
        }
    }

    /// MATLAB's view convention: view(0, 0) looks along +y from -y, view(90, 0) along -x
    /// from +x, view(az, 90) straight down; the camera keeps the poster's aspect.
    #[test]
    fn view_angles_follow_matlab() {
        let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-6);
        assert!(close(view_direction(0.0, 0.0), [0.0, -1.0, 0.0]));
        assert!(close(view_direction(90.0, 0.0), [1.0, 0.0, 0.0]));
        assert!(close(
            view_direction(-37.5, 90.0).map(|v| (v * 1e5).round() / 1e5),
            [0.0, 0.0, 1.0]
        ));
        let pack = bounds([-1.2, -0.1, 0.0], [0.1, 0.7, 0.2]);
        let state = shared_camera(
            std::iter::once(pack),
            std::iter::once(pack),
            1.0,
            Some(view_direction(-30.0, 30.0)),
            1.6,
            &Layout::default(),
        );
        let camera = &state.axes[0];
        assert_eq!(camera.aspect_ratio, 1.6);
        let from_target = unit(sub(camera.position, camera.target));
        assert!(close(
            from_target.map(|v| (v * 1e4).round() / 1e4),
            view_direction(-30.0, 30.0).map(|v| (v * 1e4).round() / 1e4)
        ));
    }

    /// Framing the result: a module at one end of a long pack. The camera frames the padded
    /// module inside the margin and closer than the whole pack would, and its clip planes still
    /// reach every corner of the pack.
    #[test]
    fn result_frames_pad_the_result_and_clip_planes_reach_the_context() {
        let module = bounds([-1.03, 0.07, 0.03], [-0.80, 0.49, 0.13]);
        let pack = bounds([-1.2, -0.6, 0.0], [1.2, 0.6, 0.25]);
        let direction = Some(view_direction(-60.0, 18.0));
        let whole = shared_camera(
            std::iter::once(pack),
            std::iter::once(pack),
            1.0,
            direction,
            1.6,
            &Layout::default(),
        );
        let state = shared_camera(
            std::iter::once(module),
            std::iter::once(pack),
            1.5,
            direction,
            1.6,
            &Layout::default(),
        );
        let camera = &state.axes[0];
        let distance = |camera: &PlotCameraState| {
            let offset = sub(camera.position, camera.target);
            dot(offset, offset).sqrt()
        };
        assert!(distance(camera) < 0.5 * distance(&whole.axes[0]));
        let grown = {
            let mid = (module.min + module.max) * 0.5;
            let half = (module.max - module.min) * 0.5 * 1.5;
            BoundingBox {
                min: mid - half,
                max: mid + half,
            }
        };
        let (left, right, bottom, top) = extent(&projected(&state, &[grown]));
        for side in [-left, right, -bottom, top] {
            assert!(side <= EDGE + 1e-4);
        }
        assert!(
            right >= EDGE - 1e-4 || top >= EDGE - 1e-4,
            "the padded module fills the frame"
        );
        let PlotCameraProjection::Perspective { near, far, .. } = camera.projection else {
            panic!("snapshot cameras are perspective");
        };
        let forward = unit(sub(camera.target, camera.position));
        for point in corners(pack) {
            let depth = dot(sub(point.to_f32(), camera.position), forward);
            assert!(depth < far, "pack corner beyond the far plane");
        }
        assert!(near > 0.0 && near >= far / 5000.0 * 0.999);
    }
}
