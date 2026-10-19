//! Transient graphics awareness in render coordinates. No networking or account
//! policy belongs here. Hosts bind an immutable resource revision before passing
//! these values to the same overlay used by GPU and snapshot rendering.
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PointerPosition {
    Axes { axes: String, x: f64, y: f64 },
    View { view: String, u: f64, v: f64 },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PresencePointer {
    pub position: PointerPosition,
    pub color: [u8; 3],
    pub opacity: f32,
    pub label: Option<String>,
    /// Host-controlled progress makes live and snapshot frames deterministic.
    pub click_progress: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub click_position: Option<PointerPosition>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PresenceFrame {
    /// Local host fence, never sent to another collaborator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<PresenceTarget>,
    pub revision: String,
    pub pointers: Vec<PresencePointer>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum PresenceTarget {
    #[serde(rename = "figure")]
    Figure { handle: u32 },
    #[serde(rename = "geometryScene")]
    GeometryScene { handle: u32 },
}
impl PresenceFrame {
    pub fn validate(&self) -> bool {
        self.revision.len() <= 160
            && self.pointers.len() <= 32
            && self.pointers.iter().all(|pointer| {
                pointer.opacity.is_finite()
                    && (0.0..=1.0).contains(&pointer.opacity)
                    && pointer.label.as_ref().is_none_or(|name| {
                        name.chars().count() <= 80 && !name.chars().any(char::is_control)
                    })
                    && pointer.click_progress.is_none_or(|progress| {
                        progress.is_finite() && (0.0..=1.0).contains(&progress)
                    })
                    && [&pointer.position]
                        .into_iter()
                        .chain(pointer.click_position.iter())
                        .all(|position| match position {
                            PointerPosition::Axes { axes, x, y } => {
                                axes.len() <= 160 && x.is_finite() && y.is_finite()
                            }
                            PointerPosition::View { view, u, v } => {
                                view.len() <= 512
                                    && u.is_finite()
                                    && v.is_finite()
                                    && (0.0..=1.0).contains(u)
                                    && (0.0..=1.0).contains(v)
                            }
                        })
            })
    }
}

/// The actual rendered axes rectangle (logical pixels) and GPU transform. Using
/// outer panel dimensions would make padding, subplots and letterboxing drift.
#[derive(Clone, Debug)]
pub struct PresenceProjection {
    pub axes: String,
    pub rect: [f32; 4], // x, y, width, height in logical pixels
    pub view_projection: Mat4,
    /// The same visible data bounds used by the 2D grid and tick renderer.
    /// Matrix unprojection can differ when a host fits its camera separately.
    pub data_bounds: Option<[f64; 4]>,
    pub is_3d: bool,
    pub x_log: bool,
    pub y_log: bool,
}
impl PresenceProjection {
    fn valid(&self) -> bool {
        self.rect.iter().all(|value| value.is_finite())
            && self.rect[2] > 0.0
            && self.rect[3] > 0.0
            && self.view_projection.is_finite()
            && self.data_bounds.is_none_or(|[left, right, bottom, top]| {
                [left, right, bottom, top]
                    .iter()
                    .all(|value| value.is_finite())
                    && right > left
                    && top > bottom
                    && (!self.x_log || left > 0.0)
                    && (!self.y_log || bottom > 0.0)
            })
    }
    pub fn view_signature(&self) -> String {
        let mut result = format!(
            "{}:{:08x}:",
            self.axes,
            (self.rect[2] / self.rect[3]).to_bits()
        );
        for value in self.view_projection.to_cols_array() {
            result.push_str(&format!("{:08x}", value.to_bits()));
        }
        result
    }
    pub fn capture(&self, pixel: [f32; 2]) -> Option<PointerPosition> {
        if !self.valid() {
            return None;
        }
        let u = (pixel[0] - self.rect[0]) / self.rect[2];
        let v = (pixel[1] - self.rect[1]) / self.rect[3];
        if !u.is_finite()
            || !v.is_finite()
            || !(0.0..=1.0).contains(&u)
            || !(0.0..=1.0).contains(&v)
        {
            return None;
        }
        if self.is_3d {
            return Some(PointerPosition::View {
                view: self.view_signature(),
                u: u as f64,
                v: v as f64,
            });
        }
        if let Some([left, right, bottom, top]) = self.data_bounds {
            let interpolate = |lo: f64, hi: f64, fraction: f32, logarithmic: bool| {
                if logarithmic {
                    10f64.powf(lo.log10() + (hi.log10() - lo.log10()) * fraction as f64)
                } else {
                    lo + (hi - lo) * fraction as f64
                }
            };
            let x = interpolate(left, right, u, self.x_log);
            let y = interpolate(top, bottom, v, self.y_log);
            return (x.is_finite() && y.is_finite()).then(|| PointerPosition::Axes {
                axes: self.axes.clone(),
                x,
                y,
            });
        }
        let inverse = self.view_projection.inverse();
        let near = inverse.project_point3(Vec3::new(u * 2.0 - 1.0, 1.0 - v * 2.0, 0.0));
        let far = inverse.project_point3(Vec3::new(u * 2.0 - 1.0, 1.0 - v * 2.0, 1.0));
        let delta = far - near;
        if !near.is_finite() || !delta.is_finite() || delta.z.abs() < f32::EPSILON {
            return None;
        }
        let world = near + delta * (-near.z / delta.z);
        let x = if self.x_log {
            10f64.powf(world.x as f64)
        } else {
            world.x as f64
        };
        let y = if self.y_log {
            10f64.powf(world.y as f64)
        } else {
            world.y as f64
        };
        (x.is_finite() && y.is_finite()).then(|| PointerPosition::Axes {
            axes: self.axes.clone(),
            x,
            y,
        })
    }
    pub fn project(&self, pointer: &PointerPosition) -> Option<[f32; 2]> {
        if !self.valid() {
            return None;
        }
        let (u, v) = match pointer {
            PointerPosition::Axes { axes, x, y } if !self.is_3d && axes == &self.axes => {
                if let Some([left, right, bottom, top]) = self.data_bounds {
                    let fraction = |value: f64, lo: f64, hi: f64, logarithmic: bool| {
                        if logarithmic {
                            (value.log10() - lo.log10()) / (hi.log10() - lo.log10())
                        } else {
                            (value - lo) / (hi - lo)
                        }
                    };
                    let u = fraction(*x, left, right, self.x_log);
                    let v = fraction(*y, top, bottom, self.y_log);
                    if !u.is_finite() || !v.is_finite() {
                        return None;
                    }
                    return ((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).then(|| {
                        [
                            self.rect[0] + u as f32 * self.rect[2],
                            self.rect[1] + v as f32 * self.rect[3],
                        ]
                    });
                }
                let x = if self.x_log { x.log10() } else { *x };
                let y = if self.y_log { y.log10() } else { *y };
                if !x.is_finite() || !y.is_finite() {
                    return None;
                }
                let point = self.view_projection * Vec3::new(x as f32, y as f32, 0.0).extend(1.0);
                if !point.is_finite() || point.w <= 0.0 {
                    return None;
                }
                (
                    (point.x / point.w + 1.0) / 2.0,
                    (1.0 - point.y / point.w) / 2.0,
                )
            }
            PointerPosition::View { view, u, v }
                if self.is_3d && view == &self.view_signature() =>
            {
                (*u as f32, *v as f32)
            }
            _ => return None,
        };
        if !u.is_finite()
            || !v.is_finite()
            || !(0.0..=1.0).contains(&u)
            || !(0.0..=1.0).contains(&v)
        {
            return None;
        }
        Some([
            self.rect[0] + u * self.rect[2],
            self.rect[1] + v * self.rect[3],
        ])
    }
}

#[cfg(feature = "egui-overlay")]
pub fn render_presence(
    ctx: &egui::Context,
    revision: &str,
    frame: &PresenceFrame,
    projections: &[PresenceProjection],
) {
    if frame.revision != revision || !frame.validate() {
        return;
    }
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("collaborator_pointers"),
    ));
    let mut labels = 0;
    for pointer in &frame.pointers {
        let Some(pixel) = projections
            .iter()
            .find_map(|projection| projection.project(&pointer.position))
        else {
            continue;
        };
        let point = egui::pos2(pixel[0], pixel[1]);
        let alpha = (pointer.opacity * 255.0).round() as u8;
        let color = egui::Color32::from_rgba_unmultiplied(
            pointer.color[0],
            pointer.color[1],
            pointer.color[2],
            alpha,
        );
        // Readable on any plot: a soft shadow and a dark halo sit under the
        // white-edged pointer, so one of its two edges always contrasts.
        let arrow = |offset: egui::Vec2| -> Vec<egui::Pos2> {
            [(0.0, 0.0), (4.6, 17.2), (8.0, 10.4), (17.2, 8.0)]
                .iter()
                .map(|&(x, y)| point + offset + egui::vec2(x, y))
                .collect()
        };
        let shade = |a: f32| egui::Color32::from_black_alpha((a * pointer.opacity * 255.0) as u8);
        for (offset, a) in [(egui::vec2(1.5, 2.5), 0.22), (egui::vec2(0.75, 1.25), 0.32)] {
            painter.add(egui::Shape::convex_polygon(
                arrow(offset),
                shade(a),
                egui::Stroke::NONE,
            ));
        }
        painter.add(egui::Shape::convex_polygon(
            arrow(egui::Vec2::ZERO),
            egui::Color32::TRANSPARENT,
            egui::Stroke::new(3.0, shade(0.45)),
        ));
        painter.add(egui::Shape::convex_polygon(
            arrow(egui::Vec2::ZERO),
            color,
            egui::Stroke::new(1.5, egui::Color32::from_white_alpha(alpha)),
        ));
        if let Some(progress) = pointer.click_progress {
            let click = pointer.click_position.as_ref().unwrap_or(&pointer.position);
            if let Some(pixel) = projections
                .iter()
                .find_map(|projection| projection.project(click))
            {
                painter.circle_stroke(
                    egui::pos2(pixel[0], pixel[1]),
                    5.0 + progress * 9.0,
                    egui::Stroke::new(1.5, color.gamma_multiply(1.0 - progress)),
                );
            }
        }
        if let Some(label) = &pointer.label {
            if labels < 3 {
                // A name tag in the collaborator's color carries its own
                // background; text flips to near-black on light colors.
                let [r, g, b] = pointer.color.map(|c| c as f32 / 255.0);
                let light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.6;
                let text_color = if light {
                    egui::Color32::from_rgba_unmultiplied(17, 24, 39, alpha)
                } else {
                    egui::Color32::from_white_alpha(alpha)
                };
                let galley = painter.layout_no_wrap(
                    label.clone(),
                    egui::FontId::proportional(12.0),
                    text_color,
                );
                let padding = egui::vec2(6.0, 3.0);
                let origin = point + egui::vec2(14.0, 18.0);
                let tag = egui::Rect::from_min_size(origin, galley.size() + padding * 2.0);
                painter.rect_filled(tag.translate(egui::vec2(0.0, 1.5)), 5.0, shade(0.3));
                painter.rect_filled(tag.expand(1.0), 6.0, shade(0.35));
                painter.rect_filled(tag, 5.0, color);
                painter.galley(origin + padding, galley, text_color);
                labels += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn projection() -> PresenceProjection {
        PresenceProjection {
            axes: "axes:0".into(),
            rect: [40.0, 30.0, 400.0, 200.0],
            view_projection: Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            data_bounds: None,
            is_3d: false,
            x_log: false,
            y_log: false,
        }
    }
    #[test]
    fn axes_round_trip_uses_actual_padding_and_resizes_without_drift() {
        let first = projection();
        let point = first.capture([140.0, 80.0]).unwrap();
        assert_eq!(first.project(&point), Some([140.0, 80.0]));
        let resized = PresenceProjection {
            rect: [80.0, 60.0, 800.0, 400.0],
            ..first
        };
        assert_eq!(resized.project(&point), Some([280.0, 160.0]));
        assert!(resized.capture([0.0, 0.0]).is_none());
    }
    #[test]
    fn data_origin_tracks_the_visible_axes_on_differently_sized_clients() {
        let sender = PresenceProjection {
            data_bounds: Some([0.0, 6.0, -1.0, 1.0]),
            // Deliberately unrelated camera matrix: ticks and grid are drawn
            // from visible data bounds, so the pointer must use those bounds.
            view_projection: Mat4::from_scale(Vec3::splat(0.25)),
            ..projection()
        };
        let origin = sender.capture([40.0, 130.0]).expect("origin on sender");
        assert!(matches!(&origin, PointerPosition::Axes { x, y, .. } if *x == 0.0 && *y == 0.0));
        let receiver = PresenceProjection {
            rect: [72.0, 45.0, 630.0, 315.0],
            data_bounds: Some([0.0, 6.0, -1.17, 0.83]),
            ..sender
        };
        let pixel = receiver.project(&origin).expect("origin on receiver");
        assert!((pixel[0] - 72.0).abs() < 0.001);
        assert!((pixel[1] - (45.0 + 315.0 * 0.83 / 2.0)).abs() < 0.001);
    }
    #[test]
    fn log_axes_reject_invalid_values_and_preserve_positive_data_coordinates() {
        let view = PresenceProjection {
            x_log: true,
            ..projection()
        };
        let point = view.capture([240.0, 130.0]).unwrap();
        assert!(matches!(point, PointerPosition::Axes { x: 1.0, .. }));
        assert_eq!(view.project(&point), Some([240.0, 130.0]));
        assert!(view
            .project(&PointerPosition::Axes {
                axes: view.axes.clone(),
                x: 0.0,
                y: 0.0
            })
            .is_none());
    }
    #[test]
    fn three_dimensional_cursors_require_exact_camera_and_aspect_compatibility() {
        let view = PresenceProjection {
            is_3d: true,
            ..projection()
        };
        let point = view.capture([240.0, 130.0]).unwrap();
        assert_eq!(view.project(&point), Some([240.0, 130.0]));
        let other = PresenceProjection {
            rect: [0.0, 0.0, 400.0, 400.0],
            ..view.clone()
        };
        assert!(other.project(&point).is_none());
        let other = PresenceProjection {
            view_projection: view.view_projection * Mat4::from_rotation_z(0.1),
            ..view
        };
        assert!(other.project(&point).is_none());
    }
}
