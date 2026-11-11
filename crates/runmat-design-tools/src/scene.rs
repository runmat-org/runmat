//! The model scene the Desktop's model canvas draws, built from a part or assembly display:
//! the renderer's figure-scene JSON (mesh plots with one region per face or occurrence, and the
//! model's edges as scene lines), and the camera that frames it along a named view.
//!
//! This follows the model canvas (`model-display.ts` `buildModelScene` and `model-canvas.tsx`
//! `frameCamera`) value for value: the same colours, regions, triangle budget, box stand-ins
//! for the smallest occurrences past it, view directions and frame fill, so a render matches
//! what the Desktop shows.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Face colour of design models (`CAD_FACE_RGBA`).
pub const FACE_RGBA: [f32; 4] = [0.66, 0.72, 0.8, 1.0];
/// Edge colour of design models (`CAD_EDGE_RGBA`).
pub const EDGE_RGBA: [f32; 4] = [0.58, 0.72, 0.88, 0.95];
/// Triangles one mesh plot holds.
const TRIANGLES_PER_PLOT: usize = 120_000;
/// Triangles drawn at full detail; past it the smallest occurrences draw as boxes.
pub const TRIANGLE_BUDGET: usize = 250_000;
/// Box corners within the view, as a fraction of its half-width and half-height.
pub const FRAME_FILL: f64 = 0.8;

/// A standard view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NamedView {
    Isometric,
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
}

impl NamedView {
    /// The direction from the target to the eye, and the up vector (the renderer's presets).
    pub fn orientation(self) -> ([f64; 3], [f64; 3]) {
        let s = 1.0 / 3f64.sqrt();
        match self {
            NamedView::Isometric => ([s, -s, s], [0.0, 0.0, 1.0]),
            NamedView::Top => ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            NamedView::Bottom => ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
            NamedView::Front => ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            NamedView::Back => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            NamedView::Left => ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            NamedView::Right => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        }
    }
}

/// How faces and edges draw (the canvas's display modes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DisplayMode {
    /// Shaded faces with the model's edges over them.
    #[default]
    ShadedEdges,
    /// Shaded faces only.
    Shaded,
    /// Edges only.
    Wireframe,
    /// Faint faces with every edge (the canvas's hidden-lines mode).
    HiddenLines,
}

/// An explicit camera, in model metres.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CameraSpec {
    /// Where the eye looks from. With `fit` it gives only the direction.
    pub position: [f64; 3],
    #[serde(default)]
    pub target: Option<[f64; 3]>,
    #[serde(default)]
    pub up: Option<[f64; 3]>,
    /// Vertical field of view in degrees (default 45, as the Desktop's camera).
    #[serde(default)]
    pub fov_deg: Option<f64>,
    /// Frame the whole model along the camera's direction (default true).
    #[serde(default)]
    pub fit: Option<bool>,
}

/// Which picture of a design to render.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ViewRequest {
    /// A standard view (default `isometric`).
    #[serde(default)]
    pub view: Option<NamedView>,
    /// An explicit camera instead of a standard view.
    #[serde(default)]
    pub camera: Option<CameraSpec>,
    #[serde(default)]
    pub display: Option<DisplayMode>,
    /// Faces or occurrences to highlight, by name or path (`*` globs), as a selection.
    #[serde(default)]
    pub highlight: Option<Vec<String>>,
    /// `light` (default) or `dark`, as the Desktop's themes.
    #[serde(default)]
    pub theme: Option<String>,
    /// Pixels (default 1280 × 800, at most 4096).
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RenderInput {
    /// Workspace path of the `.part` or `.assembly`.
    pub path: String,
    #[serde(flatten)]
    pub view: ViewRequest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RenderOutput {
    pub path: String,
    pub mime_type: String,
    /// The PNG, base64.
    pub png_base64: String,
    pub width: u32,
    pub height: u32,
    /// `gpu` (the Desktop's wgpu renderer, headless) or `cpu_reference` (its CPU rasterizer,
    /// used when no GPU adapter is available; see `notes`).
    pub renderer: String,
    /// The camera used, so a client can ask for a nearby view.
    pub camera: Value,
    pub triangles: usize,
    /// Occurrences drawn as their boxes because the model was over the triangle budget.
    pub boxed_occurrences: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// A region of the scene: a face (`face:<body>:<face>`) or an occurrence (`occ:<n>`).
#[derive(Clone, Debug, PartialEq)]
pub struct SceneRegion {
    pub id: String,
    /// What it picks as: the face's reference or the occurrence's path.
    pub label: String,
}

/// The built scene.
pub struct ModelScene {
    /// The figure (`FigureScene` JSON).
    pub figure: Value,
    pub regions: Vec<SceneRegion>,
    pub bounds: Option<([f64; 3], [f64; 3])>,
    pub triangles: usize,
    pub boxed: usize,
}

struct Plot {
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[usize; 3]>,
    regions: Vec<(String, String, String, usize, usize)>,
    label: String,
    color: [f32; 4],
}

impl Plot {
    fn new(label: String, color: [f32; 4]) -> Self {
        Plot {
            vertices: Vec::new(),
            triangles: Vec::new(),
            regions: Vec::new(),
            label,
            color,
        }
    }
    fn open(&mut self, id: &str, label: &str, tag: &str) {
        self.regions.push((
            id.to_owned(),
            label.to_owned(),
            tag.to_owned(),
            self.triangles.len(),
            0,
        ));
    }
    fn close(&mut self) {
        let n = self.triangles.len();
        if let Some(r) = self.regions.last_mut() {
            r.4 = n - r.3;
        }
    }
    fn push(&mut self, positions: &[f64], triangles: &[usize], t: Option<&[f64]>) -> usize {
        let base = self.vertices.len();
        for p in positions.chunks_exact(3) {
            self.vertices.push(apply(t, p[0], p[1], p[2]));
        }
        for tri in triangles.chunks_exact(3) {
            self.triangles
                .push([base + tri[0], base + tri[1], base + tri[2]]);
        }
        triangles.len() / 3
    }
    fn json(&self, index: usize) -> Value {
        json!({
            "kind": "mesh",
            "vertices": self.vertices,
            "triangles": self.triangles,
            "mesh_id": format!("design_{index}"),
            "face_color_rgba": self.color,
            "edge_color_rgba": [0.08, 0.1, 0.13, 1.0],
            "face_alpha": self.color[3],
            "edge_alpha": 0,
            "edge_width": 0,
            "edge_mode": "none",
            "axes_index": 0,
            "label": self.label,
            "regions": self.regions.iter().map(|(id, label, tag, start, count)| json!({
                "regionId": id, "label": label, "tag": tag,
                "triangleRanges": [{ "start": start, "count": count }],
            })).collect::<Vec<_>>(),
            "highlight_color_rgba": [0.98, 0.78, 0.22, 1.0],
            "visible": true,
        })
    }
}

/// Micrometre rounding, as the canvas does.
fn um(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

fn apply(t: Option<&[f64]>, x: f64, y: f64, z: f64) -> [f64; 3] {
    let Some(t) = t else {
        return [um(x), um(y), um(z)];
    };
    let g = |i: usize, d: f64| t.get(i).copied().unwrap_or(d);
    [
        um(g(0, 1.0) * x + g(1, 0.0) * y + g(2, 0.0) * z + g(9, 0.0)),
        um(g(3, 0.0) * x + g(4, 1.0) * y + g(5, 0.0) * z + g(10, 0.0)),
        um(g(6, 0.0) * x + g(7, 0.0) * y + g(8, 1.0) * z + g(11, 0.0)),
    ]
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn indices(v: &Value) -> Vec<usize> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default()
}

/// The reference a face picks as (`faceReference`).
pub fn face_reference(body: &Value, face: &Value) -> String {
    face["reference"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| {
            format!(
                "feature.{}.face[{}:{}]",
                face["features"][0].as_str().unwrap_or("body"),
                body["index"].as_u64().unwrap_or(0),
                face["index"].as_u64().unwrap_or(0)
            )
        })
}

const BOX_FACES: [[usize; 4]; 6] = [
    [0, 2, 6, 4],
    [1, 5, 7, 3],
    [0, 4, 5, 1],
    [2, 3, 7, 6],
    [0, 1, 3, 2],
    [4, 6, 7, 5],
];

fn bounds_of(v: &Value) -> Option<([f64; 3], [f64; 3])> {
    let min = floats(&v["min"]);
    let max = floats(&v["max"]);
    (min.len() == 3 && max.len() == 3).then(|| ([min[0], min[1], min[2]], [max[0], max[1], max[2]]))
}

fn box_mesh(b: ([f64; 3], [f64; 3])) -> (Vec<f64>, Vec<usize>) {
    let mut positions = Vec::new();
    let mut triangles = Vec::new();
    let corner = |k: usize| -> [f64; 3] {
        [0, 1, 2].map(|a| if (k >> a) & 1 == 1 { b.1[a] } else { b.0[a] })
    };
    for quad in BOX_FACES {
        let base = positions.len() / 3;
        for k in quad {
            positions.extend(corner(k));
        }
        triangles.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (positions, triangles)
}

struct Lines {
    positions: Vec<[f64; 3]>,
}

impl Lines {
    fn polyline(&mut self, points: &[f64], t: Option<&[f64]>) {
        let mut i = 0;
        while i + 5 < points.len() {
            self.positions
                .push(apply(t, points[i], points[i + 1], points[i + 2]));
            self.positions
                .push(apply(t, points[i + 3], points[i + 4], points[i + 5]));
            i += 3;
        }
    }
}

/// The scene of a display (`{ kind: "part" | "assembly", data }`).
pub fn build_model_scene(display: &Value) -> ModelScene {
    let mut scene = build_at_budget(display, TRIANGLE_BUDGET);
    let mut budget = TRIANGLE_BUDGET / 2;
    // The renderer's import limit, with headroom (the canvas halves the budget the same way).
    while scene.figure.to_string().len() > 15 * 1024 * 1024 && budget >= 1000 {
        scene = build_at_budget(display, budget);
        budget /= 2;
    }
    scene
}

fn build_at_budget(display: &Value, budget: usize) -> ModelScene {
    let kind = display["kind"].as_str().unwrap_or("part");
    let data = &display["data"];
    let mut regions = Vec::new();
    let mut plots: Vec<Plot> = Vec::new();
    let mut lines = Lines {
        positions: Vec::new(),
    };
    let mut current = Plot::new("Model".into(), FACE_RGBA);
    let mut triangles = 0;
    let mut boxed = 0;
    let rollover = |current: &mut Plot, plots: &mut Vec<Plot>| {
        if current.triangles.len() >= TRIANGLES_PER_PLOT {
            let next = Plot::new(format!("Model {}", plots.len() + 2), FACE_RGBA);
            plots.push(std::mem::replace(current, next));
        }
    };
    let empty = Vec::new();
    if kind == "part" {
        for body in data["bodies"].as_array().unwrap_or(&empty) {
            for face in body["faces"].as_array().unwrap_or(&empty) {
                let id = format!(
                    "face:{}:{}",
                    body["index"].as_u64().unwrap_or(0),
                    face["index"].as_u64().unwrap_or(0)
                );
                let reference = face_reference(body, face);
                current.open(&id, &reference, "face");
                triangles += current.push(
                    &floats(&face["positions"]),
                    &indices(&face["triangles"]),
                    None,
                );
                current.close();
                regions.push(SceneRegion {
                    id,
                    label: reference,
                });
                rollover(&mut current, &mut plots);
            }
            for edge in body["edges"].as_array().unwrap_or(&empty) {
                lines.polyline(&floats(&edge["points"]), None);
            }
        }
    } else {
        let prototypes = data["prototypes"].as_array().unwrap_or(&empty);
        let occurrences = data["occurrences"].as_array().unwrap_or(&empty);
        let proto =
            |occ: &Value| prototypes.get(occ["prototype"].as_u64().unwrap_or(u64::MAX) as usize);
        let cost = |occ: &Value| -> usize {
            match proto(occ) {
                Some(p) if p["shape"] == "mesh" => p["part"]["bodies"].as_array().map_or(0, |b| {
                    b.iter()
                        .map(|b| b["triangle_count"].as_u64().unwrap_or(0) as usize)
                        .sum()
                }),
                _ => 12,
            }
        };
        let extent = |occ: &Value| -> f64 {
            let b = proto(occ).and_then(|p| {
                if p["shape"] == "mesh" {
                    bounds_of(&p["part"]["bounds"])
                } else if p["shape"] == "box" {
                    bounds_of(&p["bounds"])
                } else {
                    None
                }
            });
            b.map_or(0.0, |(lo, hi)| {
                ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
            })
        };
        let mut order: Vec<usize> = (0..occurrences.len()).collect();
        order.sort_by(|&a, &b| {
            extent(&occurrences[b])
                .partial_cmp(&extent(&occurrences[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut coarse = std::collections::BTreeSet::new();
        let mut spent = 0;
        for n in order {
            let c = cost(&occurrences[n]);
            if spent + c > budget && c > 12 {
                coarse.insert(n);
            } else {
                spent += c;
            }
        }
        boxed = coarse.len();
        for (n, occ) in occurrences.iter().enumerate() {
            let Some(p) = proto(occ) else { continue };
            let t = floats(&occ["transform"]);
            let t = (t.len() >= 12).then_some(t.as_slice());
            let path = occ["path"].as_str().unwrap_or_default().to_owned();
            let id = format!("occ:{n}");
            let as_box = if coarse.contains(&n) {
                if p["shape"] == "mesh" {
                    bounds_of(&p["part"]["bounds"])
                } else {
                    None
                }
            } else if p["shape"] == "box" {
                bounds_of(&p["bounds"])
            } else {
                None
            };
            if coarse.contains(&n) && as_box.is_none() {
                continue;
            }
            current.open(&id, &path, "occurrence");
            if let Some(b) = as_box {
                let (positions, tris) = box_mesh(b);
                triangles += current.push(&positions, &tris, t);
                for q in 0..6 {
                    for k in 0..4 {
                        let a = (q * 4 + k) * 3;
                        let b = (q * 4 + (k + 1) % 4) * 3;
                        lines.polyline(
                            &[
                                positions[a],
                                positions[a + 1],
                                positions[a + 2],
                                positions[b],
                                positions[b + 1],
                                positions[b + 2],
                            ],
                            t,
                        );
                    }
                }
            } else if p["shape"] == "mesh" {
                for body in p["part"]["bodies"].as_array().unwrap_or(&empty) {
                    for face in body["faces"].as_array().unwrap_or(&empty) {
                        triangles += current.push(
                            &floats(&face["positions"]),
                            &indices(&face["triangles"]),
                            t,
                        );
                    }
                    for edge in body["edges"].as_array().unwrap_or(&empty) {
                        lines.polyline(&floats(&edge["points"]), t);
                    }
                }
            }
            current.close();
            regions.push(SceneRegion { id, label: path });
            rollover(&mut current, &mut plots);
        }
    }
    if !current.triangles.is_empty() {
        plots.push(current);
    }
    let bounds = bounds_of(&data["bounds"]);
    let mut figure = json!({
        "schemaVersion": 7,
        "layout": { "axesRows": 1, "axesCols": 1, "axesIndices": plots.iter().map(|_| 0).collect::<Vec<_>>() },
        "metadata": {
            "gridEnabled": true,
            "legendEnabled": false,
            "colorbarEnabled": false,
            "axisEqual": true,
            "backgroundRgba": [1, 1, 1, 1],
            "legendEntries": [],
            "xLabel": "X",
            "yLabel": "Y",
        },
        "plots": plots.iter().enumerate().map(|(i, p)| p.json(i)).collect::<Vec<_>>(),
    });
    if !lines.positions.is_empty() {
        let colors: Vec<[f32; 4]> = lines.positions.iter().map(|_| EDGE_RGBA).collect();
        figure["geometryLines"] = json!([{
            "positions": lines.positions,
            "colors": colors,
            "indices": null,
            "meshId": null,
            "label": "Edges",
            "visible": true,
            "wireframeOnly": true,
        }]);
    }
    ModelScene {
        figure,
        regions,
        bounds,
        triangles,
        boxed,
    }
}

/// A framed camera: eye, target, up and the projection's extents, in model metres.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FramedCamera {
    pub position: [f64; 3],
    pub target: [f64; 3],
    pub up: [f64; 3],
    /// `orthographic` or `perspective`.
    pub projection: String,
    /// Orthographic half-extents (`right`, `top`) or the perspective field of view (radians).
    pub half_width: f64,
    pub half_height: f64,
    pub fov: f64,
    pub near: f64,
    pub far: f64,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    if l < 1e-12 {
        a
    } else {
        [a[0] / l, a[1] / l, a[2] / l]
    }
}

/// The camera that frames `bounds` looking along `-back` (`frameCamera`): every corner of the
/// box inside `FRAME_FILL` of the view, true proportions kept.
pub fn frame_camera(
    bounds: ([f64; 3], [f64; 3]),
    back: [f64; 3],
    up_hint: [f64; 3],
    aspect: f64,
    perspective_fov: Option<f64>,
) -> FramedCamera {
    let (lo, hi) = bounds;
    let centre = [0, 1, 2].map(|a| (lo[a] + hi[a]) / 2.0);
    let half = [0, 1, 2].map(|a| (hi[a] - lo[a]).abs() / 2.0);
    let radius = dot(half, half).sqrt().max(1e-6);
    let back = normalize(back);
    let view = if dot(back, back).sqrt() > 0.5 {
        back
    } else {
        normalize([1.0, -1.0, 1.0])
    };
    let up_hint = if dot(view, up_hint).abs() > 0.999 {
        if view[2].abs() > 0.999 {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        }
    } else {
        up_hint
    };
    let right = normalize(cross(up_hint, view));
    let up = cross(view, right);
    let corners: Vec<[f64; 3]> = (0..8)
        .map(|k| {
            [
                if k & 1 == 1 { half[0] } else { -half[0] },
                if k & 2 == 2 { half[1] } else { -half[1] },
                if k & 4 == 4 { half[2] } else { -half[2] },
            ]
        })
        .collect();
    let a = aspect.max(1e-3);
    if let Some(fov) = perspective_fov {
        let tan_y = (fov / 2.0).tan() * FRAME_FILL;
        let tan_x = tan_y * a;
        let mut distance: f64 = 1e-6;
        for c in &corners {
            let toward = dot(*c, view);
            distance = distance
                .max(toward + dot(*c, right).abs() / tan_x)
                .max(toward + dot(*c, up).abs() / tan_y);
        }
        let position = [0, 1, 2].map(|i| centre[i] + view[i] * distance);
        return FramedCamera {
            position,
            target: centre,
            up,
            projection: "perspective".into(),
            half_width: 0.0,
            half_height: 0.0,
            fov,
            near: (distance * 1e-3).max(distance - radius * 2.0),
            far: distance + radius * 4.0,
        };
    }
    let mut w: f64 = 1e-6;
    let mut h: f64 = 1e-6;
    for c in &corners {
        w = w.max(dot(*c, right).abs());
        h = h.max(dot(*c, up).abs());
    }
    h = h.max(w / a) / FRAME_FILL;
    w = h * a;
    let distance = radius * 4.0;
    let position = [0, 1, 2].map(|i| centre[i] + view[i] * distance);
    FramedCamera {
        position,
        target: centre,
        up,
        projection: "orthographic".into(),
        half_width: w,
        half_height: h,
        fov: 0.0,
        near: distance - radius * 2.0,
        far: distance + radius * 2.0,
    }
}

/// The camera a request asks for, framed on `bounds`.
pub fn camera_for(
    request: &ViewRequest,
    bounds: ([f64; 3], [f64; 3]),
    aspect: f64,
) -> FramedCamera {
    if let Some(c) = &request.camera {
        let target = c
            .target
            .unwrap_or([0, 1, 2].map(|a| (bounds.0[a] + bounds.1[a]) / 2.0));
        let up = c.up.unwrap_or([0.0, 0.0, 1.0]);
        let fov = c.fov_deg.unwrap_or(45.0).clamp(5.0, 120.0).to_radians();
        let mut framed = frame_camera(bounds, sub(c.position, target), up, aspect, Some(fov));
        if !c.fit.unwrap_or(true) {
            let reach = framed.far - framed.near;
            let distance = dot(sub(c.position, target), sub(c.position, target)).sqrt();
            framed.position = c.position;
            framed.target = target;
            framed.near = (distance * 1e-3).max(distance - reach);
            framed.far = distance + reach;
        }
        return framed;
    }
    let (back, up) = request.view.unwrap_or(NamedView::Isometric).orientation();
    frame_camera(bounds, back, up, aspect, Some(45f64.to_radians()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_front_view_frames_the_box_at_the_fill() {
        let cam = frame_camera(
            ([0.0, 0.0, 0.0], [0.1, 0.02, 0.05]),
            NamedView::Front.orientation().0,
            NamedView::Front.orientation().1,
            1.0,
            None,
        );
        assert_eq!(cam.projection, "orthographic");
        // Looking along +y: the box is 100 mm wide and 50 mm tall; width governs.
        assert!((cam.half_width - 0.05 / FRAME_FILL).abs() < 1e-9, "{cam:?}");
        assert!(cam.position[1] < 0.0);
        assert!((cam.up[2] - 1.0).abs() < 1e-9);
    }
}
