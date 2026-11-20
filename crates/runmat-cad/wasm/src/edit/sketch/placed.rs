//! A sketch tool's point fields given as a point of the model: a click on the sketch plane or in
//! the view (`point(x mm, y mm, z mm)`), the origin (`origin:point`, `origin.point`), or a vertex.
//! Each is written as the sketch point where it sits seen straight onto the sketch plane,
//! `{ x, y }` in sketch millimetres, the value a click with a drawing tool sends.
//!
//! A pick of a sketch point (`sketch.<sketch>.l1.end`) is left for `picked_points`, which writes
//! it as that point so the tool shares it.

use super::POINT_FIELDS;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{json, Map, Value};

/// A sketch's plane in model coordinates (metres): its origin and its x and y directions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SketchFrame {
    pub origin: [f64; 3],
    pub x: [f64; 3],
    pub y: [f64; 3],
}

impl SketchFrame {
    /// The frame a display carries for sketch `id` (`display.sketches[].frame`).
    pub(crate) fn of_display(display: &Value, id: &str) -> Option<Self> {
        let sketch = display["sketches"]
            .as_array()?
            .iter()
            .find(|s| s["id"] == id)?;
        let v = |k: &str| -> Option<[f64; 3]> {
            let a = sketch["frame"][k].as_array()?;
            Some([
                a.first()?.as_f64()?,
                a.get(1)?.as_f64()?,
                a.get(2)?.as_f64()?,
            ])
        };
        Some(Self {
            origin: v("origin")?,
            x: v("x")?,
            y: v("y")?,
        })
    }

    /// Where a model point (metres) sits on the plane, in sketch millimetres.
    fn sketch_mm(&self, p: [f64; 3]) -> (f64, f64) {
        let d = [0, 1, 2].map(|i| p[i] - self.origin[i]);
        let dot = |a: [f64; 3]| d[0] * a[0] + d[1] * a[1] + d[2] * a[2];
        // The display's frames are single precision: a tenth of a micrometre is what a pick means.
        let round = |v: f64| (v * 1.0e7).round() / 1.0e4 + 0.0;
        (round(dot(self.x)), round(dot(self.y)))
    }
}

/// The model point (metres) a pick names without help: a clicked point or the origin.
fn free_point(pick: &str) -> Option<[f64; 3]> {
    if matches!(pick, "origin:point" | "origin.point") {
        return Some([0.0; 3]);
    }
    let inner = pick.trim().strip_prefix("point(")?.strip_suffix(')')?;
    let coords: Vec<f64> = inner
        .split(',')
        .map(|c| {
            c.trim()
                .strip_suffix("mm")
                .unwrap_or(c)
                .trim()
                .parse::<f64>()
        })
        .collect::<Result<_, _>>()
        .ok()?;
    match coords.as_slice() {
        [x, y, z] => Some([x / 1000.0, y / 1000.0, z / 1000.0]),
        _ => None,
    }
}

/// The single pick a point field holds (`"…"` or `["…"]`).
fn single_pick(value: Option<&Value>) -> Option<&str> {
    match value? {
        Value::String(s) => Some(s.as_str()),
        Value::Array(a) if a.len() == 1 => a[0].as_str(),
        _ => None,
    }
}

/// `values` with each point field that names a model point written as `{ x, y }` on sketch `id`'s
/// plane. `vertex` finds a vertex pick's model point; `frame` is asked once, and only when a
/// field needs it. A pick that names no point is left as it is, for the tool to refuse.
pub(crate) fn placed_points(
    values: &Map<String, Value>,
    id: &str,
    vertex: impl Fn(&str) -> Option<[f64; 3]>,
    frame: impl FnOnce() -> Option<SketchFrame>,
) -> Result<Map<String, Value>, EditFailure> {
    let mut out = values.clone();
    let mut frame = Some(frame);
    let mut known: Option<SketchFrame> = None;
    for key in POINT_FIELDS {
        let Some(pick) = single_pick(values.get(key)) else {
            continue;
        };
        if pick.starts_with("sketch.") {
            continue;
        }
        let Some(at) = free_point(pick).or_else(|| vertex(pick)) else {
            continue;
        };
        if known.is_none() {
            known = frame.take().and_then(|f| f());
        }
        let plane = known.ok_or_else(|| {
            super::model(format!(
                "sketch {id} isn't drawn, so `{pick}` can't be placed on its plane"
            ))
        })?;
        let (x, y) = plane.sketch_mm(at);
        out.insert(key.to_owned(), json!({ "x": x, "y": y }));
    }
    Ok(out)
}
