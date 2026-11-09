//! One drawing layout, in sheet millimetres (y up), that the Desktop's sheet and every export
//! (PDF, DXF, DWG) draw as given. The compile puts it in the projection as `layouts`: sheet name
//! → its primitives, its dimensions (for exports that write native dimension entities), and the
//! findings the layout made (a repeated dimension, a view that doesn't fit).
//!
//! The layout runs in this order, each step treating what came before as occupied:
//! 1. **Sheet**: the border with its zones and centring marks, the title block (always, with
//!    standard fields when the format names none), the projection symbol ([`border`]).
//! 2. **Tables** snap to the sheet's corners and zones; a table placed by hand stays ([`tables`]).
//! 3. **Hand-placed annotations** keep their spot, and views keep clear of them.
//! 4. **Views** sit in projection alignment with their parent, a gap apart that scales with the
//!    sheet; section, detail, auxiliary and broken-out views take the free space nearest their
//!    parent. Each view reserves room for its label and its dimension rows ([`views`]).
//! 5. **Dimensions** stack outward from the view at the standard's offsets, shortest nearest;
//!    a dimension that repeats one already drawn is a finding and is drawn once ([`dims`]).
//! 6. **Callouts** (hole callouts, frames, finishes, notes, weld symbols, datum targets,
//!    balloons) take the free spot nearest what they point at, where their label overlaps
//!    nothing and their leader crosses no text ([`callouts`], [`solver`]).
//!
//! Line weights, text heights, arrowheads and dimension offsets come from the sheet's drafting
//! standard (`styles`, [`runmat_cad::drawing::standard`]). The layout is deterministic: the
//! same projection lays out the same way.

mod border;
mod angular;
mod callouts;
mod dims;
pub(crate) mod marks;
pub(crate) mod solver;
mod tables;
mod views;

#[cfg(test)]
mod tests;

use serde::Serialize;
use serde_json::{Map, Value};

pub type Xy = [f64; 2];

/// One drawn mark: the pick ID (`view:<name>`, `annotation:<id>`, `table:<name>`, `edge:…`,
/// `title`, or empty), what it is, its layer and line style, and its shape.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Prim {
    pub id: String,
    pub kind: &'static str,
    pub layer: &'static str,
    pub style: &'static str,
    pub shape: Shape,
    /// The stroke width (mm) the drafting standard gives this line.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Shape {
    Rect { x: f64, y: f64, w: f64, h: f64 },
    Circle { x: f64, y: f64, r: f64 },
    Polyline {
        points: Vec<f64>,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        closed: bool,
    },
    /// A filled region: the outer ring, then its holes.
    Polygon { rings: Vec<Vec<f64>> },
    Text {
        x: f64,
        y: f64,
        text: String,
        size: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        anchor: Option<&'static str>,
        /// Degrees, counter-clockwise.
        #[serde(skip_serializing_if = "Option::is_none")]
        rotation: Option<f64>,
    },
}

/// The drafting standard's numbers for one sheet (ISO 129-1 or ASME Y14.5, from `styles`).
#[derive(Clone, Debug)]
pub struct Std {
    pub iso: bool,
    pub text: f64,
    pub arrow: f64,
    pub arrow_width: f64,
    pub gap: f64,
    pub over: f64,
    pub first: f64,
    pub step: f64,
    /// ASME: figures read from the bottom of the sheet, in a break in the dimension line.
    pub unidirectional: bool,
    pub visible: f64,
    pub hidden: f64,
    pub center: f64,
    pub thin: f64,
    pub thick: f64,
}

impl Std {
    /// The style the compile gives a sheet, else ISO's.
    pub fn of(style: &Value) -> Std {
        let n = |k: &str, d: f64| style[k].as_f64().unwrap_or(d);
        let lines = &style["lines"];
        let w = |k: &str, d: f64| lines[k].as_f64().unwrap_or(d);
        let iso = style["standard"].as_str() != Some("asme");
        Std {
            iso,
            text: n("text_height", 3.5),
            arrow: n("arrow_length", 3.0),
            arrow_width: n("arrow_width", 1.0),
            gap: n("extension_gap", 1.0),
            over: n("extension_overshoot", 2.0),
            first: n("first_offset", 10.0),
            step: n("offset_step", 8.0),
            unidirectional: style["text"].as_str() == Some("unidirectional"),
            visible: w("visible", 0.5),
            hidden: w("hidden", 0.25),
            center: w("center", 0.25),
            thin: w("dimension", 0.25),
            thick: w("thick", 0.7),
        }
    }

    /// The stroke width for a line style.
    pub fn width(&self, style: &str) -> Option<f64> {
        match style {
            "visible" => Some(self.visible),
            "hidden" | "benddown" => Some(self.hidden),
            "center" | "bendup" => Some(self.center),
            "dimink" | "hatch" => Some(self.thin),
            "thick" => Some(self.thick),
            "frame" => Some(self.visible * 0.7),
            _ => None,
        }
    }

    /// The width of `text` set `size` high (the sheet font: 0.6 of the height per character).
    pub fn text_width(text: &str, size: f64) -> f64 {
        text.chars().count() as f64 * size * 0.6
    }
}

/// A dimension as placed: exports write it as a native dimension entity whose block holds the
/// primitives drawn for `id`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlacedDimension {
    pub id: String,
    pub from: Xy,
    pub to: Xy,
    /// A point on the dimension line.
    pub line: Xy,
    /// The measured direction (radians: 0 horizontal, pi/2 vertical).
    pub rotation: f64,
    pub text: String,
    pub text_at: Xy,
    /// The measured value in model millimetres.
    pub measurement: f64,
}

/// Something the layout found: a repeated dimension, a view or callout with no free room.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Finding {
    pub code: &'static str,
    pub item: String,
    pub message: String,
}

/// One sheet laid out.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SheetLayout {
    pub size: Xy,
    pub primitives: Vec<Prim>,
    pub dimensions: Vec<PlacedDimension>,
    pub findings: Vec<Finding>,
    /// Facts about missing data (a view whose model isn't open).
    pub notes: Vec<String>,
    /// Where each item sits, as its `at` would place it (a view's centre, a callout's elbow, a
    /// note's or frame's corner, a table's corner, a dimension's line): moving an item the
    /// layout placed starts from here.
    pub placed: std::collections::BTreeMap<String, Xy>,
}

/// Every sheet of a compiled drawing projection, laid out: `{ <sheet>: SheetLayout }`.
pub fn layouts(data: &Value, displays: &Map<String, Value>) -> Value {
    let mut out = Map::new();
    for sheet in data["document"]["sheets"].as_array().into_iter().flatten() {
        let Some(name) = sheet["name"].as_str() else { continue };
        let laid = sheet_layout(data, displays, name);
        out.insert(name.to_owned(), serde_json::to_value(laid).unwrap_or(Value::Null));
    }
    Value::Object(out)
}

/// Sheet `name` of a compiled drawing projection, laid out.
pub fn sheet_layout(data: &Value, displays: &Map<String, Value>, name: &str) -> SheetLayout {
    let mut ctx = Ctx::new(data, displays, name);
    border::draw(&mut ctx);
    tables::place(&mut ctx);
    callouts::reserve_manual(&mut ctx);
    let mut planned = views::measure(&mut ctx);
    let plans = dims::plan(&mut ctx, &planned);
    views::place(&mut ctx, &mut planned, &plans.rooms);
    views::draw(&mut ctx, &planned);
    dims::draw(&mut ctx, &planned, plans);
    callouts::draw(&mut ctx, &planned);
    for p in &mut ctx.out.primitives {
        if p.width.is_none() {
            p.width = ctx.std.width(p.style);
        }
    }
    ctx.out
}

/// The layout's working state for one sheet.
pub(crate) struct Ctx<'a> {
    pub data: &'a Value,
    pub displays: &'a Map<String, Value>,
    pub sheet: String,
    pub std: Std,
    /// The sheet's scale (numerator / denominator).
    pub scale: f64,
    pub size: Xy,
    /// The drawing frame's inside: everything drawn stays in it.
    pub frame: solver::BoxR,
    pub occ: solver::Occupancy,
    pub out: SheetLayout,
}

impl<'a> Ctx<'a> {
    fn new(data: &'a Value, displays: &'a Map<String, Value>, sheet: &str) -> Ctx<'a> {
        let doc = &data["document"];
        let sheet_json = doc["sheets"].as_array().and_then(|s| s.iter().find(|s| s["name"] == sheet)).cloned().unwrap_or(Value::Null);
        let format = doc["formats"].as_array().and_then(|f| f.iter().find(|f| f["name"] == sheet_json["format"])).cloned().unwrap_or(Value::Null);
        // The format's own size: a named sheet's, or a custom sheet's width and height (A3 when unreadable).
        let size = serde_json::from_value::<runmat_cad::drawing::SheetSize>(format["size"].clone())
            .ok()
            .and_then(|size| size.mm())
            .unwrap_or_else(|| runmat_cad::drawing::title_layout::sheet_size("A3"));
        let ratio = floats(&sheet_json["scale"]);
        let scale = if ratio.len() == 2 && ratio[1] != 0.0 { ratio[0] / ratio[1] } else { 1.0 };
        let std = Std::of(&data["styles"][sheet]);
        let frame = border::frame_box(size, std.iso);
        Ctx {
            data,
            displays,
            sheet: sheet.to_owned(),
            std,
            scale,
            size,
            frame,
            occ: solver::Occupancy::new(frame),
            out: SheetLayout { size, ..Default::default() },
        }
    }

    pub fn doc(&self) -> &'a Value {
        &self.data["document"]
    }

    pub fn push(&mut self, prims: Vec<Prim>) {
        self.out.primitives.extend(prims);
    }

    pub fn finding(&mut self, code: &'static str, item: &str, message: String) {
        self.out.findings.push(Finding { code, item: item.to_owned(), message });
    }

    /// The text the compile gives an annotation (its value, frame cells, callout).
    pub fn text_of(&self, id: &str) -> String {
        self.data["text"][id].as_str().unwrap_or(id).to_owned()
    }
}

/// A JSON array of numbers.
pub fn floats(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default()
}

/// A JSON `[x, y]`.
pub fn xy(v: &Value) -> Option<Xy> {
    let f = floats(v);
    (f.len() >= 2).then(|| [f[0], f[1]])
}

/// A `[x, y]` a drawing item was placed at by hand (an unplaced item carries `[0, 0]` or none).
pub fn placed_at(v: &Value) -> Option<Xy> {
    xy(&v["at"]).filter(|p| p[0] != 0.0 || p[1] != 0.0)
}

/// A scale as a drawing labels it (`2:1`, `1:2`).
pub fn scale_text(scale: f64) -> String {
    if scale >= 1.0 {
        let s = runmat_cad::drawing::standard::metric_text(scale, 3);
        format!("{s}:1")
    } else {
        format!("1:{}", runmat_cad::drawing::standard::metric_text(1.0 / scale, 2))
    }
}
