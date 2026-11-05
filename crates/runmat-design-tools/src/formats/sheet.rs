//! A 2D drafting sheet: layers, line types, blocks and entities in sheet millimetres. Drawings
//! and flat patterns are laid out into one [`Sheet`]; the DXF and DWG writers write the same
//! sheet, and their readers give it back, so the two files carry the same content.

/// A layer: name, AutoCAD color index and line type name.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub name: String,
    pub color: i16,
    pub linetype: String,
}

/// A line type: name, description and dash pattern (positive dashes, negative gaps, mm).
#[derive(Clone, Debug, PartialEq)]
pub struct Linetype {
    pub name: String,
    pub description: String,
    pub pattern: Vec<f64>,
}

/// A named block (the title block, a datum symbol) with its base point and entities.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub name: String,
    pub base: [f64; 2],
    pub entities: Vec<Entity>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DimensionKind {
    /// Two extension-line origins and the dimension line position; `rotation` (radians) is
    /// the measured direction (0 = horizontal, pi/2 = vertical).
    Linear { from: [f64; 2], to: [f64; 2], line: [f64; 2], rotation: f64 },
    /// Center and a point on the circle.
    Radius { center: [f64; 2], on: [f64; 2] },
    /// Two opposite points on the circle.
    Diameter { from: [f64; 2], to: [f64; 2] },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dimension {
    pub kind: DimensionKind,
    /// Where the text sits.
    pub text_at: [f64; 2],
    /// The measured value (mm); the text shows `text` when set, else the value.
    pub measurement: f64,
    pub text: Option<String>,
    /// The anonymous block (`*D1`, …) holding the drawn lines, arrows and text.
    pub block: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Line { a: [f64; 2], b: [f64; 2] },
    /// Counter-clockwise from `start` to `end` (radians).
    Arc { center: [f64; 2], radius: f64, start: f64, end: f64 },
    Circle { center: [f64; 2], radius: f64 },
    Polyline { points: Vec<[f64; 2]>, closed: bool },
    /// Single-line text, its insertion point at the left baseline.
    Text { at: [f64; 2], height: f64, rotation: f64, text: String },
    /// Paragraph text, its insertion point at the top left; `\P` separates paragraphs.
    MText { at: [f64; 2], height: f64, width: f64, text: String },
    Dimension(Dimension),
    /// A filled or patterned area bounded by closed loops. `pattern` is `SOLID` or a
    /// predefined pattern name (`ANSI31`) drawn at `angle` (radians) and `scale`.
    Hatch { loops: Vec<Vec<[f64; 2]>>, pattern: String, angle: f64, scale: f64 },
    /// An insertion of a block by name.
    Insert { block: String, at: [f64; 2], scale: f64, rotation: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    pub layer: String,
    pub shape: Shape,
}

/// A whole sheet: paper size (mm), tables, blocks and the entities of the sheet itself.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Sheet {
    pub paper: [f64; 2],
    pub layers: Vec<Layer>,
    pub linetypes: Vec<Linetype>,
    pub blocks: Vec<Block>,
    pub entities: Vec<Entity>,
}

impl Sheet {
    pub fn add(&mut self, layer: &str, shape: Shape) {
        self.entities.push(Entity { layer: layer.into(), shape });
    }
    pub fn line(&mut self, layer: &str, a: [f64; 2], b: [f64; 2]) {
        self.add(layer, Shape::Line { a, b });
    }
    pub fn text(&mut self, layer: &str, at: [f64; 2], height: f64, text: &str) {
        self.add(layer, Shape::Text { at, height, rotation: 0.0, text: text.into() });
    }
    pub fn rect(&mut self, layer: &str, x: f64, y: f64, w: f64, h: f64) {
        self.add(layer, Shape::Polyline { points: vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h]], closed: true });
    }
    /// Add a dimension and its anonymous block: extension lines, the dimension line, two
    /// arrowheads and the text (the value to two decimals unless the dimension has text).
    pub fn dimension(&mut self, layer: &str, mut d: Dimension) {
        let name = format!("*D{}", self.blocks.iter().filter(|b| b.name.starts_with("*D")).count() + 1);
        d.block = name.clone();
        let mut b = Sheet::default();
        let arrow = |b: &mut Sheet, tip: [f64; 2], dir: [f64; 2]| {
            let (l, w) = (2.5, 0.6);
            let back = [tip[0] - dir[0] * l, tip[1] - dir[1] * l];
            let n = [-dir[1] * w, dir[0] * w];
            b.add(layer, Shape::Polyline { points: vec![tip, [back[0] + n[0], back[1] + n[1]], [back[0] - n[0], back[1] - n[1]]], closed: true });
        };
        let (a, z) = match &d.kind {
            DimensionKind::Linear { from, to, line, rotation } => {
                let dir = [rotation.cos(), rotation.sin()];
                let foot = |p: [f64; 2]| {
                    let t = (p[0] - line[0]) * dir[0] + (p[1] - line[1]) * dir[1];
                    [line[0] + dir[0] * t, line[1] + dir[1] * t]
                };
                let (fa, fb) = (foot(*from), foot(*to));
                b.line(layer, *from, fa);
                b.line(layer, *to, fb);
                (fa, fb)
            }
            DimensionKind::Radius { center, on } => (*center, *on),
            DimensionKind::Diameter { from, to } => (*from, *to),
        };
        b.line(layer, a, z);
        let len = ((z[0] - a[0]).powi(2) + (z[1] - a[1]).powi(2)).sqrt().max(1e-9);
        let dir = [(z[0] - a[0]) / len, (z[1] - a[1]) / len];
        arrow(&mut b, z, dir);
        if !matches!(d.kind, DimensionKind::Radius { .. }) {
            arrow(&mut b, a, [-dir[0], -dir[1]]);
        }
        let text = d.text.clone().unwrap_or_else(|| format!("{:.2}", d.measurement));
        let rotation = if let DimensionKind::Linear { rotation, .. } = d.kind { rotation } else { 0.0 };
        b.add(layer, Shape::Text { at: d.text_at, height: 2.5, rotation, text });
        self.blocks.push(Block { name, base: [0.0, 0.0], entities: b.entities });
        self.add(layer, Shape::Dimension(d));
    }
    /// The extents of every entity (blocks placed by their inserts' points only).
    pub fn extents(&self) -> ([f64; 2], [f64; 2]) {
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        let mut see = |p: [f64; 2]| {
            lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            hi = [hi[0].max(p[0]), hi[1].max(p[1])];
        };
        for e in &self.entities {
            match &e.shape {
                Shape::Line { a, b } => {
                    see(*a);
                    see(*b);
                }
                Shape::Arc { center, radius, .. } | Shape::Circle { center, radius } => {
                    see([center[0] - radius, center[1] - radius]);
                    see([center[0] + radius, center[1] + radius]);
                }
                Shape::Polyline { points, .. } => points.iter().for_each(|p| see(*p)),
                Shape::Hatch { loops, .. } => loops.iter().flatten().for_each(|p| see(*p)),
                Shape::Text { at, .. } | Shape::MText { at, .. } | Shape::Insert { at, .. } => see(*at),
                Shape::Dimension(d) => see(d.text_at),
            }
        }
        if !lo[0].is_finite() {
            return ([0.0, 0.0], self.paper);
        }
        (lo, hi)
    }
}

/// The standard line types a sheet uses: continuous, dashed (hidden lines) and center.
pub fn standard_linetypes() -> Vec<Linetype> {
    vec![
        Linetype { name: "DASHED".into(), description: "__ __ __".into(), pattern: vec![3.0, -1.5] },
        Linetype { name: "CENTER".into(), description: "____ _ ____ _".into(), pattern: vec![8.0, -1.5, 1.5, -1.5] },
    ]
}

/// A sheet as sorted, rounded records (to 1e-3 mm and 1e-4 rad) of what both DXF and DWG
/// carry: paragraph text as one text per paragraph, hatches as closed boundary polylines and
/// dimensions by their block, kind and points. Two files with the same records draw the same.
pub fn canonical(sheet: &Sheet) -> Vec<String> {
    fn r(v: f64) -> String {
        let s = format!("{:.3}", v);
        if s == "-0.000" { "0.000".into() } else { s }
    }
    fn p(v: [f64; 2]) -> String {
        format!("({},{})", r(v[0]), r(v[1]))
    }
    fn ang(a: f64) -> String {
        let t = std::f64::consts::TAU;
        format!("{:.4}", ((a % t) + t) % t)
    }
    fn records(scope: &str, entities: &[Entity], out: &mut Vec<String>) {
        for e in entities {
            let l = format!("{scope}/{}", e.layer);
            match &e.shape {
                Shape::Line { a, b } => out.push(format!("{l} line {} {}", p(*a), p(*b))),
                Shape::Arc { center, radius, start, end } => out.push(format!("{l} arc {} {} {} {}", p(*center), r(*radius), ang(*start), ang(*end))),
                Shape::Circle { center, radius } => out.push(format!("{l} circle {} {}", p(*center), r(*radius))),
                Shape::Polyline { points, closed } => out.push(format!("{l} polyline {closed} {}", points.iter().map(|q| p(*q)).collect::<Vec<_>>().join(" "))),
                Shape::Text { at, height, rotation, text } => out.push(format!("{l} text {} {} {} {text}", p(*at), r(*height), ang(*rotation))),
                Shape::MText { at, height, text, .. } => {
                    for (i, line) in text.split("\\P").enumerate() {
                        out.push(format!("{l} text {} {} {} {line}", p([at[0], at[1] - height - i as f64 * height * 1.6]), r(*height), ang(0.0)));
                    }
                }
                Shape::Dimension(d) => {
                    let pts = match &d.kind {
                        DimensionKind::Linear { from, to, line, rotation } => format!("linear {} {} {} {}", p(*from), p(*to), p(*line), ang(*rotation)),
                        DimensionKind::Radius { center, on } => format!("radius {} {}", p(*center), p(*on)),
                        DimensionKind::Diameter { from, to } => format!("diameter {} {}", p(*from), p(*to)),
                    };
                    out.push(format!("{l} dimension {} {pts} {} {:?}", d.block, p(d.text_at), d.text));
                }
                Shape::Hatch { loops, .. } => {
                    for lp in loops {
                        out.push(format!("{l} polyline true {}", lp.iter().map(|q| p(*q)).collect::<Vec<_>>().join(" ")));
                    }
                }
                Shape::Insert { block, at, scale, rotation } => out.push(format!("{l} insert {block} {} {} {}", p(*at), r(*scale), ang(*rotation))),
            }
        }
    }
    let mut out = Vec::new();
    records("", &sheet.entities, &mut out);
    for b in &sheet.blocks {
        out.push(format!("block {} {}", b.name, p(b.base)));
        records(&b.name, &b.entities, &mut out);
    }
    for l in &sheet.layers {
        out.push(format!("layer {} {} {}", l.name, l.color, l.linetype));
    }
    for t in &sheet.linetypes {
        out.push(format!("linetype {} {}", t.name, t.pattern.iter().map(|d| r(*d)).collect::<Vec<_>>().join(" ")));
    }
    out.sort();
    out
}
