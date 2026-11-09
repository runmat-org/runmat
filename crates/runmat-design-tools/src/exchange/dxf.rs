//! DXF export (R12 ASCII, `AC1009`) of a laid-out [`Sheet`], and a reader for the files it
//! writes. R12 has no paragraph text or hatch entities: paragraph text is written one TEXT per
//! paragraph, and a hatch as its closed boundary polylines on the hatch's layer.

use crate::formats::sheet::{Block, Dimension, DimensionKind, Entity, Shape, Sheet};
use crate::{Files, ToolError};
use std::fmt::Write;

use super::sheet::{drawing_sheet, flat_pattern};

/// DXF text: `%%d`, `%%c` and `%%p` for degree, diameter and plus-minus, `\U+XXXX` for any
/// other character outside ASCII.
fn encode(text: &str) -> String {
    let mut s = String::new();
    for c in text.chars() {
        match c {
            '°' => s.push_str("%%d"),
            '⌀' | 'Ø' => s.push_str("%%c"),
            '±' => s.push_str("%%p"),
            c if c.is_ascii() => s.push(c),
            c => {
                let _ = write!(s, "\\U+{:04X}", c as u32);
            }
        }
    }
    s
}

fn decode(text: &str) -> String {
    let t = text
        .replace("%%d", "°")
        .replace("%%c", "⌀")
        .replace("%%p", "±");
    let mut out = String::new();
    let mut rest = t.as_str();
    while let Some(i) = rest.find("\\U+") {
        out.push_str(&rest[..i]);
        let hex = rest.get(i + 3..i + 7).unwrap_or("");
        match u32::from_str_radix(hex, 16).ok().and_then(char::from_u32) {
            Some(c) if hex.len() == 4 => {
                out.push(c);
                rest = &rest[i + 7..];
            }
            _ => {
                out.push_str("\\U+");
                rest = &rest[i + 3..];
            }
        }
    }
    out.push_str(rest);
    out
}

struct Out(String);

impl Out {
    fn g(&mut self, code: i32, value: impl std::fmt::Display) {
        let _ = write!(self.0, "{code}\n{value}\n");
    }
    fn f(&mut self, code: i32, v: f64) {
        self.g(code, format!("{v:.6}"));
    }
    fn xy(&mut self, code: i32, p: [f64; 2]) {
        self.f(code, p[0]);
        self.f(code + 10, p[1]);
        self.f(code + 20, 0.0);
    }
    fn entity(&mut self, e: &Entity) {
        let l = e.layer.as_str();
        let head = |o: &mut Out, kind: &str| {
            o.g(0, kind);
            o.g(8, l);
        };
        match &e.shape {
            Shape::Line { a, b } => {
                head(self, "LINE");
                self.xy(10, *a);
                self.xy(11, *b);
            }
            Shape::Arc {
                center,
                radius,
                start,
                end,
            } => {
                head(self, "ARC");
                self.xy(10, *center);
                self.f(40, *radius);
                self.f(50, start.to_degrees());
                self.f(51, end.to_degrees());
            }
            Shape::Circle { center, radius } => {
                head(self, "CIRCLE");
                self.xy(10, *center);
                self.f(40, *radius);
            }
            Shape::Polyline { points, closed } => self.polyline(l, points, *closed),
            Shape::Text {
                at,
                height,
                rotation,
                text,
            } => {
                head(self, "TEXT");
                self.xy(10, *at);
                self.f(40, *height);
                self.g(1, encode(text));
                if *rotation != 0.0 {
                    self.f(50, rotation.to_degrees());
                }
            }
            Shape::MText {
                at, height, text, ..
            } => {
                for (i, line) in text.split("\\P").enumerate() {
                    head(self, "TEXT");
                    self.xy(10, [at[0], at[1] - height - i as f64 * height * 1.6]);
                    self.f(40, *height);
                    self.g(1, encode(line));
                }
            }
            Shape::Dimension(d) => {
                head(self, "DIMENSION");
                self.g(2, &d.block);
                let (def, p13, p14, flag, rot) = match &d.kind {
                    DimensionKind::Linear {
                        from,
                        to,
                        line,
                        rotation,
                    } => (*line, *from, *to, 0, *rotation),
                    DimensionKind::Radius { center, on } => (*center, *on, *on, 4, 0.0),
                    DimensionKind::Diameter { from, to } => (*from, *to, *to, 3, 0.0),
                };
                self.xy(10, def);
                self.xy(11, d.text_at);
                self.g(70, flag);
                if let Some(t) = &d.text {
                    self.g(1, encode(t));
                }
                match d.kind {
                    DimensionKind::Linear { .. } => {
                        self.xy(13, p13);
                        self.xy(14, p14);
                        self.f(50, rot.to_degrees());
                    }
                    _ => {
                        self.xy(15, p13);
                    }
                }
            }
            Shape::Hatch { loops, .. } => {
                for lp in loops {
                    self.polyline(l, lp, true);
                }
            }
            Shape::Insert {
                block,
                at,
                scale,
                rotation,
            } => {
                head(self, "INSERT");
                self.g(2, block);
                self.xy(10, *at);
                if *scale != 1.0 {
                    self.f(41, *scale);
                    self.f(42, *scale);
                }
                if *rotation != 0.0 {
                    self.f(50, rotation.to_degrees());
                }
            }
        }
    }
    fn polyline(&mut self, layer: &str, points: &[[f64; 2]], closed: bool) {
        self.g(0, "POLYLINE");
        self.g(8, layer);
        self.g(66, 1);
        self.xy(10, [0.0, 0.0]);
        self.g(70, closed as i32);
        for p in points {
            self.g(0, "VERTEX");
            self.g(8, layer);
            self.xy(10, *p);
        }
        self.g(0, "SEQEND");
        self.g(8, layer);
    }
}

/// A sheet as R12 DXF.
pub fn write(sheet: &Sheet) -> String {
    let mut o = Out(String::new());
    let (lo, hi) = sheet.extents();
    o.g(0, "SECTION");
    o.g(2, "HEADER");
    o.g(9, "$ACADVER");
    o.g(1, "AC1009");
    o.g(9, "$INSUNITS");
    o.g(70, 4);
    o.g(9, "$EXTMIN");
    o.xy(10, lo);
    o.g(9, "$EXTMAX");
    o.xy(10, hi);
    o.g(0, "ENDSEC");
    o.g(0, "SECTION");
    o.g(2, "TABLES");
    o.g(0, "TABLE");
    o.g(2, "LTYPE");
    o.g(70, sheet.linetypes.len() + 1);
    for (name, description, pattern) in std::iter::once(("CONTINUOUS", "Solid line", &[][..]))
        .chain(
            sheet
                .linetypes
                .iter()
                .map(|t| (t.name.as_str(), t.description.as_str(), &t.pattern[..])),
        )
    {
        o.g(0, "LTYPE");
        o.g(2, name);
        o.g(70, 0);
        o.g(3, description);
        o.g(72, 65);
        o.g(73, pattern.len());
        o.f(40, pattern.iter().map(|d| d.abs()).sum());
        for d in pattern {
            o.f(49, *d);
        }
    }
    o.g(0, "ENDTAB");
    o.g(0, "TABLE");
    o.g(2, "LAYER");
    o.g(70, sheet.layers.len());
    for l in &sheet.layers {
        o.g(0, "LAYER");
        o.g(2, &l.name);
        o.g(70, 0);
        o.g(62, l.color);
        o.g(6, &l.linetype);
    }
    o.g(0, "ENDTAB");
    o.g(0, "ENDSEC");
    o.g(0, "SECTION");
    o.g(2, "BLOCKS");
    for b in &sheet.blocks {
        o.g(0, "BLOCK");
        o.g(8, "0");
        o.g(2, &b.name);
        o.g(70, if b.name.starts_with('*') { 1 } else { 0 });
        o.xy(10, b.base);
        o.g(3, &b.name);
        for e in &b.entities {
            o.entity(e);
        }
        o.g(0, "ENDBLK");
        o.g(8, "0");
    }
    o.g(0, "ENDSEC");
    o.g(0, "SECTION");
    o.g(2, "ENTITIES");
    for e in &sheet.entities {
        o.entity(e);
    }
    o.g(0, "ENDSEC");
    o.g(0, "EOF");
    o.0
}

/// Read a DXF this writer wrote back into a sheet (paragraph text comes back as single-line
/// text, hatches as their boundary polylines).
pub fn read(text: &str) -> Result<Sheet, String> {
    let lines: Vec<&str> = text.lines().collect();
    let pairs: Vec<(i32, &str)> = lines
        .chunks_exact(2)
        .map(|p| {
            Ok((
                p[0].trim()
                    .parse::<i32>()
                    .map_err(|_| format!("bad group code {:?}", p[0]))?,
                p[1].trim_end(),
            ))
        })
        .collect::<Result<_, String>>()?;
    let mut sheet = Sheet::default();
    let mut i = 0;
    let mut section = "";
    let mut block: Option<Block> = None;
    while i < pairs.len() {
        let (code, value) = pairs[i];
        if code != 0 {
            i += 1;
            continue;
        }
        // The record: this 0 group and every pair to the next 0 group.
        let end = (i + 1..pairs.len())
            .find(|&k| pairs[k].0 == 0)
            .unwrap_or(pairs.len());
        let rec = &pairs[i + 1..end];
        let s = |c: i32| rec.iter().find(|p| p.0 == c).map(|p| p.1).unwrap_or("");
        let f = |c: i32| s(c).parse::<f64>().unwrap_or(0.0);
        let xy = |c: i32| [f(c), f(c + 10)];
        let layer = s(8).to_owned();
        let mut shape = None;
        match value {
            "SECTION" => {
                section = if s(2) == "ENTITIES" {
                    "ENTITIES"
                } else if s(2) == "BLOCKS" {
                    "BLOCKS"
                } else {
                    ""
                }
            }
            "LAYER" => sheet.layers.push(crate::formats::sheet::Layer {
                name: s(2).into(),
                color: f(62) as i16,
                linetype: s(6).into(),
            }),
            "LTYPE" if s(2) != "CONTINUOUS" => {
                sheet.linetypes.push(crate::formats::sheet::Linetype {
                    name: s(2).into(),
                    description: s(3).into(),
                    pattern: rec
                        .iter()
                        .filter(|p| p.0 == 49)
                        .filter_map(|p| p.1.parse().ok())
                        .collect(),
                })
            }
            "BLOCK" => {
                block = Some(Block {
                    name: s(2).into(),
                    base: xy(10),
                    entities: Vec::new(),
                })
            }
            "ENDBLK" => sheet.blocks.extend(block.take()),
            "LINE" => {
                shape = Some(Shape::Line {
                    a: xy(10),
                    b: xy(11),
                })
            }
            "ARC" => {
                shape = Some(Shape::Arc {
                    center: xy(10),
                    radius: f(40),
                    start: f(50).to_radians(),
                    end: f(51).to_radians(),
                })
            }
            "CIRCLE" => {
                shape = Some(Shape::Circle {
                    center: xy(10),
                    radius: f(40),
                })
            }
            "TEXT" => {
                shape = Some(Shape::Text {
                    at: xy(10),
                    height: f(40),
                    rotation: f(50).to_radians(),
                    text: decode(s(1)),
                })
            }
            "INSERT" => {
                let scale = if s(41).is_empty() { 1.0 } else { f(41) };
                shape = Some(Shape::Insert {
                    block: s(2).into(),
                    at: xy(10),
                    scale,
                    rotation: f(50).to_radians(),
                })
            }
            "DIMENSION" => {
                let text = rec.iter().find(|p| p.0 == 1).map(|p| decode(p.1));
                let kind = match f(70) as i32 & 7 {
                    4 => DimensionKind::Radius {
                        center: xy(10),
                        on: xy(15),
                    },
                    3 => DimensionKind::Diameter {
                        from: xy(10),
                        to: xy(15),
                    },
                    _ => DimensionKind::Linear {
                        from: xy(13),
                        to: xy(14),
                        line: xy(10),
                        rotation: f(50).to_radians(),
                    },
                };
                shape = Some(Shape::Dimension(Dimension {
                    kind,
                    text_at: xy(11),
                    measurement: 0.0,
                    text,
                    block: s(2).into(),
                }))
            }
            "POLYLINE" => {
                let closed = f(70) as i32 & 1 == 1;
                let mut points = Vec::new();
                let mut k = end;
                while k < pairs.len() && pairs[k] == (0, "VERTEX") {
                    let e = (k + 1..pairs.len())
                        .find(|&j| pairs[j].0 == 0)
                        .unwrap_or(pairs.len());
                    let g = |c: i32| {
                        pairs[k + 1..e]
                            .iter()
                            .find(|p| p.0 == c)
                            .and_then(|p| p.1.parse::<f64>().ok())
                            .unwrap_or(0.0)
                    };
                    points.push([g(10), g(20)]);
                    k = e;
                }
                shape = Some(Shape::Polyline { points, closed });
            }
            _ => {}
        }
        if let Some(shape) = shape {
            let e = Entity { layer, shape };
            match (&mut block, section) {
                (Some(b), _) => b.entities.push(e),
                (None, "ENTITIES") => sheet.entities.push(e),
                _ => {}
            }
        }
        i = end;
    }
    Ok(sheet)
}

/// A drawing sheet as DXF: (sheet name, DXF text, notes).
pub(super) fn drawing_dxf(
    path: &str,
    text: &str,
    files: &Files,
    sheet: Option<&str>,
    configuration: Option<&str>,
) -> Result<(String, String, Vec<String>), ToolError> {
    let (name, sheet, notes) = drawing_sheet(path, text, files, sheet, configuration)?;
    Ok((name, write(&sheet), notes))
}

/// The DXF of a part's flat pattern, when it has one.
pub(super) fn flat_pattern_dxf(path: &str, text: &str) -> Result<String, ToolError> {
    Ok(write(&flat_pattern(path, text)?))
}
