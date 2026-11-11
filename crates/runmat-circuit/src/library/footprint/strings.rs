//! Footprint-generator strings: one string names a standard package and every dimension the
//! generator needs (`QFN24_4mm_P0.5_EP2.6`, `SOIC8`, `R0402@A`).
//!
//! | Family | Canonical form | Example |
//! |---|---|---|
//! | chip R, C, L, D, F, LED | `<kind><imperial size>` | `R0402`, `LED0805` |
//! | SOIC | `SOIC<pins>`, `SOIC<pins>W` (7.5 mm body) | `SOIC8`, `SOIC16W` |
//! | TSSOP | `TSSOP<pins>` | `TSSOP20` |
//! | MSOP | `MSOP<pins>` | `MSOP10` |
//! | QFP | `QFP<pins>_<body>mm_P<pitch>` | `QFP48_7mm_P0.5` |
//! | QFN | `QFN<pins>_<body>mm_P<pitch>[_EP<pad>]` | `QFN24_4mm_P0.5_EP2.6` |
//! | DFN | `DFN<pins>_<w>x<l>mm_P<pitch>[_EP<w>x<l>]` | `DFN8_3x3mm_P0.65_EP1.6x2.4` |
//! | BGA | `BGA<balls>_<rows>x<cols>_P<pitch>[_BALL<d>]` | `BGA256_16x16_P1` |
//! | SOT-23 | `SOT23-<pins>` | `SOT23-6` |
//! | SOT-223 | `SOT223` | `SOT223` |
//! | Pin header | `PinHeader<cols>x<rows>[_P<pitch>]` | `PinHeader1x04` |
//! | SMD connector | `SMDConn<pins>_P<pitch>[_Pad<w>x<l>][_MP<w>x<l>]` | `SMDConn4_P1_MP1.2x1.8` |
//! | Mounting hole | `MountingHole_<drill>mm_Pad<d>` | `MountingHole_3.2mm_Pad6` |
//!
//! Letters are not case-sensitive on reading, and `mm` after a size is optional. `@A` (most
//! material) or `@C` (least) picks the IPC-7351B density for the families it applies to; the
//! default is `@B`, nominal. [`canonical`] writes the form above, defaults left out.

use super::{bga, chip, dfn, mounting_hole, msop, pin_header, qfn, qfp, smd_connector, soic, sot223, sot23, tssop, Density};
use crate::geom::{mm, Point};
use crate::library::{Package, PadShape};
use runmat_design_syntax::json::format_f64;
use serde::Serialize;

/// A package a generator string names, with every dimension it takes.
#[derive(Clone, Debug, PartialEq)]
enum Spec {
    Chip { kind: String, size: String },
    Soic { pins: u32, wide: bool },
    Tssop { pins: u32 },
    Msop { pins: u32 },
    Qfp { pins: u32, body: f64, pitch: f64 },
    Qfn { pins: u32, body: f64, pitch: f64, ep: Option<f64> },
    Dfn { pins: u32, body: (f64, f64), pitch: f64, ep: Option<(f64, f64)> },
    Bga { rows: u32, cols: u32, pitch: f64, ball: f64 },
    Sot23 { pins: u32 },
    Sot223,
    PinHeader { cols: u32, rows: u32, pitch: f64 },
    SmdConn { pins: u32, pitch: f64, pad: (f64, f64), mounting: Option<(f64, f64)> },
    MountingHole { drill: f64, pad: f64 },
}

/// One parameter of a described package.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Parameter {
    pub name: &'static str,
    pub value: serde_json::Value,
    /// `mm` for a length, empty for a count or a word.
    pub unit: &'static str,
}

/// What a generator string names: its family, the parameters the generator takes, the
/// density, and the canonical string that names the same package.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Description {
    /// The family's word (`qfn`, `chip`, `pin_header`).
    pub family: &'static str,
    /// The family as the docs name it (`QFN`, `Chip passive`).
    pub label: &'static str,
    pub parameters: Vec<Parameter>,
    /// `A`, `B` or `C`; absent for families the density doesn't change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub density: Option<char>,
    pub canonical: String,
}

fn n(v: f64) -> String {
    format_f64(v)
}

fn pair(text: &str) -> Option<(f64, f64)> {
    let text = text.trim_end_matches("mm");
    match text.split_once('x') {
        Some((a, b)) => Some((a.parse().ok()?, b.parse().ok()?)),
        None => text.parse().ok().map(|v| (v, v)),
    }
}

fn length(text: &str) -> Option<f64> {
    text.trim_end_matches("mm").parse().ok().filter(|v: &f64| *v > 0.0)
}

/// The value after `prefix` in one of `fields`, by the first field that starts with it.
fn field<'a>(fields: &[&'a str], prefix: &str) -> Option<&'a str> {
    fields
        .iter()
        .find_map(|f| f.strip_prefix(prefix).filter(|v| v.starts_with(|c: char| c.is_ascii_digit() || c == '.')))
}

fn count(text: &str) -> Option<u32> {
    text.parse().ok().filter(|v| *v > 0)
}

impl Spec {
    fn parse(lower: &str) -> Option<Spec> {
        let mut parts = lower.split('_');
        let head = parts.next()?;
        let rest: Vec<&str> = parts.collect();
        for kind in ["led", "r", "c", "l", "d", "f"] {
            if let Some(code) = head.strip_prefix(kind) {
                if !code.is_empty() && code.chars().all(|c| c.is_ascii_digit()) && rest.is_empty() {
                    return Some(Spec::Chip { kind: kind.to_ascii_uppercase(), size: code.into() });
                }
            }
        }
        if let Some(r) = head.strip_prefix("soic") {
            return Some(Spec::Soic { pins: count(r.trim_end_matches('w'))?, wide: r.ends_with('w') });
        }
        if let Some(r) = head.strip_prefix("tssop") {
            return Some(Spec::Tssop { pins: count(r)? });
        }
        if let Some(r) = head.strip_prefix("msop") {
            return Some(Spec::Msop { pins: count(r)? });
        }
        if let Some(r) = head.strip_prefix("qfp") {
            return Some(Spec::Qfp { pins: count(r)?, body: length(rest.first()?)?, pitch: length(field(&rest, "p")?)? });
        }
        if let Some(r) = head.strip_prefix("qfn") {
            let ep = match field(&rest, "ep") {
                Some(e) => Some(length(e)?),
                None => None,
            };
            return Some(Spec::Qfn { pins: count(r)?, body: length(rest.first()?)?, pitch: length(field(&rest, "p")?)?, ep });
        }
        if let Some(r) = head.strip_prefix("dfn") {
            let ep = match field(&rest, "ep") {
                Some(e) => Some(pair(e)?),
                None => None,
            };
            return Some(Spec::Dfn { pins: count(r)?, body: pair(rest.first()?)?, pitch: length(field(&rest, "p")?)?, ep });
        }
        if head.starts_with("bga") {
            let (r, c) = rest.first()?.split_once('x')?;
            let pitch = length(field(&rest, "p")?)?;
            let ball = match field(&rest, "ball") {
                Some(b) => length(b)?,
                None => pitch * 0.5,
            };
            return Some(Spec::Bga { rows: count(r)?, cols: count(c)?, pitch, ball });
        }
        if let Some(r) = head.strip_prefix("sot23") {
            let pins = if r.is_empty() { 3 } else { count(r.trim_start_matches('-'))? };
            return [3, 5, 6].contains(&pins).then_some(Spec::Sot23 { pins });
        }
        if head == "sot223" && rest.is_empty() {
            return Some(Spec::Sot223);
        }
        if let Some(r) = head.strip_prefix("pinheader") {
            let (c, rows) = r.split_once('x')?;
            let pitch = match field(&rest, "p") {
                Some(p) => length(p)?,
                None => 2.54,
            };
            return Some(Spec::PinHeader { cols: count(c)?, rows: count(rows)?, pitch });
        }
        if let Some(r) = head.strip_prefix("smdconn") {
            let pitch = length(field(&rest, "p")?)?;
            let pad = match field(&rest, "pad") {
                Some(p) => pair(p)?,
                None => (round2(pitch * 0.6), 1.5),
            };
            let mounting = match field(&rest, "mp") {
                Some(m) => Some(pair(m)?),
                None => None,
            };
            return Some(Spec::SmdConn { pins: count(r)?, pitch, pad, mounting });
        }
        if head == "mountinghole" {
            let drill = length(rest.first()?)?;
            let pad = match field(&rest, "pad") {
                Some(p) => length(p)?,
                None => round2(drill * 1.875),
            };
            return (pad > drill).then_some(Spec::MountingHole { drill, pad });
        }
        None
    }

    fn generate(&self, d: Density) -> Option<Package> {
        Some(match self {
            Spec::Chip { kind, size } => chip(kind, size, d)?,
            Spec::Soic { pins, wide } => soic(*pins, !wide, d),
            Spec::Tssop { pins } => tssop(*pins, d),
            Spec::Msop { pins } => msop(*pins, d),
            Spec::Qfp { pins, body, pitch } => qfp(*pins, *body, *pitch, d),
            Spec::Qfn { pins, body, pitch, ep } => qfn(*pins, *body, *pitch, *ep, d),
            Spec::Dfn { pins, body, pitch, ep } => dfn(*pins, *body, *pitch, *ep, d),
            Spec::Bga { rows, cols, pitch, ball } => {
                let b = (*cols as f64 * pitch + pitch, *rows as f64 * pitch + pitch);
                bga(*rows, *cols, *pitch, *ball, b, |_, _| true, d)
            }
            Spec::Sot23 { pins } => sot23(*pins, d)?,
            Spec::Sot223 => sot223(d),
            Spec::PinHeader { cols, rows, pitch } => pin_header(*rows, *cols, *pitch),
            Spec::SmdConn { pins, pitch, pad, mounting } => {
                let body = ((*pins as f64 - 1.0) * pitch + 3.0, 3.0);
                let mps: Vec<(Point, PadShape, Option<i64>)> = match mounting {
                    Some((w, l)) => [-1.0, 1.0]
                        .iter()
                        .map(|s| (Point::mm(s * (body.0 / 2.0 - w / 2.0), body.1 / 2.0), PadShape::Rect { w: mm(*w), h: mm(*l) }, None))
                        .collect(),
                    None => vec![],
                };
                smd_connector("SMDConn", *pins, *pitch, *pad, &mps, body)
            }
            Spec::MountingHole { drill, pad } => mounting_hole(*drill, *pad),
        })
    }

    /// Whether the IPC density changes this package.
    fn takes_density(&self) -> bool {
        !matches!(self, Spec::PinHeader { .. } | Spec::SmdConn { .. } | Spec::MountingHole { .. })
    }

    fn canonical(&self) -> String {
        let ep = |e: Option<(f64, f64)>| e.map(|(w, l)| format!("_EP{}x{}", n(w), n(l))).unwrap_or_default();
        match self {
            Spec::Chip { kind, size } => format!("{kind}{size}"),
            Spec::Soic { pins, wide } => format!("SOIC{pins}{}", if *wide { "W" } else { "" }),
            Spec::Tssop { pins } => format!("TSSOP{pins}"),
            Spec::Msop { pins } => format!("MSOP{pins}"),
            Spec::Qfp { pins, body, pitch } => format!("QFP{pins}_{}mm_P{}", n(*body), n(*pitch)),
            Spec::Qfn { pins, body, pitch, ep } => {
                format!("QFN{pins}_{}mm_P{}{}", n(*body), n(*pitch), ep.map(|e| format!("_EP{}", n(e))).unwrap_or_default())
            }
            Spec::Dfn { pins, body, pitch, ep: e } => format!("DFN{pins}_{}x{}mm_P{}{}", n(body.0), n(body.1), n(*pitch), ep(*e)),
            Spec::Bga { rows, cols, pitch, ball } => {
                let b = if *ball == pitch * 0.5 { String::new() } else { format!("_BALL{}", n(*ball)) };
                format!("BGA{}_{rows}x{cols}_P{}{b}", rows * cols, n(*pitch))
            }
            Spec::Sot23 { pins } => format!("SOT23-{pins}"),
            Spec::Sot223 => "SOT223".into(),
            Spec::PinHeader { cols, rows, pitch } => {
                format!("PinHeader{cols}x{rows:02}{}", if *pitch == 2.54 { String::new() } else { format!("_P{}", n(*pitch)) })
            }
            Spec::SmdConn { pins, pitch, pad, mounting } => {
                let p = if *pad == (round2(pitch * 0.6), 1.5) { String::new() } else { format!("_Pad{}x{}", n(pad.0), n(pad.1)) };
                let m = mounting.map(|(w, l)| format!("_MP{}x{}", n(w), n(l))).unwrap_or_default();
                format!("SMDConn{pins}_P{}{p}{m}", n(*pitch))
            }
            Spec::MountingHole { drill, pad } => format!("MountingHole_{}mm_Pad{}", n(*drill), n(*pad)),
        }
    }

    fn family(&self) -> (&'static str, &'static str) {
        match self {
            Spec::Chip { .. } => ("chip", "Chip passive"),
            Spec::Soic { .. } => ("soic", "SOIC"),
            Spec::Tssop { .. } => ("tssop", "TSSOP"),
            Spec::Msop { .. } => ("msop", "MSOP"),
            Spec::Qfp { .. } => ("qfp", "QFP"),
            Spec::Qfn { .. } => ("qfn", "QFN"),
            Spec::Dfn { .. } => ("dfn", "DFN"),
            Spec::Bga { .. } => ("bga", "BGA"),
            Spec::Sot23 { .. } => ("sot23", "SOT-23"),
            Spec::Sot223 => ("sot223", "SOT-223"),
            Spec::PinHeader { .. } => ("pin_header", "Pin header"),
            Spec::SmdConn { .. } => ("smd_connector", "SMD connector"),
            Spec::MountingHole { .. } => ("mounting_hole", "Mounting hole"),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let count = |name, v: u32| Parameter { name, value: v.into(), unit: "" };
        let len = |name, v: f64| Parameter { name, value: v.into(), unit: "mm" };
        match self {
            Spec::Chip { kind, size } => vec![
                Parameter { name: "kind", value: kind.clone().into(), unit: "" },
                Parameter { name: "size", value: size.clone().into(), unit: "" },
            ],
            Spec::Soic { pins, wide } => vec![count("pins", *pins), len("body_width", if *wide { 7.5 } else { 3.9 })],
            Spec::Tssop { pins } | Spec::Msop { pins } => vec![count("pins", *pins)],
            Spec::Qfp { pins, body, pitch } => vec![count("pins", *pins), len("body", *body), len("pitch", *pitch)],
            Spec::Qfn { pins, body, pitch, ep } => {
                let mut v = vec![count("pins", *pins), len("body", *body), len("pitch", *pitch)];
                v.extend(ep.map(|e| len("exposed_pad", e)));
                v
            }
            Spec::Dfn { pins, body, pitch, ep } => {
                let mut v = vec![count("pins", *pins), len("body_width", body.0), len("body_length", body.1), len("pitch", *pitch)];
                if let Some((w, l)) = ep {
                    v.extend([len("exposed_pad_width", *w), len("exposed_pad_length", *l)]);
                }
                v
            }
            Spec::Bga { rows, cols, pitch, ball } => vec![count("rows", *rows), count("cols", *cols), len("pitch", *pitch), len("ball", *ball)],
            Spec::Sot23 { pins } => vec![count("pins", *pins)],
            Spec::Sot223 => vec![],
            Spec::PinHeader { cols, rows, pitch } => vec![count("cols", *cols), count("rows", *rows), len("pitch", *pitch)],
            Spec::SmdConn { pins, pitch, pad, mounting } => {
                let mut v = vec![count("pins", *pins), len("pitch", *pitch), len("pad_width", pad.0), len("pad_length", pad.1)];
                if let Some((w, l)) = mounting {
                    v.extend([len("mounting_pad_width", *w), len("mounting_pad_length", *l)]);
                }
                v
            }
            Spec::MountingHole { drill, pad } => vec![len("drill", *drill), len("pad", *pad)],
        }
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn split(s: &str) -> Option<(Spec, Density)> {
    let (body, density) = match s.trim().rsplit_once('@') {
        Some((b, d)) => (
            b,
            match d.to_ascii_uppercase().as_str() {
                "A" => Density::A,
                "B" => Density::B,
                "C" => Density::C,
                _ => return None,
            },
        ),
        None => (s.trim(), Density::B),
    };
    Some((Spec::parse(&body.to_ascii_lowercase())?, density))
}

/// The package a generator string names, or `None` when it names none.
pub fn from_string(s: &str) -> Option<Package> {
    let (spec, density) = split(s)?;
    spec.generate(density)
}

/// What a generator string names: family, parameters, density and canonical string. `None`
/// when the string names no package the generator makes.
pub fn describe(s: &str) -> Option<Description> {
    let (spec, density) = split(s)?;
    spec.generate(density)?;
    let (family, label) = spec.family();
    let takes = spec.takes_density();
    let mut canonical = spec.canonical();
    if takes && density != Density::B {
        canonical.push_str(if density == Density::A { "@A" } else { "@C" });
    }
    Some(Description {
        family,
        label,
        parameters: spec.parameters(),
        density: takes.then_some(match density {
            Density::A => 'A',
            Density::B => 'B',
            Density::C => 'C',
        }),
        canonical,
    })
}

/// The canonical form of a generator string; any other text (a footprint file) unchanged.
pub fn canonical(s: &str) -> String {
    describe(s).map(|d| d.canonical).unwrap_or_else(|| s.to_owned())
}

#[cfg(test)]
#[path = "strings_tests.rs"]
mod tests;
