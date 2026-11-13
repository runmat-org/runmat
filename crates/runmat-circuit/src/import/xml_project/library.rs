//! XML project libraries as the schematic and board embed them: packages become footprints
//! ([`Package`]) and device sets become catalog parts, one per device and technology.

use super::super::Report;
use crate::geom::{mm, Coord, Point};
use crate::import::xml::Element;
use crate::library::catalog::{PartPin, PartRecord};
use crate::library::{make, CopperSpan, Graphic, Package, PadDef, PadKind, PadShape};
use std::collections::BTreeMap;

/// An XML project rotation (`R90`, `MR180`, `SR45`): degrees counter-clockwise and whether it mirrors.
pub fn rotation(rot: Option<&str>) -> (f64, bool) {
    let r = rot.unwrap_or("R0");
    let mirror = r.contains('M');
    let deg = r
        .trim_start_matches(|c: char| c.is_ascii_alphabetic())
        .parse()
        .unwrap_or(0.0);
    (deg, mirror)
}

pub fn point(e: &Element, x: &str, y: &str) -> Point {
    Point::new(mm(e.num(x).unwrap_or(0.0)), mm(e.num(y).unwrap_or(0.0)))
}

pub fn len(e: &Element, name: &str) -> Coord {
    mm(e.num(name).unwrap_or(0.0))
}

/// The arc a `curve`d wire draws from `a` to `b` (degrees, counter-clockwise when positive),
/// as its mid point.
pub fn arc_mid(a: Point, b: Point, curve: f64) -> Point {
    let (ax, ay, bx, by) = (a.x as f64, a.y as f64, b.x as f64, b.y as f64);
    let (mx, my) = ((ax + bx) / 2.0, (ay + by) / 2.0);
    let (dx, dy) = (bx - ax, by - ay);
    let half = curve.to_radians() / 2.0;
    // The centre sits on the chord's perpendicular, left of a→b for a counter-clockwise arc.
    let k = 0.5 / half.tan();
    let (cx, cy) = (mx - dy * k, my + dx * k);
    let (rx, ry) = (ax - cx, ay - cy);
    let (s, c) = half.sin_cos();
    Point::new(
        (cx + rx * c - ry * s).round() as Coord,
        (cy + rx * s + ry * c).round() as Coord,
    )
}

/// A wire as a footprint or board graphic: a line, or an arc when curved.
pub fn wire_graphic(w: &Element) -> Graphic {
    let (a, b) = (point(w, "x1", "y1"), point(w, "x2", "y2"));
    let width = len(w, "width");
    match w.num("curve").filter(|c| c.abs() > 1e-9) {
        Some(curve) => Graphic::Arc {
            start: a,
            mid: arc_mid(a, b, curve),
            end: b,
            width,
        },
        None => Graphic::Line {
            points: vec![a, b],
            width,
        },
    }
}

/// XML project layers: 1 Top … 16 Bottom; 21/22 placement (silk), 51/52 documentation, 20 dimension.
pub const TOP: u32 = 1;
pub const BOTTOM: u32 = 16;
pub const DIMENSION: u32 = 20;
pub const T_PLACE: u32 = 21;
pub const B_PLACE: u32 = 22;
pub const T_DOCU: u32 = 51;
pub const B_DOCU: u32 = 52;
pub const MILLING: u32 = 46;

pub fn layer(e: &Element) -> u32 {
    e.num("layer").map(|l| l as u32).unwrap_or(0)
}

/// A through-hole pad's copper size when the file leaves it to the design rules
/// (`diameter="0"`): the drill plus a restring of a quarter of it, between 0.25 and 0.5 mm.
fn auto_diameter(drill: Coord) -> Coord {
    drill + 2 * (drill / 4).clamp(mm(0.25), mm(0.5))
}

fn no(e: &Element, name: &str) -> bool {
    e.attr(name) == Some("no")
}

/// A package as a footprint. `key` names it in the pool and in the footprint file.
pub fn package(key: &str, p: &Element, report: &mut Report) -> Package {
    let mut pads = Vec::new();
    let (mut silk, mut fab) = (Vec::new(), Vec::new());
    let mut copper: Vec<&Element> = Vec::new();
    for item in &p.children {
        match item.name.as_str() {
            "smd" => {
                let (w, h) = (len(item, "dx"), len(item, "dy"));
                let roundness = item.num("roundness").unwrap_or(0.0).clamp(0.0, 100.0);
                let shape = if roundness >= 100.0 && w == h {
                    PadShape::Circle { d: w }
                } else if roundness >= 100.0 {
                    PadShape::Oval { w, h }
                } else if roundness > 0.0 {
                    PadShape::RoundRect {
                        w,
                        h,
                        ratio: roundness / 200.0,
                    }
                } else {
                    PadShape::Rect { w, h }
                };
                let mut pad = PadDef::smd(
                    item.attr("name").unwrap_or(""),
                    point(item, "x", "y"),
                    shape,
                );
                pad.rotation = rotation(item.attr("rot")).0;
                pad.copper = if layer(item) == BOTTOM {
                    CopperSpan::Back
                } else {
                    CopperSpan::Front
                };
                pad.mask = !no(item, "stop");
                pad.paste = !no(item, "cream");
                pads.push(pad);
            }
            "pad" => {
                let drill = len(item, "drill");
                let d = match len(item, "diameter") {
                    0 => auto_diameter(drill),
                    d => d,
                };
                let shape = match item.attr("shape").unwrap_or("round") {
                    "square" => PadShape::Rect { w: d, h: d },
                    "octagon" => PadShape::Chamfered {
                        w: d,
                        h: d,
                        chamfer: (d as f64 * 0.2929) as Coord,
                        corners: 15,
                    },
                    "long" => PadShape::Oval { w: 2 * d, h: d },
                    "offset" => {
                        report.approximated(format!("package {key}: pad {} has an offset shape; it becomes an oval centred on the drill", item.attr("name").unwrap_or("")));
                        PadShape::Oval { w: 2 * d, h: d }
                    }
                    _ => PadShape::Circle { d },
                };
                let mut pad = PadDef::tht(
                    item.attr("name").unwrap_or(""),
                    point(item, "x", "y"),
                    shape,
                    drill,
                );
                pad.rotation = rotation(item.attr("rot")).0;
                pad.mask = !no(item, "stop");
                pads.push(pad);
            }
            "hole" => {
                let drill = len(item, "drill");
                pads.push(PadDef::npth("", point(item, "x", "y"), drill));
            }
            "wire" | "circle" | "rectangle" | "polygon" => {
                let Some(g) = graphic(item) else { continue };
                match layer(item) {
                    T_PLACE => silk.push(g),
                    T_DOCU => fab.push(g),
                    B_PLACE | B_DOCU => {
                        report.not_carried(format!(
                            "package {key}: bottom-side graphics in a top-side package"
                        ));
                    }
                    TOP | BOTTOM if item.name == "polygon" => copper.push(item),
                    _ => {}
                }
            }
            // `>NAME` and `>VALUE` texts: the board places the designator and value itself.
            "text" | "description" => {}
            other => report.not_carried(format!("package {key}: <{other}>")),
        }
    }
    // A copper polygon in a package (a castellation's half pad, a pad's extension) is more
    // copper of the pad whose centre it covers: a pad of the same name, so it carries that
    // pad's net.
    for poly in copper {
        let points: Vec<Point> = poly.all("vertex").map(|v| point(v, "x", "y")).collect();
        let host = pads
            .iter()
            .find(|p| {
                p.kind != PadKind::Npth
                    && !p.name.is_empty()
                    && crate::geom::poly::ring_contains(&points, p.at) != Some(false)
            })
            .map(|p| (p.name.clone(), p.at));
        match host {
            Some((name, at)) if points.len() >= 3 => {
                if poly
                    .all("vertex")
                    .any(|v| v.num("curve").is_some_and(|c| c.abs() > 1e-9))
                {
                    report.approximated(format!(
                        "package {key}: curved copper polygon edges are carried as straight ones"
                    ));
                }
                let mut outline: Vec<Point> = points.iter().map(|q| *q - at).collect();
                if crate::geom::poly::area2(&outline) < 0 {
                    outline.reverse();
                }
                let mut pad = PadDef::smd(
                    &name,
                    at,
                    PadShape::Custom {
                        points: outline,
                        holes: vec![],
                    },
                );
                pad.copper = if layer(poly) == BOTTOM {
                    CopperSpan::Back
                } else {
                    CopperSpan::Front
                };
                pad.paste = false;
                pads.push(pad);
            }
            _ => report.not_carried(format!(
                "package {key}: a copper polygon that covers no pad"
            )),
        }
    }
    let smd = !pads.is_empty() && pads.iter().all(|p| p.kind == PadKind::Smd);
    Package {
        uuid: make::uuid("package", key),
        name: key.to_owned(),
        description: p
            .child("description")
            .map(|d| d.text.trim().to_owned())
            .unwrap_or_default(),
        pads,
        courtyard: vec![],
        silk,
        fab,
        model: None,
        body: None,
        smd,
    }
}

/// A wire, circle, rectangle or polygon as a graphic (outside copper).
pub fn graphic(item: &Element) -> Option<Graphic> {
    let width = len(item, "width");
    Some(match item.name.as_str() {
        "wire" => wire_graphic(item),
        "circle" => Graphic::Circle {
            center: point(item, "x", "y"),
            radius: len(item, "radius"),
            width,
            fill: width == 0,
        },
        "rectangle" => {
            let (a, b) = (point(item, "x1", "y1"), point(item, "x2", "y2"));
            let deg = rotation(item.attr("rot")).0;
            if deg.rem_euclid(180.0) == 0.0 {
                Graphic::Rect {
                    a,
                    b,
                    width: 0,
                    fill: true,
                }
            } else {
                let c = Point::new((a.x + b.x) / 2, (a.y + b.y) / 2);
                let t = crate::geom::Transform::rotation(deg);
                let corners = [
                    Point::new(a.x, a.y),
                    Point::new(b.x, a.y),
                    Point::new(b.x, b.y),
                    Point::new(a.x, b.y),
                ];
                let points = corners.iter().map(|p| t.apply(*p - c) + c).collect();
                Graphic::Polygon {
                    points,
                    width: 0,
                    fill: true,
                }
            }
        }
        "polygon" => Graphic::Polygon {
            points: item.all("vertex").map(|v| point(v, "x", "y")).collect(),
            width,
            fill: true,
        },
        _ => return None,
    })
}

/// A pin name the part can carry: The format's `@n` suffix (several pads of one signal shown as
/// one pin name) is kept, so each pin stays its own.
fn pin_type(direction: Option<&str>) -> &'static str {
    match direction.unwrap_or("io") {
        "nc" => "no_connect",
        "in" => "input",
        "out" => "output",
        "oc" => "open_collector",
        "pwr" => "power_in",
        "pas" => "passive",
        "hiz" => "tri_state",
        "sup" => "power_out",
        _ => "bidirectional",
    }
}

/// A device set's device: its catalog part (pins as `gate.pin` when a pin name repeats across
/// gates) and, per gate and pin, the part's pin name.
pub struct DevicePart {
    pub part: PartRecord,
    /// (gate, symbol pin) → part pin.
    pub pins: BTreeMap<(String, String), String>,
}

/// The attributes that carry a manufacturer part number, in the order they're tried.
const MPN_KEYS: &[&str] = &[
    "MPN",
    "MANUFACTURER_PART_NUMBER",
    "MFR_PART_NUMBER",
    "PARTNO",
    "PART_NUMBER",
    "MPN1",
    "MF_PN",
    "MP",
];
const MAKER_KEYS: &[&str] = &["MANUFACTURER", "MANUFACTURER_NAME", "MFR", "MF", "VENDOR"];

/// The value of the first attribute named in `keys` (case-insensitive) that has one.
pub fn find_attr(attrs: &BTreeMap<String, String>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| {
        attrs
            .iter()
            .find(|(name, v)| name.eq_ignore_ascii_case(k) && !v.trim().is_empty())
            .map(|(_, v)| v.trim().to_owned())
    })
}

/// The part a device makes. `symbols` is the library's symbols by name (for pin directions),
/// `package_file` the footprint file the part's package names, `attrs` the technology's and the
/// placed part's attributes (the part's win).
#[allow(clippy::too_many_arguments)]
pub fn device_part(
    library: &str,
    deviceset: &Element,
    device: &Element,
    symbols: &BTreeMap<String, &Element>,
    package_file: &str,
    attrs: &BTreeMap<String, String>,
    fallback_mpn: &str,
) -> DevicePart {
    let gates: Vec<(String, String)> = deviceset
        .list("gates", "gate")
        .map(|g| {
            (
                g.attr("name").unwrap_or("").to_owned(),
                g.attr("symbol").unwrap_or("").to_owned(),
            )
        })
        .collect();
    let direction = |gate: &str, pin: &str| -> Option<String> {
        let symbol = gates
            .iter()
            .find(|(g, _)| g == gate)
            .map(|(_, s)| s.as_str())?;
        symbols
            .get(symbol)?
            .all("pin")
            .find(|p| p.attr("name") == Some(pin))
            .map(|p| p.attr("direction").unwrap_or("io").to_owned())
    };
    let connects: Vec<(String, String, Vec<String>)> = device
        .list("connects", "connect")
        .map(|c| {
            (
                c.attr("gate").unwrap_or("").to_owned(),
                c.attr("pin").unwrap_or("").to_owned(),
                c.attr("pad")
                    .unwrap_or("")
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect(),
            )
        })
        .collect();
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, pin, _) in &connects {
        *seen.entry(pin.as_str()).or_default() += 1;
    }
    let mut pins = BTreeMap::new();
    let mut part_pins = Vec::new();
    for (gate, pin, pads) in &connects {
        let name = if seen[pin.as_str()] > 1 {
            format!("{gate}.{pin}")
        } else {
            pin.clone()
        };
        pins.insert((gate.clone(), pin.clone()), name.clone());
        part_pins.push(PartPin {
            name,
            pads: pads.clone(),
            kind: pin_type(direction(gate, pin).as_deref()).into(),
        });
    }
    let mpn = find_attr(attrs, MPN_KEYS).unwrap_or_else(|| fallback_mpn.to_owned());
    let prefix = deviceset
        .attr("prefix")
        .unwrap_or("")
        .trim_end_matches(|c: char| !c.is_ascii_alphabetic())
        .to_owned();
    let description = deviceset
        .child("description")
        .map(|d| first_line(&d.text))
        .unwrap_or_default();
    let mut params = BTreeMap::new();
    for (k, v) in attrs {
        if MPN_KEYS
            .iter()
            .chain(MAKER_KEYS)
            .any(|m| k.eq_ignore_ascii_case(m))
            || v.trim().is_empty()
        {
            continue;
        }
        params.insert(param_key(k), v.clone());
    }
    let part = PartRecord {
        mpn,
        manufacturer: find_attr(attrs, MAKER_KEYS).unwrap_or_default(),
        description,
        category: String::new(),
        prefix,
        package: package_file.to_owned(),
        pins: part_pins,
        params,
        datasheet: None,
        spice: None,
        ..Default::default()
    };
    let _ = library;
    DevicePart { part, pins }
}

/// A parameter name the `.circuit` grammar takes as a bare word (`PROD_ID` → `prod_id`).
pub fn param_key(k: &str) -> String {
    let mut out: String = k
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    if out.starts_with(|c: char| c.is_ascii_digit()) || out.is_empty() {
        out.insert(0, 'p');
    }
    out
}

/// The first line of a description, with the rich-text markup the format stores removed.
pub fn first_line(html: &str) -> String {
    let mut out = String::new();
    let mut tag = false;
    for c in html.chars() {
        match c {
            '<' => tag = true,
            '>' => {
                tag = false;
                if !out.trim().is_empty() {
                    break;
                }
            }
            _ if !tag => out.push(c),
            _ => {}
        }
    }
    crate::import::xml::unescape(out.trim())
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_owned()
}
