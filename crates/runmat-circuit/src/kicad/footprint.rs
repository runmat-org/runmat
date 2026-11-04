//! Footprints: `(footprint "Lib:Name" …)` nodes ⇄ [`Package`].
//!
//! Inside a board file, pad positions are footprint-local but pad angles are absolute
//! (footprint rotation included), and a back-side footprint is stored already flipped. On
//! import we recover the top-side package: for a back footprint the stored local coordinates,
//! read with y up, *are* the top-side geometry, and our rotation is KiCad's plus 180° (we
//! mirror about the y axis, KiCad about the x axis).

use super::sexpr::SExpr;
use super::{at_of, len, mm_of, pts_node, pts_of, uuid_node, xy};
use crate::geom::poly::{rect_ring, Ring};
use crate::geom::{Coord, Point, Rect};
use crate::layout::Side;
use crate::library::{
    Body, CopperSpan, Drill, Graphic, Model3d, Package, PadDef, PadKind, PadShape,
};
use std::collections::BTreeMap;

/// A footprint read from a board: the package plus its placement and pad nets.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedFootprint {
    pub package: Package,
    pub at: Point,
    pub rotation: f64,
    pub side: Side,
    pub reference: String,
    pub value: String,
    pub locked: bool,
    /// Pad name → (net code or name text).
    pub pad_nets: BTreeMap<String, String>,
    pub properties: BTreeMap<String, String>,
    /// KiCad schematic path (`(path "/uuid/uuid")`) linking to the symbol.
    pub path: Option<String>,
}

pub(crate) fn graphic(node: &SExpr, local: &dyn Fn(Point) -> Point) -> Option<(String, Graphic)> {
    let layer = node.child_arg("layer")?.to_string();
    let width = node
        .child("stroke")
        .and_then(|s| mm_of(s.child("width")))
        .or_else(|| mm_of(node.child("width")))
        .unwrap_or(0);
    let p = |h: &str| local(at_of(node.child(h)).0);
    let fill = node
        .child("fill")
        .map(|f| f.arg(0) == Some("solid") || f.arg(0) == Some("yes"))
        .unwrap_or(false);
    let g = match node.head()? {
        "fp_line" | "gr_line" => Graphic::Line {
            points: vec![p("start"), p("end")],
            width,
        },
        "fp_rect" | "gr_rect" => Graphic::Rect {
            a: p("start"),
            b: p("end"),
            width,
            fill,
        },
        "fp_circle" | "gr_circle" => {
            let c = p("center");
            let e = p("end");
            Graphic::Circle {
                center: c,
                radius: c.dist(e).round() as Coord,
                width,
                fill,
            }
        }
        "fp_arc" | "gr_arc" => Graphic::Arc {
            start: p("start"),
            mid: p("mid"),
            end: p("end"),
            width,
        },
        "fp_poly" | "gr_poly" => Graphic::Polygon {
            points: pts_of(node.child("pts")).into_iter().map(local).collect(),
            width,
            fill,
        },
        "fp_text" => Graphic::Text {
            at: local(at_of(node.child("at")).0),
            text: node.arg(1).unwrap_or("").to_string(),
            size: node
                .child("effects")
                .and_then(|e| e.child("font"))
                .and_then(|f| f.child("size"))
                .and_then(|s| s.arg_f(0))
                .map(|v| (v * 1e6) as Coord)
                .unwrap_or(1_000_000),
            angle: at_of(node.child("at")).1,
        },
        _ => return None,
    };
    Some((layer, g))
}

fn pad_shape(node: &SExpr, size: (Coord, Coord)) -> PadShape {
    let (w, h) = size;
    match node.arg(2) {
        Some("circle") => PadShape::Circle { d: w },
        Some("oval") => PadShape::Oval { w, h },
        Some("roundrect") => PadShape::RoundRect {
            w,
            h,
            ratio: node.child_f("roundrect_rratio").unwrap_or(0.25),
        },
        Some("trapezoid") => PadShape::Trapezoid {
            w,
            h,
            delta: node
                .child("rect_delta")
                .and_then(|d| d.arg_f(0))
                .map(|v| (v * 1e6) as Coord)
                .unwrap_or(0),
        },
        Some("custom") => custom_shape(node, size).unwrap_or(PadShape::Rect { w, h }),
        _ => {
            if let Some(r) = node.child_f("chamfer_ratio") {
                let mut corners = 0u8;
                if let Some(c) = node.child("chamfer") {
                    for (name, bit) in [
                        ("top_left", 1u8),
                        ("top_right", 2),
                        ("bottom_left", 4),
                        ("bottom_right", 8),
                    ] {
                        if c.has_flag(name) {
                            corners |= bit;
                        }
                    }
                }
                if corners != 0 {
                    return PadShape::Chamfered {
                        w,
                        h,
                        chamfer: (w.min(h) as f64 * r) as Coord,
                        corners,
                    };
                }
            }
            PadShape::Rect { w, h }
        }
    }
}

/// A custom pad: the anchor (`(options (anchor rect|circle))`, sized by the pad's `size`)
/// unioned with every primitive: `gr_poly` (filled, with its stroke), `gr_line`, `gr_arc`
/// (start/mid/end, or the older centre/end/angle form), `gr_circle` (filled disk or stroked
/// ring), `gr_rect`, `gr_curve` (cubic Bézier, flattened), each stroked by its `width`.
/// Coordinates are pad-local. A pad is one copper island: primitives that touch nothing
/// else are dropped, keeping the largest island.
fn custom_shape(node: &SExpr, size: (Coord, Coord)) -> Option<PadShape> {
    use crate::geom::poly::{boolean, capsule_ring, circle_ring, BoolOp};
    use crate::geom::shape::{arc_from_three, arc_points};
    let tol = crate::geom::ARC_TOL;
    let (w, h) = size;
    let mut rings: Vec<Ring> = Vec::new();
    let anchor = node
        .child("options")
        .and_then(|o| o.child_arg("anchor"))
        .unwrap_or("rect");
    if w > 0 && h > 0 {
        if anchor == "circle" {
            rings.push(circle_ring(Point::ORIGIN, w / 2, tol));
        } else {
            rings.push(rect_ring(Rect::new(
                Point::new(-w / 2, -h / 2),
                Point::new(w / 2, h / 2),
            )));
        }
    }
    let width_of = |g: &SExpr| {
        mm_of(g.child("width"))
            .or_else(|| mm_of(g.child("stroke").and_then(|s| s.child("width"))))
            .unwrap_or(0)
    };
    let filled = |g: &SExpr| {
        g.child("fill")
            .and_then(|f| f.arg(0).or_else(|| f.child_arg("type")))
            .is_some_and(|v| v == "yes" || v == "solid")
    };
    let stroke = |pts: &[Point], wd: Coord, closed: bool, out: &mut Vec<Ring>| {
        if wd <= 0 {
            return;
        }
        let n = pts.len();
        let m = if closed { n } else { n.saturating_sub(1) };
        for i in 0..m {
            out.push(capsule_ring(pts[i], pts[(i + 1) % n], wd / 2, tol));
        }
    };
    let prims = node.child("primitives")?;
    for g in prims.items().iter().skip(1) {
        let wd = width_of(g);
        match g.head() {
            Some("gr_poly") => {
                let pts = pts_of(g.child("pts"));
                if pts.len() >= 3 && (filled(g) || g.child("fill").is_none()) {
                    rings.push(pts.clone());
                }
                stroke(&pts, wd, true, &mut rings);
            }
            Some("gr_line") => {
                let (a, _) = at_of(g.child("start"));
                let (b, _) = at_of(g.child("end"));
                stroke(&[a, b], wd, false, &mut rings);
            }
            Some("gr_arc") => {
                let (s0, _) = at_of(g.child("start"));
                let (e0, _) = at_of(g.child("end"));
                let pts = if let Some(m) = g.child("mid") {
                    let (m, _) = at_of(Some(m));
                    match arc_from_three(s0, m, e0) {
                        Some((c, sw)) => arc_points(c, s0, sw, tol),
                        None => vec![s0, e0],
                    }
                } else {
                    // Older form: start is the centre, end the arc start, angle the sweep
                    // (clockwise on screen, so counter-clockwise with y up).
                    let sweep = g.child_f("angle").unwrap_or(0.0);
                    arc_points(s0, e0, sweep, tol)
                };
                stroke(&pts, wd.max(1), false, &mut rings);
            }
            Some("gr_circle") => {
                let (c, _) = at_of(g.child("center"));
                let (e, _) = at_of(g.child("end"));
                let r = c.dist(e) as Coord;
                rings.push(circle_ring(c, r + wd / 2, tol));
                if !filled(g) && r > wd / 2 {
                    // A stroked circle is a ring: its inside stays open unless another
                    // primitive covers it (non-zero union).
                    let mut inner = circle_ring(c, r - wd / 2, tol);
                    inner.reverse();
                    rings.push(inner);
                }
            }
            Some("gr_rect") => {
                let (a, _) = at_of(g.child("start"));
                let (b, _) = at_of(g.child("end"));
                let r = rect_ring(Rect::new(a, b));
                if filled(g) {
                    rings.push(r.clone());
                }
                stroke(&r, wd, true, &mut rings);
            }
            Some("gr_curve") => {
                let c = pts_of(g.child("pts"));
                if c.len() == 4 {
                    let n = 32;
                    let pts: Vec<Point> = (0..=n)
                        .map(|k| {
                            let t = k as f64 / n as f64;
                            let u = 1.0 - t;
                            let f = |a: Coord, b: Coord, cc: Coord, d: Coord| {
                                u * u * u * a as f64
                                    + 3.0 * u * u * t * b as f64
                                    + 3.0 * u * t * t * cc as f64
                                    + t * t * t * d as f64
                            };
                            Point::from_f(
                                f(c[0].x, c[1].x, c[2].x, c[3].x),
                                f(c[0].y, c[1].y, c[2].y, c[3].y),
                            )
                        })
                        .collect();
                    stroke(&pts, wd.max(1), false, &mut rings);
                }
            }
            _ => {}
        }
    }
    let set = boolean(&rings, &[], BoolOp::Union);
    let poly = set
        .polys
        .iter()
        .max_by(|a, b| a.area().total_cmp(&b.area()))?
        .clone();
    Some(PadShape::Custom {
        points: poly.outer,
        holes: poly.holes,
    })
}

/// One ring for a polygon with holes (primitives can only add area): each hole is spliced in
/// through a zero-width slit from the nearest outer vertex.
fn keyhole(outer: &[Point], holes: &[Vec<Point>]) -> Vec<Point> {
    let mut ring = outer.to_vec();
    for h in holes.iter().filter(|h| h.len() >= 3) {
        let (mut bi, mut bj, mut bd) = (0, 0, f64::INFINITY);
        for (i, p) in ring.iter().enumerate() {
            for (j, q) in h.iter().enumerate() {
                let d = p.dist(*q);
                if d < bd {
                    (bi, bj, bd) = (i, j, d);
                }
            }
        }
        let mut next = ring[..=bi].to_vec();
        next.extend(h[bj..].iter().chain(&h[..=bj]).copied());
        next.extend_from_slice(&ring[bi..]);
        ring = next;
    }
    ring
}

/// Reads a footprint node. `in_board` selects the board convention (absolute pad angles,
/// back side stored flipped).
pub fn read(node: &SExpr, in_board: bool) -> PlacedFootprint {
    let name = node.arg(0).unwrap_or("").to_string();
    let (at, rot_k) = at_of(node.child("at"));
    let side = if node.child_arg("layer") == Some("B.Cu") {
        Side::Bottom
    } else {
        Side::Top
    };
    let back = in_board && side == Side::Bottom;
    // Local KiCad coordinates → package coordinates (y up), un-flipping back footprints.
    let local = move |p: Point| if back { Point::new(p.x, -p.y) } else { p };
    let rotation = if back {
        crate::geom::normalize_deg(rot_k + 180.0)
    } else {
        rot_k
    };
    let mut pads = Vec::new();
    let mut pad_nets = BTreeMap::new();
    for p in node.children("pad") {
        let pname = p.arg(0).unwrap_or("").to_string();
        let kind = match p.arg(1) {
            Some("thru_hole") => PadKind::Tht,
            Some("np_thru_hole") => PadKind::Npth,
            Some("connect") => PadKind::Connector,
            _ => PadKind::Smd,
        };
        let (pat, pang) = at_of(p.child("at"));
        let size = p
            .child("size")
            .map(|s| {
                (
                    (s.arg_f(0).unwrap_or(0.0) * 1e6) as Coord,
                    (s.arg_f(1).unwrap_or(0.0) * 1e6) as Coord,
                )
            })
            .unwrap_or((0, 0));
        let drill = p.child("drill").map(|d| {
            let oval = d.has_flag("oval");
            let (a, b) = if oval {
                (d.arg_f(1), d.arg_f(2))
            } else {
                (d.arg_f(0), None)
            };
            Drill {
                diameter: (a.unwrap_or(0.0) * 1e6) as Coord,
                slot: b.map(|v| (v * 1e6) as Coord),
                plated: kind != PadKind::Npth,
            }
        });
        let layers: Vec<&str> = p
            .child("layers")
            .map(|l| l.items().iter().skip(1).filter_map(|x| x.atom()).collect())
            .unwrap_or_default();
        let has = |n: &str| {
            layers
                .iter()
                .any(|l| *l == n || l.starts_with('*') && n.ends_with(&l[1..]))
        };
        let (front, backl) = if side == Side::Top || !in_board {
            ("F", "B")
        } else {
            ("B", "F")
        };
        let copper = if layers.contains(&"*.Cu")
            || (has(&format!("{front}.Cu")) && has(&format!("{backl}.Cu")))
        {
            CopperSpan::All
        } else if has(&format!("{backl}.Cu")) && !has(&format!("{front}.Cu")) {
            CopperSpan::Back
        } else {
            CopperSpan::Front
        };
        let local_rot = if in_board { pang - rot_k } else { pang };
        let local_rot = if back { -local_rot } else { local_rot };
        let pd = PadDef {
            name: pname.clone(),
            kind,
            shape: pad_shape(p, size),
            at: local(pat),
            rotation: crate::geom::normalize_deg(local_rot),
            drill,
            copper,
            mask: layers.iter().any(|l| l.ends_with(".Mask")),
            paste: layers.iter().any(|l| l.ends_with(".Paste")),
            mask_margin: mm_of(p.child("solder_mask_margin")),
            paste_margin: mm_of(p.child("solder_paste_margin")),
            thermal_pad: false,
        };
        if let Some(n) = p.child("net") {
            let key = n.arg(1).or(n.arg(0)).unwrap_or("").to_string();
            if !key.is_empty() {
                pad_nets.insert(pname, key);
            }
        }
        pads.push(pd);
    }
    let mut silk = Vec::new();
    let mut fab = Vec::new();
    let mut crt: Vec<Point> = Vec::new();
    let mut crt_rings: Vec<Ring> = Vec::new();
    for c in node.items().iter().skip(1) {
        let Some((layer, g)) = graphic(c, &local) else {
            continue;
        };
        if layer.ends_with("SilkS") {
            silk.push(g);
        } else if layer.ends_with("Fab") {
            fab.push(g);
        } else if layer.ends_with("CrtYd") {
            match g {
                Graphic::Rect { a, b, .. } => crt_rings.push(rect_ring(Rect::new(a, b))),
                Graphic::Polygon { points, .. } => crt_rings.push(points),
                Graphic::Line { points, .. } => crt.extend(points),
                Graphic::Circle { center, radius, .. } => {
                    crt_rings.push(crate::geom::poly::circle_ring(center, radius, 10_000))
                }
                Graphic::Arc {
                    start, mid, end, ..
                } => crt.extend([start, mid, end]),
                Graphic::Text { .. } => {}
            }
        }
    }
    if !crt.is_empty() {
        crt_rings.push(rect_ring(Rect::from_points(crt)));
    }
    for r in &mut crt_rings {
        if crate::geom::poly::area2(r) < 0 {
            r.reverse();
        }
    }
    let model = node.child("model").map(|m| {
        let v3 = |h: &str, d: f64| {
            m.child(h)
                .and_then(|x| x.child("xyz"))
                .map(|x| {
                    (
                        x.arg_f(0).unwrap_or(d),
                        x.arg_f(1).unwrap_or(d),
                        x.arg_f(2).unwrap_or(d),
                    )
                })
                .unwrap_or((d, d, d))
        };
        Model3d {
            path: m.arg(0).unwrap_or("").to_string(),
            offset: v3("offset", 0.0),
            rotation: v3("rotate", 0.0),
            scale: v3("scale", 1.0),
            origin: "footprint".into(),
        }
    });
    let mut properties = BTreeMap::new();
    let mut reference = String::new();
    let mut value = String::new();
    for pr in node.children("property") {
        let (k, v) = (pr.arg(0).unwrap_or(""), pr.arg(1).unwrap_or(""));
        match k {
            "Reference" => reference = v.to_string(),
            "Value" => value = v.to_string(),
            _ => {
                properties.insert(k.to_string(), v.to_string());
            }
        }
    }
    for t in node.children("fp_text") {
        match t.arg(0) {
            Some("reference") if reference.is_empty() => reference = t.arg(1).unwrap_or("").into(),
            Some("value") if value.is_empty() => value = t.arg(1).unwrap_or("").into(),
            _ => {}
        }
    }
    let smd = node
        .child("attr")
        .map(|a| a.has_flag("smd"))
        .unwrap_or_else(|| pads.iter().all(|p| p.kind != PadKind::Tht));
    let package = Package {
        uuid: crate::library::make::uuid("package", &name),
        name: name.clone(),
        description: node.child_arg("descr").unwrap_or("").to_string(),
        pads,
        courtyard: crt_rings,
        silk,
        fab,
        model,
        body: None::<Body>,
        smd,
    };
    PlacedFootprint {
        package,
        at,
        rotation,
        side,
        reference,
        value,
        locked: node.has_flag("locked") || node.child("locked").is_some(),
        pad_nets,
        properties,
        path: node.child_arg("path").map(|s| s.to_string()),
    }
}

fn stroke(w: Coord) -> SExpr {
    SExpr::list(
        "stroke",
        vec![
            SExpr::list("width", vec![len(w)]),
            SExpr::list("type", vec![SExpr::sym("solid")]),
        ],
    )
}

pub(crate) fn graphic_node(
    g: &Graphic,
    layer: &str,
    local: &dyn Fn(Point) -> Point,
    seed: &str,
) -> SExpr {
    let l = SExpr::list("layer", vec![SExpr::str(layer)]);
    let fill = |f: bool| SExpr::list("fill", vec![SExpr::sym(if f { "solid" } else { "none" })]);
    match g {
        Graphic::Line { points, width } => SExpr::list(
            "fp_line",
            vec![
                xy("start", local(points[0])),
                xy("end", local(*points.last().unwrap())),
                stroke(*width),
                l,
                uuid_node(seed),
            ],
        ),
        Graphic::Rect {
            a,
            b,
            width,
            fill: f,
        } => SExpr::list(
            "fp_rect",
            vec![
                xy("start", local(*a)),
                xy("end", local(*b)),
                stroke(*width),
                fill(*f),
                l,
                uuid_node(seed),
            ],
        ),
        Graphic::Circle {
            center,
            radius,
            width,
            fill: f,
        } => SExpr::list(
            "fp_circle",
            vec![
                xy("center", local(*center)),
                xy("end", local(*center + Point::new(*radius, 0))),
                stroke(*width),
                fill(*f),
                l,
                uuid_node(seed),
            ],
        ),
        Graphic::Arc {
            start,
            mid,
            end,
            width,
        } => SExpr::list(
            "fp_arc",
            vec![
                xy("start", local(*start)),
                xy("mid", local(*mid)),
                xy("end", local(*end)),
                stroke(*width),
                l,
                uuid_node(seed),
            ],
        ),
        Graphic::Polygon {
            points,
            width,
            fill: f,
        } => SExpr::list(
            "fp_poly",
            vec![
                pts_node(&points.iter().map(|p| local(*p)).collect::<Vec<_>>()),
                stroke(*width),
                fill(*f),
                l,
                uuid_node(seed),
            ],
        ),
        Graphic::Text {
            at,
            text,
            size,
            angle,
        } => SExpr::list(
            "fp_text",
            vec![
                SExpr::sym("user"),
                SExpr::str(text),
                SExpr::list(
                    "at",
                    vec![len(local(*at).x), len(-local(*at).y), SExpr::num(*angle)],
                ),
                l,
                uuid_node(seed),
                SExpr::list(
                    "effects",
                    vec![SExpr::list(
                        "font",
                        vec![
                            SExpr::list("size", vec![len(*size), len(*size)]),
                            SExpr::list("thickness", vec![len(size / 7)]),
                        ],
                    )],
                ),
            ],
        ),
    }
}

/// Pad node in library or board convention.
pub fn pad_node(
    p: &PadDef,
    side: Side,
    fp_rot_k: f64,
    in_board: bool,
    net: Option<(i64, &str)>,
    seed: &str,
) -> SExpr {
    let back = in_board && side == Side::Bottom;
    let local = |q: Point| if back { Point::new(q.x, -q.y) } else { q };
    let kind = match p.kind {
        PadKind::Smd => "smd",
        PadKind::Tht => "thru_hole",
        PadKind::Npth => "np_thru_hole",
        PadKind::Connector => "connect",
    };
    let (w, h) = p.shape.size();
    let (shape, extra): (&str, Vec<SExpr>) = match &p.shape {
        PadShape::Circle { .. } => ("circle", vec![]),
        PadShape::Rect { .. } => ("rect", vec![]),
        PadShape::Oval { .. } => ("oval", vec![]),
        PadShape::RoundRect { ratio, .. } => (
            "roundrect",
            vec![SExpr::list("roundrect_rratio", vec![SExpr::num(*ratio)])],
        ),
        PadShape::Trapezoid { delta, .. } => (
            "trapezoid",
            vec![SExpr::list(
                "rect_delta",
                vec![len(*delta), SExpr::num(0.0)],
            )],
        ),
        PadShape::Chamfered {
            chamfer, corners, ..
        } => {
            let mut c = Vec::new();
            for (name, bit) in [
                ("top_left", 1u8),
                ("top_right", 2),
                ("bottom_left", 4),
                ("bottom_right", 8),
            ] {
                if corners & bit != 0 {
                    c.push(SExpr::sym(name));
                }
            }
            (
                "rect",
                vec![
                    SExpr::list(
                        "chamfer_ratio",
                        vec![SExpr::num(*chamfer as f64 / w.min(h).max(1) as f64)],
                    ),
                    SExpr::list("chamfer", c),
                ],
            )
        }
        PadShape::Custom { points, holes } => (
            "custom",
            vec![
                SExpr::list(
                    "options",
                    vec![
                        SExpr::list("clearance", vec![SExpr::sym("outline")]),
                        SExpr::list("anchor", vec![SExpr::sym("circle")]),
                    ],
                ),
                SExpr::list(
                    "primitives",
                    vec![SExpr::list(
                        "gr_poly",
                        vec![
                            pts_node(&keyhole(points, holes)),
                            SExpr::list("width", vec![SExpr::num(0.0)]),
                            SExpr::list("fill", vec![SExpr::sym("yes")]),
                        ],
                    )],
                ),
            ],
        ),
    };
    let rot = if back { -p.rotation } else { p.rotation };
    let ang = if in_board {
        crate::geom::normalize_deg(rot + fp_rot_k)
    } else {
        rot
    };
    let at = local(p.at);
    let mut at_items = vec![len(at.x), len(-at.y)];
    if ang != 0.0 {
        at_items.push(SExpr::num(ang));
    }
    let (f, b) = if back { ("B", "F") } else { ("F", "B") };
    let layers: Vec<SExpr> = match (p.kind, p.copper) {
        (PadKind::Tht, _) | (_, CopperSpan::All) => vec![SExpr::str("*.Cu"), SExpr::str("*.Mask")],
        (PadKind::Npth, _) => vec![SExpr::str("*.Cu"), SExpr::str("*.Mask")],
        (_, CopperSpan::Back) => {
            let mut v = vec![SExpr::str(&format!("{b}.Cu"))];
            if p.mask {
                v.push(SExpr::str(&format!("{b}.Mask")));
            }
            v
        }
        _ => {
            let mut v = vec![SExpr::str(&format!("{f}.Cu"))];
            if p.paste {
                v.push(SExpr::str(&format!("{f}.Paste")));
            }
            if p.mask {
                v.push(SExpr::str(&format!("{f}.Mask")));
            }
            v
        }
    };
    let mut items = vec![
        SExpr::str(&p.name),
        SExpr::sym(kind),
        SExpr::sym(shape),
        SExpr::list("at", at_items),
        // A custom pad's anchor is part of its shape: keep it a 10 µm dot at the origin.
        if matches!(p.shape, PadShape::Custom { .. }) {
            SExpr::list("size", vec![len(10_000), len(10_000)])
        } else {
            SExpr::list("size", vec![len(w), len(h)])
        },
    ];
    if let Some(d) = &p.drill {
        items.push(match d.slot {
            Some(s) => SExpr::list("drill", vec![SExpr::sym("oval"), len(d.diameter), len(s)]),
            None => SExpr::list("drill", vec![len(d.diameter)]),
        });
    }
    items.push(SExpr::list("layers", layers));
    items.extend(extra);
    if let Some(m) = p.mask_margin {
        items.push(SExpr::list("solder_mask_margin", vec![len(m)]));
    }
    if let Some(m) = p.paste_margin {
        items.push(SExpr::list("solder_paste_margin", vec![len(m)]));
    }
    if let Some((code, name)) = net {
        items.push(SExpr::list(
            "net",
            vec![SExpr::sym(&code.to_string()), SExpr::str(name)],
        ));
    }
    items.push(uuid_node(seed));
    SExpr::list("pad", items)
}

/// A placed footprint and a pad-name → `(net code, net name)` lookup.
pub type Placement<'a> = (
    &'a crate::layout::Footprint,
    &'a dyn Fn(&str) -> Option<(i64, String)>,
);

/// Writes a footprint node. For a library file pass `placement = None`.
pub fn write(pkg: &Package, placement: Option<Placement<'_>>) -> SExpr {
    let in_board = placement.is_some();
    let (side, rot_k, at, reference, value) = match &placement {
        Some((f, _)) => {
            let rk = if f.side == Side::Bottom {
                crate::geom::normalize_deg(f.rotation - 180.0)
            } else {
                f.rotation
            };
            (f.side, rk, f.at, f.reference.clone(), f.value.clone())
        }
        None => (
            Side::Top,
            0.0,
            Point::ORIGIN,
            "REF**".into(),
            pkg.name.clone(),
        ),
    };
    let back = in_board && side == Side::Bottom;
    let local = move |q: Point| if back { Point::new(q.x, -q.y) } else { q };
    let flip_layer = |l: &str| -> String {
        if back {
            l.replacen("F.", "B.", 1)
        } else {
            l.to_string()
        }
    };
    let mut items = vec![SExpr::str(&pkg.name)];
    items.push(SExpr::list(
        "layer",
        vec![SExpr::str(if back { "B.Cu" } else { "F.Cu" })],
    ));
    if in_board {
        let mut a = vec![len(at.x), len(-at.y)];
        if rot_k != 0.0 {
            a.push(SExpr::num(rot_k));
        }
        items.push(SExpr::list("at", a));
    }
    if !pkg.description.is_empty() {
        items.push(SExpr::list("descr", vec![SExpr::str(&pkg.description)]));
    }
    let prop = |k: &str, v: &str, layer: &str, y: f64| {
        SExpr::list(
            "property",
            vec![
                SExpr::str(k),
                SExpr::str(v),
                SExpr::list("at", vec![SExpr::num(0.0), SExpr::num(y)]),
                SExpr::list("layer", vec![SExpr::str(layer)]),
                SExpr::list(
                    "effects",
                    vec![SExpr::list(
                        "font",
                        vec![
                            SExpr::list("size", vec![SExpr::num(1.0), SExpr::num(1.0)]),
                            SExpr::list("thickness", vec![SExpr::num(0.15)]),
                        ],
                    )],
                ),
            ],
        )
    };
    items.push(prop("Reference", &reference, &flip_layer("F.SilkS"), -1.5));
    items.push(prop("Value", &value, &flip_layer("F.Fab"), 1.5));
    if let Some((f, _)) = &placement {
        for (k, v) in &f.properties {
            items.push(prop(k, v, &flip_layer("F.Fab"), 0.0));
        }
    }
    items.push(SExpr::list(
        "attr",
        vec![SExpr::sym(if pkg.smd { "smd" } else { "through_hole" })],
    ));
    for (i, g) in pkg.silk.iter().enumerate() {
        items.push(graphic_node(
            g,
            &flip_layer("F.SilkS"),
            &local,
            &format!("{}silk{i}{reference}", pkg.name),
        ));
    }
    for (i, g) in pkg.fab.iter().enumerate() {
        items.push(graphic_node(
            g,
            &flip_layer("F.Fab"),
            &local,
            &format!("{}fab{i}{reference}", pkg.name),
        ));
    }
    for (i, r) in pkg.courtyard.iter().enumerate() {
        let g = Graphic::Polygon {
            points: r.clone(),
            width: 50_000,
            fill: false,
        };
        items.push(graphic_node(
            &g,
            &flip_layer("F.CrtYd"),
            &local,
            &format!("{}crt{i}{reference}", pkg.name),
        ));
    }
    for p in &pkg.pads {
        let net = placement.as_ref().and_then(|(_, f)| f(&p.name));
        items.push(pad_node(
            p,
            side,
            rot_k,
            in_board,
            net.as_ref().map(|(c, n)| (*c, n.as_str())),
            &format!("{}pad{}{reference}", pkg.name, p.name),
        ));
    }
    if let Some(m) = &pkg.model {
        let v3 = |h: &str, v: (f64, f64, f64)| {
            SExpr::list(
                h,
                vec![SExpr::list(
                    "xyz",
                    vec![SExpr::num(v.0), SExpr::num(v.1), SExpr::num(v.2)],
                )],
            )
        };
        items.push(SExpr::list(
            "model",
            vec![
                SExpr::str(&m.path),
                v3("offset", m.offset),
                v3("scale", m.scale),
                v3("rotate", m.rotation),
            ],
        ));
    }
    SExpr::list("footprint", items)
}

/// Reads a `.kicad_mod` file.
pub fn read_mod(text: &str) -> Result<Package, String> {
    let e = super::sexpr::parse(text).map_err(|e| e.to_string())?;
    if e.head() != Some("footprint") && e.head() != Some("module") {
        return Err("not a footprint file".into());
    }
    Ok(read(&e, false).package)
}

/// Writes a `.kicad_mod` file (KiCad 8 format).
pub fn write_mod(pkg: &Package) -> String {
    let mut e = write(pkg, None);
    if let Some(v) = e.items_mut() {
        v.insert(2, SExpr::list("version", vec![SExpr::sym("20240108")]));
        v.insert(3, SExpr::list("generator", vec![SExpr::str("runmat")]));
    }
    super::sexpr::write(&e)
}
