//! `footprint <name> = { … }`: the text form of a [`Package`], in a library `.circuit` or a
//! design's own file. A generated package prints to the same form.
//!
//! ```text
//! footprint gps_module_lcc18 = {
//!     description = "18-pad LCC GPS module, 10.1 x 9.7 mm",
//!     pads = [
//!         { number = "1", shape = rect, size = [0.8, 1.5], at = [-4.05, -3.81], layers = smd },
//!         { number = "19", shape = circle, size = [1.6, 1.6], at = [0, 0], drill = 1.0, layers = through }
//!     ],
//!     courtyard = rect([-5.3, -5.1], [5.3, 5.1]),
//!     silk = [line([-5.05, -4.85], [5.05, -4.85]), circle([-5.6, -4.6], 0.15)],
//!     fab = [rect([-5.05, -4.85], [5.05, 4.85])],
//!     body = { height = 2.4, model = "gps_module.step" }
//! }
//! ```
//!
//! Lengths are mm. A pad's `layers` is `smd` (top copper, paste and mask), `through` (every
//! copper layer and mask, with a plated `drill`), `hole` (an unplated mechanical hole) or
//! `edge` (edge-connector fingers: copper and mask, no paste). Drawings take `line`, `rect`,
//! `circle`, `arc`, `polygon` and `text`. The courtyard is closed: a rectangle, a polygon, a
//! circle, or lines and arcs that join end to end.

use super::printer::{list, record, PrintError};
use super::values::{angle, boolean, len, mm_text, num, point, pt_text, text, val, Fields, Val, V};
use super::*;
use crate::geom::poly::{rect_ring, Ring};
use crate::geom::{mm, Coord, Point, Rect};
use crate::library::make::uuid;
use crate::library::{Body, CopperSpan, Drill, Graphic, Model3d, Package, PadDef, PadKind, PadShape};

/// Silkscreen line width when a drawing gives none, mm.
pub const SILK_WIDTH: f64 = 0.12;
/// Fabrication drawing line width when a drawing gives none, mm.
pub const FAB_WIDTH: f64 = 0.10;
/// The corner ratio of a `roundrect` pad that gives none.
pub const ROUND_RATIO: f64 = 0.25;

/// A written value whose lists may hold drawing calls (`rect([0, 0], [1, 1])`).
pub(crate) enum Drawn {
    Call { name: String, args: Vec<Val>, named: Vec<(String, Val)>, span: Span },
    List(Vec<Drawn>),
    Plain(Val),
}

pub(crate) fn drawn(expr: &Expr) -> Read<Drawn> {
    match expr {
        Expr::Call(call) => {
            let span = expr.syntax().span();
            let name = match call.callee() {
                Some(Expr::Reference(r)) => r.segments().join("."),
                _ => return Err(diagnostic(span, "a drawing is `line`, `rect`, `circle`, `arc`, `polygon` or `text`")),
            };
            let mut args = Vec::new();
            let mut named = Vec::new();
            for a in call.arguments() {
                match a {
                    ast::Argument::Positional(e) => args.push(val(&e)?),
                    ast::Argument::Named(n) => {
                        let key = n.name().map(|n| n.text()).unwrap_or_default();
                        let value = n.value().ok_or_else(|| diagnostic(span, format!("`{key}` needs a value")))?;
                        named.push((key, val(&value)?));
                    }
                }
            }
            Ok(Drawn::Call { name, args, named, span })
        }
        Expr::List(l) => Ok(Drawn::List(l.items().iter().map(drawn).collect::<Read<_>>()?)),
        other => Ok(Drawn::Plain(val(other)?)),
    }
}

/// The drawing calls a field holds: one call or a list of them.
fn calls(d: Drawn) -> Read<Vec<(String, Vec<Val>, Vec<(String, Val)>, Span)>> {
    match d {
        Drawn::Call { name, args, named, span } => Ok(vec![(name, args, named, span)]),
        Drawn::List(items) => {
            let mut out = Vec::new();
            for item in items {
                out.extend(calls(item)?);
            }
            Ok(out)
        }
        Drawn::Plain(v) => Err(diagnostic(v.span, "expected a drawing: `line`, `rect`, `circle`, `arc`, `polygon` or `text`")),
    }
}

/// One drawing call as a [`Graphic`], with `width` defaulting to `default_width` mm.
pub(crate) fn graphic(name: &str, args: &[Val], named: &[(String, Val)], span: Span, default_width: f64) -> Read<Graphic> {
    let opt = |key: &str| named.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    for (k, v) in named {
        if !["width", "fill", "size", "angle"].contains(&k.as_str()) {
            return Err(diagnostic(v.span, format!("`{name}` has no `{k}`; it takes width, fill, size and angle")));
        }
    }
    let width = match opt("width") {
        Some(v) => len(v)?,
        None => mm(default_width),
    };
    let fill = opt("fill").map(boolean).transpose()?.unwrap_or(false);
    let arity = |n: usize, form: &str| -> Read<()> {
        if args.len() == n {
            Ok(())
        } else {
            Err(diagnostic(span, format!("`{name}` is `{form}`")))
        }
    };
    Ok(match name {
        "line" => {
            if args.len() < 2 {
                return Err(diagnostic(span, "`line` is `line([x, y], [x, y], …)`"));
            }
            Graphic::Line { points: args.iter().map(point).collect::<Read<_>>()?, width }
        }
        "rect" => {
            arity(2, "rect([x, y], [x, y])")?;
            Graphic::Rect { a: point(&args[0])?, b: point(&args[1])?, width, fill }
        }
        "circle" => {
            arity(2, "circle([x, y], radius)")?;
            Graphic::Circle { center: point(&args[0])?, radius: len(&args[1])?, width, fill }
        }
        "arc" => {
            arity(3, "arc([start], [mid], [end])")?;
            Graphic::Arc { start: point(&args[0])?, mid: point(&args[1])?, end: point(&args[2])?, width }
        }
        "polygon" => {
            arity(1, "polygon([[x, y], …])")?;
            Graphic::Polygon { points: super::values::ring(&args[0])?, width, fill }
        }
        "text" => {
            arity(2, "text([x, y], \"words\", size = 1)")?;
            Graphic::Text {
                at: point(&args[0])?,
                text: text(&args[1])?,
                size: opt("size").map(len).transpose()?.unwrap_or(mm(1.0)),
                angle: opt("angle").map(angle).transpose()?.unwrap_or(0.0),
            }
        }
        other => return Err(diagnostic(span, format!("`{other}` is not a drawing; use line, rect, circle, arc, polygon or text"))),
    })
}

/// A drawing call as written, `width` left out when it is `default_width` mm.
pub(crate) fn graphic_text(g: &Graphic, default_width: f64) -> String {
    let mut named = Vec::new();
    let width = |w: Coord, named: &mut Vec<String>| {
        if w != mm(default_width) {
            named.push(format!("width = {}", mm_text(w)));
        }
    };
    let fill = |f: bool, named: &mut Vec<String>| {
        if f {
            named.push("fill = true".into());
        }
    };
    let (name, args) = match g {
        Graphic::Line { points, width: w } => {
            width(*w, &mut named);
            ("line", points.iter().map(|p| pt_text(*p)).collect::<Vec<_>>())
        }
        Graphic::Rect { a, b, width: w, fill: f } => {
            width(*w, &mut named);
            fill(*f, &mut named);
            ("rect", vec![pt_text(*a), pt_text(*b)])
        }
        Graphic::Circle { center, radius, width: w, fill: f } => {
            width(*w, &mut named);
            fill(*f, &mut named);
            ("circle", vec![pt_text(*center), mm_text(*radius)])
        }
        Graphic::Arc { start, mid, end, width: w } => {
            width(*w, &mut named);
            ("arc", vec![pt_text(*start), pt_text(*mid), pt_text(*end)])
        }
        Graphic::Polygon { points, width: w, fill: f } => {
            width(*w, &mut named);
            fill(*f, &mut named);
            ("polygon", vec![super::values::ring_text(points)])
        }
        Graphic::Text { at, text, size, angle } => {
            if *size != mm(1.0) {
                named.push(format!("size = {}", mm_text(*size)));
            }
            if *angle != 0.0 {
                named.push(format!("angle = {}", format_f64(*angle)));
            }
            ("text", vec![pt_text(*at), quote(text)])
        }
    };
    let mut all = args;
    all.extend(named);
    format!("{name}({})", all.join(", "))
}

/// A list of drawings, one per line when long.
pub(crate) fn graphics_text(gs: &[Graphic], default_width: f64) -> String {
    let items: Vec<String> = gs.iter().map(|g| graphic_text(g, default_width)).collect();
    let inline = list(&items);
    if inline.len() <= 84 {
        return inline;
    }
    format!("[\n{INDENT}{}\n]", items.join(&format!(",\n{INDENT}")))
}

/// Pad layer words: the pad kind and its copper span.
const LAYERS: &[(&str, PadKind)] = &[("smd", PadKind::Smd), ("through", PadKind::Tht), ("hole", PadKind::Npth), ("edge", PadKind::Connector)];

const CORNERS: &[(&str, u8)] = &[("top_left", 1), ("top_right", 2), ("bottom_left", 4), ("bottom_right", 8)];

fn size_of(v: &Val) -> Read<(Coord, Coord)> {
    match &v.v {
        V::List(items) if items.len() == 2 => Ok((len(&items[0])?, len(&items[1])?)),
        V::Num(_) | V::Qty(..) => {
            let d = len(v)?;
            Ok((d, d))
        }
        _ => Err(diagnostic(v.span, "a size is `[width, height]`")),
    }
}

fn pad_of(v: &Val, footprint: &str) -> Read<PadDef> {
    let mut f = Fields::of(v, "pad")?;
    let number = f.opt("number", text)?.unwrap_or_default();
    let layers = f.opt("layers", text)?.unwrap_or_else(|| "smd".into());
    let kind = LAYERS
        .iter()
        .find(|(w, _)| *w == layers)
        .map(|(_, k)| *k)
        .ok_or_else(|| diagnostic(v.span, format!("footprint `{footprint}`: pad `{number}` has layers `{layers}`; a pad is `smd`, `through`, `hole` or `edge`")))?;
    let shape_word = f.opt("shape", text)?.unwrap_or_else(|| "rect".into());
    let size = f.opt("size", size_of)?;
    let need_size = |what: &str| size.ok_or_else(|| diagnostic(v.span, format!("footprint `{footprint}`: pad `{number}` is {what} and needs `size = [w, h]`")));
    let shape = match shape_word.as_str() {
        "rect" => {
            let (w, h) = need_size("a rect")?;
            PadShape::Rect { w, h }
        }
        "roundrect" => {
            let (w, h) = need_size("a roundrect")?;
            PadShape::RoundRect { w, h, ratio: f.or("ratio", ROUND_RATIO, num)? }
        }
        "circle" => PadShape::Circle { d: need_size("a circle")?.0 },
        "oval" => {
            let (w, h) = need_size("an oval")?;
            PadShape::Oval { w, h }
        }
        "trapezoid" => {
            let (w, h) = need_size("a trapezoid")?;
            PadShape::Trapezoid { w, h, delta: f.need("delta").and_then(len)? }
        }
        "chamfered" => {
            let (w, h) = need_size("a chamfered rect")?;
            let mut corners = 0u8;
            for c in super::values::texts(f.need("corners")?)? {
                corners |= CORNERS
                    .iter()
                    .find(|(w, _)| *w == c)
                    .map(|(_, b)| *b)
                    .ok_or_else(|| diagnostic(v.span, format!("`{c}` is not a corner; use top_left, top_right, bottom_left or bottom_right")))?;
            }
            PadShape::Chamfered { w, h, chamfer: f.need("chamfer").and_then(len)?, corners }
        }
        "custom" => PadShape::Custom {
            points: super::values::ring(f.need("polygon")?)?,
            holes: f.opt("holes", super::values::rings)?.unwrap_or_default(),
        },
        other => {
            return Err(diagnostic(v.span, format!("footprint `{footprint}`: pad `{number}` has shape `{other}`; a pad is rect, roundrect, circle, oval, trapezoid, chamfered or custom")))
        }
    };
    let at = f.or("at", Point::ORIGIN, point)?;
    let drill = match f.get("drill") {
        None => None,
        Some(d) => Some(match &d.v {
            V::Rec(_) => {
                let mut df = Fields::of(d, "drill")?;
                let drill = Drill { diameter: len(df.need("diameter")?)?, slot: df.opt("slot", len)?, plated: kind == PadKind::Tht };
                df.finish()?;
                drill
            }
            _ => Drill { diameter: len(d)?, slot: None, plated: kind == PadKind::Tht },
        }),
    };
    match kind {
        PadKind::Tht | PadKind::Npth if drill.is_none() => {
            return Err(diagnostic(v.span, format!("footprint `{footprint}`: pad `{number}` goes through the board and needs a `drill`")))
        }
        PadKind::Smd | PadKind::Connector if drill.is_some() => {
            return Err(diagnostic(v.span, format!("footprint `{footprint}`: pad `{number}` is `{layers}` and takes no `drill`; use `layers = through`")))
        }
        _ => {}
    }
    let copper = match (kind, f.opt("side", text)?.as_deref()) {
        (PadKind::Tht | PadKind::Npth, _) => CopperSpan::All,
        (_, Some("back")) => CopperSpan::Back,
        (_, Some("front") | None) => CopperSpan::Front,
        (_, Some(other)) => return Err(diagnostic(v.span, format!("`side = {other}`: a pad's side is `front` or `back`"))),
    };
    let pad = PadDef {
        name: number,
        kind,
        shape,
        at,
        rotation: f.or("rotation", 0.0, angle)?,
        drill,
        copper,
        mask: f.or("mask", true, boolean)?,
        paste: f.or("paste", kind == PadKind::Smd, boolean)?,
        mask_margin: f.opt("mask_expand", len)?,
        paste_margin: f.opt("paste_expand", len)?,
        thermal_pad: f.or("thermal", false, boolean)?,
    };
    f.finish()?;
    Ok(pad)
}

fn pad_text(p: &PadDef) -> Result<String, PrintError> {
    let mut r: Vec<(String, String)> = vec![("number".into(), quote(&p.name))];
    let mut put = |k: &str, v: String| r.push((k.into(), v));
    let size = |w: Coord, h: Coord| format!("[{}, {}]", mm_text(w), mm_text(h));
    match &p.shape {
        PadShape::Rect { w, h } => {
            put("shape", "rect".into());
            put("size", size(*w, *h));
        }
        PadShape::RoundRect { w, h, ratio } => {
            put("shape", "roundrect".into());
            put("size", size(*w, *h));
            if *ratio != ROUND_RATIO {
                put("ratio", format_f64(*ratio));
            }
        }
        PadShape::Circle { d } => {
            put("shape", "circle".into());
            put("size", size(*d, *d));
        }
        PadShape::Oval { w, h } => {
            put("shape", "oval".into());
            put("size", size(*w, *h));
        }
        PadShape::Trapezoid { w, h, delta } => {
            put("shape", "trapezoid".into());
            put("size", size(*w, *h));
            put("delta", mm_text(*delta));
        }
        PadShape::Chamfered { w, h, chamfer, corners } => {
            put("shape", "chamfered".into());
            put("size", size(*w, *h));
            put("chamfer", mm_text(*chamfer));
            let words: Vec<String> = CORNERS.iter().filter(|(_, b)| corners & b != 0).map(|(w, _)| w.to_string()).collect();
            put("corners", list(&words));
        }
        PadShape::Custom { points, holes } => {
            put("shape", "custom".into());
            put("polygon", super::values::ring_text(points));
            if !holes.is_empty() {
                put("holes", super::values::rings_text(holes));
            }
        }
    }
    put("at", pt_text(p.at));
    if p.rotation != 0.0 {
        put("rotation", format_f64(p.rotation));
    }
    let layers = LAYERS.iter().find(|(_, k)| *k == p.kind).map(|(w, _)| *w).unwrap_or("smd");
    put("layers", layers.into());
    if let Some(d) = &p.drill {
        match d.slot {
            None => put("drill", mm_text(d.diameter)),
            Some(s) => put("drill", format!("{{ diameter = {}, slot = {} }}", mm_text(d.diameter), mm_text(s))),
        }
    }
    if p.copper == CopperSpan::Back && matches!(p.kind, PadKind::Smd | PadKind::Connector) {
        put("side", "back".into());
    }
    if !p.mask {
        put("mask", "false".into());
    }
    if p.paste != (p.kind == PadKind::Smd) {
        put("paste", p.paste.to_string());
    }
    if let Some(m) = p.mask_margin {
        put("mask_expand", mm_text(m));
    }
    if let Some(m) = p.paste_margin {
        put("paste_expand", mm_text(m));
    }
    if p.thermal_pad {
        put("thermal", "true".into());
    }
    let inner: Vec<String> = r.iter().map(|(k, v)| format!("{k} = {v}")).collect();
    Ok(format!("{{ {} }}", inner.join(", ")))
}

/// Joins courtyard lines and arcs end to end into closed rings; `None` names an open end.
fn chain(segments: Vec<Vec<Point>>) -> Result<Vec<Ring>, Point> {
    let mut left = segments;
    let mut rings = Vec::new();
    while let Some(first) = left.pop() {
        let mut ring = first;
        loop {
            let (start, end) = (ring[0], *ring.last().unwrap());
            if ring.len() > 2 && start == end {
                ring.pop();
                rings.push(ring);
                break;
            }
            let Some(i) = left.iter().position(|s| s[0] == end || *s.last().unwrap() == end) else {
                return Err(end);
            };
            let mut next = left.swap_remove(i);
            if next[0] != end {
                next.reverse();
            }
            ring.extend(next.into_iter().skip(1));
        }
    }
    Ok(rings)
}

/// Points along an arc through `start`, `mid` and `end`, for a courtyard edge.
fn arc_points(start: Point, mid: Point, end: Point) -> Vec<Point> {
    let (ax, ay) = start.to_f();
    let (bx, by) = mid.to_f();
    let (cx, cy) = end.to_f();
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-9 {
        return vec![start, end];
    }
    let ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
    let r = ((ax - ux).powi(2) + (ay - uy).powi(2)).sqrt();
    let a0 = (ay - uy).atan2(ax - ux);
    let a1 = (by - uy).atan2(bx - ux);
    let a2 = (cy - uy).atan2(cx - ux);
    let tau = std::f64::consts::TAU;
    let ccw = |from: f64, to: f64| (to - from).rem_euclid(tau);
    let sweep = if ccw(a0, a1) <= ccw(a0, a2) { ccw(a0, a2) } else { -(tau - ccw(a0, a2)) };
    let n = 16;
    let mut out = vec![start];
    for i in 1..n {
        let a = a0 + sweep * i as f64 / n as f64;
        out.push(Point::from_f(ux + r * a.cos(), uy + r * a.sin()));
    }
    out.push(end);
    out
}

fn courtyard_of(d: Drawn, footprint: &str) -> Read<Vec<Ring>> {
    let mut rings = Vec::new();
    let mut segments = Vec::new();
    let mut at = Span::default();
    for (name, args, named, span) in calls(d)? {
        at = span;
        match graphic(&name, &args, &named, span, 0.0)? {
            Graphic::Rect { a, b, .. } => rings.push(rect_ring(Rect::new(a, b))),
            Graphic::Polygon { points, .. } => rings.push(points),
            Graphic::Circle { center, radius, .. } => rings.push(
                (0..32)
                    .map(|i| {
                        let a = std::f64::consts::TAU * i as f64 / 32.0;
                        center + Point::from_f(radius as f64 * a.cos(), radius as f64 * a.sin())
                    })
                    .collect(),
            ),
            Graphic::Line { points, .. } => segments.push(points),
            Graphic::Arc { start, mid, end, .. } => segments.push(arc_points(start, mid, end)),
            Graphic::Text { .. } => return Err(diagnostic(span, format!("footprint `{footprint}`: a courtyard is an outline, not text"))),
        }
    }
    let joined = chain(segments).map_err(|p| {
        diagnostic(at, format!("footprint `{footprint}`: the courtyard is not closed; its outline ends at [{}, {}]", mm_text(p.x), mm_text(p.y)))
    })?;
    rings.extend(joined);
    Ok(rings)
}

fn courtyard_text(rings: &[Ring]) -> String {
    let one = |r: &Ring| {
        if r.len() == 4 {
            let b = Rect::from_points(r.iter().copied());
            if rect_ring(b) == *r {
                return format!("rect({}, {})", pt_text(b.min), pt_text(b.max));
            }
        }
        format!("polygon({})", super::values::ring_text(r))
    };
    match rings {
        [r] => one(r),
        rs => list(&rs.iter().map(one).collect::<Vec<_>>()),
    }
}

/// The body outline a footprint's `body` takes when it gives no `size`: the fab drawing's
/// extent, else the courtyard's, else the pads'.
fn derived_size(fab: &[Graphic], courtyard: &[Ring], pads: &[PadDef]) -> (Coord, Coord) {
    let mut pts: Vec<Point> = Vec::new();
    for g in fab {
        match g {
            Graphic::Line { points, .. } | Graphic::Polygon { points, .. } => pts.extend(points),
            Graphic::Rect { a, b, .. } => pts.extend([*a, *b]),
            Graphic::Circle { center, radius, .. } => pts.extend([*center - Point::new(*radius, *radius), *center + Point::new(*radius, *radius)]),
            Graphic::Arc { start, mid, end, .. } => pts.extend([*start, *mid, *end]),
            Graphic::Text { .. } => {}
        }
    }
    if pts.is_empty() {
        pts.extend(courtyard.iter().flatten());
    }
    if pts.is_empty() {
        for p in pads {
            let (w, h) = p.shape.size();
            pts.extend([p.at - Point::new(w / 2, h / 2), p.at + Point::new(w / 2, h / 2)]);
        }
    }
    if pts.is_empty() {
        return (0, 0);
    }
    let r = Rect::from_points(pts);
    (r.width(), r.height())
}

fn triple(v: &Val) -> Read<(f64, f64, f64)> {
    match &v.v {
        V::List(items) if items.len() == 3 => Ok((num(&items[0])?, num(&items[1])?, num(&items[2])?)),
        _ => Err(diagnostic(v.span, "expected `[x, y, z]`")),
    }
}

fn triple_text(t: (f64, f64, f64)) -> String {
    format!("[{}, {}, {}]", format_f64(t.0), format_f64(t.1), format_f64(t.2))
}

/// Reads `footprint <name> = { … }`.
pub(super) fn read(name: String, expr: Option<Expr>, span: Span) -> Read<Package> {
    let mut description = String::new();
    let mut pads = Vec::new();
    let mut courtyard = Vec::new();
    let mut silk = Vec::new();
    let mut fab = Vec::new();
    let mut body_val: Option<Val> = None;
    for (field, value) in record_of(expr, span)? {
        match field.as_str() {
            "description" => description = text(&val(&value)?)?,
            "pads" => {
                let v = val(&value)?;
                pads = super::values::list(&v)?.iter().map(|p| pad_of(p, &name)).collect::<Read<_>>()?;
            }
            "courtyard" => courtyard = courtyard_of(drawn(&value)?, &name)?,
            "silk" => silk = layer(drawn(&value)?, SILK_WIDTH)?,
            "fab" => fab = layer(drawn(&value)?, FAB_WIDTH)?,
            "body" => body_val = Some(val(&value)?),
            other => {
                return Err(diagnostic(
                    value.span(),
                    format!("a footprint has no `{other}`; it takes description, pads, courtyard, silk, fab and body"),
                ))
            }
        }
    }
    if pads.is_empty() {
        return Err(diagnostic(span, format!("footprint `{name}` needs its `pads`")));
    }
    let (mut body, mut model) = (None, None);
    if let Some(v) = body_val {
        let mut f = Fields::of(&v, "body")?;
        if let Some(height) = f.opt("height", len)? {
            let (width, length) = match f.opt("size", size_of)? {
                Some(s) => s,
                None => derived_size(&fab, &courtyard, &pads),
            };
            body = Some(Body { width, length, height });
        } else if f.get("size").is_some() {
            return Err(diagnostic(v.span, format!("footprint `{name}`: a body with a `size` needs its `height`")));
        }
        if let Some(path) = f.opt("model", text)? {
            model = Some(Model3d {
                path,
                offset: f.or("model_offset", (0.0, 0.0, 0.0), triple)?,
                rotation: f.or("model_rotation", (0.0, 0.0, 0.0), triple)?,
                scale: f.or("model_scale", (1.0, 1.0, 1.0), triple)?,
                origin: f.or("model_origin", "footprint".to_string(), text)?,
            });
        }
        f.finish()?;
    }
    let smd = pads.iter().all(|p| matches!(p.kind, PadKind::Smd | PadKind::Connector));
    Ok(Package {
        uuid: uuid("package", &name),
        name,
        description,
        pads,
        courtyard,
        silk,
        fab,
        model,
        body,
        smd,
    })
}

fn layer(d: Drawn, width: f64) -> Read<Vec<Graphic>> {
    calls(d)?.into_iter().map(|(n, a, named, s)| graphic(&n, &a, &named, s, width)).collect()
}

/// The value written after `footprint <name> = `, which [`read`] reads back to `package`
/// (named `name`).
pub fn print(package: &Package) -> Result<String, PrintError> {
    let mut entries: Vec<(String, String)> = Vec::new();
    if !package.description.is_empty() {
        entries.push(("description".into(), quote(&package.description)));
    }
    if package.pads.is_empty() {
        return Err(PrintError(format!("footprint `{}` has no pads", package.name)));
    }
    let pads = package.pads.iter().map(pad_text).collect::<Result<Vec<_>, _>>()?;
    entries.push(("pads".into(), format!("[\n{INDENT}{}\n]", pads.join(&format!(",\n{INDENT}")))));
    if !package.courtyard.is_empty() {
        entries.push(("courtyard".into(), courtyard_text(&package.courtyard)));
    }
    if !package.silk.is_empty() {
        entries.push(("silk".into(), graphics_text(&package.silk, SILK_WIDTH)));
    }
    if !package.fab.is_empty() {
        entries.push(("fab".into(), graphics_text(&package.fab, FAB_WIDTH)));
    }
    let mut body: Vec<(String, String)> = Vec::new();
    if let Some(b) = &package.body {
        if (b.width, b.length) != derived_size(&package.fab, &package.courtyard, &package.pads) {
            body.push(("size".into(), format!("[{}, {}]", mm_text(b.width), mm_text(b.length))));
        }
        body.push(("height".into(), mm_text(b.height)));
    }
    if let Some(m) = &package.model {
        body.push(("model".into(), quote(&m.path)));
        if m.offset != (0.0, 0.0, 0.0) {
            body.push(("model_offset".into(), triple_text(m.offset)));
        }
        if m.rotation != (0.0, 0.0, 0.0) {
            body.push(("model_rotation".into(), triple_text(m.rotation)));
        }
        if m.scale != (1.0, 1.0, 1.0) {
            body.push(("model_scale".into(), triple_text(m.scale)));
        }
        if m.origin != "footprint" {
            body.push(("model_origin".into(), super::values::name_text(&m.origin)));
        }
    }
    if !body.is_empty() {
        entries.push(("body".into(), record(&body)));
    }
    Ok(record(&entries))
}

/// The whole statement, `footprint <name> = { … }`.
pub fn statement(name: &str, package: &Package) -> Result<String, PrintError> {
    Ok(format!("{} {} = {}", words::FOOTPRINT, super::printer::name(name), print(package)?))
}
