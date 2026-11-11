//! Packages: pads, shapes and footprint definitions, read and printed.

use super::*;
use crate::geom::{Coord, Point};
use crate::ids::Uuid;
use crate::library::{Body, CopperSpan, Drill, Model3d, Package, PadDef, PadKind, PadShape};
use runmat_design_syntax::ast::{self, Headed};
use runmat_types::Span;

/* --- packages ------------------------------------------------------------------------------ */

pub(super) fn pad_kind_name(k: PadKind) -> &'static str {
    match k {
        PadKind::Smd => "smd",
        PadKind::Tht => "tht",
        PadKind::Npth => "npth",
        PadKind::Connector => "edge",
    }
}

pub(super) fn copper_span_name(c: CopperSpan) -> &'static str {
    match c {
        CopperSpan::Front => "front",
        CopperSpan::Back => "back",
        CopperSpan::All => "all",
    }
}

/// The pad a kind starts from; a pad statement writes only what differs.
pub(super) fn pad_default(
    kind: PadKind,
    name: &str,
    at: Point,
    shape: PadShape,
    drill: Coord,
) -> PadDef {
    match kind {
        PadKind::Smd => PadDef::smd(name, at, shape),
        PadKind::Connector => PadDef {
            kind: PadKind::Connector,
            paste: false,
            ..PadDef::smd(name, at, shape)
        },
        PadKind::Tht => PadDef::tht(name, at, shape, drill),
        PadKind::Npth => PadDef {
            shape,
            ..PadDef::npth(name, at, drill)
        },
    }
}

pub(super) fn read_shape(f: &mut Fields, span: Span) -> Read<PadShape> {
    let size = |f: &mut Fields| -> Read<(Coord, Coord)> {
        let p = point(f.need("size")?)?;
        Ok((p.x, p.y))
    };
    let sv = f.need("shape")?;
    Ok(match text(sv)?.as_str() {
        "circle" => PadShape::Circle {
            d: len(f.need("d")?)?,
        },
        "rect" => {
            let (w, h) = size(f)?;
            PadShape::Rect { w, h }
        }
        "roundrect" => {
            let (w, h) = size(f)?;
            PadShape::RoundRect {
                w,
                h,
                ratio: num(f.need("ratio")?)?,
            }
        }
        "oval" => {
            let (w, h) = size(f)?;
            PadShape::Oval { w, h }
        }
        "trapezoid" => {
            let (w, h) = size(f)?;
            PadShape::Trapezoid {
                w,
                h,
                delta: len(f.need("delta")?)?,
            }
        }
        "chamfered" => {
            let (w, h) = size(f)?;
            PadShape::Chamfered {
                w,
                h,
                chamfer: len(f.need("chamfer")?)?,
                corners: u8::try_from(whole(f.need("corners")?)?)
                    .map_err(|_| diagnostic(span, "corners is a bit mask 0 to 15"))?,
            }
        }
        "custom" => PadShape::Custom {
            points: ring(f.need("points")?)?,
            holes: f.or("holes", Vec::new(), rings)?,
        },
        other => {
            return Err(diagnostic(
                sv.span,
                format!("`{other}` is not a pad shape; expected circle, rect, roundrect, oval, trapezoid, chamfered or custom"),
            ))
        }
    })
}

pub(super) fn shape_fields(s: &PadShape, r: &mut Rec) {
    let size = |r: &mut Rec, w: Coord, h: Coord| {
        r.put("size", pt_text(Point::new(w, h)));
    };
    match s {
        PadShape::Circle { d } => {
            r.put("shape", "circle").len("d", *d);
        }
        PadShape::Rect { w, h } => {
            r.put("shape", "rect");
            size(r, *w, *h);
        }
        PadShape::RoundRect { w, h, ratio } => {
            r.put("shape", "roundrect");
            size(r, *w, *h);
            r.put("ratio", f64_text(*ratio));
        }
        PadShape::Oval { w, h } => {
            r.put("shape", "oval");
            size(r, *w, *h);
        }
        PadShape::Trapezoid { w, h, delta } => {
            r.put("shape", "trapezoid");
            size(r, *w, *h);
            r.len("delta", *delta);
        }
        PadShape::Chamfered {
            w,
            h,
            chamfer,
            corners,
        } => {
            r.put("shape", "chamfered");
            size(r, *w, *h);
            r.len("chamfer", *chamfer)
                .put("corners", corners.to_string());
        }
        PadShape::Custom { points, holes } => {
            r.put("shape", "custom").put("points", ring_text(points));
            if !holes.is_empty() {
                r.put("holes", rings_text(holes));
            }
        }
    }
}

pub(super) fn read_pad(name: String, v: &Val) -> Read<PadDef> {
    let mut f = Fields::of(v, "pad")?;
    let kv = f.get("kind");
    let kind = match kv.map(text).transpose()?.as_deref() {
        None | Some("smd") => PadKind::Smd,
        Some("tht") => PadKind::Tht,
        Some("npth") => PadKind::Npth,
        Some("edge") => PadKind::Connector,
        Some(other) => {
            return Err(diagnostic(
                kv.unwrap().span,
                format!("`{other}` is not a pad kind; expected smd, tht, npth or edge"),
            ))
        }
    };
    let shape = read_shape(&mut f, v.span)?;
    let at = f.or("at", Point::ORIGIN, point)?;
    let drill = match f.get("drill") {
        None => None,
        Some(d) if matches!(&d.v, V::Word(w) if w == "none") => Some(None),
        Some(d) => Some(Some(len(d)?)),
    };
    let mut p = pad_default(kind, &name, at, shape, drill.flatten().unwrap_or(0));
    match drill {
        Some(None) => p.drill = None,
        Some(Some(d)) => {
            let plated = p.drill.as_ref().is_none_or(|x| x.plated);
            p.drill = Some(Drill {
                diameter: d,
                slot: None,
                plated,
            });
        }
        None if p.drill.is_some() => {
            return Err(diagnostic(v.span, "a through-hole pad needs `drill`"))
        }
        None => {}
    }
    if let Some(drill) = p.drill.as_mut() {
        drill.slot = f.opt("slot", len)?;
        drill.plated = f.or("plated", drill.plated, boolean)?;
    }
    p.rotation = f.or("rotation", 0.0, angle)?;
    p.copper = match f.get("copper") {
        None => p.copper,
        Some(c) => match text(c)?.as_str() {
            "front" => CopperSpan::Front,
            "back" => CopperSpan::Back,
            "all" => CopperSpan::All,
            other => {
                return Err(diagnostic(
                    c.span,
                    format!("`{other}` is not a copper span; expected front, back or all"),
                ))
            }
        },
    };
    p.mask = f.or("mask", p.mask, boolean)?;
    p.paste = f.or("paste", p.paste, boolean)?;
    p.mask_margin = f.opt("mask_margin", len)?;
    p.paste_margin = f.opt("paste_margin", len)?;
    p.thermal_pad = f.or("thermal", false, boolean)?;
    f.finish()?;
    Ok(p)
}

pub(super) fn pad_text(p: &PadDef) -> String {
    let d = pad_default(
        p.kind,
        &p.name,
        p.at,
        p.shape.clone(),
        p.drill.as_ref().map_or(0, |d| d.diameter),
    );
    let mut r = Rec::new();
    if p.kind != PadKind::Smd {
        r.put("kind", pad_kind_name(p.kind));
    }
    shape_fields(&p.shape, &mut r);
    if p.at != Point::ORIGIN {
        r.pt("at", p.at);
    }
    if p.rotation != 0.0 {
        r.put("rotation", f64_text(p.rotation));
    }
    match (&p.drill, &d.drill) {
        (None, Some(_)) => {
            r.put("drill", "none");
        }
        (None, None) => {}
        (Some(x), base) => {
            r.len("drill", x.diameter);
            if let Some(s) = x.slot {
                r.len("slot", s);
            }
            if x.plated != base.as_ref().is_none_or(|b| b.plated) {
                r.put("plated", x.plated.to_string());
            }
        }
    }
    if p.copper != d.copper {
        r.put("copper", copper_span_name(p.copper));
    }
    if p.mask != d.mask {
        r.put("mask", p.mask.to_string());
    }
    if p.paste != d.paste {
        r.put("paste", p.paste.to_string());
    }
    if let Some(m) = p.mask_margin {
        r.len("mask_margin", m);
    }
    if let Some(m) = p.paste_margin {
        r.len("paste_margin", m);
    }
    r.flag("thermal", p.thermal_pad);
    r.text()
}

pub(super) fn floats3(v: &Val) -> Read<(f64, f64, f64)> {
    match list(v)? {
        [a, b, c] => Ok((num(a)?, num(b)?, num(c)?)),
        _ => Err(diagnostic(v.span, "expected `[x, y, z]`")),
    }
}

pub(super) fn floats3_text(t: (f64, f64, f64)) -> String {
    format!("[{}, {}, {}]", f64_text(t.0), f64_text(t.1), f64_text(t.2))
}

pub(super) fn read_package(b: &ast::BlockStatement, key: &str) -> Read<Package> {
    let mut p = Package {
        uuid: Uuid::NIL,
        name: key.to_owned(),
        description: String::new(),
        pads: Vec::new(),
        courtyard: Vec::new(),
        silk: Vec::new(),
        fab: Vec::new(),
        model: None,
        body: None,
        smd: false,
    };
    for statement in b.block().map(|b| b.statements()).unwrap_or_default() {
        let Some(h) = head(&statement)? else {
            return Err(diagnostic(
                statement.span(),
                "expected a package statement such as `pad`",
            ));
        };
        let value = need_value(&h)?;
        match (h.word.as_str(), h.name.as_ref()) {
            (words::NAME, None) => p.name = text(value)?,
            (words::UUID, None) => {
                let t = text(value)?;
                p.uuid = Uuid::parse(&t)
                    .ok_or_else(|| diagnostic(value.span, format!("`{t}` is not a UUID")))?;
            }
            (words::DESCRIPTION, None) => p.description = text(value)?,
            (words::SMD, None) => p.smd = boolean(value)?,
            (words::PAD, Some((name, _))) => p.pads.push(read_pad(name.clone(), value)?),
            (words::COURTYARD, None) => p.courtyard.push(ring(value)?),
            (words::SILK | words::FAB, Some((kind, _))) => {
                let mut f = Fields::of(value, "graphic")?;
                let g = read_graphic(kind, &mut f, h.span)?;
                f.finish()?;
                if h.word == words::SILK {
                    p.silk.push(g);
                } else {
                    p.fab.push(g);
                }
            }
            (words::MODEL, None) => {
                let mut f = Fields::of(value, "3D model")?;
                p.model = Some(Model3d {
                    path: text(f.need("path")?)?,
                    offset: f.or("offset", (0.0, 0.0, 0.0), floats3)?,
                    rotation: f.or("rotation", (0.0, 0.0, 0.0), floats3)?,
                    scale: f.or("scale", (1.0, 1.0, 1.0), floats3)?,
                    origin: f.or("origin", "footprint".to_owned(), text)?,
                });
                f.finish()?;
            }
            (words::BODY, None) => {
                let mut f = Fields::of(value, "body")?;
                p.body = Some(Body {
                    width: len(f.need("width")?)?,
                    length: len(f.need("length")?)?,
                    height: len(f.need("height")?)?,
                });
                f.finish()?;
            }
            (word, _) => {
                return Err(diagnostic(
                    h.span,
                    format!("`{word}` does not belong in a package; expected name, uuid, description, smd, pad, courtyard, silk, fab, model or body"),
                ))
            }
        }
    }
    Ok(p)
}
