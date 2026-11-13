//! Entities (specification 20.4): the common entity data and handles, and LINE, ARC, CIRCLE,
//! LWPOLYLINE, TEXT, MTEXT, DIMENSION (linear, radius, diameter), HATCH, INSERT, BLOCK and
//! ENDBLK, written from and read back into [`Shape`]s.

use super::bits::BitReader;
use super::stream::{ObjectIn, ObjectOut};
use crate::formats::sheet::{Dimension, DimensionKind, Shape};

type R<T> = Result<T, String>;

pub const TEXT: u16 = 0x01;
pub const INSERT: u16 = 0x07;
pub const BLOCK: u16 = 0x04;
pub const ENDBLK: u16 = 0x05;
pub const ARC: u16 = 0x11;
pub const CIRCLE: u16 = 0x12;
pub const LINE: u16 = 0x13;
pub const DIM_LINEAR: u16 = 0x15;
pub const DIM_RADIUS: u16 = 0x19;
pub const DIM_DIAMETER: u16 = 0x1A;
pub const MTEXT: u16 = 0x2C;
/// Class numbers (AcDb:Classes): LWPOLYLINE, HATCH, LAYOUT.
pub const LWPOLYLINE: u16 = 500;
pub const HATCH: u16 = 501;
pub const LAYOUT: u16 = 502;

/// The handles an entity needs from the database.
pub struct Refs<'a> {
    /// The owning block header for entities inside a block (`None` in model space).
    pub owner: Option<u64>,
    pub layer: u64,
    pub style: u64,
    pub dimstyle: u64,
    pub appid: u64,
    pub block: &'a dyn Fn(&str) -> u64,
}

/// Common entity data (20.4.1, R2018): no graphic, entity mode, no reactors or extension
/// dictionary, color, line type, plot style and material by layer, visible, line weight by
/// layer. The handles: the owner (inside blocks) and the layer.
pub fn start(kind: u16, handle: u64, refs: &Refs) -> ObjectOut {
    let mut o = ObjectOut::new(kind, handle);
    let d = &mut o.data;
    d.b(false);
    d.bb(if refs.owner.is_some() { 0 } else { 2 });
    d.bl(0);
    d.b(true);
    d.b(false);
    d.b(true);
    d.bs(256);
    d.bd(1.0);
    d.bb(0);
    d.bb(0);
    d.bb(0);
    d.rc(0);
    d.b(false);
    d.b(false);
    d.b(false);
    d.bs(0);
    d.rc(29);
    if let Some(owner) = refs.owner {
        o.h(4, owner);
    }
    o.h(5, refs.layer);
    o
}

fn p3(o: &mut ObjectOut, p: [f64; 2]) {
    o.data.bd(p[0]);
    o.data.bd(p[1]);
    o.data.bd(0.0);
}

fn rd2(o: &mut ObjectOut, p: [f64; 2]) {
    o.data.rd(p[0]);
    o.data.rd(p[1]);
}

/// One entity's bytes.
pub fn write(handle: u64, shape: &Shape, refs: &Refs) -> Vec<u8> {
    let kind = match shape {
        Shape::Line { .. } => LINE,
        Shape::Arc { .. } => ARC,
        Shape::Circle { .. } => CIRCLE,
        Shape::Polyline { .. } => LWPOLYLINE,
        Shape::Text { .. } => TEXT,
        Shape::MText { .. } => MTEXT,
        Shape::Dimension(d) => match d.kind {
            DimensionKind::Linear { .. } => DIM_LINEAR,
            DimensionKind::Radius { .. } => DIM_RADIUS,
            DimensionKind::Diameter { .. } => DIM_DIAMETER,
        },
        Shape::Hatch { .. } => HATCH,
        Shape::Insert { .. } => INSERT,
    };
    let mut o = start(kind, handle, refs);
    match shape {
        Shape::Line { a, b } => {
            let d = &mut o.data;
            d.b(true);
            d.rd(a[0]);
            d.dd(b[0], a[0]);
            d.rd(a[1]);
            d.dd(b[1], a[1]);
            d.bt(0.0);
            d.be([0.0, 0.0, 1.0]);
        }
        Shape::Arc {
            center,
            radius,
            start,
            end,
        } => {
            p3(&mut o, *center);
            o.data.bd(*radius);
            o.data.bt(0.0);
            o.data.be([0.0, 0.0, 1.0]);
            o.data.bd(*start);
            o.data.bd(*end);
        }
        Shape::Circle { center, radius } => {
            p3(&mut o, *center);
            o.data.bd(*radius);
            o.data.bt(0.0);
            o.data.be([0.0, 0.0, 1.0]);
        }
        Shape::Polyline { points, closed } => {
            let d = &mut o.data;
            d.bs(if *closed { 512 } else { 0 });
            d.bl(points.len() as i32);
            let mut prev = [0.0, 0.0];
            for (i, p) in points.iter().enumerate() {
                if i == 0 {
                    d.rd(p[0]);
                    d.rd(p[1]);
                } else {
                    d.dd(p[0], prev[0]);
                    d.dd(p[1], prev[1]);
                }
                prev = *p;
            }
        }
        Shape::Text {
            at,
            height,
            rotation,
            text,
        } => {
            // Data flags: no elevation, alignment point, oblique angle, width factor or
            // justification; the rotation when it isn't zero.
            let flags = 0xF7 | if *rotation == 0.0 { 0x08 } else { 0 };
            o.data.rc(flags);
            rd2(&mut o, *at);
            o.data.be([0.0, 0.0, 1.0]);
            o.data.bt(0.0);
            if *rotation != 0.0 {
                o.data.rd(*rotation);
            }
            o.data.rd(*height);
            o.t(text);
            o.h(5, refs.style);
        }
        Shape::MText {
            at,
            height,
            width,
            text,
        } => {
            p3(&mut o, *at);
            for c in [0.0, 0.0, 1.0, 1.0, 0.0, 0.0] {
                o.data.bd(c);
            }
            o.data.bd(*width);
            o.data.bd(0.0);
            o.data.bd(*height);
            o.data.bs(1); // attachment: top left
            o.data.bs(1); // left to right
            o.data.bd(*height);
            o.data.bd(*width);
            o.t(text);
            o.h(5, refs.style);
            o.data.bs(1);
            o.data.bd(1.0);
            o.data.b(false);
            o.data.bl(0); // no background
            o.data.b(true); // not annotative: the redundant fields follow
            o.data.bs(0);
            o.data.b(true);
            o.h(5, refs.appid);
            o.data.bl(1);
            for c in [1.0, 0.0, 0.0] {
                o.data.bd(c);
            }
            p3(&mut o, *at);
            for c in [*width, 0.0, *width, *height] {
                o.data.bd(c);
            }
            o.data.bs(0); // no columns
        }
        Shape::Dimension(dim) => {
            let d = &mut o.data;
            d.rc(0);
            for c in [0.0, 0.0, 1.0] {
                d.bd(c);
            }
            d.rd(dim.text_at[0]);
            d.rd(dim.text_at[1]);
            d.bd(0.0);
            d.rc(0);
            o.strings.tu(dim.text.as_deref().unwrap_or(""));
            let d = &mut o.data;
            for c in [0.0, 0.0, 1.0, 1.0, 1.0, 0.0] {
                d.bd(c);
            }
            d.bs(5);
            d.bs(1);
            d.bd(1.0);
            d.bd(dim.measurement);
            d.b(false);
            d.b(false);
            d.b(false);
            d.rd(0.0);
            d.rd(0.0);
            match dim.kind {
                DimensionKind::Linear {
                    from,
                    to,
                    line,
                    rotation,
                } => {
                    p3(&mut o, from);
                    p3(&mut o, to);
                    p3(&mut o, line);
                    o.data.bd(0.0);
                    o.data.bd(rotation);
                }
                DimensionKind::Radius { center, on } => {
                    p3(&mut o, center);
                    p3(&mut o, on);
                    o.data.bd(0.0);
                }
                DimensionKind::Diameter { from, to } => {
                    p3(&mut o, to);
                    p3(&mut o, from);
                    o.data.bd(0.0);
                }
            }
            o.h(5, refs.dimstyle);
            o.h(5, (refs.block)(&dim.block));
        }
        Shape::Hatch {
            loops,
            pattern,
            angle,
            scale,
        } => {
            let solid = pattern == "SOLID";
            let d = &mut o.data;
            for _ in 0..2 {
                d.bl(0);
            }
            d.bd(0.0);
            d.bd(0.0);
            d.bl(0);
            d.bd(0.0);
            d.bl(0);
            o.strings.tu("");
            let d = &mut o.data;
            d.bd(0.0);
            for c in [0.0, 0.0, 1.0] {
                d.bd(c);
            }
            o.strings.tu(pattern);
            let d = &mut o.data;
            d.b(solid);
            d.b(false);
            d.bl(loops.len() as i32);
            for (i, lp) in loops.iter().enumerate() {
                d.bl(if i == 0 { 3 } else { 2 }); // a polyline path (the first external)
                d.b(false);
                d.b(true);
                d.bl(lp.len() as i32);
                for p in lp {
                    d.rd(p[0]);
                    d.rd(p[1]);
                }
                d.bl(0);
            }
            d.bs(1); // outermost
            d.bs(1); // predefined
            if !solid {
                d.bd(*angle);
                d.bd(*scale);
                d.b(false);
                // ANSI31: one family of lines at 45 degrees, 3.175 mm apart.
                d.bs(1);
                d.bd(std::f64::consts::FRAC_PI_4 + angle);
                d.bd(0.0);
                d.bd(0.0);
                let step = 3.175 * scale / std::f64::consts::SQRT_2;
                d.bd(-step);
                d.bd(step);
                d.bs(0);
            }
            d.bl(0);
        }
        Shape::Insert {
            block,
            at,
            scale,
            rotation,
        } => {
            p3(&mut o, *at);
            if *scale == 1.0 {
                o.data.bb(3);
            } else {
                o.data.bb(2);
                o.data.rd(*scale);
            }
            o.data.bd(*rotation);
            for c in [0.0, 0.0, 1.0] {
                o.data.bd(c);
            }
            o.data.b(false);
            o.h(5, (refs.block)(block));
        }
    }
    o.finish()
}

/// BLOCK (with its name) or ENDBLK, owned by its block header.
pub fn block_marker(kind: u16, handle: u64, name: Option<&str>, refs: &Refs) -> Vec<u8> {
    let mut o = start(kind, handle, refs);
    if let Some(n) = name {
        o.t(n);
    }
    o.finish()
}

/// What reading an entity gives: its layer handle, the handles of the block it names (for
/// inserts and dimensions) and its shape.
pub struct EntityIn {
    pub layer: u64,
    pub block: Option<u64>,
    pub shape: Option<Shape>,
}

fn r3(d: &mut BitReader) -> R<[f64; 2]> {
    let p = [d.bd()?, d.bd()?];
    d.bd()?;
    Ok(p)
}

/// Read an entity's common data and, for the kinds this writer writes, its shape.
pub fn read(o: &ObjectIn) -> R<EntityIn> {
    let mut d = o.data();
    let mut s = o.strings();
    let mut h = o.handles();
    if d.b()? {
        return Err("entity graphics are not read".into());
    }
    let mode = d.bb()?;
    let reactors = d.bl()?;
    let no_xdic = d.b()?;
    d.b()?;
    d.b()?;
    let color = d.bs()?;
    if color as u16 & 0x8000 != 0 {
        d.bl()?;
    }
    if color as u16 & 0x2000 != 0 {
        d.bl()?;
    }
    d.bd()?;
    let ltype = d.bb()?;
    let plot = d.bb()?;
    let material = d.bb()?;
    d.rc()?;
    let (v1, v2, v3) = (d.b()?, d.b()?, d.b()?);
    d.bs()?;
    d.rc()?;
    if mode == 0 {
        h.h()?;
    }
    for _ in 0..reactors {
        h.h()?;
    }
    if !no_xdic {
        h.h()?;
    }
    if color as u16 & 0x4000 != 0 {
        h.h()?;
    }
    let layer = h.h()?.value;
    for (flag, present) in [(ltype, true), (material, true), (plot, true)] {
        if present && flag == 3 {
            h.h()?;
        }
    }
    for v in [v1, v2, v3] {
        if v {
            h.h()?;
        }
    }
    let mut block = None;
    let shape = match o.kind {
        LINE => {
            let flat = d.b()?;
            let x0 = d.rd()?;
            let x1 = d.dd(x0)?;
            let y0 = d.rd()?;
            let y1 = d.dd(y0)?;
            if !flat {
                let z = d.rd()?;
                d.dd(z)?;
            }
            Some(Shape::Line {
                a: [x0, y0],
                b: [x1, y1],
            })
        }
        ARC => {
            let center = r3(&mut d)?;
            let radius = d.bd()?;
            d.bt()?;
            d.be()?;
            Some(Shape::Arc {
                center,
                radius,
                start: d.bd()?,
                end: d.bd()?,
            })
        }
        CIRCLE => {
            let center = r3(&mut d)?;
            Some(Shape::Circle {
                center,
                radius: d.bd()?,
            })
        }
        LWPOLYLINE => {
            let flag = d.bs()?;
            if flag & 4 != 0 {
                d.bd()?;
            }
            if flag & 8 != 0 {
                d.bd()?;
            }
            if flag & 2 != 0 {
                d.bd()?;
            }
            if flag & 1 != 0 {
                d.be()?;
            }
            let n = d.bl()?.max(0) as usize;
            let mut points: Vec<[f64; 2]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = if i == 0 {
                    [d.rd()?, d.rd()?]
                } else {
                    let q = points[i - 1];
                    [d.dd(q[0])?, d.dd(q[1])?]
                };
                points.push(p);
            }
            Some(Shape::Polyline {
                points,
                closed: flag & 512 != 0,
            })
        }
        TEXT => {
            let flags = d.rc()?;
            if flags & 1 == 0 {
                d.rd()?;
            }
            let at = [d.rd()?, d.rd()?];
            if flags & 2 == 0 {
                d.dd(at[0])?;
                d.dd(at[1])?;
            }
            d.be()?;
            d.bt()?;
            if flags & 4 == 0 {
                d.rd()?;
            }
            let rotation = if flags & 8 == 0 { d.rd()? } else { 0.0 };
            let height = d.rd()?;
            Some(Shape::Text {
                at,
                height,
                rotation,
                text: s.tu()?,
            })
        }
        MTEXT => {
            let at = r3(&mut d)?;
            for _ in 0..6 {
                d.bd()?;
            }
            let width = d.bd()?;
            d.bd()?;
            let height = d.bd()?;
            Some(Shape::MText {
                at,
                height,
                width,
                text: s.tu()?,
            })
        }
        DIM_LINEAR | DIM_RADIUS | DIM_DIAMETER => {
            d.rc()?;
            for _ in 0..3 {
                d.bd()?;
            }
            let text_at = [d.rd()?, d.rd()?];
            d.bd()?;
            d.rc()?;
            let text = Some(s.tu()?).filter(|t| !t.is_empty());
            for _ in 0..6 {
                d.bd()?;
            }
            d.bs()?;
            d.bs()?;
            d.bd()?;
            let measurement = d.bd()?;
            for _ in 0..3 {
                d.b()?;
            }
            d.rd()?;
            d.rd()?;
            let (a, b) = (r3(&mut d)?, r3(&mut d)?);
            let kind = match o.kind {
                DIM_LINEAR => {
                    let line = r3(&mut d)?;
                    d.bd()?;
                    DimensionKind::Linear {
                        from: a,
                        to: b,
                        line,
                        rotation: d.bd()?,
                    }
                }
                DIM_RADIUS => DimensionKind::Radius { center: a, on: b },
                _ => DimensionKind::Diameter { from: b, to: a },
            };
            h.h()?;
            block = Some(h.h()?.value);
            Some(Shape::Dimension(Dimension {
                kind,
                text_at,
                measurement,
                text,
                block: String::new(),
            }))
        }
        HATCH => {
            d.bl()?;
            d.bl()?;
            d.bd()?;
            d.bd()?;
            d.bl()?;
            d.bd()?;
            let colors = d.bl()?;
            for _ in 0..colors {
                d.bd()?;
                d.bs()?;
                d.bl()?;
                d.rc()?;
            }
            s.tu()?;
            d.bd()?;
            for _ in 0..3 {
                d.bd()?;
            }
            let pattern = s.tu()?;
            let solid = d.b()?;
            d.b()?;
            let paths = d.bl()?;
            let mut loops = Vec::new();
            for _ in 0..paths {
                let flag = d.bl()?;
                if flag & 2 == 0 {
                    return Err("hatch edge paths are not read".into());
                }
                let bulges = d.b()?;
                d.b()?;
                let n = d.bl()?;
                let mut lp = Vec::new();
                for _ in 0..n {
                    lp.push([d.rd()?, d.rd()?]);
                    if bulges {
                        d.bd()?;
                    }
                }
                d.bl()?;
                loops.push(lp);
            }
            d.bs()?;
            d.bs()?;
            let (angle, scale) = if solid {
                (0.0, 1.0)
            } else {
                (d.bd()?, d.bd()?)
            };
            Some(Shape::Hatch {
                loops,
                pattern,
                angle,
                scale,
            })
        }
        INSERT => {
            let at = r3(&mut d)?;
            let scale = match d.bb()? {
                3 => 1.0,
                2 => d.rd()?,
                1 => {
                    d.dd(1.0)?;
                    d.dd(1.0)?;
                    1.0
                }
                _ => {
                    let x = d.rd()?;
                    d.dd(x)?;
                    d.dd(x)?;
                    x
                }
            };
            let rotation = d.bd()?;
            block = Some(h.h()?.value);
            Some(Shape::Insert {
                block: String::new(),
                at,
                scale,
                rotation,
            })
        }
        _ => None,
    };
    Ok(EntityIn {
        layer,
        block,
        shape,
    })
}
