//! Graphics: lines, arcs, circles, polygons and text, read and printed.

use super::*;
use crate::library::Graphic;
use runmat_types::Span;

/* --- graphics ------------------------------------------------------------------------------ */

pub(super) fn read_graphic(kind: &str, f: &mut Fields, span: Span) -> Read<Graphic> {
    let width = |f: &mut Fields| f.or("width", 0, len);
    let fill = |f: &mut Fields| f.or("fill", false, boolean);
    Ok(match kind {
        "line" => Graphic::Line {
            points: ring(f.need("points")?)?,
            width: width(f)?,
        },
        "rect" => Graphic::Rect {
            a: point(f.need("from")?)?,
            b: point(f.need("to")?)?,
            width: width(f)?,
            fill: fill(f)?,
        },
        "circle" => Graphic::Circle {
            center: point(f.need("center")?)?,
            radius: len(f.need("radius")?)?,
            width: width(f)?,
            fill: fill(f)?,
        },
        "arc" => Graphic::Arc {
            start: point(f.need("from")?)?,
            mid: point(f.need("mid")?)?,
            end: point(f.need("to")?)?,
            width: width(f)?,
        },
        "polygon" => Graphic::Polygon {
            points: ring(f.need("points")?)?,
            width: width(f)?,
            fill: fill(f)?,
        },
        "text" => Graphic::Text {
            at: point(f.need("at")?)?,
            text: text(f.need("text")?)?,
            size: len(f.need("size")?)?,
            angle: f.or("angle", 0.0, angle)?,
        },
        other => {
            return Err(diagnostic(
                span,
                format!(
                    "`{other}` is not a graphic; expected line, rect, circle, arc, polygon or text"
                ),
            ))
        }
    })
}

/// The graphic's kind word and its fields after `prefix` (a board graphic's layer).
pub(super) fn graphic_text(g: &Graphic, mut r: Rec) -> (&'static str, String) {
    let kind = match g {
        Graphic::Line { points, width } => {
            r.put("points", ring_text(points)).len("width", *width);
            "line"
        }
        Graphic::Rect { a, b, width, fill } => {
            r.pt("from", *a)
                .pt("to", *b)
                .len("width", *width)
                .flag("fill", *fill);
            "rect"
        }
        Graphic::Circle {
            center,
            radius,
            width,
            fill,
        } => {
            r.pt("center", *center)
                .len("radius", *radius)
                .len("width", *width)
                .flag("fill", *fill);
            "circle"
        }
        Graphic::Arc {
            start,
            mid,
            end,
            width,
        } => {
            r.pt("from", *start)
                .pt("mid", *mid)
                .pt("to", *end)
                .len("width", *width);
            "arc"
        }
        Graphic::Polygon {
            points,
            width,
            fill,
        } => {
            r.put("points", ring_text(points))
                .len("width", *width)
                .flag("fill", *fill);
            "polygon"
        }
        Graphic::Text {
            at,
            text,
            size,
            angle,
        } => {
            r.pt("at", *at)
                .put("text", name_text(text))
                .len("size", *size);
            if *angle != 0.0 {
                r.put("angle", f64_text(*angle));
            }
            "text"
        }
    };
    (kind, r.text())
}
