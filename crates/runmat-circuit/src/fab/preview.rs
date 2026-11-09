//! One Gerber layer drawn as SVG, from the same image the Gerber file writes, so a preview
//! shows the layer as fabricated: flashes, strokes with round ends and filled regions in one
//! colour (`currentColor`), on the board outline's frame. The mask layer is drawn as its
//! openings (the file's negative image). Coordinates are millimetres with y up, flipped once
//! by the root group.

use super::{image, Aperture, GerberLayer, Sink};
use crate::geom::poly::Ring;
use crate::geom::{to_mm, Point};
use crate::layout::Board;
use std::fmt::Write as _;

struct Svg {
    apertures: Vec<Aperture>,
    body: String,
}

fn n(v: f64) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".into()
    } else {
        s.to_owned()
    }
}

fn p(pt: Point) -> (String, String) {
    (n(to_mm(pt.x)), n(to_mm(pt.y)))
}

impl Sink for Svg {
    fn aperture(&mut self, a: Aperture, _function: &str) -> usize {
        if let Some(i) = self.apertures.iter().position(|x| *x == a) {
            return i;
        }
        self.apertures.push(a);
        self.apertures.len() - 1
    }
    fn object(&mut self, _attrs: &[(&str, String)]) {}
    fn flash(&mut self, ap: usize, at: Point) {
        let (x, y) = (to_mm(at.x), to_mm(at.y));
        let _ = match self.apertures[ap] {
            Aperture::Circle(d) => writeln!(
                self.body,
                r#"<circle cx="{}" cy="{}" r="{}"/>"#,
                n(x),
                n(y),
                n(to_mm(d) / 2.0)
            ),
            Aperture::Rect(w, h) => {
                let (w, h) = (to_mm(w), to_mm(h));
                writeln!(
                    self.body,
                    r#"<rect x="{}" y="{}" width="{}" height="{}"/>"#,
                    n(x - w / 2.0),
                    n(y - h / 2.0),
                    n(w),
                    n(h)
                )
            }
            Aperture::Obround(w, h) => {
                let (w, h) = (to_mm(w), to_mm(h));
                let r = w.min(h) / 2.0;
                writeln!(
                    self.body,
                    r#"<rect x="{}" y="{}" width="{}" height="{}" rx="{}"/>"#,
                    n(x - w / 2.0),
                    n(y - h / 2.0),
                    n(w),
                    n(h),
                    n(r)
                )
            }
        };
    }
    fn draw(&mut self, ap: usize, pts: &[Point]) {
        let width = match self.apertures[ap] {
            Aperture::Circle(d) => to_mm(d),
            Aperture::Rect(w, h) | Aperture::Obround(w, h) => to_mm(w.min(h)),
        };
        if pts.len() == 1 {
            return self.flash(ap, pts[0]);
        }
        let d: Vec<String> = pts
            .iter()
            .map(|&q| {
                let (x, y) = p(q);
                format!("{x},{y}")
            })
            .collect();
        let _ = writeln!(
            self.body,
            r#"<polyline points="{}" fill="none" stroke="currentColor" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"/>"#,
            d.join(" "),
            n(width)
        );
    }
    fn region(&mut self, ring: &Ring, _function: &str) {
        if ring.len() < 3 {
            return;
        }
        let mut d = String::new();
        for (i, q) in ring.iter().enumerate() {
            let (x, y) = p(*q);
            let _ = write!(d, "{}{x} {y}", if i == 0 { "M" } else { "L" });
        }
        let _ = writeln!(self.body, r#"<path d="{d}Z"/>"#);
    }
}

/// `layer` of `board` as an SVG document: the image in `currentColor`, the board outline as a
/// thin frame (`data-role="board-outline"`; no class, so a page's stylesheet doesn't restyle it), framed by the outline with a small margin.
pub fn layer_svg(board: &Board, layer: GerberLayer) -> String {
    let mut svg = Svg {
        apertures: Vec::new(),
        body: String::new(),
    };
    image(board, layer, &mut svg);
    let b = board.outline.bbox();
    let (x0, y0, x1, y1) = (
        to_mm(b.min.x),
        to_mm(b.min.y),
        to_mm(b.max.x),
        to_mm(b.max.y),
    );
    let margin = ((x1 - x0).max(y1 - y0) * 0.03).max(0.5);
    let mut outline = Svg {
        apertures: Vec::new(),
        body: String::new(),
    };
    image(board, GerberLayer::Profile, &mut outline);
    format!(
        concat!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {} {}" data-layer="{}">"#,
            r#"<g transform="scale(1,-1)">"#,
            r#"<g data-role="board-outline" fill="none" stroke="currentColor" opacity="0.45">{}</g>"#,
            r#"<g data-role="layer-image" fill="currentColor">{}</g>"#,
            "</g></svg>"
        ),
        n(x0 - margin),
        n(-(y1 + margin)),
        n(x1 - x0 + 2.0 * margin),
        n(y1 - y0 + 2.0 * margin),
        super::file_function(board, layer).0,
        outline.body,
        svg.body,
    )
}

/// A fabrication or drill drawing as SVG, drawn as its Gerber draws it (strokes, and text in
/// the stroke font) in `currentColor`, framed by the drawing's extent with a small margin. The
/// fabrication drawing's title block names the board revision.
pub fn drawing_svg(drawing: &super::drawing::Drawing) -> String {
    let mut svg = Svg {
        apertures: Vec::new(),
        body: String::new(),
    };
    drawing.emit(&mut svg);
    let b = drawing.extent();
    let (x0, y0, x1, y1) = (
        to_mm(b.min.x),
        to_mm(b.min.y),
        to_mm(b.max.x),
        to_mm(b.max.y),
    );
    let margin = ((x1 - x0).max(y1 - y0) * 0.02).max(0.5);
    format!(
        concat!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {} {}" data-layer="{}">"#,
            r#"<g transform="scale(1,-1)"><g data-role="layer-image" fill="currentColor">{}</g></g></svg>"#
        ),
        n(x0 - margin),
        n(-(y1 + margin)),
        n(x1 - x0 + 2.0 * margin),
        n(y1 - y0 + 2.0 * margin),
        drawing
            .title
            .replace('&', "&amp;")
            .replace('"', "&quot;")
            .replace('<', "&lt;")
            .replace('>', "&gt;"),
        svg.body,
    )
}
