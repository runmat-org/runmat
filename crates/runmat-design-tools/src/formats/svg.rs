//! A [`Sheet`] as one SVG 1.1 file at true size in millimetres (`width="420mm"`, a viewBox in
//! millimetres): the entities the PDF, DXF and DWG writers write, so the four files carry one
//! drawing. The writer follows the PDF writer's rules ([`super::pdf`]):
//!
//! - Lines, polylines, circles and arcs stroke in black. Visible edges are 0.5 mm, everything
//!   else 0.25 mm, and each layer's line type draws as its dash pattern.
//! - Hatches: `SOLID` fills; a patterned hatch draws 45° lines 2.5 mm apart, clipped to its
//!   loops (even-odd).
//! - Blocks draw where they are inserted, and dimensions draw from their anonymous blocks.
//! - Text sets in a monospace font at the sheet's height, from its left baseline.
//!
//! The sheet's y axis points up and SVG's points down, so every point is written as
//! `(x, paper height − y)` and angles turn the other way.

use super::sheet::{Block, Entity, Shape, Sheet};
use std::fmt::Write as _;

/// One SVG file of the sheet.
pub fn write(sheet: &Sheet) -> String {
    let [w, h] = sheet.paper;
    let mut svg = Svg {
        out: String::new(),
        sheet,
        height: h,
        clips: 0,
    };
    let _ = writeln!(
        svg.out,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"{w}mm\" height=\"{h}mm\" viewBox=\"0 0 {w} {h}\">\n<rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>\n<g fill=\"none\" stroke=\"black\" stroke-linecap=\"round\" stroke-linejoin=\"round\">"
    );
    for e in &sheet.entities {
        svg.entity(e, &Transform::IDENTITY, 0);
    }
    svg.out.push_str("</g>\n</svg>\n");
    svg.out
}

#[derive(Clone, Copy)]
struct Transform {
    at: [f64; 2],
    scale: f64,
    rotation: f64,
}

impl Transform {
    const IDENTITY: Transform = Transform {
        at: [0.0, 0.0],
        scale: 1.0,
        rotation: 0.0,
    };
    fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.rotation.sin_cos();
        [
            self.at[0] + self.scale * (c * p[0] - s * p[1]),
            self.at[1] + self.scale * (s * p[0] + c * p[1]),
        ]
    }
    fn then(&self, inner: &Transform) -> Transform {
        Transform {
            at: self.apply(inner.at),
            scale: self.scale * inner.scale,
            rotation: self.rotation + inner.rotation,
        }
    }
}

struct Svg<'a> {
    out: String,
    sheet: &'a Sheet,
    height: f64,
    clips: usize,
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

impl Svg<'_> {
    fn block(&self, name: &str) -> Option<&Block> {
        self.sheet.blocks.iter().find(|b| b.name == name)
    }

    /// A sheet point in SVG space.
    fn at(&self, p: [f64; 2]) -> String {
        format!("{:.4},{:.4}", p[0], self.height - p[1])
    }

    /// The layer's stroke attributes: width and dash pattern (mm).
    fn stroke(&self, layer: &str) -> String {
        let width = if layer == "VISIBLE" || layer == "OUTLINE" {
            0.5
        } else {
            0.25
        };
        let linetype = self
            .sheet
            .layers
            .iter()
            .find(|l| l.name == layer)
            .map(|l| l.linetype.as_str())
            .unwrap_or("CONTINUOUS");
        let dashes: Vec<String> = self
            .sheet
            .linetypes
            .iter()
            .find(|t| t.name == linetype)
            .map(|t| {
                t.pattern
                    .iter()
                    .map(|v| format!("{:.3}", v.abs().max(0.2)))
                    .collect()
            })
            .unwrap_or_default();
        if dashes.is_empty() {
            format!("stroke-width=\"{width}\"")
        } else {
            format!(
                "stroke-width=\"{width}\" stroke-dasharray=\"{}\"",
                dashes.join(" ")
            )
        }
    }

    fn polyline(&mut self, layer: &str, pts: &[[f64; 2]], closed: bool) {
        let points = pts
            .iter()
            .map(|p| self.at(*p))
            .collect::<Vec<_>>()
            .join(" ");
        let tag = if closed { "polygon" } else { "polyline" };
        let stroke = self.stroke(layer);
        let _ = writeln!(self.out, "<{tag} points=\"{points}\" {stroke}/>");
    }

    fn entity(&mut self, e: &Entity, t: &Transform, depth: usize) {
        let p = |q: [f64; 2]| t.apply(q);
        match &e.shape {
            Shape::Line { a, b } => self.polyline(&e.layer, &[p(*a), p(*b)], false),
            Shape::Polyline { points, closed } => {
                let pts: Vec<[f64; 2]> = points.iter().map(|q| p(*q)).collect();
                self.polyline(&e.layer, &pts, *closed);
            }
            Shape::Circle { center, radius } => {
                let c = p(*center);
                let stroke = self.stroke(&e.layer);
                let _ = writeln!(
                    self.out,
                    "<circle cx=\"{:.4}\" cy=\"{:.4}\" r=\"{:.4}\" {stroke}/>",
                    c[0],
                    self.height - c[1],
                    radius * t.scale
                );
            }
            Shape::Arc {
                center,
                radius,
                start,
                end,
            } => {
                let mut sweep = end - start;
                if sweep <= 0.0 {
                    sweep += std::f64::consts::TAU;
                }
                let (c, r, a0) = (p(*center), radius * t.scale, start + t.rotation);
                let from = [c[0] + r * a0.cos(), c[1] + r * a0.sin()];
                let to = [c[0] + r * (a0 + sweep).cos(), c[1] + r * (a0 + sweep).sin()];
                let large = if sweep > std::f64::consts::PI { 1 } else { 0 };
                // Counter-clockwise on the sheet is clockwise in SVG's flipped y: sweep flag 0.
                let stroke = self.stroke(&e.layer);
                let _ = writeln!(
                    self.out,
                    "<path d=\"M{} A{r:.4},{r:.4} 0 {large} 0 {}\" {stroke}/>",
                    self.at(from),
                    self.at(to)
                );
            }
            Shape::Text {
                at,
                height,
                rotation,
                text,
            } => self.text(p(*at), height * t.scale, rotation + t.rotation, text),
            Shape::MText {
                at, height, text, ..
            } => {
                // The insertion point is the top left; paragraphs run down 1.5 heights apart.
                let h = height * t.scale;
                for (i, line) in text.split("\\P").flat_map(|l| l.split('\n')).enumerate() {
                    let q = p(*at);
                    self.text([q[0], q[1] - h - i as f64 * h * 1.5], h, t.rotation, line);
                }
            }
            Shape::Hatch { loops, pattern, .. } => {
                let pts: Vec<Vec<[f64; 2]>> = loops
                    .iter()
                    .map(|l| l.iter().map(|q| p(*q)).collect())
                    .collect();
                self.hatch(&pts, pattern == "SOLID");
            }
            Shape::Insert {
                block,
                at,
                scale,
                rotation,
            } if depth < 8 => {
                let Some(b) = self.block(block).cloned() else {
                    return;
                };
                let inner = Transform {
                    at: [at[0] - b.base[0] * scale, at[1] - b.base[1] * scale],
                    scale: *scale,
                    rotation: *rotation,
                };
                let t = t.then(&inner);
                for e in &b.entities {
                    self.entity(e, &t, depth + 1);
                }
            }
            Shape::Dimension(d) if depth < 8 => {
                let Some(b) = self.block(&d.block).cloned() else {
                    return;
                };
                for e in &b.entities {
                    self.entity(e, t, depth + 1);
                }
            }
            _ => {}
        }
    }

    /// A solid fill, or 45° lines 2.5 mm apart clipped to the loops (even-odd).
    fn hatch(&mut self, loops: &[Vec<[f64; 2]>], solid: bool) {
        let d: String = loops
            .iter()
            .filter(|l| !l.is_empty())
            .map(|l| {
                format!(
                    "M{}Z",
                    l.iter().map(|q| self.at(*q)).collect::<Vec<_>>().join(" L")
                )
            })
            .collect();
        if solid {
            let _ = writeln!(
                self.out,
                "<path d=\"{d}\" fill=\"black\" fill-rule=\"evenodd\" stroke=\"none\"/>"
            );
            return;
        }
        self.clips += 1;
        let id = format!("h{}", self.clips);
        let _ = writeln!(
            self.out,
            "<clipPath id=\"{id}\"><path d=\"{d}\" clip-rule=\"evenodd\"/></clipPath>"
        );
        let pts: Vec<[f64; 2]> = loops.iter().flatten().copied().collect();
        let lo = pts
            .iter()
            .fold([f64::MAX; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]);
        let hi = pts
            .iter()
            .fold([f64::MIN; 2], |m, p| [m[0].max(p[0]), m[1].max(p[1])]);
        let span = (hi[0] - lo[0]) + (hi[1] - lo[1]);
        let spacing = 2.5 * std::f64::consts::SQRT_2;
        let mut lines = String::new();
        let mut x = lo[0] - (hi[1] - lo[1]);
        while x <= hi[0] {
            let _ = write!(
                lines,
                "M{} L{}",
                self.at([x, lo[1]]),
                self.at([x + span, lo[1] + span])
            );
            x += spacing;
        }
        let _ = writeln!(
            self.out,
            "<path d=\"{lines}\" stroke-width=\"0.18\" clip-path=\"url(#{id})\"/>"
        );
    }

    /// Text at its left baseline `at`, turned by `rotation` (radians, counter-clockwise).
    fn text(&mut self, at: [f64; 2], h: f64, rotation: f64, text: &str) {
        if text.is_empty() {
            return;
        }
        let (x, y) = (at[0], self.height - at[1]);
        let turn = if rotation.abs() > 1e-9 {
            format!(
                " transform=\"rotate({:.4} {x:.4} {y:.4})\"",
                -rotation.to_degrees()
            )
        } else {
            String::new()
        };
        let _ = writeln!(
            self.out,
            "<text x=\"{x:.4}\" y=\"{y:.4}\" font-family=\"monospace\" font-size=\"{h:.4}\" fill=\"black\" stroke=\"none\"{turn}>{}</text>",
            escape(text)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sheet_writes_at_true_size_with_y_up() {
        let mut sheet = Sheet {
            paper: [420.0, 297.0],
            ..Default::default()
        };
        sheet.line("VISIBLE", [10.0, 10.0], [110.0, 10.0]);
        sheet.add(
            "NOTES",
            Shape::Text {
                at: [20.0, 30.0],
                height: 3.5,
                rotation: 0.0,
                text: "A < B".into(),
            },
        );
        let svg = write(&sheet);
        assert!(
            svg.contains("width=\"420mm\" height=\"297mm\" viewBox=\"0 0 420 297\""),
            "{svg}"
        );
        assert!(
            svg.contains("points=\"10.0000,287.0000 110.0000,287.0000\" stroke-width=\"0.5\""),
            "{svg}"
        );
        assert!(svg.contains(">A &lt; B</text>"), "{svg}");
    }
}
