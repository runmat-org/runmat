//! A [`Sheet`] as a one-page vector PDF (ISO 32000-1), at true size in millimetres: the same
//! entities the DXF and DWG writers write, so the three files carry one drawing.
//!
//! - Lines, polylines, circles and arcs stroke in black. Visible edges are 0.5 mm, everything
//!   else 0.25 mm (ISO 128-20's two line widths). Each layer's line type draws as its dash
//!   pattern.
//! - Hatches: `SOLID` fills; patterned hatches (ANSI31) draw 45° lines clipped to their loops.
//! - Blocks draw where they are inserted, and dimensions draw from their anonymous blocks.
//! - Text sets in Courier, one of the standard 14 fonts every reader has, so nothing is
//!   embedded and each character is 0.6 of the text height wide, as the sheet lays text out.
//!   Drawing symbols the font lacks (⌀, ⊥, ⌖, ∥, ↧, Ⓜ and the other GD&T characteristics) draw as
//!   lines in their character cell.

use super::sheet::{Block, Entity, Shape, Sheet};
use std::fmt::Write as _;

const PT: f64 = 72.0 / 25.4;

/// One PDF file of the sheet.
pub fn write(sheet: &Sheet) -> Vec<u8> {
    write_pages(std::slice::from_ref(sheet))
}

/// One PDF file with a page per sheet, in order, each at its own paper size.
pub fn write_pages(sheets: &[Sheet]) -> Vec<u8> {
    // Objects: 1 catalog, 2 pages, 3 font, then a page and its content stream per sheet.
    let mut objects = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
        String::new(),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>".to_owned(),
    ];
    let mut kids = Vec::new();
    for sheet in sheets {
        let mut page = Page { out: String::new(), sheet };
        page.out.push_str("1 J 1 j 0 0 0 RG 0 0 0 rg\n");
        // Page space in millimetres.
        let _ = writeln!(page.out, "{PT:.6} 0 0 {PT:.6} 0 0 cm");
        for e in &sheet.entities {
            page.entity(e, &Transform::IDENTITY, 0);
        }
        let content = page.out;
        let [w, h] = sheet.paper.map(|v| v * PT);
        let (page_id, content_id) = (objects.len() + 1, objects.len() + 2);
        kids.push(format!("{page_id} 0 R"));
        objects.push(format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:.3} {h:.3}] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"));
        objects.push(format!("<< /Length {} >>\nstream\n{content}endstream", content.len()));
    }
    objects[1] = format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids.join(" "), kids.len());
    let mut pdf = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
    }
    let xref = pdf.len();
    let mut tail = format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
    for o in offsets {
        let _ = writeln!(tail, "{o:010} 00000 n ");
    }
    let _ = write!(tail, "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objects.len() + 1);
    pdf.extend_from_slice(tail.as_bytes());
    pdf
}

#[derive(Clone, Copy)]
struct Transform {
    at: [f64; 2],
    scale: f64,
    rotation: f64,
}

impl Transform {
    const IDENTITY: Transform = Transform { at: [0.0, 0.0], scale: 1.0, rotation: 0.0 };
    fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.rotation.sin_cos();
        [self.at[0] + self.scale * (c * p[0] - s * p[1]), self.at[1] + self.scale * (s * p[0] + c * p[1])]
    }
    fn then(&self, inner: &Transform) -> Transform {
        Transform { at: self.apply(inner.at), scale: self.scale * inner.scale, rotation: self.rotation + inner.rotation }
    }
}

struct Page<'a> {
    out: String,
    sheet: &'a Sheet,
}

/// The line width (mm) a symbol drawn in text of height `h` (mm) strokes with: h/14, the
/// stem of the writer's Courier, so a stroked Ω weighs as the letters beside it.
fn stroke_width(h: f64) -> f64 {
    h / 14.0
}

impl Page<'_> {
    fn block(&self, name: &str) -> Option<&Block> {
        self.sheet.blocks.iter().find(|b| b.name == name)
    }

    /// The layer's stroke: width and dash pattern (mm).
    fn stroke(&mut self, layer: &str) {
        let width = if layer == "VISIBLE" || layer == "OUTLINE" { 0.5 } else { 0.25 };
        let linetype = self.sheet.layers.iter().find(|l| l.name == layer).map(|l| l.linetype.as_str()).unwrap_or("CONTINUOUS");
        let dashes: Vec<f64> = self
            .sheet
            .linetypes
            .iter()
            .find(|t| t.name == linetype)
            .map(|t| t.pattern.iter().map(|v| v.abs().max(0.2)).collect())
            .unwrap_or_default();
        let pattern = dashes.iter().map(|d| format!("{d:.3}")).collect::<Vec<_>>().join(" ");
        let _ = writeln!(self.out, "{width} w [{pattern}] 0 d");
    }

    fn entity(&mut self, e: &Entity, t: &Transform, depth: usize) {
        let p = |q: [f64; 2]| t.apply(q);
        match &e.shape {
            Shape::Line { a, b } => {
                self.stroke(&e.layer);
                self.path(&[p(*a), p(*b)], false);
                self.out.push_str("S\n");
            }
            Shape::Polyline { points, closed } => {
                self.stroke(&e.layer);
                let pts: Vec<[f64; 2]> = points.iter().map(|q| p(*q)).collect();
                self.path(&pts, *closed);
                self.out.push_str("S\n");
            }
            Shape::Circle { center, radius } => {
                self.stroke(&e.layer);
                self.arc(p(*center), radius * t.scale, 0.0, std::f64::consts::TAU, true);
                self.out.push_str("S\n");
            }
            Shape::Arc { center, radius, start, end } => {
                self.stroke(&e.layer);
                let mut sweep = end - start;
                if sweep <= 0.0 {
                    sweep += std::f64::consts::TAU;
                }
                self.arc(p(*center), radius * t.scale, start + t.rotation, sweep, false);
                self.out.push_str("S\n");
            }
            Shape::Text { at, height, rotation, text } => self.text(p(*at), height * t.scale, rotation + t.rotation, text),
            Shape::MText { at, height, text, .. } => {
                // The insertion point is the top left; paragraphs run down 1.5 heights apart.
                let h = height * t.scale;
                for (i, line) in text.split("\\P").flat_map(|l| l.split('\n')).enumerate() {
                    let q = p(*at);
                    self.text([q[0], q[1] - h - i as f64 * h * 1.5], h, t.rotation, line);
                }
            }
            Shape::Hatch { loops, pattern, .. } => {
                let pts: Vec<Vec<[f64; 2]>> = loops.iter().map(|l| l.iter().map(|q| p(*q)).collect()).collect();
                self.hatch(&pts, pattern == "SOLID");
            }
            Shape::Insert { block, at, scale, rotation } if depth < 8 => {
                let Some(b) = self.block(block).cloned() else { return };
                let inner = Transform { at: [at[0] - b.base[0] * scale, at[1] - b.base[1] * scale], scale: *scale, rotation: *rotation };
                let t = t.then(&inner);
                for e in &b.entities {
                    self.entity(e, &t, depth + 1);
                }
            }
            Shape::Dimension(d) if depth < 8 => {
                let Some(b) = self.block(&d.block).cloned() else { return };
                for e in &b.entities {
                    self.entity(e, t, depth + 1);
                }
            }
            _ => {}
        }
    }

    fn path(&mut self, pts: &[[f64; 2]], closed: bool) {
        for (i, q) in pts.iter().enumerate() {
            let _ = writeln!(self.out, "{:.4} {:.4} {}", q[0], q[1], if i == 0 { "m" } else { "l" });
        }
        if closed {
            self.out.push_str("h\n");
        }
    }

    /// An arc from angle `a0` sweeping `sweep` (radians, counter-clockwise) as cubic Béziers of
    /// at most 90° each.
    fn arc(&mut self, c: [f64; 2], r: f64, a0: f64, sweep: f64, closed: bool) {
        let n = (sweep / std::f64::consts::FRAC_PI_2).ceil().max(1.0) as usize;
        let step = sweep / n as f64;
        let k = 4.0 / 3.0 * (step / 4.0).tan();
        let at = |a: f64| [c[0] + r * a.cos(), c[1] + r * a.sin()];
        let p0 = at(a0);
        let _ = writeln!(self.out, "{:.4} {:.4} m", p0[0], p0[1]);
        for i in 0..n {
            let (a, b) = (a0 + step * i as f64, a0 + step * (i + 1) as f64);
            let (pa, pb) = (at(a), at(b));
            let c1 = [pa[0] - k * r * a.sin(), pa[1] + k * r * a.cos()];
            let c2 = [pb[0] + k * r * b.sin(), pb[1] - k * r * b.cos()];
            let _ = writeln!(self.out, "{:.4} {:.4} {:.4} {:.4} {:.4} {:.4} c", c1[0], c1[1], c2[0], c2[1], pb[0], pb[1]);
        }
        if closed {
            self.out.push_str("h\n");
        }
    }

    /// A solid fill, or 45° lines 2.5 mm apart clipped to the loops (even-odd).
    fn hatch(&mut self, loops: &[Vec<[f64; 2]>], solid: bool) {
        self.out.push_str("q\n");
        for l in loops {
            self.path(l, true);
        }
        if solid {
            self.out.push_str("f*\nQ\n");
            return;
        }
        self.out.push_str("W* n\n0.18 w [] 0 d\n");
        let pts: Vec<[f64; 2]> = loops.iter().flatten().copied().collect();
        let lo = pts.iter().fold([f64::MAX; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]);
        let hi = pts.iter().fold([f64::MIN; 2], |m, p| [m[0].max(p[0]), m[1].max(p[1])]);
        let span = (hi[0] - lo[0]) + (hi[1] - lo[1]);
        let spacing = 2.5 * std::f64::consts::SQRT_2;
        let mut x = lo[0] - (hi[1] - lo[1]);
        while x <= hi[0] {
            let _ = writeln!(self.out, "{:.4} {:.4} m {:.4} {:.4} l S", x, lo[1], x + span, lo[1] + span);
            x += spacing;
        }
        self.out.push_str("Q\n");
    }

    /// Text at its left baseline `at`: runs of the font's characters as text, the symbols it
    /// lacks drawn in their 0.6-height cells.
    fn text(&mut self, at: [f64; 2], h: f64, rotation: f64, text: &str) {
        let (s, c) = rotation.sin_cos();
        let cell = 0.6 * h;
        let mut run = String::new();
        let mut run_start = 0usize;
        let chars: Vec<char> = text.chars().collect();
        let flush = |page: &mut Self, run: &mut String, start: usize| {
            if run.is_empty() {
                return;
            }
            let x = start as f64 * cell;
            let o = [at[0] + c * x, at[1] + s * x];
            let _ = writeln!(page.out, "BT /F1 {h:.3} Tf {c:.5} {s:.5} {:.5} {c:.5} {:.4} {:.4} Tm ({run}) Tj ET", -s, o[0], o[1]);
            run.clear();
        };
        for (i, ch) in chars.iter().enumerate() {
            match win_ansi(*ch) {
                Some(b) => {
                    if run.is_empty() {
                        run_start = i;
                    }
                    match b {
                        b'(' | b')' | b'\\' => {
                            run.push('\\');
                            run.push(b as char);
                        }
                        0x20..=0x7e => run.push(b as char),
                        _ => {
                            let _ = write!(run, "\\{b:03o}");
                        }
                    }
                }
                None => {
                    flush(self, &mut run, run_start);
                    let x = i as f64 * cell;
                    let origin = [at[0] + c * x, at[1] + s * x];
                    self.symbol(*ch, origin, h, rotation);
                }
            }
        }
        flush(self, &mut run, run_start);
    }

    /// A drawing symbol in the character cell at `origin` (left baseline), `h` the text height.
    /// Its strokes are [`stroke_width`] wide.
    fn symbol(&mut self, ch: char, origin: [f64; 2], h: f64, rotation: f64) {
        let t = Transform { at: origin, scale: h, rotation };
        // Cell units: 0.6 wide, the cap height 0.72 tall; the middle at (0.3, 0.36).
        let (m, r) = ([0.3, 0.36], 0.26);
        let mut lines: Vec<Vec<[f64; 2]>> = Vec::new();
        let mut circles: Vec<([f64; 2], f64)> = Vec::new();
        let mut letter: Option<char> = None;
        match ch {
            '⌀' => {
                circles.push((m, r));
                lines.push(vec![[0.0, 0.0], [0.6, 0.72]]);
            }
            '⊥' => lines.extend([vec![[0.3, 0.0], [0.3, 0.72]], vec![[0.0, 0.0], [0.6, 0.0]]]),
            '∥' => lines.extend([vec![[0.05, 0.0], [0.3, 0.72]], vec![[0.3, 0.0], [0.55, 0.72]]]),
            '⏥' => lines.push(vec![[0.0, 0.12], [0.42, 0.12], [0.6, 0.6], [0.18, 0.6], [0.0, 0.12]]),
            '⏤' => lines.push(vec![[0.0, 0.36], [0.6, 0.36]]),
            '⌖' => {
                circles.push((m, 0.2));
                lines.extend([vec![[0.0, 0.36], [0.6, 0.36]], vec![[0.3, 0.0], [0.3, 0.72]]]);
            }
            '○' => circles.push((m, r)),
            '◎' => circles.extend([(m, r), (m, 0.13)]),
            '⌭' => {
                circles.push((m, 0.2));
                lines.extend([vec![[0.0, 0.0], [0.25, 0.72]], vec![[0.35, 0.0], [0.6, 0.72]]]);
            }
            '⌒' => lines.push((0..=12).map(|i| { let a = std::f64::consts::PI * i as f64 / 12.0; [0.3 + 0.3 * a.cos(), 0.15 + 0.4 * a.sin()] }).collect()),
            '⌓' => {
                let mut arc: Vec<[f64; 2]> = (0..=12).map(|i| { let a = std::f64::consts::PI * i as f64 / 12.0; [0.3 + 0.3 * a.cos(), 0.15 + 0.4 * a.sin()] }).collect();
                arc.push(arc[0]);
                lines.push(arc);
            }
            '∠' => lines.push(vec![[0.6, 0.6], [0.0, 0.0], [0.6, 0.0]]),
            // Ohm (component values: `5.1 kΩ`): a horseshoe open at the bottom, on two feet.
            '\u{3a9}' | '\u{2126}' => {
                let (a0, a1) = (-0.3 * std::f64::consts::PI, 1.3 * std::f64::consts::PI);
                let mut arc: Vec<[f64; 2]> = vec![[0.05, 0.0]];
                arc.extend((0..=16).rev().map(|i| {
                    let a = a0 + (a1 - a0) * i as f64 / 16.0;
                    [0.3 + 0.24 * a.cos(), 0.4 + 0.3 * a.sin()]
                }));
                arc.push([0.55, 0.0]);
                lines.push(arc);
            }
            '⌯' => lines.extend([vec![[0.0, 0.12], [0.6, 0.12]], vec![[0.0, 0.36], [0.6, 0.36]], vec![[0.0, 0.6], [0.6, 0.6]]]),
            '↗' => lines.extend([vec![[0.05, 0.05], [0.55, 0.65]], vec![[0.3, 0.6], [0.55, 0.65], [0.5, 0.4]]]),
            '⌰' => lines.extend([vec![[0.0, 0.05], [0.4, 0.65]], vec![[0.2, 0.05], [0.6, 0.65]], vec![[0.0, 0.05], [0.6, 0.05]]]),
            '↧' => lines.extend([vec![[0.05, 0.72], [0.55, 0.72]], vec![[0.3, 0.72], [0.3, 0.0]], vec![[0.12, 0.2], [0.3, 0.0], [0.48, 0.2]]]),
            'Ⓜ' | 'Ⓛ' | 'Ⓢ' | 'Ⓕ' | 'Ⓟ' | 'Ⓣ' | 'Ⓤ' => {
                circles.push((m, 0.33));
                letter = Some(match ch {
                    'Ⓜ' => 'M',
                    'Ⓛ' => 'L',
                    'Ⓢ' => 'S',
                    'Ⓕ' => 'F',
                    'Ⓟ' => 'P',
                    'Ⓣ' => 'T',
                    _ => 'U',
                });
            }
            _ => letter = Some('?'),
        }
        // The strokes match the font's weight: h/14, Courier's stem at its height (0.25 mm at
        // a drawing's 3.5 mm text, 0.09 mm at a schematic value's 1.27 mm).
        let _ = writeln!(self.out, "{:.3} w [] 0 d", stroke_width(h));
        for l in lines {
            let pts: Vec<[f64; 2]> = l.iter().map(|q| t.apply(*q)).collect();
            self.path(&pts, false);
            self.out.push_str("S\n");
        }
        for (c, rr) in circles {
            self.arc(t.apply(c), rr * h, 0.0, std::f64::consts::TAU, true);
            self.out.push_str("S\n");
        }
        if let Some(l) = letter {
            let small = if ch == '?' { 1.0 } else { 0.55 };
            let p = t.apply([0.3 - 0.3 * small, 0.36 - 0.3 * small]);
            self.text(p, h * small, rotation, &l.to_string());
        }
    }
}

/// A character's byte in WinAnsiEncoding, or none when the font lacks it.
fn win_ansi(c: char) -> Option<u8> {
    let u = c as u32;
    match u {
        0x20..=0x7e => Some(u as u8),
        0xa0..=0xff => Some(u as u8),
        _ => match c {
            '€' => Some(0x80),
            '‚' => Some(0x82),
            '„' => Some(0x84),
            '…' => Some(0x85),
            '‘' => Some(0x91),
            '’' => Some(0x92),
            '“' => Some(0x93),
            '”' => Some(0x94),
            '•' => Some(0x95),
            '–' => Some(0x96),
            '—' => Some(0x97),
            '™' => Some(0x99),
            _ => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sheet_writes_a_well_formed_page_with_its_text_and_symbols() {
        let mut s = Sheet { paper: [420.0, 297.0], ..Sheet::default() };
        s.line("VISIBLE", [10.0, 10.0], [410.0, 10.0]);
        s.text("NOTES", [20.0, 20.0], 3.5, "⌀4.1 +0.018/0 (A)");
        let pdf = write(&s);
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.starts_with("%PDF-1.4"));
        assert!(text.ends_with("%%EOF\n"));
        // The page is A3 in points.
        assert!(text.contains("/MediaBox [0 0 1190.551 841.890]"), "{text}");
        // The diameter draws as geometry; the rest sets as text, the parentheses escaped.
        assert!(text.contains("(4.1 +0.018/0 \\(A\\)) Tj"), "{text}");
        // An ohm sign draws as strokes; the text around it stays text.
        let mut o = Sheet { paper: [420.0, 297.0], ..Sheet::default() };
        o.text("NOTES", [20.0, 20.0], 3.5, "5.1 kΩ");
        let ohm = String::from_utf8_lossy(&write(&o)).into_owned();
        assert!(ohm.contains("(5.1 k) Tj") && !ohm.contains("(?) Tj"), "{ohm}");
        // Its strokes weigh as the text: 3.5 mm text strokes at 0.25 mm, 1.27 mm text at 0.091 mm.
        assert!(ohm.contains("0.250 w [] 0 d"), "{ohm}");
        let mut small = Sheet { paper: [420.0, 297.0], ..Sheet::default() };
        small.text("NOTES", [20.0, 20.0], 1.27, "10 kΩ");
        assert!(String::from_utf8_lossy(&write(&small)).contains("0.091 w [] 0 d"));
        // The xref offsets point at their objects (bytes: the header's comment is binary).
        let find = |hay: &[u8], needle: &[u8]| hay.windows(needle.len()).rposition(|w| w == needle).unwrap();
        let after = find(&pdf, b"startxref\n") + b"startxref\n".len();
        let xref: usize = std::str::from_utf8(&pdf[after..]).unwrap().lines().next().unwrap().parse().unwrap();
        let table = std::str::from_utf8(&pdf[xref..]).unwrap();
        assert!(table.starts_with("xref"));
        for (i, line) in table.lines().skip(3).take(5).enumerate() {
            let at: usize = line[..10].parse().unwrap();
            assert!(pdf[at..].starts_with(format!("{} 0 obj", i + 1).as_bytes()), "object {}", i + 1);
        }
    }
}
