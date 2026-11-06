//! A title-block logo as outlines: the shapes of an SVG file (`rect`, `polygon`, `path`, with
//! their `transform`) flattened to closed loops, so the sheet, PDF, SVG, DXF and DWG all draw
//! the same lines. Fills, strokes and colors are not read: a logo prints as its outline.
//!
//! Path data takes the `M L H V C S Q T Z` commands, absolute and relative; curves flatten to
//! [`CURVE_STEPS`] segments. An arc (`A`) is drawn as a straight segment to its end point.

use serde::Serialize;

/// Segments per curve when a path's Bézier curves are flattened.
pub const CURVE_STEPS: usize = 12;

/// A logo's outlines in a box `size` wide and high (the file's units), y up, origin at the lower
/// left. Each loop is closed and lists its points as `[x, y]`.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Outline {
    pub size: [f64; 2],
    pub loops: Vec<Vec<[f64; 2]>>,
}

impl Outline {
    /// The loops scaled to fit a `w` × `h` box centered on `center`, keeping the aspect ratio.
    pub fn fitted(&self, center: [f64; 2], w: f64, h: f64) -> Vec<Vec<[f64; 2]>> {
        if self.size[0] <= 0.0 || self.size[1] <= 0.0 {
            return Vec::new();
        }
        let s = (w / self.size[0]).min(h / self.size[1]);
        let origin = [center[0] - self.size[0] * s / 2.0, center[1] - self.size[1] * s / 2.0];
        self.loops
            .iter()
            .map(|l| l.iter().map(|p| [origin[0] + p[0] * s, origin[1] + p[1] * s]).collect())
            .collect()
    }
}

type Affine = [f64; 6];
const IDENTITY: Affine = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn compose(a: Affine, b: Affine) -> Affine {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}

fn apply(m: &Affine, p: [f64; 2]) -> [f64; 2] {
    [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]]
}

fn numbers(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<f64>| {
        if let Ok(v) = cur.parse() {
            out.push(v);
        }
        cur.clear();
    };
    for c in text.chars() {
        let starts_new = (c == '-' || c == '+') && !cur.is_empty() && !cur.ends_with(['e', 'E'])
            || c == '.' && cur.contains('.') && !cur.contains(['e', 'E']);
        if starts_new {
            flush(&mut cur, &mut out);
        }
        if c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E') {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// A `transform` attribute: `matrix`, `translate`, `scale` and `rotate(a [cx cy])`.
fn transform(text: &str) -> Affine {
    let mut m = IDENTITY;
    for part in text.split(')') {
        let Some((name, args)) = part.split_once('(') else { continue };
        let v = numbers(args);
        let t = match (name.trim().trim_start_matches(',').trim(), v.as_slice()) {
            ("matrix", [a, b, c, d, e, f]) => [*a, *b, *c, *d, *e, *f],
            ("translate", [x]) => [1.0, 0.0, 0.0, 1.0, *x, 0.0],
            ("translate", [x, y]) => [1.0, 0.0, 0.0, 1.0, *x, *y],
            ("scale", [s]) => [*s, 0.0, 0.0, *s, 0.0, 0.0],
            ("scale", [x, y]) => [*x, 0.0, 0.0, *y, 0.0, 0.0],
            ("rotate", [a, rest @ ..]) => {
                let (s, c) = a.to_radians().sin_cos();
                let r = [c, s, -s, c, 0.0, 0.0];
                match rest {
                    [cx, cy] => compose(compose([1.0, 0.0, 0.0, 1.0, *cx, *cy], r), [1.0, 0.0, 0.0, 1.0, -cx, -cy]),
                    _ => r,
                }
            }
            _ => IDENTITY,
        };
        m = compose(m, t);
    }
    m
}

/// The value of attribute `name` in a tag's text.
fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let mut at = 0;
    while let Some(i) = tag[at..].find(name) {
        let start = at + i;
        let before = tag[..start].chars().last();
        let rest = tag[start + name.len()..].trim_start();
        if before.is_some_and(char::is_whitespace) {
            if let Some(rest) = rest.strip_prefix('=') {
                let rest = rest.trim_start();
                let quote = rest.chars().next()?;
                if quote == '"' || quote == '\'' {
                    let body = &rest[1..];
                    return body.find(quote).map(|end| &body[..end]);
                }
            }
        }
        at = start + name.len();
    }
    None
}

fn number(tag: &str, name: &str) -> f64 {
    attribute(tag, name).and_then(|v| numbers(v).first().copied()).unwrap_or(0.0)
}

/// A path's subpaths as point lists (each closed when drawn).
fn path_loops(d: &str) -> Vec<Vec<[f64; 2]>> {
    let mut tokens: Vec<(char, Vec<f64>)> = Vec::new();
    let mut start = None;
    for (i, c) in d.char_indices() {
        if c.is_ascii_alphabetic() && c != 'e' && c != 'E' {
            if let Some((cmd, from)) = start {
                tokens.push((cmd, numbers(&d[from..i])));
            }
            start = Some((c, i + 1));
        }
    }
    if let Some((cmd, from)) = start {
        tokens.push((cmd, numbers(&d[from..])));
    }
    let mut loops = Vec::new();
    let mut current: Vec<[f64; 2]> = Vec::new();
    let mut p = [0.0, 0.0];
    let mut first = p;
    // The last control point, for the smooth curves (S, T).
    let mut control: Option<[f64; 2]> = None;
    let finish = |current: &mut Vec<[f64; 2]>, loops: &mut Vec<Vec<[f64; 2]>>| {
        if current.len() >= 3 {
            loops.push(std::mem::take(current));
        } else {
            current.clear();
        }
    };
    for (cmd, args) in tokens {
        let rel = cmd.is_ascii_lowercase();
        let at = |p: [f64; 2], x: f64, y: f64| if rel { [p[0] + x, p[1] + y] } else { [x, y] };
        match cmd.to_ascii_uppercase() {
            'M' => {
                finish(&mut current, &mut loops);
                for (k, xy) in args.chunks_exact(2).enumerate() {
                    p = at(p, xy[0], xy[1]);
                    if k == 0 {
                        first = p;
                    }
                    current.push(p);
                }
                control = None;
            }
            'L' => {
                for xy in args.chunks_exact(2) {
                    p = at(p, xy[0], xy[1]);
                    current.push(p);
                }
                control = None;
            }
            'H' => {
                for x in &args {
                    p = [if rel { p[0] + x } else { *x }, p[1]];
                    current.push(p);
                }
                control = None;
            }
            'V' => {
                for y in &args {
                    p = [p[0], if rel { p[1] + y } else { *y }];
                    current.push(p);
                }
                control = None;
            }
            'C' | 'S' => {
                let smooth = cmd.eq_ignore_ascii_case(&'S');
                for c in args.chunks_exact(if smooth { 4 } else { 6 }) {
                    let (c1, c2, end) = if smooth {
                        let c1 = control.map(|k| [2.0 * p[0] - k[0], 2.0 * p[1] - k[1]]).unwrap_or(p);
                        (c1, at(p, c[0], c[1]), at(p, c[2], c[3]))
                    } else {
                        (at(p, c[0], c[1]), at(p, c[2], c[3]), at(p, c[4], c[5]))
                    };
                    for k in 1..=CURVE_STEPS {
                        let t = k as f64 / CURVE_STEPS as f64;
                        let u = 1.0 - t;
                        let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                        current.push([
                            w[0] * p[0] + w[1] * c1[0] + w[2] * c2[0] + w[3] * end[0],
                            w[0] * p[1] + w[1] * c1[1] + w[2] * c2[1] + w[3] * end[1],
                        ]);
                    }
                    control = Some(c2);
                    p = end;
                }
            }
            'Q' | 'T' => {
                let smooth = cmd.eq_ignore_ascii_case(&'T');
                for c in args.chunks_exact(if smooth { 2 } else { 4 }) {
                    let (q, end) = if smooth {
                        (control.map(|k| [2.0 * p[0] - k[0], 2.0 * p[1] - k[1]]).unwrap_or(p), at(p, c[0], c[1]))
                    } else {
                        (at(p, c[0], c[1]), at(p, c[2], c[3]))
                    };
                    for k in 1..=CURVE_STEPS {
                        let t = k as f64 / CURVE_STEPS as f64;
                        let u = 1.0 - t;
                        current.push([
                            u * u * p[0] + 2.0 * u * t * q[0] + t * t * end[0],
                            u * u * p[1] + 2.0 * u * t * q[1] + t * t * end[1],
                        ]);
                    }
                    control = Some(q);
                    p = end;
                }
            }
            'A' => {
                for a in args.chunks_exact(7) {
                    p = at(p, a[5], a[6]);
                    current.push(p);
                }
                control = None;
            }
            'Z' => {
                finish(&mut current, &mut loops);
                p = first;
                control = None;
            }
            _ => {}
        }
    }
    finish(&mut current, &mut loops);
    loops
}

/// Reads the outlines of an SVG file's `rect`, `polygon` and `path` shapes. Errors name what's
/// missing: a file with no `viewBox` (or `width` and `height`), or with no shape.
pub fn outline(svg: &str) -> Result<Outline, String> {
    let root = svg.find("<svg").map(|i| &svg[i..svg[i..].find('>').map_or(svg.len(), |e| i + e)]).ok_or("the file isn't an SVG (no <svg> element)")?;
    let view = attribute(root, "viewBox").map(numbers).filter(|v| v.len() == 4);
    let [x0, y0, w, h] = match view {
        Some(v) => [v[0], v[1], v[2], v[3]],
        None => [0.0, 0.0, number(root, "width"), number(root, "height")],
    };
    if w <= 0.0 || h <= 0.0 {
        return Err("the SVG has no viewBox or size".into());
    }
    // Into the box: y flips so the logo reads upright on a sheet whose y points up.
    let to_box: Affine = [1.0, 0.0, 0.0, -1.0, -x0, y0 + h];
    let mut loops = Vec::new();
    let mut rest = svg;
    while let Some(i) = rest.find('<') {
        let tail = &rest[i + 1..];
        let end = tail.find('>').unwrap_or(tail.len());
        let tag = &tail[..end];
        rest = &tail[end.min(tail.len())..];
        let name = tag.split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("");
        let m = compose(to_box, attribute(tag, "transform").map(transform).unwrap_or(IDENTITY));
        let shapes: Vec<Vec<[f64; 2]>> = match name {
            "rect" => {
                let (x, y, rw, rh) = (number(tag, "x"), number(tag, "y"), number(tag, "width"), number(tag, "height"));
                if rw <= 0.0 || rh <= 0.0 {
                    continue;
                }
                vec![vec![[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]]]
            }
            "polygon" => {
                let v = attribute(tag, "points").map(numbers).unwrap_or_default();
                vec![v.chunks_exact(2).map(|p| [p[0], p[1]]).collect()]
            }
            "path" => attribute(tag, "d").map(path_loops).unwrap_or_default(),
            _ => continue,
        };
        loops.extend(shapes.into_iter().filter(|l| l.len() >= 3).map(|l| l.into_iter().map(|p| apply(&m, p)).collect::<Vec<_>>()));
    }
    if loops.is_empty() {
        return Err("the SVG has no rect, polygon or path to outline".into());
    }
    Ok(Outline { size: [w, h], loops })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rects_rotate_about_their_center_and_flip_upright() {
        let svg = r#"<svg width="100" height="50" viewBox="0 0 100 50"><rect x="10" y="0" width="20" height="10"/><rect y="40" width="10" height="20" transform="rotate(-90 0 40)"/></svg>"#;
        let o = outline(svg).unwrap();
        assert_eq!(o.size, [100.0, 50.0]);
        assert_eq!(o.loops.len(), 2);
        // The first rect sits at the top of the file, so at the top of the box.
        assert_eq!(o.loops[0][0], [10.0, 50.0]);
        // rotate(-90 0 40) turns the 10 × 20 rect into a 20 × 10 one, 30 to 40 down the file,
        // so 10 to 20 up the box.
        let xs: Vec<f64> = o.loops[1].iter().map(|p| (p[0] * 1e9).round() / 1e9).collect();
        let ys: Vec<f64> = o.loops[1].iter().map(|p| (p[1] * 1e9).round() / 1e9).collect();
        assert_eq!(xs.iter().cloned().fold(f64::MIN, f64::max), 20.0, "{:?}", o.loops[1]);
        assert_eq!(ys.iter().cloned().fold(f64::MIN, f64::max), 20.0, "{:?}", o.loops[1]);
        assert_eq!(ys.iter().cloned().fold(f64::MAX, f64::min), 10.0, "{:?}", o.loops[1]);
    }

    #[test]
    fn a_path_reads_moves_lines_and_curves_relative_and_absolute() {
        let svg = r#"<svg viewBox="0 0 10 10"><path d="M1 1H9V9C9 9 5 10 1 9ZM2 2l2 0 0 2-2 0z"/></svg>"#;
        let o = outline(svg).unwrap();
        assert_eq!(o.loops.len(), 2);
        assert_eq!(o.loops[0].len(), 3 + CURVE_STEPS);
        assert_eq!(o.loops[1], vec![[2.0, 8.0], [4.0, 8.0], [4.0, 6.0], [2.0, 6.0]]);
        assert_eq!(numbers("1.5.5-2e-1"), vec![1.5, 0.5, -0.2]);
    }

    #[test]
    fn the_runmat_logo_outlines_its_six_tiles_and_its_letters() {
        let o = outline(include_str!("../../tests/fixtures/runmat-logo.svg")).unwrap();
        assert_eq!(o.size, [828.0, 140.0]);
        assert!(o.loops.len() >= 6 + 6, "{} loops", o.loops.len());
        let fitted = o.fitted([60.0, 10.0], 100.0, 10.0);
        let max_x = fitted.iter().flatten().map(|p| p[0]).fold(f64::MIN, f64::max);
        let min_y = fitted.iter().flatten().map(|p| p[1]).fold(f64::MAX, f64::min);
        assert!(max_x <= 110.0 + 1e-6 && min_y >= 5.0 - 1e-6, "fits the cell: {max_x} {min_y}");
        assert!(outline("<svg viewBox=\"0 0 1 1\"></svg>").is_err());
    }
}
