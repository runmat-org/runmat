//! A single-stroke (monoline) font for legend and copper text in fabrication outputs.
//!
//! Original glyph design, on a grid four units wide and six tall (baseline 0, cap height 6,
//! descenders to −2), drawn as polylines; a one-point polyline is a dot. Lowercase letters are
//! drawn as small capitals at two thirds of the cap height. Characters outside the set are
//! drawn as an empty box, so missing glyphs are visible.
//!
//! [`strokes`] lays out a string: `size` is the cap height, the advance is `size × 0.9`, lines
//! advance by `size × 1.6`, and the block is centred on the anchor (KiCad's default
//! justification), then mirrored (bottom-side text) and rotated counter-clockwise.

use crate::geom::{Coord, Point};

/// Glyph table: polylines of grid points, two characters per point: `0`–`9` for 0…9, `a` for
/// −1 and `b` for −2. Polylines are separated by spaces.
fn glyph(c: char) -> Option<&'static str> {
    Some(match c {
        'A' => "002640 1333",
        'B' => "00063645443303 33424130 3000",
        'C' => "4536160501103041",
        'D' => "00062644422000",
        'E' => "46060040 0333",
        'F' => "460600 0333",
        'G' => "45361605011030414323",
        'H' => "0006 4046 0343",
        'I' => "1636 2620 1030",
        'J' => "4641301001",
        'K' => "0006 4602 1340",
        'L' => "060040",
        'M' => "0006234640",
        'N' => "00064046",
        'O' => "100105163645413010",
        'P' => "00063645443303",
        'Q' => "100105163645413010 2240",
        'R' => "00063645443303 2340",
        'S' => "453616050413334241301001",
        'T' => "0646 2620",
        'U' => "060110304146",
        'V' => "062046",
        'W' => "0610243046",
        'X' => "0046 0640",
        'Y' => "062346 2320",
        'Z' => "06460040",
        '0' => "100105163645413010 0145",
        '1' => "152620 1030",
        '2' => "05163645440040",
        '3' => "05163645443323 3342413010 1001",
        '4' => "30360242",
        '5' => "460603334241301001",
        '6' => "36160501103041423303",
        '7' => "064610",
        '8' => "130405163645443313020110304142 4233",
        '9' => "43130405163645413010",
        '.' => "20",
        ',' => "21201a",
        ':' => "21 24",
        ';' => "24 21201a",
        '-' => "1333",
        '+' => "0343 2125",
        '_' => "0a4a",
        '/' => "0046",
        '\\' => "0640",
        '(' => "36141230",
        ')' => "16343210",
        '[' => "36161030",
        ']' => "16363010",
        '{' => "36252413222130",
        '}' => "16252433222110",
        '<' => "460340",
        '>' => "064300",
        '=' => "0444 0242",
        '!' => "2622 20",
        '?' => "05163645442322 20",
        '"' => "1614 3634",
        '\'' => "2624",
        '`' => "1624",
        '#' => "1016 3036 0242 0444",
        '$' => "453616050413334241301001 272a",
        '%' => "0046 0616151406 3141403031",
        '&' => "4014152635340201102042",
        '*' => "2125 0244 0442",
        '@' => "3234241322324345361605011040",
        '^' => "142634",
        '~' => "03143243",
        '|' => "2a27",
        '°' => "1626251516",
        'µ' => "0b04 01103041 4440",
        'Ω' => "001011030516364543313040",
        '±' => "0444 2226 0040",
        ' ' => "",
        _ => return None,
    })
}

fn coord(b: u8) -> i64 {
    match b {
        b'a' => -1,
        b'b' => -2,
        d => (d - b'0') as i64,
    }
}

/// Polylines of one glyph in grid units (y up), and whether it was found.
fn glyph_lines(c: char) -> (Vec<Vec<(f64, f64)>>, bool) {
    let (key, small) = if c.is_lowercase() {
        (c.to_uppercase().next().unwrap_or(c), true)
    } else {
        (c, false)
    };
    let sy = if small { 4.0 / 6.0 } else { 1.0 };
    match glyph(key).or_else(|| glyph(c)) {
        Some(g) => (
            g.split(' ')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.as_bytes()
                        .chunks(2)
                        .filter(|p| p.len() == 2)
                        .map(|p| (coord(p[0]) as f64, coord(p[1]) as f64 * sy))
                        .collect()
                })
                .collect(),
            true,
        ),
        None => (
            vec![vec![
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 6.0),
                (0.0, 6.0),
                (0.0, 0.0),
            ]],
            false,
        ),
    }
}

/// Whether every character of `text` has a glyph.
pub fn covers(text: &str) -> bool {
    text.chars().all(|c| c == '\n' || glyph_lines(c).1)
}

/// The text as polylines in board coordinates (see the module docs for the layout).
pub fn strokes(text: &str, at: Point, size: Coord, angle: f64, mirrored: bool) -> Vec<Vec<Point>> {
    let u = size as f64 / 6.0;
    let advance = 5.4; // grid units: 4 wide plus spacing (0.9 × size)
    let line = 9.6; // 1.6 × size
    let lines: Vec<&str> = text.split('\n').collect();
    let height = 6.0 + line * (lines.len() as f64 - 1.0);
    let (s, c) = angle.to_radians().sin_cos();
    let mut out = Vec::new();
    for (li, l) in lines.iter().enumerate() {
        let n = l.chars().count() as f64;
        let width = if n > 0.0 { advance * n - 1.4 } else { 0.0 };
        let x0 = -width / 2.0;
        let y0 = height / 2.0 - 6.0 - line * li as f64;
        for (k, ch) in l.chars().enumerate() {
            for pl in glyph_lines(ch).0 {
                let pts: Vec<Point> = pl
                    .iter()
                    .map(|&(gx, gy)| {
                        let mut x = (x0 + advance * k as f64 + gx) * u;
                        let y = (y0 + gy) * u;
                        if mirrored {
                            x = -x;
                        }
                        Point::from_f(at.x as f64 + x * c - y * s, at.y as f64 + x * s + y * c)
                    })
                    .collect();
                out.push(pts);
            }
        }
    }
    out
}

/// The default stroke width for text of cap height `size` (KiCad uses 0.15 mm for 1 mm).
pub fn default_thickness(size: Coord) -> Coord {
    (size * 3 / 20).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::mm;

    #[test]
    fn printable_ascii_is_covered() {
        for b in 0x20u8..0x7f {
            let c = b as char;
            assert!(covers(&c.to_string()), "{c:?} has no glyph");
            for pl in glyph_lines(c).0 {
                for (x, y) in pl {
                    assert!(
                        (-0.0..=4.0).contains(&x) && (-2.0..=7.0).contains(&y),
                        "{c:?}"
                    );
                }
            }
        }
        assert!(covers("µΩ±°"));
        assert!(!covers("☃"));
    }

    #[test]
    fn layout_centres_rotates_and_mirrors() {
        // Cells are centred on the anchor (full-width glyphs fill their cells).
        let s = strokes("HM", Point::ORIGIN, mm(1.0), 0.0, false);
        let all: Vec<Point> = s.iter().flatten().copied().collect();
        let r = crate::geom::Rect::from_points(all.iter().copied());
        assert!(
            (r.center().x).abs() <= 2 && (r.center().y).abs() <= 2,
            "{r:?}"
        );
        assert!((r.height() - mm(1.0)).abs() <= 2);
        let m = strokes("HM", Point::ORIGIN, mm(1.0), 0.0, true);
        assert_eq!(m[0][0].x, -s[0][0].x);
        let rot = strokes("HM", Point::ORIGIN, mm(1.0), 90.0, false);
        let rr = crate::geom::Rect::from_points(rot.iter().flatten().copied());
        assert!((rr.width() - r.height()).abs() <= 2 && (rr.height() - r.width()).abs() <= 2);
        // Small capitals for lowercase.
        let lo = strokes("r", Point::ORIGIN, mm(1.0), 0.0, false);
        let lr = crate::geom::Rect::from_points(lo.iter().flatten().copied());
        assert!(lr.height() < mm(0.7));
    }
}
