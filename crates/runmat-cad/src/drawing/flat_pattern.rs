//! Flat patterns for drawings and cutting: a [`FlatLayout`] (from a flat pattern feature's
//! record) becomes a 2D sheet of outline curves and bend lines in millimetres, a bend table,
//! and a DXF file for laser, punch and waterjet programming.
//!
//! The sheet is the layout's xy plane seen from its +z side, so an `Up` bend turns towards the
//! viewer. Bend lines carry their note (`UP 90° R3 BA 5.6`), which a drawing places beside the
//! line and the DXF writes as text on the bend's layer.
//!
//! DXF layers: `OUTLINE` (the cut path, outer boundary and holes), `BEND_UP` and `BEND_DOWN`
//! (bend centre lines), `BEND_NOTES` (the notes). Units are millimetres (`$INSUNITS` 4).

use crate::model::sheet_metal::{BendDirection, FlatLayout};
use runmat_geometry_kernel::op::{ProfileSegment, SegmentKind};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt::Write;

/// One curve of the sheet, in millimetres.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "curve")]
pub enum SheetCurve {
    Line {
        start: [f64; 2],
        end: [f64; 2],
    },
    /// Counter-clockwise from `start_angle` to `end_angle` (degrees).
    Arc {
        center: [f64; 2],
        radius: f64,
        start_angle: f64,
        end_angle: f64,
    },
    Circle {
        center: [f64; 2],
        radius: f64,
    },
    /// Ellipses and splines, sampled to a polyline within the chord tolerance.
    Polyline {
        points: Vec<[f64; 2]>,
    },
}

/// A row of the bend table, in the order of the bend lines.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BendTableRow {
    /// `B1`, `B2`, … as the drawing tags the lines.
    pub tag: String,
    pub key: String,
    pub feature: String,
    pub direction: BendDirection,
    /// Degrees, millimetres.
    pub angle: f64,
    pub inner_radius: f64,
    pub allowance: f64,
    pub deduction: f64,
}

/// A bend line of the sheet with its note.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetBendLine {
    pub tag: String,
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub direction: BendDirection,
    pub note: String,
}

/// A flat pattern ready to draw: millimetres in the layout's xy plane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlatPatternSheet {
    /// The outline loops (the first is the outer boundary).
    pub outline: Vec<Vec<SheetCurve>>,
    pub bend_lines: Vec<SheetBendLine>,
    pub bend_table: Vec<BendTableRow>,
    /// The outline's extents, `[x_min, y_min]` and `[x_max, y_max]`.
    pub min: [f64; 2],
    pub max: [f64; 2],
    pub thickness: f64,
}

/// Chord tolerance (mm) for sampling ellipses and splines.
const CHORD_MM: f64 = 0.01;

fn mm(p: [f64; 2]) -> [f64; 2] {
    [p[0] * 1e3, p[1] * 1e3]
}

/// Formats a number with at most `places` decimals and no trailing zeros.
fn num(v: f64, places: usize) -> String {
    let s = format!("{v:.places$}");
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    };
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

/// The sheet of a flat layout.
pub fn flat_pattern_sheet(layout: &FlatLayout) -> FlatPatternSheet {
    let outline: Vec<Vec<SheetCurve>> = layout
        .loops
        .iter()
        .map(|l| l.iter().map(sheet_curve).collect())
        .collect();
    let (lo, hi) = layout.extents();
    let mut bend_lines = Vec::new();
    let mut bend_table = Vec::new();
    for (k, b) in layout.bend_lines.iter().enumerate() {
        let tag = format!("B{}", k + 1);
        let dir = match b.direction {
            BendDirection::Up => "UP",
            BendDirection::Down => "DOWN",
        };
        let note = format!(
            "{tag} {dir} {}° R{} BA {}",
            num(b.angle.to_degrees(), 2),
            num(b.inner_radius * 1e3, 3),
            num(b.allowance * 1e3, 3)
        );
        bend_lines.push(SheetBendLine {
            tag: tag.clone(),
            start: mm(b.start),
            end: mm(b.end),
            direction: b.direction,
            note,
        });
        bend_table.push(BendTableRow {
            tag,
            key: b.key.clone(),
            feature: b.feature.0.clone(),
            direction: b.direction,
            angle: b.angle.to_degrees(),
            inner_radius: b.inner_radius * 1e3,
            allowance: b.allowance * 1e3,
            deduction: b.deduction * 1e3,
        });
    }
    FlatPatternSheet {
        outline,
        bend_lines,
        bend_table,
        min: mm(lo),
        max: mm(hi),
        thickness: layout.depth.abs() * 1e3,
    }
}

fn sheet_curve(seg: &ProfileSegment) -> SheetCurve {
    let (s, e) = (mm(seg.start), mm(seg.end));
    match &seg.kind {
        SegmentKind::Line => SheetCurve::Line { start: s, end: e },
        SegmentKind::Arc { center, clockwise } => {
            let c = mm(*center);
            let a0 = (s[1] - c[1]).atan2(s[0] - c[0]).to_degrees();
            let a1 = (e[1] - c[1]).atan2(e[0] - c[0]).to_degrees();
            let (start_angle, end_angle) = if *clockwise { (a1, a0) } else { (a0, a1) };
            SheetCurve::Arc {
                center: c,
                radius: (s[0] - c[0]).hypot(s[1] - c[1]),
                start_angle,
                end_angle,
            }
        }
        SegmentKind::Circle { center, radius } => SheetCurve::Circle {
            center: mm(*center),
            radius: radius * 1e3,
        },
        SegmentKind::Ellipse {
            center,
            major,
            ratio,
            clockwise,
        } => {
            let (c, a) = (mm(*center), mm(*major));
            let ra = a[0].hypot(a[1]);
            let rb = ra * ratio;
            let rot = a[1].atan2(a[0]);
            // Parameter of a point on the ellipse.
            let param = |p: [f64; 2]| {
                let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
                let (u, v) = (
                    dx * rot.cos() + dy * rot.sin(),
                    -dx * rot.sin() + dy * rot.cos(),
                );
                (v / rb).atan2(u / ra)
            };
            let (t0, t1) = (param(s), param(e));
            let mut sweep = (t1 - t0).rem_euclid(2.0 * PI);
            if sweep < 1e-12 {
                sweep = 2.0 * PI;
            }
            if *clockwise {
                sweep -= 2.0 * PI;
            }
            // Chord error of a step dt at radius ra: ra (1 − cos(dt/2)).
            let dt = 2.0 * (1.0 - CHORD_MM / ra.max(CHORD_MM)).clamp(-1.0, 1.0).acos();
            let n = ((sweep.abs() / dt.max(1e-3)).ceil() as usize).clamp(8, 4096);
            let points = (0..=n)
                .map(|k| {
                    let t = t0 + sweep * k as f64 / n as f64;
                    let (u, v) = (ra * t.cos(), rb * t.sin());
                    [
                        c[0] + u * rot.cos() - v * rot.sin(),
                        c[1] + u * rot.sin() + v * rot.cos(),
                    ]
                })
                .collect();
            SheetCurve::Polyline { points }
        }
        SegmentKind::Spline {
            degree,
            poles,
            weights,
            knots,
            multiplicities,
        } => {
            let poles: Vec<[f64; 2]> = poles.iter().map(|p| mm(*p)).collect();
            let flat: Vec<f64> = knots
                .iter()
                .zip(multiplicities)
                .flat_map(|(k, m)| std::iter::repeat_n(*k, *m as usize))
                .collect();
            let p = *degree as usize;
            let span: f64 = poles
                .windows(2)
                .map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]))
                .sum();
            let n = ((span / CHORD_MM).sqrt().ceil() as usize * 4).clamp(16, 4096);
            let (u0, u1) = (flat[p], flat[flat.len() - p - 1]);
            let points = (0..=n)
                .map(|k| {
                    let u = u0 + (u1 - u0) * k as f64 / n as f64;
                    de_boor(p, &flat, &poles, weights, u)
                })
                .collect();
            SheetCurve::Polyline { points }
        }
    }
}

/// A point of a (rational) B-spline by de Boor's algorithm.
fn de_boor(p: usize, knots: &[f64], poles: &[[f64; 2]], weights: &[f64], u: f64) -> [f64; 2] {
    let n = poles.len();
    let mut k = p;
    while k + 1 < n && u >= knots[k + 1] {
        k += 1;
    }
    let w = |i: usize| if weights.is_empty() { 1.0 } else { weights[i] };
    let mut d: Vec<[f64; 3]> = (0..=p)
        .map(|j| {
            let i = j + k - p;
            [poles[i][0] * w(i), poles[i][1] * w(i), w(i)]
        })
        .collect();
    for r in 1..=p {
        for j in (r..=p).rev() {
            let i = j + k - p;
            let den = knots[i + p + 1 - r] - knots[i];
            let a = if den.abs() < 1e-300 {
                0.0
            } else {
                (u - knots[i]) / den
            };
            for c in 0..3 {
                d[j][c] = (1.0 - a) * d[j - 1][c] + a * d[j][c];
            }
        }
    }
    [d[p][0] / d[p][2], d[p][1] / d[p][2]]
}

impl FlatPatternSheet {
    /// The sheet as an ASCII DXF (AC1009 entities, millimetres).
    pub fn to_dxf(&self) -> String {
        let mut out = String::new();
        let pair = |out: &mut String, code: i32, value: &str| {
            let _ = writeln!(out, "{code}\n{value}");
        };
        let f = |v: f64| num(v, 6);
        pair(&mut out, 0, "SECTION");
        pair(&mut out, 2, "HEADER");
        pair(&mut out, 9, "$INSUNITS");
        pair(&mut out, 70, "4");
        pair(&mut out, 9, "$EXTMIN");
        pair(&mut out, 10, &f(self.min[0]));
        pair(&mut out, 20, &f(self.min[1]));
        pair(&mut out, 9, "$EXTMAX");
        pair(&mut out, 10, &f(self.max[0]));
        pair(&mut out, 20, &f(self.max[1]));
        pair(&mut out, 0, "ENDSEC");
        pair(&mut out, 0, "SECTION");
        pair(&mut out, 2, "TABLES");
        pair(&mut out, 0, "TABLE");
        pair(&mut out, 2, "LAYER");
        pair(&mut out, 70, "4");
        for (name, color) in [
            ("OUTLINE", 7),
            ("BEND_UP", 3),
            ("BEND_DOWN", 1),
            ("BEND_NOTES", 8),
        ] {
            pair(&mut out, 0, "LAYER");
            pair(&mut out, 2, name);
            pair(&mut out, 70, "0");
            pair(&mut out, 62, &color.to_string());
            pair(&mut out, 6, "CONTINUOUS");
        }
        pair(&mut out, 0, "ENDTAB");
        pair(&mut out, 0, "ENDSEC");
        pair(&mut out, 0, "SECTION");
        pair(&mut out, 2, "ENTITIES");
        let line = |out: &mut String, layer: &str, a: [f64; 2], b: [f64; 2]| {
            pair(out, 0, "LINE");
            pair(out, 8, layer);
            pair(out, 10, &f(a[0]));
            pair(out, 20, &f(a[1]));
            pair(out, 11, &f(b[0]));
            pair(out, 21, &f(b[1]));
        };
        for c in self.outline.iter().flatten() {
            match c {
                SheetCurve::Line { start, end } => line(&mut out, "OUTLINE", *start, *end),
                SheetCurve::Arc {
                    center,
                    radius,
                    start_angle,
                    end_angle,
                } => {
                    pair(&mut out, 0, "ARC");
                    pair(&mut out, 8, "OUTLINE");
                    pair(&mut out, 10, &f(center[0]));
                    pair(&mut out, 20, &f(center[1]));
                    pair(&mut out, 40, &f(*radius));
                    pair(&mut out, 50, &f(start_angle.rem_euclid(360.0)));
                    pair(&mut out, 51, &f(end_angle.rem_euclid(360.0)));
                }
                SheetCurve::Circle { center, radius } => {
                    pair(&mut out, 0, "CIRCLE");
                    pair(&mut out, 8, "OUTLINE");
                    pair(&mut out, 10, &f(center[0]));
                    pair(&mut out, 20, &f(center[1]));
                    pair(&mut out, 40, &f(*radius));
                }
                SheetCurve::Polyline { points } => {
                    for w in points.windows(2) {
                        line(&mut out, "OUTLINE", w[0], w[1]);
                    }
                }
            }
        }
        let height = (self.max[0] - self.min[0])
            .max(self.max[1] - self.min[1])
            .max(1.0)
            / 60.0;
        for b in &self.bend_lines {
            let layer = match b.direction {
                BendDirection::Up => "BEND_UP",
                BendDirection::Down => "BEND_DOWN",
            };
            line(&mut out, layer, b.start, b.end);
            let mid = [(b.start[0] + b.end[0]) / 2.0, (b.start[1] + b.end[1]) / 2.0];
            let angle = (b.end[1] - b.start[1])
                .atan2(b.end[0] - b.start[0])
                .to_degrees();
            pair(&mut out, 0, "TEXT");
            pair(&mut out, 8, "BEND_NOTES");
            pair(&mut out, 10, &f(mid[0]));
            pair(&mut out, 20, &f(mid[1]));
            pair(&mut out, 40, &f(height));
            // DXF text writes the degree sign as `%%d`.
            pair(&mut out, 1, &b.note.replace('°', "%%d"));
            pair(&mut out, 50, &f(angle));
        }
        pair(&mut out, 0, "ENDSEC");
        pair(&mut out, 0, "EOF");
        out
    }
}

/// The sheet a flat-pattern view draws: the layout's sheet, without the bend notes or the
/// bend table when the view leaves them out. `None` for any other view kind.
pub fn view_sheet(kind: &super::ViewKind, layout: &FlatLayout) -> Option<FlatPatternSheet> {
    let super::ViewKind::FlatPattern {
        bend_notes,
        bend_table,
        ..
    } = kind
    else {
        return None;
    };
    let mut sheet = flat_pattern_sheet(layout);
    if !bend_notes {
        for l in &mut sheet.bend_lines {
            l.note.clear();
        }
    }
    if !bend_table {
        sheet.bend_table.clear();
    }
    Some(sheet)
}
