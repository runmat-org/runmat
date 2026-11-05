//! Footprint generation for standard package families from IPC-7351B land-pattern equations.
//!
//! For a two-terminal row of leads with lead span `L` (toe to toe), terminal length `T` and
//! terminal width `W`, each given as min/max:
//!
//! ```text
//! S    = L − 2T                     (heel to heel); S_max = L_max − 2T_min, S_min = L_min − 2T_max
//! Z    = L_min + 2·J_T + √(C_L² + F² + P²)     outer pad extent (toe side), rounded up
//! G    = S_max − 2·J_H − √(C_S² + F² + P²)     inner pad extent (heel side), rounded down
//! X    = W_min + 2·J_S + √(C_W² + F² + P²)     pad width, rounded up
//! ```
//!
//! with tolerances `C = max − min`, fabrication tolerance `F` and placement tolerance `P`.
//! Pad length is `(Z − G)/2`, pad centres sit at `±(Z + G)/4`. Fillet goals `J_T`, `J_H`,
//! `J_S` and the courtyard excess depend on the lead style and the [`Density`] level
//! (A most material, B nominal, C least). Rounding is to 0.01 mm.
//!
//! Families: chip passives (01005…2512), SOIC/SOP/SSOP/TSSOP, QFP, QFN/DFN (with exposed
//! pad), BGA (with depopulation), SOT-23-3/5/6, SOT-223, pin headers and generic
//! through-hole connectors. Each package gets pads, courtyard, silk and fab outlines, a pin-1
//! mark, a body for 3D placeholders and a model reference by name.

use super::{Body, CopperSpan, Graphic, Model3d, Package, PadDef, PadShape};
use crate::geom::poly::{rect_ring, Ring};
use crate::geom::{mm, Coord, Point, Rect};
use crate::library::make::uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Density {
    /// Most material: wave soldering, rework, high reliability.
    A,
    /// Nominal: most reflow assemblies.
    B,
    /// Least material: high density.
    C,
}

impl Density {
    fn pick(self, a: f64, b: f64, c: f64) -> f64 {
        match self {
            Density::A => a,
            Density::B => b,
            Density::C => c,
        }
    }
    pub fn suffix(self) -> &'static str {
        match self {
            Density::A => "M",
            Density::B => "N",
            Density::C => "L",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeadStyle {
    /// Rectangular or square-end chip components ≥ 1608 metric.
    Chip,
    /// Chip components below 1608 metric.
    SmallChip,
    /// Gull-wing, pitch > 0.625 mm.
    GullWing,
    /// Gull-wing, pitch ≤ 0.625 mm.
    GullWingFine,
    /// No-lead (QFN/DFN/SON) with pull-back.
    NoLead,
    /// J-lead.
    JLead,
}

/// Fillet goals `(toe, heel, side)` and courtyard excess, mm.
pub fn fillets(style: LeadStyle, d: Density) -> (f64, f64, f64, f64) {
    match style {
        LeadStyle::Chip => (
            d.pick(0.55, 0.35, 0.15),
            d.pick(0.0, -0.05, -0.10),
            d.pick(0.05, 0.0, -0.05),
            d.pick(0.5, 0.25, 0.1),
        ),
        LeadStyle::SmallChip => (
            d.pick(0.15, 0.10, 0.05),
            d.pick(0.0, 0.0, 0.0),
            d.pick(0.0, 0.0, 0.0),
            d.pick(0.2, 0.15, 0.1),
        ),
        LeadStyle::GullWing => (
            d.pick(0.55, 0.35, 0.15),
            d.pick(0.45, 0.35, 0.25),
            d.pick(0.05, 0.03, 0.01),
            d.pick(0.5, 0.25, 0.1),
        ),
        LeadStyle::GullWingFine => (
            d.pick(0.55, 0.35, 0.15),
            d.pick(0.45, 0.35, 0.25),
            d.pick(0.01, -0.02, -0.04),
            d.pick(0.5, 0.25, 0.1),
        ),
        LeadStyle::NoLead => (
            d.pick(0.40, 0.30, 0.20),
            d.pick(0.0, 0.0, 0.0),
            d.pick(-0.04, -0.04, -0.04),
            d.pick(0.5, 0.25, 0.1),
        ),
        LeadStyle::JLead => (
            d.pick(-0.10, -0.20, -0.30),
            d.pick(0.55, 0.35, 0.15),
            d.pick(0.05, 0.03, 0.01),
            d.pick(0.5, 0.25, 0.1),
        ),
    }
}

/// Fabrication and placement tolerances, mm.
pub const FAB_TOL: f64 = 0.05;
pub const PLACE_TOL: f64 = 0.025;

/// A min/max range in millimetres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    pub min: f64,
    pub max: f64,
}

impl Range {
    pub fn new(min: f64, max: f64) -> Self {
        Range { min, max }
    }
    pub fn nom_tol(nom: f64, tol: f64) -> Self {
        Range {
            min: nom - tol,
            max: nom + tol,
        }
    }
    pub fn nom(&self) -> f64 {
        (self.min + self.max) / 2.0
    }
    pub fn tol(&self) -> f64 {
        self.max - self.min
    }
}

/// Result of the IPC calculation, mm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Land {
    pub z: f64,
    pub g: f64,
    pub x: f64,
    pub courtyard_excess: f64,
}

impl Land {
    pub fn pad_length(&self) -> f64 {
        (self.z - self.g) / 2.0
    }
    pub fn pad_center(&self) -> f64 {
        (self.z + self.g) / 4.0
    }
}

fn round_up(v: f64) -> f64 {
    (v * 100.0 - 1e-9).ceil() / 100.0
}
fn round_down(v: f64) -> f64 {
    (v * 100.0 + 1e-9).floor() / 100.0
}

/// The IPC-7351B land for one lead row pair.
pub fn ipc_land(span: Range, terminal: Range, width: Range, style: LeadStyle, d: Density) -> Land {
    let (jt, jh, js, cy) = fillets(style, d);
    let f2p2 = FAB_TOL * FAB_TOL + PLACE_TOL * PLACE_TOL;
    let s_max = span.max - 2.0 * terminal.min;
    let s_min = span.min - 2.0 * terminal.max;
    let cl = span.tol();
    let cs = s_max - s_min;
    let cw = width.tol();
    let z = round_up(span.min + 2.0 * jt + (cl * cl + f2p2).sqrt());
    let g = round_down((s_max - 2.0 * jh - (cs * cs + f2p2).sqrt()).max(0.0));
    let x = round_up(width.min + 2.0 * js + (cw * cw + f2p2).sqrt());
    Land {
        z,
        g,
        x,
        courtyard_excess: cy,
    }
}

const SILK_W: f64 = 0.12;
const FAB_W: f64 = 0.10;

fn courtyard(pads: &[PadDef], body: Rect, excess: Coord) -> Vec<Ring> {
    let mut r = body;
    for p in pads {
        let (w, h) = p.shape.size();
        let (w, h) = if (p.rotation / 90.0).round() as i64 % 2 != 0 {
            (h, w)
        } else {
            (w, h)
        };
        r = r.union(Rect::new(
            Point::new(p.at.x - w / 2, p.at.y - h / 2),
            Point::new(p.at.x + w / 2, p.at.y + h / 2),
        ));
    }
    let grid = mm(0.01);
    let snap_dn = |v: Coord| v.div_euclid(grid) * grid;
    let snap_up = |v: Coord| -(-v).div_euclid(grid) * grid;
    let r = r.inflate(excess);
    vec![rect_ring(Rect::new(
        Point::new(snap_dn(r.min.x), snap_dn(r.min.y)),
        Point::new(snap_up(r.max.x), snap_up(r.max.y)),
    ))]
}

fn body_rect(w: f64, l: f64) -> Rect {
    Rect::new(Point::mm(-w / 2.0, -l / 2.0), Point::mm(w / 2.0, l / 2.0))
}

fn outline(r: Rect, width: f64) -> Graphic {
    Graphic::Rect {
        a: r.min,
        b: r.max,
        width: mm(width),
        fill: false,
    }
}

/// Silk lines along the body edges, clipped back from pads by `clear`.
fn silk_around(body: Rect, pads: &[PadDef], clear: Coord) -> Vec<Graphic> {
    let mut out = Vec::new();
    let lw = mm(SILK_W);
    let off = lw / 2 + mm(0.05);
    let r = body.inflate(off);
    let pad_boxes: Vec<Rect> = pads
        .iter()
        .map(|p| {
            let (w, h) = p.shape.size();
            Rect::new(
                Point::new(p.at.x - w / 2, p.at.y - h / 2),
                Point::new(p.at.x + w / 2, p.at.y + h / 2),
            )
            .inflate(clear + lw / 2)
        })
        .collect();
    let edges = [
        (r.min, Point::new(r.max.x, r.min.y)),
        (Point::new(r.max.x, r.min.y), r.max),
        (r.max, Point::new(r.min.x, r.max.y)),
        (Point::new(r.min.x, r.max.y), r.min),
    ];
    for (a, b) in edges {
        // Split the edge into runs that stay outside every pad box.
        let horizontal = a.y == b.y;
        let (lo, hi) = if horizontal {
            (a.x.min(b.x), a.x.max(b.x))
        } else {
            (a.y.min(b.y), a.y.max(b.y))
        };
        let mut blocked: Vec<(Coord, Coord)> = pad_boxes
            .iter()
            .filter(|pb| {
                if horizontal {
                    a.y >= pb.min.y && a.y <= pb.max.y
                } else {
                    a.x >= pb.min.x && a.x <= pb.max.x
                }
            })
            .map(|pb| {
                if horizontal {
                    (pb.min.x, pb.max.x)
                } else {
                    (pb.min.y, pb.max.y)
                }
            })
            .collect();
        blocked.sort();
        let mut cur = lo;
        for (s, e) in blocked.into_iter().chain(std::iter::once((hi, hi))) {
            if s > cur + lw {
                let seg_end = s.min(hi);
                let pts = if horizontal {
                    vec![Point::new(cur, a.y), Point::new(seg_end, a.y)]
                } else {
                    vec![Point::new(a.x, cur), Point::new(a.x, seg_end)]
                };
                out.push(Graphic::Line {
                    points: pts,
                    width: lw,
                });
            }
            cur = cur.max(e);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn finish(
    name: String,
    description: String,
    pads: Vec<PadDef>,
    body_w: f64,
    body_l: f64,
    height: f64,
    excess: f64,
    pin1: Option<Point>,
    smd: bool,
) -> Package {
    let body = body_rect(body_w, body_l);
    let mut silk = silk_around(body, &pads, mm(0.2));
    if let Some(p) = pin1 {
        silk.push(Graphic::Circle {
            center: p,
            radius: mm(0.1),
            width: mm(SILK_W),
            fill: true,
        });
    }
    let mut fab = vec![outline(body, FAB_W)];
    fab.push(Graphic::Text {
        at: Point::ORIGIN,
        text: "${REFERENCE}".into(),
        size: mm((body_w.min(body_l) * 0.4).clamp(0.2, 1.0)),
        angle: 0.0,
    });
    Package {
        uuid: uuid("package", &name),
        courtyard: courtyard(&pads, body, mm(excess)),
        model: Some(Model3d {
            path: format!("${{RUNMAT_3DMODELS}}/{name}.step"),
            offset: (0.0, 0.0, 0.0),
            rotation: (0.0, 0.0, 0.0),
            scale: (1.0, 1.0, 1.0),
            origin: "footprint".into(),
        }),
        body: Some(Body {
            width: mm(body_w),
            length: mm(body_l),
            height: mm(height),
        }),
        name,
        description,
        pads,
        silk,
        fab,
        smd,
    }
}

/// Chip size codes: (imperial, metric, L, W, T_min, T_max, height), mm, with ± length/width
/// tolerance.
/// (imperial, metric, L, W, T_min, T_max, ±tol, height).
type ChipRow = (&'static str, &'static str, f64, f64, f64, f64, f64, f64);
const CHIPS: &[ChipRow] = &[
    ("01005", "0402", 0.40, 0.20, 0.07, 0.14, 0.02, 0.13),
    ("0201", "0603", 0.60, 0.30, 0.10, 0.20, 0.03, 0.26),
    ("0402", "1005", 1.00, 0.50, 0.15, 0.35, 0.05, 0.35),
    ("0603", "1608", 1.60, 0.80, 0.20, 0.50, 0.15, 0.45),
    ("0805", "2012", 2.00, 1.25, 0.25, 0.75, 0.20, 0.60),
    ("1206", "3216", 3.20, 1.60, 0.25, 0.75, 0.20, 0.60),
    ("1210", "3225", 3.20, 2.50, 0.25, 0.75, 0.20, 0.60),
    ("1812", "4532", 4.50, 3.20, 0.25, 0.85, 0.20, 0.80),
    ("2010", "5025", 5.00, 2.50, 0.35, 0.85, 0.20, 0.60),
    ("2512", "6332", 6.30, 3.20, 0.35, 0.85, 0.20, 0.60),
];

/// Two-terminal chip (`kind` is `R`, `C`, `L`, `D`, `LED`, `F` and only sets the name).
/// `code` is imperial (`0402`) or metric (`1005Metric`).
pub fn chip(kind: &str, code: &str, density: Density) -> Option<Package> {
    let metric = code.strip_suffix("Metric");
    let row = CHIPS.iter().find(|c| match metric {
        Some(m) => c.1 == m,
        None => c.0 == code,
    })?;
    let &(imp, met, l, w, tmin, tmax, tol, h) = row;
    let span = Range::nom_tol(l, tol);
    let width = Range::nom_tol(w, tol);
    let style = if l < 1.6 {
        LeadStyle::SmallChip
    } else {
        LeadStyle::Chip
    };
    let land = ipc_land(span, Range::new(tmin, tmax), width, style, density);
    let c = land.pad_center();
    let pl = land.pad_length();
    let shape = PadShape::RoundRect {
        w: mm(pl),
        h: mm(land.x),
        ratio: 0.25,
    };
    let pads = vec![
        PadDef::smd("1", Point::mm(-c, 0.0), shape.clone()),
        PadDef::smd("2", Point::mm(c, 0.0), shape),
    ];
    let polar = matches!(kind, "D" | "LED");
    let name = format!("{kind}_{imp}_{met}Metric_IPC{}", density.suffix());
    Some(finish(
        name,
        format!("{kind} chip {imp} ({met} metric), IPC-7351B density {density:?}"),
        pads,
        l,
        w,
        h,
        land.courtyard_excess,
        polar.then(|| Point::mm(-c - pl / 2.0 - 0.25, 0.0)),
        true,
    ))
}

/// Parameters of a dual- or quad-row gull-wing, J-lead or no-lead package.
#[derive(Clone, Debug, PartialEq)]
pub struct LeadedSpec {
    pub family: String,
    pub pins: u32,
    /// `2` (SOIC/SOP/DFN, pins along y on the left and right) or `4` (QFP/QFN).
    pub rows: u32,
    pub pitch: f64,
    /// Body (x, y), mm.
    pub body: (f64, f64),
    pub height: f64,
    /// Toe-to-toe span across the rows (x for 2-row; both axes for 4-row).
    pub span: Range,
    /// Second span for 4-row packages with a non-square span (y axis).
    pub span_y: Option<Range>,
    pub terminal: Range,
    pub lead_width: Range,
    pub style: LeadStyle,
    /// Exposed thermal pad (x, y) with optional paste division count.
    pub exposed_pad: Option<(f64, f64)>,
}

/// Dual- or quad-row package; pins counted counter-clockwise from pin 1 at top-left.
pub fn leaded(spec: &LeadedSpec, density: Density) -> Package {
    let LeadedSpec {
        pins,
        rows,
        pitch,
        body,
        span,
        terminal,
        lead_width,
        style,
        ..
    } = spec.clone();
    let style = match style {
        LeadStyle::GullWing if pitch <= 0.625 => LeadStyle::GullWingFine,
        s => s,
    };
    let land_x = ipc_land(span, terminal, lead_width, style, density);
    let land_y = ipc_land(
        spec.span_y.unwrap_or(span),
        terminal,
        lead_width,
        style,
        density,
    );
    let mut pads = Vec::new();
    let per_row = pins / rows;
    let x_pad = PadShape::RoundRect {
        w: mm(land_x.pad_length()),
        h: mm(land_x.x),
        ratio: 0.25,
    };
    let y_pad = PadShape::RoundRect {
        w: mm(land_y.x),
        h: mm(land_y.pad_length()),
        ratio: 0.25,
    };
    let row_pos = |i: u32| (i as f64 - (per_row as f64 - 1.0) / 2.0) * pitch;
    let mut n = 1;
    let push = |pads: &mut Vec<PadDef>, n: &mut u32, at: Point, s: &PadShape| {
        pads.push(PadDef::smd(&n.to_string(), at, s.clone()));
        *n += 1;
    };
    // Internal coordinates are y-up: pin 1 is top-left (positive y), numbering runs
    // counter-clockwise.
    if rows == 2 {
        for i in 0..per_row {
            push(
                &mut pads,
                &mut n,
                Point::mm(-land_x.pad_center(), -row_pos(i)),
                &x_pad,
            );
        }
        for i in 0..per_row {
            push(
                &mut pads,
                &mut n,
                Point::mm(land_x.pad_center(), row_pos(i)),
                &x_pad,
            );
        }
    } else {
        // Left (top→bottom), bottom (left→right), right (bottom→top), top (right→left).
        for i in 0..per_row {
            push(
                &mut pads,
                &mut n,
                Point::mm(-land_x.pad_center(), -row_pos(i)),
                &x_pad,
            );
        }
        for i in 0..per_row {
            push(
                &mut pads,
                &mut n,
                Point::mm(row_pos(i), -land_y.pad_center()),
                &y_pad,
            );
        }
        for i in 0..per_row {
            push(
                &mut pads,
                &mut n,
                Point::mm(land_x.pad_center(), row_pos(i)),
                &x_pad,
            );
        }
        for i in 0..per_row {
            push(
                &mut pads,
                &mut n,
                Point::mm(-row_pos(i), land_y.pad_center()),
                &y_pad,
            );
        }
    }
    if let Some((ex, ey)) = spec.exposed_pad {
        let mut ep = PadDef::smd(
            &(pins + 1).to_string(),
            Point::ORIGIN,
            PadShape::RoundRect {
                w: mm(ex),
                h: mm(ey),
                ratio: 0.05,
            },
        );
        ep.thermal_pad = true;
        // Paste reduced to about 60 % of the pad area to avoid floating the part.
        ep.paste_margin = Some(-mm(ex.min(ey) * 0.1));
        pads.push(ep);
    }
    let first = pads[0].at;
    let ep = spec
        .exposed_pad
        .map(|(x, y)| format!("_EP{x}x{y}mm"))
        .unwrap_or_default();
    let name = format!(
        "{}-{}_{}x{}mm_P{}mm{}_IPC{}",
        spec.family,
        pins,
        body.0,
        body.1,
        pitch,
        ep,
        density.suffix()
    );
    finish(
        name,
        format!(
            "{} {pins} pins, pitch {pitch} mm, IPC-7351B density {density:?}",
            spec.family
        ),
        pads,
        body.0,
        body.1,
        spec.height,
        land_x.courtyard_excess,
        Some(Point::new(
            first.x - mm(land_x.pad_length() / 2.0 + 0.3),
            first.y,
        )),
        true,
    )
}

/// SOIC-N (JEDEC MS-012/013): `narrow` is the 3.9 mm body, else 7.5 mm.
pub fn soic(pins: u32, narrow: bool, density: Density) -> Package {
    let (bw, span) = if narrow {
        (3.9, Range::new(5.8, 6.2))
    } else {
        (7.5, Range::new(10.0, 10.65))
    };
    let bl = (pins / 2) as f64 * 1.27 + if narrow { 0.1 } else { 0.5 };
    leaded(
        &LeadedSpec {
            family: "SOIC".into(),
            pins,
            rows: 2,
            pitch: 1.27,
            body: (bw, bl),
            height: if narrow { 1.75 } else { 2.65 },
            span,
            span_y: None,
            terminal: Range::new(0.4, 1.27),
            lead_width: Range::new(0.31, 0.51),
            style: LeadStyle::GullWing,
            exposed_pad: None,
        },
        density,
    )
}

/// TSSOP-N, 4.4 mm body, 0.65 mm pitch (JEDEC MO-153).
pub fn tssop(pins: u32, density: Density) -> Package {
    leaded(
        &LeadedSpec {
            family: "TSSOP".into(),
            pins,
            rows: 2,
            pitch: 0.65,
            body: (4.4, (pins / 2) as f64 * 0.65 + 0.3),
            height: 1.2,
            span: Range::new(6.2, 6.6),
            span_y: None,
            terminal: Range::new(0.45, 0.75),
            lead_width: Range::new(0.19, 0.30),
            style: LeadStyle::GullWing,
            exposed_pad: None,
        },
        density,
    )
}

/// MSOP-N / VSSOP-N, 3.0 mm body, 0.5 mm pitch (JEDEC MO-187).
pub fn msop(pins: u32, density: Density) -> Package {
    leaded(
        &LeadedSpec {
            family: "MSOP".into(),
            pins,
            rows: 2,
            pitch: 0.5,
            body: (3.0, 3.0),
            height: 1.1,
            span: Range::new(4.75, 5.05),
            span_y: None,
            terminal: Range::new(0.4, 0.7),
            lead_width: Range::new(0.17, 0.27),
            style: LeadStyle::GullWing,
            exposed_pad: None,
        },
        density,
    )
}

/// A plated mounting hole: drill `drill` mm in a round pad of `pad` mm, as pad `1` (it takes
/// the net it is wired to, usually ground).
pub fn mounting_hole(drill: f64, pad: f64) -> Package {
    let pads = vec![PadDef::tht(
        "1",
        Point::ORIGIN,
        PadShape::Circle { d: mm(pad) },
        mm(drill),
    )];
    let mut pkg = finish(
        format!("MountingHole_{drill}mm_Pad{pad}mm"),
        format!("Mounting hole, drill {drill} mm, plated pad {pad} mm"),
        pads,
        pad,
        pad,
        0.0,
        0.25,
        None,
        false,
    );
    // The pad is the hole's whole footprint: no body outline in silk or fabrication art.
    pkg.silk.clear();
    pkg.body = None;
    pkg
}

/// QFP with square body `body` mm, `pins` total, `pitch` mm (LQFP/TQFP per JEDEC MS-026).
pub fn qfp(pins: u32, body: f64, pitch: f64, density: Density) -> Package {
    let span = Range::nom_tol(body + 2.0, 0.2);
    let lw = if pitch <= 0.5 {
        Range::new(0.17, 0.27)
    } else {
        Range::new(0.30, 0.45)
    };
    leaded(
        &LeadedSpec {
            family: "LQFP".into(),
            pins,
            rows: 4,
            pitch,
            body: (body, body),
            height: 1.6,
            span,
            span_y: None,
            terminal: Range::new(0.45, 0.75),
            lead_width: lw,
            style: LeadStyle::GullWing,
            exposed_pad: None,
        },
        density,
    )
}

/// QFN with square body, `pins` around the perimeter, optional exposed pad.
pub fn qfn(pins: u32, body: f64, pitch: f64, ep: Option<f64>, density: Density) -> Package {
    leaded(
        &LeadedSpec {
            family: "QFN".into(),
            pins,
            rows: 4,
            pitch,
            body: (body, body),
            height: 0.9,
            span: Range::nom_tol(body, 0.05),
            span_y: None,
            terminal: Range::new(0.3, 0.5),
            lead_width: Range::new(pitch * 0.36, pitch * 0.5),
            style: LeadStyle::NoLead,
            exposed_pad: ep.map(|e| (e, e)),
        },
        density,
    )
}

/// DFN/SON two-row no-lead package.
pub fn dfn(
    pins: u32,
    body: (f64, f64),
    pitch: f64,
    ep: Option<(f64, f64)>,
    density: Density,
) -> Package {
    leaded(
        &LeadedSpec {
            family: "DFN".into(),
            pins,
            rows: 2,
            pitch,
            body,
            height: 0.8,
            span: Range::nom_tol(body.0, 0.05),
            span_y: None,
            terminal: Range::new(0.3, 0.5),
            lead_width: Range::new(pitch * 0.36, pitch * 0.5),
            style: LeadStyle::NoLead,
            exposed_pad: ep,
        },
        density,
    )
}

/// BGA row letters, skipping I, O, Q, S, X, Z (JEDEC JEP95); after Y comes AA, AB, …
pub fn bga_row_name(i: u32) -> String {
    const L: &[u8] = b"ABCDEFGHJKLMNPRTUVWY";
    let n = L.len() as u32;
    if i < n {
        (L[i as usize] as char).to_string()
    } else {
        let a = (i / n) - 1;
        let b = i % n;
        format!(
            "{}{}",
            L[a as usize % L.len()] as char,
            L[b as usize] as char
        )
    }
}

/// BGA `rows × cols` at `pitch` with ball diameter `ball`; `populated(row, col)` decides
/// depopulation. Non-solder-mask-defined lands sized from the ball per density.
pub fn bga(
    rows: u32,
    cols: u32,
    pitch: f64,
    ball: f64,
    body: (f64, f64),
    populated: impl Fn(u32, u32) -> bool,
    density: Density,
) -> Package {
    let land = ball * density.pick(0.85, 0.80, 0.75);
    let land = round_up(land);
    let mut pads = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if !populated(r, c) {
                continue;
            }
            let x = (c as f64 - (cols as f64 - 1.0) / 2.0) * pitch;
            let y = -(r as f64 - (rows as f64 - 1.0) / 2.0) * pitch;
            let mut p = PadDef::smd(
                &format!("{}{}", bga_row_name(r), c + 1),
                Point::mm(x, y),
                PadShape::Circle { d: mm(land) },
            );
            p.paste = true;
            pads.push(p);
        }
    }
    let n = pads.len();
    let excess = density.pick(2.0, 1.0, 0.5);
    let name = format!(
        "BGA-{n}_{rows}x{cols}_{}x{}mm_Layout{rows}x{cols}_P{pitch}mm_Ball{ball}mm_IPC{}",
        body.0,
        body.1,
        density.suffix()
    );
    let a1 = Point::mm(-body.0 / 2.0 - 0.3, body.1 / 2.0 + 0.3);
    finish(
        name,
        format!("BGA {n} balls, {rows}x{cols} at {pitch} mm"),
        pads,
        body.0,
        body.1,
        1.2,
        excess * 0.25,
        Some(a1),
        true,
    )
}

/// SOT-23 family: 3, 5 or 6 pins.
pub fn sot23(pins: u32, density: Density) -> Option<Package> {
    let span = Range::new(2.1, 2.64);
    let land = ipc_land(
        span,
        Range::new(0.3, 0.6),
        Range::new(0.3, 0.5),
        LeadStyle::GullWing,
        density,
    );
    let (c, pl, x) = (land.pad_center(), land.pad_length(), land.x);
    let s = PadShape::RoundRect {
        w: mm(pl),
        h: mm(x),
        ratio: 0.25,
    };
    // Tables are written top-down; internal coordinates are y-up.
    let at = |x: f64, y: f64| Point::mm(x, -y);
    let pads: Vec<PadDef> = match pins {
        3 => vec![
            PadDef::smd("1", at(-c, -0.95), s.clone()),
            PadDef::smd("2", at(-c, 0.95), s.clone()),
            PadDef::smd("3", at(c, 0.0), s),
        ],
        5 => vec![
            PadDef::smd("1", at(-c, -0.95), s.clone()),
            PadDef::smd("2", at(-c, 0.0), s.clone()),
            PadDef::smd("3", at(-c, 0.95), s.clone()),
            PadDef::smd("4", at(c, 0.95), s.clone()),
            PadDef::smd("5", at(c, -0.95), s),
        ],
        6 => vec![
            PadDef::smd("1", at(-c, -0.95), s.clone()),
            PadDef::smd("2", at(-c, 0.0), s.clone()),
            PadDef::smd("3", at(-c, 0.95), s.clone()),
            PadDef::smd("4", at(c, 0.95), s.clone()),
            PadDef::smd("5", at(c, 0.0), s.clone()),
            PadDef::smd("6", at(c, -0.95), s),
        ],
        _ => return None,
    };
    Some(finish(
        format!("SOT-23-{pins}_IPC{}", density.suffix()),
        format!("SOT-23 {pins} pins, IPC-7351B density {density:?}"),
        pads,
        1.6,
        2.9,
        1.1,
        land.courtyard_excess,
        Some(at(-c - pl / 2.0 - 0.3, -0.95)),
        true,
    ))
}

/// SOT-223 (3 leads plus tab, tab is pad 4).
pub fn sot223(density: Density) -> Package {
    let span = Range::new(6.7, 7.3);
    let land = ipc_land(
        span,
        Range::new(0.75, 1.1),
        Range::new(0.6, 0.84),
        LeadStyle::GullWing,
        density,
    );
    let (c, pl, x) = (land.pad_center(), land.pad_length(), land.x);
    let s = PadShape::RoundRect {
        w: mm(pl),
        h: mm(x),
        ratio: 0.25,
    };
    let tab = PadShape::RoundRect {
        w: mm(pl),
        h: mm(3.3),
        ratio: 0.1,
    };
    let pads = vec![
        PadDef::smd("1", Point::mm(-c, 2.3), s.clone()),
        PadDef::smd("2", Point::mm(-c, 0.0), s.clone()),
        PadDef::smd("3", Point::mm(-c, -2.3), s),
        PadDef::smd("4", Point::mm(c, 0.0), tab),
    ];
    finish(
        format!("SOT-223-3_TabPin2_IPC{}", density.suffix()),
        "SOT-223 3 leads + tab".into(),
        pads,
        3.5,
        6.5,
        1.8,
        land.courtyard_excess,
        Some(Point::mm(-c - pl / 2.0 - 0.3, 2.3)),
        true,
    )
}

/// Through-hole pin header `rows × cols` at `pitch` (2.54, 2.0, 1.27 mm). Pin 1 square.
pub fn pin_header(rows: u32, cols: u32, pitch: f64) -> Package {
    let drill = if pitch >= 2.5 {
        1.0
    } else if pitch >= 2.0 {
        0.8
    } else {
        0.65
    };
    let pad = if pitch >= 2.5 {
        1.7
    } else if pitch >= 2.0 {
        1.35
    } else {
        1.0
    };
    let mut pads = Vec::new();
    let mut n = 1;
    for r in 0..rows {
        for c in 0..cols {
            let at = Point::mm(c as f64 * pitch, -(r as f64) * pitch);
            let shape = if n == 1 {
                PadShape::Rect {
                    w: mm(pad),
                    h: mm(pad),
                }
            } else {
                PadShape::Oval {
                    w: mm(pad),
                    h: mm(pad),
                }
            };
            pads.push(PadDef::tht(&n.to_string(), at, shape, mm(drill)));
            n += 1;
        }
    }
    // Centre the body over the pin field.
    let bw = cols as f64 * pitch;
    let bl = rows as f64 * pitch;
    let cx = (cols as f64 - 1.0) * pitch / 2.0;
    let cy = (rows as f64 - 1.0) * pitch / 2.0;
    let mut pkg = finish(
        format!("PinHeader_{cols}x{rows:02}_P{pitch}mm_Vertical"),
        format!("Pin header {cols}x{rows}, pitch {pitch} mm"),
        pads.into_iter()
            .map(|mut p| {
                p.at = p.at - Point::mm(cx, -cy);
                p
            })
            .collect(),
        bw,
        bl,
        8.5,
        0.5,
        None,
        false,
    );
    for p in &mut pkg.pads {
        p.copper = CopperSpan::All;
    }
    pkg
}

/// Generic connector: SMD signal pads in a row plus optional mounting pads/holes.
pub fn smd_connector(
    family: &str,
    pins: u32,
    pitch: f64,
    pad: (f64, f64),
    mounting: &[(Point, PadShape, Option<Coord>)],
    body: (f64, f64),
) -> Package {
    let mut pads = Vec::new();
    for i in 0..pins {
        let x = (i as f64 - (pins as f64 - 1.0) / 2.0) * pitch;
        let mut p = PadDef::smd(
            &(i + 1).to_string(),
            Point::mm(x, -body.1 / 2.0),
            PadShape::RoundRect {
                w: mm(pad.0),
                h: mm(pad.1),
                ratio: 0.25,
            },
        );
        p.paste = true;
        pads.push(p);
    }
    for (k, (at, shape, drill)) in mounting.iter().enumerate() {
        let p = match drill {
            Some(d) => PadDef::tht(&format!("MP{}", k + 1), *at, shape.clone(), *d),
            None => PadDef::smd(&format!("MP{}", k + 1), *at, shape.clone()),
        };
        pads.push(p);
    }
    let name = format!("{family}_1x{pins:02}_P{pitch}mm");
    finish(
        name,
        format!("{family} {pins} pins"),
        pads,
        body.0,
        body.1,
        3.0,
        0.5,
        None,
        mounting.iter().all(|m| m.2.is_none()),
    )
}

/// Parses a compact footprint string (the footprinter-style DSL):
/// `R0402`, `C0603`, `soic8`, `soic16w`, `tssop20`, `qfp48_7mm_p0.5`, `qfn32_5mm_p0.5_ep3.3`,
/// `bga256_16x16_p1.0_ball0.5`, `sot23-5`, `sot223`, `pinheader2x05_p2.54`. An optional
/// `@A`/`@B`/`@C` suffix picks the density (default B).
pub fn from_string(s: &str) -> Option<Package> {
    let (body, density) = match s.rsplit_once('@') {
        Some((b, "A")) => (b, Density::A),
        Some((b, "C")) => (b, Density::C),
        Some((b, _)) => (b, Density::B),
        None => (s, Density::B),
    };
    let lower = body.to_ascii_lowercase();
    let num = |t: &str| t.parse::<f64>().ok();
    let field = |prefix: &str| {
        lower.split('_').find_map(|t| {
            t.strip_prefix(prefix)
                .and_then(|v| v.trim_end_matches("mm").parse::<f64>().ok())
        })
    };
    for kind in ["led", "r", "c", "l", "d", "f"] {
        if let Some(code) = lower.strip_prefix(kind) {
            if code.chars().all(|c| c.is_ascii_digit()) && !code.is_empty() {
                return chip(&kind.to_ascii_uppercase(), code, density);
            }
        }
    }
    if let Some(rest) = lower.strip_prefix("soic") {
        let wide = rest.ends_with('w');
        let n = rest.trim_end_matches('w').parse().ok()?;
        return Some(soic(n, !wide, density));
    }
    if let Some(rest) = lower.strip_prefix("tssop") {
        return Some(tssop(rest.parse().ok()?, density));
    }
    if let Some(rest) = lower.strip_prefix("qfp") {
        let n = rest.split('_').next()?.parse().ok()?;
        let b = lower
            .split('_')
            .nth(1)
            .and_then(|t| num(t.trim_end_matches("mm")))?;
        return Some(qfp(n, b, field("p")?, density));
    }
    if let Some(rest) = lower.strip_prefix("qfn") {
        let n = rest.split('_').next()?.parse().ok()?;
        let b = lower
            .split('_')
            .nth(1)
            .and_then(|t| num(t.trim_end_matches("mm")))?;
        return Some(qfn(n, b, field("p")?, field("ep"), density));
    }
    if let Some(rest) = lower.strip_prefix("bga") {
        let mut it = rest.split('_');
        let _n: u32 = it.next()?.parse().ok()?;
        let (r, c) = it.next()?.split_once('x')?;
        let (r, c): (u32, u32) = (r.parse().ok()?, c.parse().ok()?);
        let p = field("p")?;
        let ball = field("ball").unwrap_or(p * 0.5);
        let b = (c as f64 * p + p, r as f64 * p + p);
        return Some(bga(r, c, p, ball, b, |_, _| true, density));
    }
    if let Some(rest) = lower.strip_prefix("sot23") {
        let n = rest.trim_start_matches('-').parse().unwrap_or(3);
        return sot23(n, density);
    }
    if lower == "sot223" {
        return Some(sot223(density));
    }
    if let Some(rest) = lower.strip_prefix("pinheader") {
        let dims = rest.split('_').next()?;
        let (c, r) = dims.split_once('x')?;
        let pitch = field("p").unwrap_or(2.54);
        return Some(pin_header(r.parse().ok()?, c.parse().ok()?, pitch));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::to_mm;

    #[test]
    fn chip_0603_density_b() {
        let p = chip("R", "0603", Density::B).unwrap();
        assert_eq!(p.pads.len(), 2);
        let (w, h) = p.pads[0].shape.size();
        // Z = 1.45 + 0.7 + √(0.3² + 0.05² + 0.025²) = 2.4552 → 2.46
        // G = (1.75 − 0.4) + 0.1 − √(0.9² + …) → 0.54
        let land = ipc_land(
            Range::nom_tol(1.6, 0.15),
            Range::new(0.2, 0.5),
            Range::nom_tol(0.8, 0.15),
            LeadStyle::Chip,
            Density::B,
        );
        assert!((land.z - 2.46).abs() < 1e-9, "{land:?}");
        assert!((to_mm(w) - land.pad_length()).abs() < 1e-6);
        assert!((to_mm(h) - land.x).abs() < 1e-6);
        assert!(p.pads[0].at.x < 0 && p.pads[1].at.x > 0);
        assert!(!p.courtyard.is_empty());
        // Density A pads are larger than C.
        let a = chip("R", "0603", Density::A).unwrap();
        let c = chip("R", "0603", Density::C).unwrap();
        assert!(a.pads[0].shape.size().0 > c.pads[0].shape.size().0);
        assert!(chip("R", "1005Metric", Density::B).is_some());
    }

    #[test]
    fn leaded_families() {
        let s = soic(8, true, Density::B);
        assert_eq!(s.pads.len(), 8);
        // Pin 1 top-left, pin 8 top-right, pin 4 bottom-left (y-up).
        assert!(s.pads[0].at.x < 0 && s.pads[0].at.y > 0);
        assert!(s.pads[7].at.x > 0 && s.pads[7].at.y > 0);
        assert!(s.pads[3].at.x < 0 && s.pads[3].at.y < 0);
        assert!((to_mm(s.pads[0].at.y - s.pads[1].at.y) - 1.27).abs() < 1e-6);
        let q = qfn(32, 5.0, 0.5, Some(3.3), Density::B);
        assert_eq!(q.pads.len(), 33);
        assert!(q.pads[32].thermal_pad);
        let l = qfp(64, 10.0, 0.5, Density::B);
        assert_eq!(l.pads.len(), 64);
        let names: std::collections::BTreeSet<_> = l.pads.iter().map(|p| p.name.clone()).collect();
        assert_eq!(names.len(), 64);
        assert!(sot23(5, Density::B).unwrap().pads.len() == 5);
        assert_eq!(sot223(Density::B).pads.len(), 4);
    }

    #[test]
    fn bga_and_headers_and_strings() {
        let b = bga(
            16,
            16,
            1.0,
            0.5,
            (17.0, 17.0),
            |r, c| !(6..10).contains(&r) || !(6..10).contains(&c),
            Density::B,
        );
        assert_eq!(b.pads.len(), 256 - 16);
        assert!(b.pads.iter().any(|p| p.name == "A1"));
        assert!(!b.pads.iter().any(|p| p.name.starts_with('I')));
        assert_eq!(bga_row_name(20), "AA");
        let h = pin_header(5, 2, 2.54);
        assert_eq!(h.pads.len(), 10);
        assert!(h.pads.iter().all(|p| p.drill.is_some()));
        for s in [
            "R0402",
            "C0603@A",
            "soic8",
            "soic16w",
            "tssop20",
            "qfp48_7mm_p0.5",
            "qfn32_5mm_p0.5_ep3.3",
            "bga256_16x16_p1.0_ball0.5",
            "sot23-5",
            "sot223",
            "pinheader2x05_p2.54",
            "LED0805",
        ] {
            assert!(from_string(s).is_some(), "{s}");
        }
        assert!(from_string("frob").is_none());
    }
}
