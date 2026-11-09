//! What the sheet already holds, and the free-space search that places a label clear of it.
//!
//! Every placed item registers boxes (a view's reserved area, a label's text, a table) and
//! segments (leaders, dimension lines). A label's candidate spot is free when its box stays in
//! the frame, overlaps no registered box, and its leader crosses no registered text.

use super::{Prim, Shape, Std, Xy};

/// An axis-aligned box, sheet mm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxR {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl BoxR {
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> BoxR {
        BoxR {
            x0: x0.min(x1),
            y0: y0.min(y1),
            x1: x0.max(x1),
            y1: y0.max(y1),
        }
    }
    pub fn around(c: Xy, w: f64, h: f64) -> BoxR {
        BoxR::new(
            c[0] - w / 2.0,
            c[1] - h / 2.0,
            c[0] + w / 2.0,
            c[1] + h / 2.0,
        )
    }
    pub fn w(&self) -> f64 {
        self.x1 - self.x0
    }
    pub fn h(&self) -> f64 {
        self.y1 - self.y0
    }
    pub fn center(&self) -> Xy {
        [(self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0]
    }
    pub fn grow(&self, m: f64) -> BoxR {
        BoxR {
            x0: self.x0 - m,
            y0: self.y0 - m,
            x1: self.x1 + m,
            y1: self.y1 + m,
        }
    }
    pub fn union(&self, o: &BoxR) -> BoxR {
        BoxR {
            x0: self.x0.min(o.x0),
            y0: self.y0.min(o.y0),
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
        }
    }
    /// Overlap with some area (touching is not overlap).
    pub fn hits(&self, o: &BoxR) -> bool {
        self.x0 < o.x1 - 1e-9
            && o.x0 < self.x1 - 1e-9
            && self.y0 < o.y1 - 1e-9
            && o.y0 < self.y1 - 1e-9
    }
    pub fn inside(&self, o: &BoxR) -> bool {
        self.x0 >= o.x0 - 1e-9
            && self.x1 <= o.x1 + 1e-9
            && self.y0 >= o.y0 - 1e-9
            && self.y1 <= o.y1 + 1e-9
    }
    pub fn contains(&self, p: Xy) -> bool {
        p[0] > self.x0 && p[0] < self.x1 && p[1] > self.y0 && p[1] < self.y1
    }
    /// Whether segment a–b passes through the box's inside.
    pub fn crossed_by(&self, a: Xy, b: Xy) -> bool {
        // Liang–Barsky clip against the box.
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let mut t0: f64 = 0.0;
        let mut t1: f64 = 1.0;
        for (p, q) in [
            (-dx, a[0] - self.x0),
            (dx, self.x1 - a[0]),
            (-dy, a[1] - self.y0),
            (dy, self.y1 - a[1]),
        ] {
            if p.abs() < 1e-12 {
                if q <= 0.0 {
                    return false;
                }
                continue;
            }
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
            if t0 >= t1 - 1e-9 {
                return false;
            }
        }
        true
    }
}

/// What a registered box holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// A view's lines and the room it keeps for its label and dimension rows.
    View,
    /// Text: a label, a figure, a callout.
    Text,
    /// A symbol or frame (a datum square, a balloon, a frame's cells).
    Mark,
    /// A table or the title block.
    Table,
}

#[derive(Clone, Debug)]
pub struct Item {
    pub b: BoxR,
    pub owner: String,
    pub class: Class,
}

#[derive(Clone, Debug)]
pub struct Occupancy {
    pub frame: BoxR,
    pub items: Vec<Item>,
    /// Leaders and dimension lines: (from, to, owner).
    pub segs: Vec<(Xy, Xy, String)>,
}

impl Occupancy {
    pub fn new(frame: BoxR) -> Occupancy {
        Occupancy {
            frame,
            items: Vec::new(),
            segs: Vec::new(),
        }
    }

    pub fn add(&mut self, b: BoxR, owner: &str, class: Class) {
        self.items.push(Item {
            b,
            owner: owner.to_owned(),
            class,
        });
    }

    pub fn seg(&mut self, a: Xy, b: Xy, owner: &str) {
        self.segs.push((a, b, owner.to_owned()));
    }

    /// Registers what `prims` draw for `owner`: each text's box, each frame or circle as a mark,
    /// and each line as a segment.
    pub fn add_prims(&mut self, prims: &[Prim], owner: &str) {
        for p in prims {
            match &p.shape {
                Shape::Text { .. } => {
                    if let Some(b) = text_box(p) {
                        self.add(b, owner, Class::Text);
                    }
                }
                Shape::Rect { x, y, w, h } => {
                    self.add(BoxR::new(*x, *y, x + w, y + h), owner, Class::Mark)
                }
                Shape::Circle { x, y, r } => {
                    self.add(BoxR::around([*x, *y], 2.0 * r, 2.0 * r), owner, Class::Mark)
                }
                Shape::Polyline { points, .. } => {
                    for k in (0..points.len().saturating_sub(3)).step_by(2) {
                        self.seg(
                            [points[k], points[k + 1]],
                            [points[k + 2], points[k + 3]],
                            owner,
                        );
                    }
                }
                Shape::Polygon { .. } => {}
            }
        }
    }

    /// Whether `b` is in the frame and overlaps nothing but `owner`'s own items (and views
    /// in `ignore_views`), with `margin` of clearance.
    pub fn free(&self, b: &BoxR, owner: &str, margin: f64, ignore_views: &[&str]) -> bool {
        if !b.inside(&self.frame) {
            return false;
        }
        let g = b.grow(margin);
        self.items.iter().all(|i| {
            i.owner == owner
                || (i.class == Class::View && ignore_views.contains(&i.owner.as_str()))
                || !g.hits(&i.b)
        }) && self
            .segs
            .iter()
            .all(|(a, z, o)| o == owner || !g.crossed_by(*a, *z))
    }

    /// Whether a leader a–b crosses text, a mark, a table, or a view other than `own_view`.
    pub fn leader_clear(&self, a: Xy, b: Xy, owner: &str, own_view: &str) -> bool {
        self.items.iter().all(|i| {
            i.owner == owner
                || (i.class == Class::View && i.owner == own_view)
                || !i.b.grow(0.5).crossed_by(a, b)
        })
    }

    /// How many registered leaders and dimension lines segment a–b crosses.
    pub fn crossings(&self, a: Xy, b: Xy, owner: &str) -> usize {
        self.segs
            .iter()
            .filter(|(p, q, o)| o != owner && segments_cross(a, b, *p, *q))
            .count()
    }
}

fn segments_cross(a: Xy, b: Xy, c: Xy, d: Xy) -> bool {
    let o = |p: Xy, q: Xy, r: Xy| (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    let (d1, d2, d3, d4) = (o(c, d, a), o(c, d, b), o(a, b, c), o(a, b, d));
    d1 * d2 < -1e-12 && d3 * d4 < -1e-12
}

/// The box a text primitive covers (its anchor and rotation taken into account).
pub fn text_box(p: &Prim) -> Option<BoxR> {
    let Shape::Text {
        x,
        y,
        text,
        size,
        anchor,
        rotation,
    } = &p.shape
    else {
        return None;
    };
    let w = Std::text_width(text, *size);
    let start = match anchor.unwrap_or("start") {
        "middle" => -w / 2.0,
        "end" => -w,
        _ => 0.0,
    };
    // Descenders below the baseline, a little room above the cap height.
    let (lo, hi) = (-size * 0.2, size * 1.05);
    let r = rotation.unwrap_or(0.0);
    if (r - 90.0).abs() < 1e-6 {
        Some(BoxR::new(x - hi, y + start, x - lo, y + start + w))
    } else if r.abs() < 1e-6 {
        Some(BoxR::new(x + start, y + lo, x + start + w, y + hi))
    } else {
        let (s, c) = r.to_radians().sin_cos();
        let pts = [[start, lo], [start + w, lo], [start, hi], [start + w, hi]]
            .map(|[u, v]| [x + u * c - v * s, y + u * s + v * c]);
        let xs = pts.map(|p| p[0]);
        let ys = pts.map(|p| p[1]);
        Some(BoxR::new(
            xs.iter().cloned().fold(f64::MAX, f64::min),
            ys.iter().cloned().fold(f64::MAX, f64::min),
            xs.iter().cloned().fold(f64::MIN, f64::max),
            ys.iter().cloned().fold(f64::MIN, f64::max),
        ))
    }
}

/// A leadered label's candidate: where its leader bends (`elbow`), which way its shoulder runs,
/// and the label box there.
#[derive(Clone, Copy, Debug)]
pub struct Spot {
    pub elbow: Xy,
    /// The shoulder runs right (else left).
    pub right: bool,
    pub label: BoxR,
}

/// A leadered label to place: the view it belongs to (its box and name), the point its leader
/// heads for, the label's box at an elbow, and where the leader's arrow lands for an elbow.
pub struct Request<'a> {
    pub owner: &'a str,
    pub view: &'a str,
    pub view_box: BoxR,
    pub anchor: Xy,
    pub label: &'a dyn Fn(Xy, bool) -> BoxR,
    pub tip: &'a dyn Fn(Xy) -> Xy,
}

/// The free spot for a leadered label nearest its anchor: elbows just outside each side of the
/// view at growing distances and offsets, each kept only when its label is free and its leader
/// crosses no text; the shortest leader with the fewest crossings wins.
pub fn place(occ: &Occupancy, req: &Request<'_>) -> Option<Spot> {
    let b = req.view_box;
    let a = req.anchor;
    let mut best: Option<(f64, Spot)> = None;
    let laterals = [
        0.0, 4.0, -4.0, 8.0, -8.0, 14.0, -14.0, 22.0, -22.0, 32.0, -32.0, 45.0, -45.0, 60.0, -60.0,
    ];
    for side in 0..4 {
        for k in 0..18 {
            let d = 5.0 + k as f64 * 4.0;
            for lat in laterals {
                let (elbow, rights): (Xy, [bool; 2]) = match side {
                    0 => (
                        [a[0] + lat, b.y1 + d],
                        if a[0] + lat >= b.center()[0] {
                            [true, false]
                        } else {
                            [false, true]
                        },
                    ),
                    1 => (
                        [a[0] + lat, b.y0 - d],
                        if a[0] + lat >= b.center()[0] {
                            [true, false]
                        } else {
                            [false, true]
                        },
                    ),
                    2 => ([b.x1 + d, a[1] + lat], [true, true]),
                    _ => ([b.x0 - d, a[1] + lat], [false, false]),
                };
                for (n, right) in rights.into_iter().enumerate() {
                    if n == 1 && rights[0] == rights[1] {
                        continue;
                    }
                    let label = (req.label)(elbow, right);
                    if !occ.free(&label, req.owner, 1.0, &[]) {
                        continue;
                    }
                    let tip = (req.tip)(elbow);
                    if !occ.leader_clear(tip, elbow, req.owner, req.view)
                        || label.grow(-0.3).crossed_by(tip, elbow)
                    {
                        continue;
                    }
                    let len = ((tip[0] - elbow[0]).powi(2) + (tip[1] - elbow[1]).powi(2)).sqrt();
                    let cost = len
                        + 0.4 * lat.abs()
                        + 25.0 * occ.crossings(tip, elbow, req.owner) as f64
                        + n as f64 * 2.0;
                    if best.as_ref().is_none_or(|(c, _)| cost < *c - 1e-9) {
                        best = Some((
                            cost,
                            Spot {
                                elbow,
                                right,
                                label,
                            },
                        ));
                    }
                }
            }
        }
    }
    best.map(|(_, s)| s)
}

/// The free spot for a box of size `w` × `h` nearest `preferred` (its center), on a grid
/// `step` apart over the frame, for which `ok` also holds.
pub fn nearest_free(
    occ: &Occupancy,
    owner: &str,
    [w, h]: Xy,
    preferred: Xy,
    step: f64,
    margin: f64,
    ok: &dyn Fn(&BoxR) -> bool,
) -> Option<BoxR> {
    let f = occ.frame;
    let (nx, ny) = (
        ((f.w() - w) / step).floor().max(0.0) as i64,
        ((f.h() - h) / step).floor().max(0.0) as i64,
    );
    let mut best: Option<(f64, BoxR)> = None;
    for i in 0..=nx {
        for j in 0..=ny {
            let c = [
                f.x0 + w / 2.0 + i as f64 * step,
                f.y0 + h / 2.0 + j as f64 * step,
            ];
            let d = (c[0] - preferred[0]).powi(2) + (c[1] - preferred[1]).powi(2);
            if best.as_ref().is_some_and(|(bd, _)| d >= *bd) {
                continue;
            }
            let b = BoxR::around(c, w, h);
            if occ.free(&b, owner, margin, &[]) && ok(&b) {
                best = Some((d, b));
            }
        }
    }
    best.map(|(_, b)| b)
}
