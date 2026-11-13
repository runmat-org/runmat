//! Drafting marks as primitives, sized by the sheet's standard (ISO 128, ISO 129-1, ISO 1101,
//! ISO 1302, ASME Y14.5): dimensions with filled arrowheads, leaders, feature control frames,
//! datum feature symbols, hatching, cutting-plane lines, detail circles, finish symbols,
//! balloons and the projection symbol.

use super::{Prim, Shape, Std, Xy};

pub fn prim(
    id: &str,
    kind: &'static str,
    layer: &'static str,
    style: &'static str,
    shape: Shape,
) -> Prim {
    Prim {
        id: id.to_owned(),
        kind,
        layer,
        style,
        shape,
        width: None,
    }
}

/// An annotation mark.
pub fn mark(id: &str, style: &'static str, shape: Shape) -> Prim {
    prim(id, "annotation", "Annotations", style, shape)
}

pub fn line(points: Vec<f64>) -> Shape {
    Shape::Polyline {
        points,
        closed: false,
    }
}

pub fn text(x: f64, y: f64, t: &str, size: f64, anchor: &'static str) -> Shape {
    Shape::Text {
        x,
        y,
        text: t.to_owned(),
        size,
        anchor: Some(anchor),
        rotation: None,
    }
}

pub fn len(a: Xy, b: Xy) -> f64 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

pub fn unit(a: Xy, b: Xy) -> Xy {
    let l = len(a, b).max(1e-9);
    [(b[0] - a[0]) / l, (b[1] - a[1]) / l]
}

impl Std {
    /// A filled arrowhead with its tip at `tip`, pointing along `dir` (unit).
    pub fn arrowhead(&self, id: &str, tip: Xy, dir: Xy) -> Prim {
        let n = [-dir[1], dir[0]];
        let back = [tip[0] - dir[0] * self.arrow, tip[1] - dir[1] * self.arrow];
        let h = self.arrow_width / 2.0;
        mark(
            id,
            "arrow",
            Shape::Polygon {
                rings: vec![vec![
                    tip[0],
                    tip[1],
                    back[0] + n[0] * h,
                    back[1] + n[1] * h,
                    back[0] - n[0] * h,
                    back[1] - n[1] * h,
                ]],
            },
        )
    }

    /// A linear dimension between feature points `a` and `b` along `axis` (0 horizontal, 1
    /// vertical), its line at `at` on the other axis. The figure sits over the line's middle
    /// (ISO: above it, turned with vertical lines; ASME: in a break, upright). When the span
    /// is too short for the figure and both arrows, the arrows go outside and point in, and the
    /// figure sits past the far end. Returns the primitives and where the figure sits.
    pub fn linear_dimension(
        &self,
        id: &str,
        a: Xy,
        b: Xy,
        axis: usize,
        at: f64,
        t: &str,
    ) -> (Vec<Prim>, Xy) {
        let other = 1 - axis;
        let p = |along: f64, across: f64| -> Xy {
            if axis == 0 {
                [along, across]
            } else {
                [across, along]
            }
        };
        let (a0, b0) = (a[axis].min(b[axis]), a[axis].max(b[axis]));
        let mut out = Vec::new();
        for q in [a, b] {
            let dir = if at >= q[other] { 1.0 } else { -1.0 };
            let (s, e) = (
                p(q[axis], q[other] + dir * self.gap),
                p(q[axis], at + dir * self.over),
            );
            out.push(mark(id, "dimink", line(vec![s[0], s[1], e[0], e[1]])));
        }
        let along: Xy = if axis == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
        let tw = Std::text_width(t, self.text);
        // The figure's extent along the line: its width, or its height when it stands upright
        // on a vertical ASME dimension.
        let upright = axis == 1 && self.unidirectional;
        let extent = if upright { self.text + 1.0 } else { tw };
        let fits = b0 - a0 >= extent + 2.0 * self.arrow + 2.0;
        let mid = (a0 + b0) / 2.0;
        let text_along = if fits {
            mid
        } else {
            b0 + self.arrow + 2.0 + extent / 2.0
        };
        let inline = self.unidirectional;
        if fits {
            if inline {
                // A break in the line for the figure.
                let half = extent / 2.0 + 1.0;
                for (s, e) in [(a0, mid - half), (mid + half, b0)] {
                    let (s, e) = (p(s, at), p(e, at));
                    out.push(mark(id, "dimink", line(vec![s[0], s[1], e[0], e[1]])));
                }
            } else {
                let (s, e) = (p(a0, at), p(b0, at));
                out.push(mark(id, "dimink", line(vec![s[0], s[1], e[0], e[1]])));
            }
            out.push(self.arrowhead(id, p(a0, at), [-along[0], -along[1]]));
            out.push(self.arrowhead(id, p(b0, at), along));
        } else {
            // Arrows outside, pointing in, the line running past both ends.
            let out_by = self.arrow + 2.0;
            let (s, e) = (p(a0 - out_by, at), p(b0 + out_by, at));
            out.push(mark(id, "dimink", line(vec![s[0], s[1], e[0], e[1]])));
            out.push(self.arrowhead(id, p(a0, at), along));
            out.push(self.arrowhead(id, p(b0, at), [-along[0], -along[1]]));
        }
        let (shape, at_text) = if axis == 0 {
            let y = if inline {
                at - self.text / 2.0
            } else {
                at + 1.0
            };
            (text(text_along, y, t, self.text, "middle"), [text_along, y])
        } else if upright {
            let x = at;
            let y = text_along - self.text / 2.0;
            (text(x, y, t, self.text, "middle"), [x, y])
        } else {
            let x = at - 1.0;
            (
                Shape::Text {
                    x,
                    y: text_along,
                    text: t.to_owned(),
                    size: self.text,
                    anchor: Some("middle"),
                    rotation: Some(90.0),
                },
                [x, text_along],
            )
        };
        out.push(mark(id, "dimtext", shape));
        (out, at_text)
    }

    /// A leader: an arrow on `tip`, a line to `elbow`, a shoulder, and `t` on the shoulder.
    pub fn leader(&self, id: &str, tip: Xy, elbow: Xy, right: bool, t: &str) -> Vec<Prim> {
        let s = if right { 1.0 } else { -1.0 };
        let shoulder = [elbow[0] + s * 3.0, elbow[1]];
        let d = unit(elbow, tip);
        vec![
            mark(
                id,
                "dimink",
                line(vec![
                    shoulder[0],
                    shoulder[1],
                    elbow[0],
                    elbow[1],
                    tip[0] - d[0] * self.arrow * 0.5,
                    tip[1] - d[1] * self.arrow * 0.5,
                ]),
            ),
            self.arrowhead(id, tip, d),
            mark(
                id,
                "dimtext",
                text(
                    shoulder[0] + s * 1.0,
                    elbow[1] - self.text / 2.0,
                    t,
                    self.text,
                    if right { "start" } else { "end" },
                ),
            ),
        ]
    }

    /// A leadered label's box: `t` on the shoulder at `elbow`.
    pub fn leader_box(&self, elbow: Xy, right: bool, t: &str) -> super::solver::BoxR {
        let w = Std::text_width(t, self.text) + 4.0;
        let (x0, x1) = if right {
            (elbow[0], elbow[0] + w)
        } else {
            (elbow[0] - w, elbow[0])
        };
        super::solver::BoxR::new(
            x0,
            elbow[1] - self.text * 0.75,
            x1,
            elbow[1] + self.text * 0.75,
        )
    }

    /// A feature control frame's cell widths (`∥|0.05|A`): the symbol cell square, the rest fit their text.
    pub fn frame_widths(&self, cells: &[&str]) -> Vec<f64> {
        let h = 2.0 * self.text;
        cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if i == 0 {
                    h
                } else {
                    h.max(Std::text_width(c, self.text) + 3.0)
                }
            })
            .collect()
    }

    /// An ISO 1101 / ASME Y14.5 feature control frame, its lower-left corner at `at`.
    pub fn control_frame(&self, id: &str, at: Xy, cells: &[&str]) -> Vec<Prim> {
        let h = 2.0 * self.text;
        let mut out = Vec::new();
        let mut x = at[0];
        for (i, (cell, w)) in cells.iter().zip(self.frame_widths(cells)).enumerate() {
            out.push(mark(id, "dimink", Shape::Rect { x, y: at[1], w, h }));
            let glyph = if i == 0 {
                self.symbol(id, cell, [x + w / 2.0, at[1] + h / 2.0], self.text)
            } else {
                None
            };
            match glyph {
                // Symbols the sheet font may lack draw as lines; the symbol stays as text too
                // (`glyphtext`), which exports write and the sheet leaves to the lines.
                Some(g) => {
                    out.extend(g.into_iter().map(|mut p| {
                        p.kind = "glyph";
                        p
                    }));
                    let mut t = mark(
                        id,
                        "glyphtext",
                        text(
                            x + w / 2.0,
                            at[1] + (h - self.text) / 2.0 + 0.4,
                            cell,
                            self.text,
                            "middle",
                        ),
                    );
                    t.kind = "glyph";
                    out.push(t);
                }
                None => out.push(mark(
                    id,
                    "dimtext",
                    text(
                        x + w / 2.0,
                        at[1] + (h - self.text) / 2.0 + 0.4,
                        cell,
                        self.text,
                        "middle",
                    ),
                )),
            }
            x += w;
        }
        out
    }

    /// A geometric characteristic drawn as lines about `c`, `h` tall (ISO 7083), or none for text.
    fn symbol(&self, id: &str, symbol: &str, c: Xy, h: f64) -> Option<Vec<Prim>> {
        let l = |pts: Vec<f64>| mark(id, "dimink", line(pts));
        let circle = |r: f64| {
            mark(
                id,
                "dimink",
                Shape::Circle {
                    x: c[0],
                    y: c[1],
                    r,
                },
            )
        };
        let [x, y] = c;
        let s = h / 2.0;
        Some(match symbol {
            "∥" => vec![
                l(vec![x - s * 0.9, y - s, x - s * 0.1, y + s]),
                l(vec![x + s * 0.1, y - s, x + s * 0.9, y + s]),
            ],
            "⊥" => vec![
                l(vec![x, y - s, x, y + s]),
                l(vec![x - s, y - s, x + s, y - s]),
            ],
            "⏤" => vec![l(vec![x - s, y, x + s, y])],
            "⏥" => vec![l(vec![
                x - s * 0.7,
                y + s * 0.5,
                x + s,
                y + s * 0.5,
                x + s * 0.7,
                y - s * 0.5,
                x - s,
                y - s * 0.5,
                x - s * 0.7,
                y + s * 0.5,
            ])],
            "∠" => vec![l(vec![
                x + s,
                y + s * 0.6,
                x - s,
                y - s * 0.7,
                x + s,
                y - s * 0.7,
            ])],
            "○" => vec![circle(s * 0.8)],
            "⌖" => vec![
                circle(s * 0.6),
                l(vec![x - s, y, x + s, y]),
                l(vec![x, y - s, x, y + s]),
            ],
            "◎" => vec![circle(s * 0.8), circle(s * 0.4)],
            _ => return None,
        })
    }

    /// A datum feature symbol: a filled triangle on the outline at `on`, a stem `away` (unit,
    /// out of the material) and a square holding the letter. Returns the primitives and the
    /// square's box.
    pub fn datum_symbol(
        &self,
        id: &str,
        on: Xy,
        away: Xy,
        letter: &str,
    ) -> (Vec<Prim>, super::solver::BoxR) {
        let bx = 2.0 * self.text;
        let n = [-away[1], away[0]];
        let tri = 1.8;
        let stem_end = [on[0] + away[0] * (tri + 4.0), on[1] + away[1] * (tri + 4.0)];
        let c = [
            stem_end[0] + away[0] * bx / 2.0,
            stem_end[1] + away[1] * bx / 2.0,
        ];
        let prims = vec![
            mark(
                id,
                "arrow",
                Shape::Polygon {
                    rings: vec![vec![
                        on[0] + n[0] * tri,
                        on[1] + n[1] * tri,
                        on[0] - n[0] * tri,
                        on[1] - n[1] * tri,
                        on[0] + away[0] * tri * 1.6,
                        on[1] + away[1] * tri * 1.6,
                    ]],
                },
            ),
            mark(
                id,
                "dimink",
                line(vec![
                    on[0] + away[0] * tri * 1.6,
                    on[1] + away[1] * tri * 1.6,
                    stem_end[0],
                    stem_end[1],
                ]),
            ),
            mark(
                id,
                "dimink",
                Shape::Rect {
                    x: c[0] - bx / 2.0,
                    y: c[1] - bx / 2.0,
                    w: bx,
                    h: bx,
                },
            ),
            mark(
                id,
                "dimtext",
                text(
                    c[0],
                    c[1] - self.text / 2.0 + 0.3,
                    letter,
                    self.text,
                    "middle",
                ),
            ),
        ];
        (
            prims,
            super::solver::BoxR::around(c, bx, bx).union(&super::solver::BoxR::around(
                on,
                2.0 * tri,
                2.0 * tri,
            )),
        )
    }

    /// An ISO 1302 surface texture symbol standing at `point` (the leader's end), its value over
    /// the long leg, and its leader to `tip`.
    pub fn surface_finish(&self, id: &str, tip: Xy, point: Xy, t: &str) -> Vec<Prim> {
        let h = 1.4 * self.text;
        let tan30 = (std::f64::consts::PI / 6.0).tan();
        let bar = Std::text_width(t, self.text) + 2.0;
        let [x, y] = point;
        let d = unit(point, tip);
        vec![
            mark(
                id,
                "dimink",
                line(vec![
                    point[0],
                    point[1],
                    tip[0] - d[0] * self.arrow * 0.5,
                    tip[1] - d[1] * self.arrow * 0.5,
                ]),
            ),
            self.arrowhead(id, tip, d),
            mark(
                id,
                "dimink",
                line(vec![
                    x - h * tan30,
                    y + h,
                    x,
                    y,
                    x + 2.0 * h * tan30,
                    y + 2.0 * h,
                    x + 2.0 * h * tan30 + bar,
                    y + 2.0 * h,
                ]),
            ),
            mark(
                id,
                "dimtext",
                text(
                    x + 2.0 * h * tan30 + 1.0,
                    y + h + 0.6,
                    t,
                    self.text,
                    "start",
                ),
            ),
        ]
    }

    /// The finish symbol's box when it stands at `point`.
    pub fn finish_box(&self, point: Xy, t: &str) -> super::solver::BoxR {
        let h = 1.4 * self.text;
        let tan30 = (std::f64::consts::PI / 6.0).tan();
        let bar = Std::text_width(t, self.text) + 2.0;
        super::solver::BoxR::new(
            point[0] - h * tan30,
            point[1],
            point[0] + 2.0 * h * tan30 + bar,
            point[1] + 2.0 * h + 0.5,
        )
    }

    /// A balloon centred on `at` with its number, its leader from the circle to a dot on `on`.
    pub fn balloon(&self, id: &str, on: Xy, at: Xy, number: &str) -> Vec<Prim> {
        let r = self.balloon_radius();
        let d = unit(at, on);
        let start = [at[0] + d[0] * r, at[1] + d[1] * r];
        vec![
            mark(
                id,
                "dimink",
                Shape::Circle {
                    x: at[0],
                    y: at[1],
                    r,
                },
            ),
            mark(
                id,
                "dimtext",
                text(at[0], at[1] - self.text / 2.0, number, self.text, "middle"),
            ),
            mark(id, "dimink", line(vec![start[0], start[1], on[0], on[1]])),
            mark(
                id,
                "arrow",
                Shape::Circle {
                    x: on[0],
                    y: on[1],
                    r: 0.6,
                },
            ),
        ]
    }

    pub fn balloon_radius(&self) -> f64 {
        self.text * 10.0 / 7.0
    }

    /// A cutting-plane line from `a` to `b` (ISO 128-44): thin chain between thick ends, arrows
    /// on the ends pointing the way the section looks (`toward`), and the letter at each arrow.
    pub fn cutting_plane(&self, id: &str, a: Xy, b: Xy, toward: Xy, letter: &str) -> Vec<Prim> {
        let u = unit(a, b);
        let mut out = vec![prim(
            id,
            "view",
            "Views",
            "center",
            line(vec![a[0], a[1], b[0], b[1]]),
        )];
        for (p, s) in [(a, 1.0), (b, -1.0)] {
            let q = [p[0] + u[0] * s * 6.0, p[1] + u[1] * s * 6.0];
            out.push(prim(
                id,
                "view",
                "Views",
                "thick",
                line(vec![p[0], p[1], q[0], q[1]]),
            ));
            let mid = [p[0] + u[0] * s * 3.0, p[1] + u[1] * s * 3.0];
            let tail = [mid[0] + toward[0] * 7.0, mid[1] + toward[1] * 7.0];
            out.push(prim(
                id,
                "view",
                "Views",
                "dimink",
                line(vec![mid[0], mid[1], tail[0], tail[1]]),
            ));
            let mut head = self.arrowhead(id, tail, toward);
            head.kind = "view";
            head.layer = "Views";
            out.push(head);
            let size = self.text * 1.4;
            out.push(prim(
                id,
                "label",
                "Views",
                "dimtext",
                text(
                    tail[0] + toward[0] * 2.5 - u[0] * s * 3.0,
                    tail[1] + toward[1] * 2.5 - size / 2.0,
                    letter,
                    size,
                    "middle",
                ),
            ));
        }
        out
    }

    /// A 45° hatch (ISO 128-50) over regions `rings` (flat x, y), even-odd, `spacing` apart.
    pub fn hatch(&self, id: &str, rings: &[Vec<f64>], spacing: f64) -> Vec<Prim> {
        let turn = |x: f64, y: f64| -> Xy {
            [
                (x + y) / std::f64::consts::SQRT_2,
                (y - x) / std::f64::consts::SQRT_2,
            ]
        };
        let back = |s: f64, t: f64| -> Xy {
            [
                (s - t) / std::f64::consts::SQRT_2,
                (s + t) / std::f64::consts::SQRT_2,
            ]
        };
        let mut segs: Vec<(Xy, Xy)> = Vec::new();
        for r in rings {
            for i in 0..(r.len() / 2).saturating_sub(1) {
                segs.push((
                    turn(r[2 * i], r[2 * i + 1]),
                    turn(r[2 * i + 2], r[2 * i + 3]),
                ));
            }
        }
        if segs.is_empty() {
            return Vec::new();
        }
        let t0 = segs
            .iter()
            .flat_map(|(a, b)| [a[1], b[1]])
            .fold(f64::MAX, f64::min);
        let t1 = segs
            .iter()
            .flat_map(|(a, b)| [a[1], b[1]])
            .fold(f64::MIN, f64::max);
        let mut out = Vec::new();
        let mut t = (t0 / spacing).ceil() * spacing + spacing / 2.0;
        while t < t1 {
            let mut xs: Vec<f64> = segs
                .iter()
                .filter(|(a, b)| (a[1] > t) != (b[1] > t))
                .map(|(a, b)| a[0] + (t - a[1]) / (b[1] - a[1]) * (b[0] - a[0]))
                .collect();
            xs.sort_by(f64::total_cmp);
            for k in (0..xs.len().saturating_sub(1)).step_by(2) {
                let (p, q) = (back(xs[k], t), back(xs[k + 1], t));
                out.push(prim(
                    id,
                    "view",
                    "Views",
                    "hatch",
                    line(vec![p[0], p[1], q[0], q[1]]),
                ));
            }
            t += spacing;
        }
        out
    }

    /// The ISO 5456-2 projection symbol with its lower-left at `at`: a truncated cone's side
    /// view and its end view, the end view on the right for third angle (left for first).
    pub fn projection_symbol(&self, at: Xy, third: bool) -> Vec<Prim> {
        let h = 2.0 * self.text;
        let [x, y] = at;
        let cone = if third { x } else { x + 10.0 };
        let ends = if third {
            x + 12.0 + h / 2.0
        } else {
            x + h / 2.0
        };
        let cy = y + h / 2.0;
        let s = |style: &'static str, shape: Shape| prim("title", "frame", "Frame", style, shape);
        vec![
            s(
                "dimink",
                Shape::Polyline {
                    closed: true,
                    points: vec![
                        cone,
                        cy - h / 4.0,
                        cone + 9.0,
                        cy - h / 2.0,
                        cone + 9.0,
                        cy + h / 2.0,
                        cone,
                        cy + h / 4.0,
                    ],
                },
            ),
            s(
                "dimink",
                Shape::Circle {
                    x: ends,
                    y: cy,
                    r: h / 2.0,
                },
            ),
            s(
                "dimink",
                Shape::Circle {
                    x: ends,
                    y: cy,
                    r: h / 4.0,
                },
            ),
            s("center", line(vec![x - 1.5, cy, ends + h / 2.0 + 1.5, cy])),
        ]
    }
}

/// `lines` (flat x, y) clipped to the disc of radius `r` about `c`.
pub fn clip_to_circle(lines: &[Vec<f64>], c: Xy, r: f64) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let inside = |x: f64, y: f64| ((x - c[0]).powi(2) + (y - c[1]).powi(2)).sqrt() <= r;
    for l in lines {
        let mut run: Vec<f64> = Vec::new();
        for i in (0..l.len().saturating_sub(3)).step_by(2) {
            let (x0, y0, x1, y1) = (l[i], l[i + 1], l[i + 2], l[i + 3]);
            let n = (((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt() / (r / 40.0))
                .ceil()
                .max(1.0) as usize;
            for k in 0..n {
                let f = |t: f64| [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t];
                let (a, b) = (f(k as f64 / n as f64), f((k + 1) as f64 / n as f64));
                if inside((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0) {
                    if run.is_empty() {
                        run.extend(a);
                    }
                    run.extend(b);
                } else if !run.is_empty() {
                    out.push(std::mem::take(&mut run));
                }
            }
        }
        if run.len() >= 4 {
            out.push(run);
        }
    }
    out
}
