//! Building-drawing marks on a view: grid lines with their bubbles, level marks and door swings.
//!
//! - **Grids** (`grids`): each grid line runs across the view as a chain line, from just past the
//!   far side of the outline out past the view's dimension rows on its own side, where its label
//!   sits in a bubble. The row of dimensions between neighbouring grids is planned with the
//!   view's other dimensions ([`super::dims`]); [`bubble_room`] keeps room beyond it.
//! - **Level marks** (`level`): a thin line at the face's height from the view's side out to a
//!   target symbol (a circle with two opposite quarters filled), the label above the line and the
//!   elevation below it, on the side the mark is placed (else the view's vertical-dimension side).
//! - **Door swings** (`door_swing`): the leaf open 90° as a visible line and the quarter arc
//!   from its end to the closed position as a thin line.

use super::marks::{line, mark, text};
use super::solver::{BoxR, Class};
use super::views::{Side, ViewPlan};
use super::{floats, placed_at, Ctx, Prim, Shape, Std, Xy};
use serde_json::Value;

/// A grid bubble's radius (sheet mm): the label's height and a half again each side.
pub fn bubble_radius(st: &Std) -> f64 {
    st.text * 1.25
}

/// The room grid bubbles take beyond a view's dimension rows: the bubble and a gap.
pub fn bubble_room(st: &Std) -> f64 {
    2.0 * bubble_radius(st) + 3.0
}

fn annotations<'a>(ctx: &Ctx<'a>, what: &str) -> Vec<&'a Value> {
    ctx.doc()["annotations"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|a| a["kind"]["annotation"] == what)
        .collect()
}

/// A model point (mm, `[x, y, z]`) on the sheet in view `v`.
fn on_sheet(v: &ViewPlan, c: &Value) -> Option<Xy> {
    let f = floats(c);
    (f.len() == 3).then(|| v.model_to_sheet([f[0], f[1], f[2]]))
}

/// Draws the sheet's level marks and door swings (before the callouts, which keep clear of them).
pub fn draw(ctx: &mut Ctx<'_>, views: &[ViewPlan]) {
    let st = ctx.std.clone();
    let data = ctx.data;
    let mut prims: Vec<(String, Vec<Prim>, Option<BoxR>)> = Vec::new();
    for a in annotations(ctx, "level") {
        let Some(v) = views
            .iter()
            .find(|v| a["view"].as_str() == Some(v.name.as_str()))
        else {
            continue;
        };
        let aid = a["id"].as_str().unwrap_or("");
        let Some(p) = data["anchors"][aid]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|c| on_sheet(v, c))
        else {
            continue;
        };
        let id = format!("annotation:{aid}");
        let g = v.geom_box();
        let label = a["kind"]["label"].as_str().unwrap_or("");
        let elevation = ctx.text_of(aid);
        let width = Std::text_width(label, st.text).max(Std::text_width(&elevation, st.text));
        let (left, end) = match placed_at(a) {
            Some(at) => (at[0] < g.center()[0], at[0]),
            None => {
                let left = v.v_side == Side::Left;
                let room = v.room[if left { 2 } else { 3 }];
                (
                    left,
                    if left {
                        g.x0 - room - 12.0
                    } else {
                        g.x1 + room + 12.0
                    },
                )
            }
        };
        let start = if left { g.x0 - 1.5 } else { g.x1 + 1.5 };
        let mut out = vec![mark(&id, "dimink", line(vec![start, p[1], end, p[1]]))];
        out.extend(level_symbol(&id, [end, p[1]], 1.6));
        // The label above the line and the height below it, beyond the symbol.
        let (x, anchor) = if left {
            (end - 3.0, "end")
        } else {
            (end + 3.0, "start")
        };
        out.push(mark(
            &id,
            "dimtext",
            text(x, p[1] + 1.0, label, st.text, anchor),
        ));
        out.push(mark(
            &id,
            "dimtext",
            text(x, p[1] - 1.0 - st.text, &elevation, st.text, anchor),
        ));
        let b = if left {
            BoxR::new(
                x - width,
                p[1] - 1.0 - st.text,
                end + 2.0,
                p[1] + 1.0 + st.text,
            )
        } else {
            BoxR::new(
                end - 2.0,
                p[1] - 1.0 - st.text,
                x + width,
                p[1] + 1.0 + st.text,
            )
        };
        ctx.out.placed.insert(id.clone(), [end, p[1]]);
        prims.push((id, out, Some(b)));
    }
    for a in annotations(ctx, "door_swing") {
        let Some(v) = views
            .iter()
            .find(|v| a["view"].as_str() == Some(v.name.as_str()))
        else {
            continue;
        };
        let aid = a["id"].as_str().unwrap_or("");
        let swing = &data["swings"][aid];
        let (Some(h), Some(open)) = (on_sheet(v, &swing["hinge"]), on_sheet(v, &swing["open"]))
        else {
            continue;
        };
        let id = format!("annotation:{aid}");
        let arc: Vec<f64> = swing["arc"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| on_sheet(v, c))
            .flatten()
            .collect();
        prims.push((
            id.clone(),
            vec![
                mark(&id, "visible", line(vec![h[0], h[1], open[0], open[1]])),
                mark(&id, "dimink", line(arc)),
            ],
            None,
        ));
    }
    for (id, out, label) in prims {
        ctx.occ.add_prims(&out, &id);
        if let Some(b) = label {
            ctx.occ.add(b, &id, Class::Text);
        }
        ctx.push(out);
    }
}

/// Draws the sheet's grids, last, so each grid line leaves a gap at every text it crosses.
pub fn draw_grids(ctx: &mut Ctx<'_>, views: &[ViewPlan]) {
    let st = ctx.std.clone();
    let data = ctx.data;
    // The text already on the sheet (figures, labels, notes), which grid lines leave a gap for.
    let texts: Vec<BoxR> = ctx
        .out
        .primitives
        .iter()
        .filter_map(super::solver::text_box)
        .map(|b| b.grow(0.8))
        .collect();
    let mut prims: Vec<(String, Vec<Prim>, Option<BoxR>)> = Vec::new();
    for a in annotations(ctx, "grids") {
        let Some(v) = views
            .iter()
            .find(|v| a["view"].as_str() == Some(v.name.as_str()))
        else {
            continue;
        };
        let aid = a["id"].as_str().unwrap_or("");
        let id = format!("annotation:{aid}");
        let labels: Vec<&str> = a["kind"]["labels"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let g = v.geom_box();
        let r = bubble_radius(&st);
        let mut out = Vec::new();
        for (i, extent) in data["extents"][aid]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            let corners: Vec<Xy> = extent
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|c| on_sheet(v, c))
                .collect();
            let [lo, hi] = match corners.as_slice() {
                [lo, hi] => [*lo, *hi],
                _ => continue,
            };
            // Seen edge-on, a grid plane is a line across the view: vertical (a grid measured
            // left to right) or horizontal.
            let vertical = (hi[0] - lo[0]).abs() < (hi[1] - lo[1]).abs();
            let side = if vertical { v.h_side } else { v.v_side };
            let room = v.room[match side {
                Side::Below => 0,
                Side::Above => 1,
                Side::Left => 2,
                Side::Right => 3,
            }];
            let (from, centre) = if vertical {
                let x = (lo[0] + hi[0]) / 2.0;
                match side {
                    Side::Below => ([x, g.y1 + 3.0], [x, g.y0 - room + r + 1.5]),
                    _ => ([x, g.y0 - 3.0], [x, g.y1 + room - r - 1.5]),
                }
            } else {
                let y = (lo[1] + hi[1]) / 2.0;
                match side {
                    Side::Left => ([g.x1 + 3.0, y], [g.x0 - room + r + 1.5, y]),
                    _ => ([g.x0 - 3.0, y], [g.x1 + room - r - 1.5, y]),
                }
            };
            let toward = super::marks::unit(from, centre);
            let end = [centre[0] - toward[0] * r, centre[1] - toward[1] * r];
            // Broken where it would cross text (a dimension's figure, a room name).
            for (a, b) in around_text(from, end, &texts) {
                out.push(mark(&id, "center", line(vec![a[0], a[1], b[0], b[1]])));
            }
            out.push(mark(
                &id,
                "dimink",
                Shape::Circle {
                    x: centre[0],
                    y: centre[1],
                    r,
                },
            ));
            let label = labels.get(i).copied().unwrap_or("");
            out.push(mark(
                &id,
                "dimtext",
                text(
                    centre[0],
                    centre[1] - st.text / 2.0,
                    label,
                    st.text,
                    "middle",
                ),
            ));
            ctx.out.placed.entry(id.clone()).or_insert(centre);
        }
        prims.push((id, out, None));
    }
    for (id, out, label) in prims {
        ctx.occ.add_prims(&out, &id);
        if let Some(b) = label {
            ctx.occ.add(b, &id, Class::Text);
        }
        ctx.push(out);
    }
}

/// The level target: a circle of radius `r` at `c` with its upper-right and lower-left quarters
/// filled.
fn level_symbol(id: &str, c: Xy, r: f64) -> Vec<Prim> {
    let quarter = |from: f64| {
        let mut ring = vec![c[0], c[1]];
        for k in 0..=8 {
            let a = from + std::f64::consts::FRAC_PI_2 * k as f64 / 8.0;
            ring.extend([c[0] + r * a.cos(), c[1] + r * a.sin()]);
        }
        mark(id, "arrow", Shape::Polygon { rings: vec![ring] })
    };
    vec![
        mark(
            id,
            "dimink",
            Shape::Circle {
                x: c[0],
                y: c[1],
                r,
            },
        ),
        quarter(0.0),
        quarter(std::f64::consts::PI),
    ]
}

/// The pieces of the straight line `a`-`b` (horizontal or vertical) outside every box in
/// `boxes` that it crosses.
fn around_text(a: Xy, b: Xy, boxes: &[BoxR]) -> Vec<(Xy, Xy)> {
    let vertical = (b[0] - a[0]).abs() < (b[1] - a[1]).abs();
    let (k, c) = if vertical { (1, a[0]) } else { (0, a[1]) };
    let (lo, hi) = (a[k].min(b[k]), a[k].max(b[k]));
    let mut cuts: Vec<(f64, f64)> = boxes
        .iter()
        .filter(|x| {
            if vertical {
                x.x0 <= c && c <= x.x1
            } else {
                x.y0 <= c && c <= x.y1
            }
        })
        .map(|x| if vertical { (x.y0, x.y1) } else { (x.x0, x.x1) })
        .filter(|(s, e)| *e > lo && *s < hi)
        .collect();
    cuts.sort_by(|p, q| p.0.total_cmp(&q.0));
    let point = |t: f64| if vertical { [c, t] } else { [t, c] };
    let mut out = Vec::new();
    let mut at = lo;
    for (s, e) in cuts {
        if s > at {
            out.push((point(at), point(s)));
        }
        at = at.max(e);
    }
    if at < hi {
        out.push((point(at), point(hi)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_line_leaves_a_gap_where_it_crosses_text() {
        let boxes = [
            BoxR::new(-2.0, 10.0, 2.0, 14.0),
            BoxR::new(5.0, 0.0, 9.0, 100.0),
        ];
        let pieces = around_text([0.0, 0.0], [0.0, 30.0], &boxes);
        assert_eq!(
            pieces,
            vec![([0.0, 0.0], [0.0, 10.0]), ([0.0, 14.0], [0.0, 30.0])]
        );
        // A line that crosses nothing stays whole.
        assert_eq!(around_text([20.0, 0.0], [20.0, 5.0], &boxes).len(), 1);
    }
}
