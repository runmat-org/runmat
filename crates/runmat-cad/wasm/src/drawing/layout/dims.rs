//! Linear, baseline, chain and ordinate dimensions, to ISO 129-1 and ASME Y14.5 spacing:
//!
//! - Each view's horizontal dimensions go on its `h_side`, vertical ones on its `v_side`
//!   ([`super::views`] picks them away from the view's parent, so dimensions don't fill the gap
//!   between views).
//! - The first dimension line sits the standard's first offset from the outline and each further
//!   row one step beyond it. Shorter dimensions sit nearer the view; dimensions that don't overlap
//!   share a row, so a chain reads as one line and a baseline set stacks a row per dimension.
//!   Ordinate dimensions sit nearest the view, each figure at the end of its extension line.
//! - A dimension measuring what another already measures (the same two references and value,
//!   in any view of the model) is drawn once; the repeat is a finding.
//! - A dimension moved by hand keeps its line where it was put.

use super::marks::{line, mark, text};
use super::views::{Side, ViewPlan};
use super::{Ctx, PlacedDimension, Shape, Std, Xy};
use serde_json::Value;
use std::collections::BTreeMap;

pub struct Dim {
    pub id: String,
    pub view: usize,
    pub ordinate: bool,
    /// 0: measured horizontally, 1: vertically.
    pub axis: usize,
    /// The two feature points, relative to the view's centre (sheet mm).
    pub p: Xy,
    pub q: Xy,
    pub corners: [Option<Vec<Xy>>; 2],
    pub text: String,
    pub measurement: f64,
    pub at: Option<Xy>,
    pub row: usize,
    /// An ordinate set's key (its zero), so the zero's figure draws once.
    pub zero: Option<String>,
}

pub struct Plans {
    pub dims: Vec<Dim>,
    /// View → room for its dimensions (below, above, left, right).
    pub rooms: BTreeMap<String, [f64; 4]>,
    /// (view, side) → each row's offset from the outline.
    offsets: BTreeMap<(usize, usize), Vec<f64>>,
}

fn side_index(s: Side) -> usize {
    match s {
        Side::Below => 0,
        Side::Above => 1,
        Side::Left => 2,
        Side::Right => 3,
    }
}

/// The axis a dimension measures along: across a planar reference seen edge-on, else the larger
/// of the offsets between its points.
fn measured_axis(corners: &[Option<Vec<Xy>>; 2], p: Xy, q: Xy) -> usize {
    for c in corners.iter().flatten() {
        let w = c.iter().map(|v| v[0]).fold(f64::MIN, f64::max) - c.iter().map(|v| v[0]).fold(f64::MAX, f64::min);
        let h = c.iter().map(|v| v[1]).fold(f64::MIN, f64::max) - c.iter().map(|v| v[1]).fold(f64::MAX, f64::min);
        if w < 1e-6 && h > 1e-6 {
            return 0;
        }
        if h < 1e-6 && w > 1e-6 {
            return 1;
        }
    }
    usize::from((q[1] - p[1]).abs() >= (q[0] - p[0]).abs())
}

/// Plans every linear-family dimension on the sheet's views: points, rows, and each view's room.
pub fn plan(ctx: &mut Ctx<'_>, views: &[ViewPlan]) -> Plans {
    let data = ctx.data;
    let mut dims: Vec<Dim> = Vec::new();
    // What each drawn dimension measures: (model, both references, value) → its id.
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let all_views = data["document"]["views"].as_array().cloned().unwrap_or_default();
    for a in data["document"]["annotations"].as_array().into_iter().flatten() {
        let kind = &a["kind"];
        let style = kind["style"].as_str().unwrap_or("");
        if kind["annotation"] != "dimension" || !matches!(style, "linear" | "ordinate" | "baseline" | "chain") {
            continue;
        }
        let id = a["id"].as_str().unwrap_or("").to_owned();
        let model = all_views.iter().find(|v| v["name"] == a["view"]).map(|v| v["model"]["file"].to_string()).unwrap_or_default();
        let Some(vi) = views.iter().position(|v| a["view"].as_str() == Some(v.name.as_str())) else {
            // On another sheet: still claims what it measures.
            continue;
        };
        let v = &views[vi];
        let refs = kind["references"].as_array().cloned().unwrap_or_default();
        let n = refs.len();
        let pairs: Vec<(usize, usize)> = match style {
            "linear" | "ordinate" if n == 2 => vec![(0, 1)],
            "baseline" => (1..n).map(|k| (0, k)).collect(),
            "chain" => (1..n).map(|k| (k - 1, k)).collect(),
            _ => Vec::new(),
        };
        let anchors = &data["anchors"][&id];
        let points: Vec<Option<Xy>> = (0..n).map(|i| model_point(&anchors[i]).map(|c| v.model_local(c))).collect();
        let corners: Vec<Option<Vec<Xy>>> = (0..n).map(|i| corners_of(&data["extents"][&id][i], v)).collect();
        let texts: Vec<String> = ctx.text_of(&id).split(" · ").map(str::to_owned).collect();
        let at = super::placed_at(a);
        for (k, (i, j)) in pairs.into_iter().enumerate() {
            let (Some(p), Some(q)) = (points[i], points[j]) else { continue };
            if super::marks::len(p, q) <= 1e-6 {
                continue;
            }
            let pair = [corners[i].clone(), corners[j].clone()];
            let axis = measured_axis(&pair, p, q);
            let measurement = (q[axis] - p[axis]).abs() / v.scale;
            let mut key_refs = [refs[i].to_string(), refs[j].to_string()];
            key_refs.sort();
            let key = format!("{model}|{}|{}|{:.3}", key_refs[0], key_refs[1], measurement);
            let label = texts.get(k).or(texts.first()).cloned().unwrap_or_else(|| id.clone());
            if seen.contains_key(&key) {
                ctx.finding("duplicate_dimension", &id, format!("The {label} dimension repeats one that measures the same two edges; it is drawn once"));
                continue;
            }
            seen.insert(key, id.clone());
            let at = at.map(|a| if axis == 0 { [a[0], a[1] + k as f64 * ctx.std.step] } else { [a[0] + k as f64 * ctx.std.step, a[1]] });
            dims.push(Dim {
                id: id.clone(),
                view: vi,
                ordinate: style == "ordinate",
                axis,
                p,
                q,
                corners: pair,
                text: label,
                measurement,
                at,
                row: 0,
                zero: (style == "ordinate").then(|| format!("{vi}|{}|{axis}", refs[0])),
            });
        }
    }
    // Rows per view and side: ordinates nearest the outline, then linear rows packed
    // shortest first, each in the nearest row where it overlaps nothing.
    let st = ctx.std.clone();
    let mut rooms: BTreeMap<String, [f64; 4]> = BTreeMap::new();
    let mut offsets = BTreeMap::new();
    for (vi, v) in views.iter().enumerate() {
        let mut room = [0.0f64; 4];
        for axis in 0..2 {
            let side = if axis == 0 { v.h_side } else { v.v_side };
            let si = side_index(side);
            let ords: Vec<usize> = (0..dims.len()).filter(|&d| dims[d].view == vi && dims[d].axis == axis && dims[d].ordinate && dims[d].at.is_none()).collect();
            let lins: Vec<usize> = (0..dims.len()).filter(|&d| dims[d].view == vi && dims[d].axis == axis && !dims[d].ordinate && dims[d].at.is_none()).collect();
            let figure = |d: &Dim| if axis == 1 && st.unidirectional || axis == 0 && !st.unidirectional { Std::text_width(&d.text, st.text) } else { st.text };
            let ord_band = ords.iter().map(|&d| figure(&dims[d])).fold(0.0, f64::max);
            // A section's cutting line runs past the view's sides with its arrows and letters:
            // the vertical dimensions start beyond them.
            let cut = axis == 1 && views.iter().any(|w| w.kind == "section" && w.parent.as_deref() == Some(v.name.as_str()));
            let base = st.first + if ords.is_empty() { 0.0 } else { ord_band + 4.0 } + if cut { 6.0 + st.text * 1.4 } else { 0.0 };
            // Each dimension's extent along its axis, its figure included.
            let spans: Vec<(f64, f64)> = lins
                .iter()
                .map(|&d| {
                    let x = &dims[d];
                    let (lo, hi) = (x.p[axis].min(x.q[axis]), x.p[axis].max(x.q[axis]));
                    let extent = if axis == 1 && st.unidirectional { st.text + 1.0 } else { Std::text_width(&x.text, st.text) };
                    if hi - lo >= extent + 2.0 * st.arrow + 2.0 { (lo, hi) } else { (lo - st.arrow - 2.0, hi + st.arrow + 3.0 + extent) }
                })
                .collect();
            let rows = pack_rows(&spans);
            for (n, &d) in lins.iter().enumerate() {
                dims[d].row = rows[n];
            }
            let count = rows.iter().copied().max().map_or(0, |m| m + 1);
            // ASME's upright figures on vertical dimensions need their width between rows.
            let mut offs = Vec::new();
            let mut off = base;
            for r in 0..count {
                offs.push(off);
                let widest = lins.iter().filter(|&&d| dims[d].row == r).map(|&d| Std::text_width(&dims[d].text, st.text)).fold(0.0, f64::max);
                off += if axis == 1 && st.unidirectional { st.step.max(widest + 3.0) } else { st.step };
            }
            let last_figure = if axis == 1 && st.unidirectional { lins.iter().map(|&d| Std::text_width(&dims[d].text, st.text) / 2.0).fold(0.0, f64::max) } else { st.text };
            let need = if count > 0 { offs[count - 1] + last_figure + 1.5 } else if !ords.is_empty() { st.first + ord_band + 1.5 } else { 0.0 };
            room[si] = room[si].max(need);
            offsets.insert((vi, si), offs);
        }
        rooms.insert(v.name.clone(), room);
    }
    Plans { dims, rooms, offsets }
}

fn model_point(v: &Value) -> Option<[f64; 3]> {
    let f = super::floats(v);
    (f.len() == 3).then(|| [f[0], f[1], f[2]])
}

fn corners_of(e: &Value, v: &ViewPlan) -> Option<Vec<Xy>> {
    let (lo, hi) = (model_point(&e[0])?, model_point(&e[1])?);
    let mut out = Vec::new();
    for x in [lo[0], hi[0]] {
        for y in [lo[1], hi[1]] {
            for z in [lo[2], hi[2]] {
                out.push(v.model_local([x, y, z]));
            }
        }
    }
    Some(out)
}

/// Rows for spans: shortest first, each in the nearest row where it overlaps nothing (touching
/// spans share a row: a chain).
pub fn pack_rows(spans: &[(f64, f64)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..spans.len()).collect();
    order.sort_by(|&a, &b| (spans[a].1 - spans[a].0).total_cmp(&(spans[b].1 - spans[b].0)).then(a.cmp(&b)));
    let mut rows: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut out = vec![0; spans.len()];
    for i in order {
        let (lo, hi) = spans[i];
        let mut r = 0;
        while rows.get(r).is_some_and(|row| row.iter().any(|&(l, h)| lo < h - 1e-6 && hi > l + 1e-6)) {
            r += 1;
        }
        if rows.len() <= r {
            rows.resize(r + 1, Vec::new());
        }
        rows[r].push((lo, hi));
        out[i] = r;
    }
    out
}

/// An extension line's foot: `point` moved across to the end of its face's box nearest `side`.
fn foot(point: Xy, corners: &Option<Vec<Xy>>, side: Side) -> Xy {
    let Some(c) = corners.as_ref().filter(|c| !c.is_empty()) else { return point };
    let xs = c.iter().map(|v| v[0]);
    let ys = c.iter().map(|v| v[1]);
    match side {
        Side::Below => [point[0], ys.fold(f64::MAX, f64::min)],
        Side::Above => [point[0], ys.fold(f64::MIN, f64::max)],
        Side::Left => [xs.fold(f64::MAX, f64::min), point[1]],
        Side::Right => [xs.fold(f64::MIN, f64::max), point[1]],
    }
}

/// Draws the planned dimensions.
pub fn draw(ctx: &mut Ctx<'_>, views: &[ViewPlan], plans: Plans) {
    let st = ctx.std.clone();
    let mut zeros_drawn: Vec<String> = Vec::new();
    for d in &plans.dims {
        let v = &views[d.view];
        let o = v.origin;
        let sheet = |p: Xy| [p[0] + o[0], p[1] + o[1]];
        let g = v.geom_box();
        let side = match d.at {
            Some(at) if d.axis == 0 => if at[1] < g.y0 { Side::Below } else { Side::Above },
            Some(at) => if at[0] < g.x0 { Side::Left } else { Side::Right },
            None => if d.axis == 0 { v.h_side } else { v.v_side },
        };
        let shift = |c: &Option<Vec<Xy>>| c.as_ref().map(|c| c.iter().map(|p| sheet(*p)).collect::<Vec<_>>());
        let a = foot(sheet(d.p), &shift(&d.corners[0]), side);
        let b = foot(sheet(d.q), &shift(&d.corners[1]), side);
        let id = format!("annotation:{}", d.id);
        let si = side_index(side);
        let edge = match side {
            Side::Below => g.y0,
            Side::Above => g.y1,
            Side::Left => g.x0,
            Side::Right => g.x1,
        };
        let out_sign = if matches!(side, Side::Below | Side::Left) { -1.0 } else { 1.0 };
        if d.ordinate {
            let at = edge + out_sign * st.first;
            let mut prims = Vec::new();
            let zero_key = d.zero.clone().unwrap_or_default();
            let ends: Vec<(Xy, String)> = if zeros_drawn.contains(&zero_key) { vec![(b, d.text.clone())] } else { vec![(a, "0".into()), (b, d.text.clone())] };
            zeros_drawn.push(zero_key);
            for (f, t) in ends {
                let other = 1 - d.axis;
                let mut s = f;
                s[other] += out_sign * st.gap;
                let mut e = f;
                e[other] = at;
                prims.push(mark(&id, "dimink", line(vec![s[0], s[1], e[0], e[1]])));
                let w = Std::text_width(&t, st.text);
                let shape = if d.axis == 0 && !st.unidirectional {
                    // Turned with its extension line, reading from the right.
                    let y = at + out_sign * (1.0 + w / 2.0);
                    Shape::Text { x: e[0] + st.text / 2.0 - 0.3, y, text: t, size: st.text, anchor: Some("middle"), rotation: Some(90.0) }
                } else if d.axis == 0 {
                    text(e[0], if out_sign < 0.0 { at - 1.0 - st.text } else { at + 1.0 }, &t, st.text, "middle")
                } else {
                    let x = at + out_sign * 1.0;
                    text(x, e[1] - st.text / 2.0, &t, st.text, if out_sign < 0.0 { "end" } else { "start" })
                };
                prims.push(mark(&id, "dimtext", shape));
            }
            ctx.occ.add_prims(&prims, &id);
            ctx.push(prims);
            continue;
        }
        let at = match d.at {
            Some(p) => p[1 - d.axis],
            None => {
                let offs = plans.offsets.get(&(d.view, si)).cloned().unwrap_or_default();
                edge + out_sign * offs.get(d.row).copied().unwrap_or(st.first + d.row as f64 * st.step)
            }
        };
        let (prims, text_at) = st.linear_dimension(&id, a, b, d.axis, at, &d.text);
        ctx.occ.add_prims(&prims, &id);
        ctx.push(prims);
        ctx.out.placed.entry(id.clone()).or_insert(if d.axis == 0 { [a[0], at] } else { [at, a[1]] });
        ctx.out.dimensions.push(PlacedDimension {
            id: d.id.clone(),
            from: a,
            to: b,
            line: if d.axis == 0 { [a[0], at] } else { [at, a[1]] },
            rotation: if d.axis == 0 { 0.0 } else { std::f64::consts::FRAC_PI_2 },
            text: d.text.clone(),
            text_at,
            measurement: d.measurement,
        });
    }
}
