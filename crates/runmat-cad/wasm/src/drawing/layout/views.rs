//! Views on the sheet: each view's lines from the compile (`hlr`), where it goes, and its label.
//!
//! - **Alignment.** A projected view stays in line with its parent (ISO 5456-2, ASME Y14.3):
//!   only its distance along the projection direction is free.
//! - **Spacing.** Views keep a gap that scales with the sheet (5 % of its short side, 8 to
//!   40 mm) between the room each reserves: its lines, its dimension rows on the sides they go,
//!   and its label under them.
//! - **Free space.** Section, detail, auxiliary and broken-out views take the free spot
//!   nearest their parent. A view placed by hand (`at`) keeps its spot while it is free; when
//!   something else holds it, the view moves to the nearest free spot and the layout says so.
//! - **Labels** go under the view: SECTION A-A, DETAIL B, VIEW C, with SCALE when the view's
//!   scale differs from the sheet's (a detail always states its scale).

use super::marks::{clip_to_circle, line, prim, text};
use super::solver::{nearest_free, BoxR, Class};
use super::{floats, xy, Ctx, Prim, Shape, Xy};
use crate::hlr::Axes;
use serde_json::Value;

/// How far a centerline or center mark runs past its circle or face (ISO 128-2: 2 to 3 mm).
pub const CENTER_OVERRUN: f64 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Below,
    Above,
    Left,
    Right,
}

/// A view as planned: its lines in view millimetres, its size on the sheet, and once placed,
/// its origin (where its centre sits).
pub struct ViewPlan {
    pub name: String,
    pub id: String,
    pub kind: String,
    pub parent: Option<String>,
    pub direction: Option<String>,
    pub at: Option<Xy>,
    pub scale: f64,
    pub edges: Vec<Vec<f64>>,
    pub hidden: Vec<Vec<f64>>,
    pub show_hidden: bool,
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub axes: Axes,
    pub seam: Value,
    /// Label lines under the view: (text, height).
    pub labels: Vec<(String, f64)>,
    /// The sides its horizontal and vertical dimensions go.
    pub h_side: Side,
    pub v_side: Side,
    /// Room kept on each side (below, above, left, right) for dimensions.
    pub room: [f64; 4],
    pub origin: Xy,
}

/// A distance as a finding says it: `1`, `2.5`, `12` (mm, to 0.5 mm, at least 0.5).
fn mm_text(mm: f64) -> String {
    let half = ((mm * 2.0).round() / 2.0).max(0.5);
    if half.fract() == 0.0 {
        format!("{half:.0}")
    } else {
        format!("{half:.1}")
    }
}

/// A view's title in a sentence (see [`ViewPlan::title`]).
fn view_title(name: &str, labels: &[(String, f64)]) -> String {
    let Some((label, _)) = labels.first().filter(|(l, _)| !l.contains(':')) else {
        let mut c = name.chars();
        let name: String = c
            .next()
            .map(|f| f.to_uppercase().chain(c).collect())
            .unwrap_or_default();
        return format!("{} view", name.replace('_', " "));
    };
    label
        .split(' ')
        .enumerate()
        .map(|(i, w)| {
            // Letters that name the view (`A-A`, `B`) stay as written.
            if w.chars().all(|c| c.is_ascii_uppercase() || c == '-')
                && (w.len() == 1 || w.contains('-'))
            {
                w.to_owned()
            } else if i == 0 {
                let lower = w.to_lowercase();
                let mut c = lower.chars();
                c.next()
                    .map(|f| f.to_uppercase().chain(c).collect())
                    .unwrap_or_default()
            } else {
                w.to_lowercase()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `Top view` → `top view`, for a title inside a sentence.
fn lower_first(title: &str) -> String {
    let mut c = title.chars();
    c.next()
        .map(|f| f.to_lowercase().chain(c).collect())
        .unwrap_or_default()
}

impl ViewPlan {
    /// How an engineer names the view in a sentence: its label in sentence case (`Top view`,
    /// `Section A-A`, `Detail B`), or for an unlabelled view its name (`Front view`).
    pub fn title(&self) -> String {
        view_title(&self.name, &self.labels)
    }

    /// A point in view millimetres on the sheet.
    pub fn to_sheet(&self, p: Xy) -> Xy {
        [
            self.origin[0] + (p[0] - self.cx) * self.scale,
            self.origin[1] + (p[1] - self.cy) * self.scale,
        ]
    }
    /// A model point (mm) on the sheet.
    pub fn model_to_sheet(&self, c: [f64; 3]) -> Xy {
        let d = |a: [f64; 3]| a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
        self.to_sheet([d(self.axes.u), d(self.axes.v)])
    }
    /// A model point relative to the view's centre, in sheet mm (before placement).
    pub fn model_local(&self, c: [f64; 3]) -> Xy {
        let d = |a: [f64; 3]| a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
        [
            (d(self.axes.u) - self.cx) * self.scale,
            (d(self.axes.v) - self.cy) * self.scale,
        ]
    }
    pub fn geom_box(&self) -> BoxR {
        BoxR::around(self.origin, self.w, self.h)
    }
    fn label_height(&self) -> f64 {
        self.labels.iter().map(|(_, s)| s * 1.5).sum::<f64>()
            + if self.labels.is_empty() { 0.0 } else { 3.0 }
    }
    /// The room the view keeps: its lines, its dimension rows, and its label under them.
    pub fn reserved(&self) -> BoxR {
        let g = self.geom_box();
        BoxR::new(
            g.x0 - self.room[2],
            g.y0 - self.room[0] - self.label_height(),
            g.x1 + self.room[3],
            g.y1 + self.room[1],
        )
    }
}

fn drawn_axes(seam: &Value) -> Option<crate::hlr::Axes> {
    let v3 = |k: &str| -> Option<[f64; 3]> {
        let f = floats(&seam["axes"][k]);
        (f.len() == 3).then(|| [f[0], f[1], f[2]])
    };
    Some(crate::hlr::Axes {
        u: v3("u")?,
        v: v3("v")?,
        depth: v3("depth")?,
    })
}

fn kind_of(view: &Value) -> String {
    view["kind"]["view"]
        .as_str()
        .unwrap_or("standard")
        .to_owned()
}

/// Every view on the sheet, measured but not yet placed.
pub fn measure(ctx: &mut Ctx<'_>) -> Vec<ViewPlan> {
    let doc = ctx.doc();
    let views_json = &doc["views"];
    let mut out = Vec::new();
    let mut first_standard = true;
    for view in views_json.as_array().into_iter().flatten() {
        if view["sheet"].as_str() != Some(ctx.sheet.as_str()) {
            continue;
        }
        let name = view["name"].as_str().unwrap_or("view").to_owned();
        let kind = kind_of(view);
        let own = floats(&view["scale"]);
        let scale = if own.len() == 2 && own[1] != 0.0 {
            own[0] / own[1]
        } else {
            ctx.scale
        };
        let seam = ctx.data["hlr"].get(&name).cloned().unwrap_or(Value::Null);
        if seam.is_null() {
            ctx.out.notes.push(format!(
                "View {name}: {} is not open, so it draws as its label only",
                view["model"]["file"].as_str().unwrap_or("its model")
            ));
        }
        let runs = |k: &str| -> Vec<Vec<f64>> {
            seam[k]
                .as_array()
                .into_iter()
                .flatten()
                .map(floats)
                .collect()
        };
        let mut edges = if seam["flat"].is_object() {
            flat_edges(&seam["flat"])
        } else {
            runs("visible")
        };
        let mut hidden = runs("hidden");
        let detail = xy(&seam["detail"]["center"]).zip(seam["detail"]["radius"].as_f64());
        let (cx, cy, w, h) = if let Some((c, r)) = detail {
            edges = clip_to_circle(&edges, c, r);
            hidden = clip_to_circle(&hidden, c, r);
            (c[0], c[1], 2.0 * r * scale, 2.0 * r * scale)
        } else {
            let pts = edges
                .iter()
                .chain(&hidden)
                .flat_map(|e| e.chunks_exact(2).map(|p| [p[0], p[1]]).collect::<Vec<_>>());
            let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
            for p in pts {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
            if lo[0] > hi[0] {
                (0.0, 0.0, 30.0, 20.0)
            } else {
                (
                    (lo[0] + hi[0]) / 2.0,
                    (lo[1] + hi[1]) / 2.0,
                    (hi[0] - lo[0]) * scale,
                    (hi[1] - lo[1]) * scale,
                )
            }
        };
        let k = &view["kind"];
        let letter = k["label"].as_str().map(str::to_uppercase);
        let big = ctx.std.text * 1.4;
        let small = ctx.std.text;
        let differs = (scale - ctx.scale).abs() > 1e-9;
        let scale_line = (format!("SCALE {}", super::scale_text(scale)), small);
        let mut labels = Vec::new();
        let titled = |t: String, labels: &mut Vec<(String, f64)>| {
            labels.push((t, big));
        };
        match kind.as_str() {
            "section" => titled(
                format!(
                    "SECTION {0}-{0}",
                    letter.clone().unwrap_or_else(|| name.to_uppercase())
                ),
                &mut labels,
            ),
            "detail" => titled(
                format!(
                    "DETAIL {}",
                    letter.clone().unwrap_or_else(|| name.to_uppercase())
                ),
                &mut labels,
            ),
            "auxiliary" => titled(
                format!(
                    "VIEW {}",
                    letter.clone().unwrap_or_else(|| name.to_uppercase())
                ),
                &mut labels,
            ),
            "broken_out" => titled("BROKEN-OUT SECTION".into(), &mut labels),
            "flat_pattern" => titled("FLAT PATTERN".into(), &mut labels),
            "projected" => {}
            _ => {
                let orientation = k["orientation"].as_str().unwrap_or("front");
                if !(first_standard && orientation != "isometric") {
                    titled(format!("{} VIEW", orientation.to_uppercase()), &mut labels);
                }
                if orientation != "isometric" {
                    first_standard = false;
                }
            }
        }
        if differs || kind == "detail" {
            labels.push(scale_line);
        }
        let parent = k["from"].as_str().map(str::to_owned);
        let direction = k["direction"].as_str().map(str::to_owned);
        let (h_side, v_side) = match (kind.as_str(), direction.as_deref()) {
            ("projected", Some("up")) => (Side::Above, Side::Left),
            ("projected", Some("right")) => (Side::Below, Side::Right),
            ("projected", Some("left")) => (Side::Below, Side::Left),
            _ => (Side::Below, Side::Left),
        };
        let display = view["display"].as_str();
        out.push(ViewPlan {
            id: format!("view:{name}"),
            // The projection the view's lines were drawn in; a view whose model isn't open has
            // none, and takes its orientation's.
            axes: drawn_axes(&seam)
                .unwrap_or_else(|| crate::hlr::axes(crate::hlr::orientation(views_json, &name, 0))),
            name,
            kind,
            parent,
            direction,
            at: super::placed_at(view),
            scale,
            edges,
            hidden,
            show_hidden: display != Some("visible_lines") && display != Some("shaded"),
            cx,
            cy,
            w,
            h,
            seam,
            labels,
            h_side,
            v_side,
            room: [0.0; 4],
            origin: [0.0, 0.0],
        });
    }
    out
}

/// Places every view (see the module notes).
pub fn place(
    ctx: &mut Ctx<'_>,
    plans: &mut [ViewPlan],
    rooms: &std::collections::BTreeMap<String, [f64; 4]>,
) {
    for p in plans.iter_mut() {
        if let Some(r) = rooms.get(&p.name) {
            p.room = *r;
        }
        // A little room all round for callouts that hug the view.
        for v in &mut p.room {
            *v += 2.0;
        }
    }
    let gap = (ctx.size[0].min(ctx.size[1]) * 0.05).clamp(8.0, 40.0);
    let f = ctx.frame;
    let step = (ctx.size[0].max(ctx.size[1]) / 160.0).max(2.0);
    let mut placed: Vec<usize> = Vec::new();
    // Views that found no room where they belong: the sheet is too full for them.
    let mut short: Vec<String> = Vec::new();
    // Roots first; then projected views, which only slide along their projection line, before
    // the views that may take any free spot.
    let mut order: Vec<usize> = (0..plans.len())
        .filter(|&i| plans[i].parent.is_none())
        .collect();
    let mut rest: Vec<usize> = (0..plans.len())
        .filter(|&i| plans[i].parent.is_some())
        .collect();
    rest.sort_by_key(|&i| plans[i].kind != "projected");
    for pass in 0..8 {
        for &i in &rest {
            if order.contains(&i) {
                continue;
            }
            let ready = plans[i].parent.as_ref().is_none_or(|p| {
                order.iter().any(|&j| &plans[j].name == p) || !plans.iter().any(|v| &v.name == p)
            });
            if ready || pass == 7 {
                order.push(i);
            }
        }
    }
    for i in order {
        let parent = plans[i]
            .parent
            .as_ref()
            .and_then(|p| placed.iter().copied().find(|&j| &plans[j].name == p));
        let v = &plans[i];
        let r = v.reserved();
        // The reserved box relative to the view's origin.
        let (l, b, rt, t) = (
            v.origin[0] - r.x0,
            v.origin[1] - r.y0,
            r.x1 - v.origin[0],
            r.y1 - v.origin[1],
        );
        let boxed = |o: Xy| BoxR::new(o[0] - l, o[1] - b, o[0] + rt, o[1] + t);
        let free = |o: Xy, occ: &super::solver::Occupancy| {
            occ.free(&boxed(o).grow(gap / 2.0), &v.id, 0.0, &[])
        };
        let mut spot: Option<Xy> = None;
        let mut preferred = v.at.unwrap_or([f.x0 + 0.3 * f.w(), f.y0 + 0.6 * f.h()]);
        if let (Some(j), "projected") = (parent, v.kind.as_str()) {
            let pv = &plans[j];
            let pr = pv.reserved();
            let dir = v.direction.as_deref().unwrap_or("right");
            let (horizontal, sign) = match dir {
                "up" => (false, 1.0),
                "down" => (false, -1.0),
                "left" => (true, -1.0),
                _ => (true, 1.0),
            };
            // Adjacent: the parent's room, the gap, then this view's room.
            let d0 = if horizontal {
                if sign > 0.0 {
                    pr.x1 - pv.origin[0] + gap + l
                } else {
                    pv.origin[0] - pr.x0 + gap + rt
                }
            } else if sign > 0.0 {
                pr.y1 - pv.origin[1] + gap + b
            } else {
                pv.origin[1] - pr.y0 + gap + t
            };
            let along = |d: f64| {
                if horizontal {
                    [pv.origin[0] + sign * d, pv.origin[1]]
                } else {
                    [pv.origin[0], pv.origin[1] + sign * d]
                }
            };
            // The authored spot keeps its distance along the projection, in line with the parent.
            if let Some(a) = v.at {
                let d = if horizontal {
                    (a[0] - pv.origin[0]) * sign
                } else {
                    (a[1] - pv.origin[1]) * sign
                };
                if d >= d0 && free(along(d), &ctx.occ) {
                    spot = Some(along(d));
                }
            }
            let mut d = d0;
            while spot.is_none() && d < ctx.size[0].max(ctx.size[1]) {
                if free(along(d), &ctx.occ) {
                    spot = Some(along(d));
                }
                d += step;
            }
            preferred = along(d0);
            if spot.is_none() {
                short.push(v.name.clone());
                ctx.finding(
                    "view_not_aligned",
                    &v.name,
                    format!(
                        "{} has no room in line with the {}; it sits in the nearest free space",
                        v.title(),
                        lower_first(&pv.title())
                    ),
                );
            }
        } else {
            if let Some(a) = v.at.filter(|a| free(*a, &ctx.occ)) {
                spot = Some(a);
            }
            if let Some(j) = parent {
                preferred = plans[j].origin;
                if v.at.is_some_and(|a| spot == Some(a)) {
                    // Kept where it was put.
                } else {
                    spot = None;
                }
            }
        }
        if spot.is_none() {
            let g = boxed([0.0, 0.0]).grow(gap / 2.0);
            let c = g.center();
            let hit = nearest_free(
                &ctx.occ,
                &v.id,
                [g.w(), g.h()],
                [preferred[0] + c[0], preferred[1] + c[1]],
                step,
                0.0,
                &|_| true,
            );
            spot = hit.map(|h| [h.center()[0] - c[0], h.center()[1] - c[1]]);
            // A full sheet: the view keeps less of a gap before it overlaps anything.
            if spot.is_none() {
                let tight = boxed([0.0, 0.0]).grow(2.0);
                let c = tight.center();
                let hit = nearest_free(
                    &ctx.occ,
                    &v.id,
                    [tight.w(), tight.h()],
                    [preferred[0] + c[0], preferred[1] + c[1]],
                    step,
                    0.0,
                    &|_| true,
                );
                spot = hit.map(|h| [h.center()[0] - c[0], h.center()[1] - c[1]]);
                if spot.is_some() {
                    short.push(v.name.clone());
                    ctx.finding("view_crowded", &v.name, format!("{} sits closer than the standard gap to the views beside it: the sheet is full", v.title()));
                }
            }
        }
        let name = v.name.clone();
        let title = v.title();
        let id = v.id.clone();
        let o = match spot {
            Some(o) => o,
            None => {
                short.push(name.clone());
                ctx.finding("view_no_room", &name, format!("{title} doesn't fit beside the other views; use a larger sheet or a smaller scale"));
                preferred
            }
        };
        if let Some(a) = v.at {
            if (a[0] - o[0]).abs() > 0.5 || (a[1] - o[1]).abs() > 0.5 {
                let moved = (o[0] - a[0]).hypot(o[1] - a[1]);
                ctx.finding(
                    "view_moved",
                    &name,
                    format!(
                        "{title} moved {} mm to clear the views, tables and notes beside it",
                        mm_text(moved)
                    ),
                );
            }
        }
        plans[i].origin = o;
        ctx.out.placed.insert(id.clone(), o);
        let reserved = plans[i].reserved().grow(gap / 2.0);
        ctx.occ.add(reserved, &id, Class::View);
        placed.push(i);
    }
    short.dedup();
    if !short.is_empty() {
        let larger = larger_sheet(ctx.size)
            .map(|n| format!(" ({n})"))
            .unwrap_or_default();
        let views = short.join(", ");
        ctx.finding("sheet_full", &ctx.sheet.clone(), format!("Sheet {} is too full for its views: use a larger sheet{larger}, or move {views} to a second sheet", ctx.sheet));
    }
}

/// The next sheet size up in the same series (ISO A4 to A0, ANSI A to E), by the sheet's size.
pub fn larger_sheet(size: Xy) -> Option<&'static str> {
    const ISO: [(&str, [f64; 2]); 5] = [
        ("A4", [297.0, 210.0]),
        ("A3", [420.0, 297.0]),
        ("A2", [594.0, 420.0]),
        ("A1", [841.0, 594.0]),
        ("A0", [1189.0, 841.0]),
    ];
    const ANSI: [(&str, [f64; 2]); 5] = [
        ("A", [279.4, 215.9]),
        ("B", [431.8, 279.4]),
        ("C", [558.8, 431.8]),
        ("D", [863.6, 558.8]),
        ("E", [1117.6, 863.6]),
    ];
    let (long, short) = (size[0].max(size[1]), size[0].min(size[1]));
    for series in [&ISO, &ANSI] {
        if let Some(i) = series
            .iter()
            .position(|(_, s)| (s[0] - long).abs() < 1.0 && (s[1] - short).abs() < 1.0)
        {
            return series.get(i + 1).map(|(n, _)| *n);
        }
    }
    None
}

/// Draws each view's lines, center marks, hatching, bend lines, label, and its marks on its
/// parent (the section's cutting line, the detail's circle). Replaces each view's reserved room
/// with its lines' box, so callouts may use what its dimensions leave free.
pub fn draw(ctx: &mut Ctx<'_>, plans: &[ViewPlan]) {
    ctx.occ.items.retain(|i| i.class != Class::View);
    for v in plans {
        ctx.occ.add(v.geom_box().grow(1.0), &v.id, Class::View);
    }
    for v in plans {
        let mut out: Vec<Prim> = Vec::new();
        // Sheet millimetres to the micrometre: finer than any print, and short in the answer.
        let place = |e: &[f64]| -> Vec<f64> {
            e.chunks_exact(2)
                .flat_map(|p| v.to_sheet([p[0], p[1]]).map(|c| (c * 1e3).round() / 1e3))
                .collect()
        };
        // Hidden lines first, so a visible edge over a hidden one draws solid (ISO 128-20).
        if v.show_hidden {
            for e in &v.hidden {
                out.push(prim(&v.id, "view", "Views", "hidden", line(place(e))));
            }
        }
        for e in &v.edges {
            out.push(prim(&v.id, "view", "Views", "visible", line(place(e))));
        }
        let detail = xy(&v.seam["detail"]["center"]).zip(v.seam["detail"]["radius"].as_f64());
        for pick in v.seam["picks"].as_array().into_iter().flatten() {
            let pts = floats(&pick["points"]);
            let pts = match detail {
                Some((c, r)) => clip_to_circle(&[pts], c, r)
                    .into_iter()
                    .next()
                    .unwrap_or_default(),
                None => pts,
            };
            if pts.len() >= 4 {
                out.push(prim(
                    &format!(
                        "edge:{}|{}",
                        v.name,
                        pick["reference"].as_str().unwrap_or("")
                    ),
                    "edge",
                    "Views",
                    "pick",
                    line(place(&pts)),
                ));
            }
        }
        let rings: Vec<Vec<f64>> = v.seam["hatch"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|r| place(&floats(r)))
            .collect();
        if !rings.is_empty() {
            let hatch = ctx.std.hatch(&v.id, &rings, 2.0);
            match detail {
                // A detail of a section hatches only inside its circle.
                Some((c, r)) => {
                    let (c, r) = (v.to_sheet(c), r * v.scale);
                    for h in hatch {
                        let Shape::Polyline { points, .. } = &h.shape else {
                            continue;
                        };
                        for run in clip_to_circle(std::slice::from_ref(points), c, r) {
                            out.push(Prim {
                                shape: line(run),
                                ..h.clone()
                            });
                        }
                    }
                }
                None => out.extend(hatch),
            }
        }
        if v.seam["flat"].is_object() {
            out.extend(bend_lines(&v.id, &v.seam["flat"], &|p| v.to_sheet(p)));
        }
        for c in v.seam["centers"].as_array().into_iter().flatten() {
            let r = c["radius"].as_f64().unwrap_or(0.0) * v.scale + CENTER_OVERRUN;
            if c["kind"] == "mark" {
                let Some(at) = xy(&c["at"]) else { continue };
                if detail.is_some_and(|(dc, dr)| {
                    ((at[0] - dc[0]).powi(2) + (at[1] - dc[1]).powi(2)).sqrt() > dr
                }) {
                    continue;
                }
                let [x, y] = v.to_sheet(at);
                out.push(prim(
                    &v.id,
                    "view",
                    "Views",
                    "center",
                    line(vec![x - r, y, x + r, y]),
                ));
                out.push(prim(
                    &v.id,
                    "view",
                    "Views",
                    "center",
                    line(vec![x, y - r, x, y + r]),
                ));
            } else if detail.is_none() {
                let (Some(a), Some(b)) = (xy(&c["from"]), xy(&c["to"])) else {
                    continue;
                };
                let (a, b) = (v.to_sheet(a), v.to_sheet(b));
                let u = super::marks::unit(a, b);
                out.push(prim(
                    &v.id,
                    "view",
                    "Views",
                    "center",
                    line(vec![
                        a[0] - u[0] * CENTER_OVERRUN,
                        a[1] - u[1] * CENTER_OVERRUN,
                        b[0] + u[0] * CENTER_OVERRUN,
                        b[1] + u[1] * CENTER_OVERRUN,
                    ]),
                ));
            }
        }
        let g = v.geom_box();
        if !g.inside(&ctx.frame) {
            ctx.out.notes.push(format!(
                "View {} at {} runs past the sheet's frame",
                v.name,
                super::scale_text(v.scale)
            ));
        }
        // The label, under the view's dimensions below it.
        let mut y = g.y0 - v.room[0] - 3.0;
        for (t, size) in &v.labels {
            y -= size;
            let p = prim(
                &v.id,
                "viewlabel",
                "Views",
                "dimtext",
                text(g.center()[0], y, t, *size, "middle"),
            );
            if let Some(b) = super::solver::text_box(&p) {
                ctx.occ.add(b, &v.id, Class::Text);
            }
            out.push(p);
            y -= size * 0.5;
        }
        // Marks on the parent: a section's cutting line, a detail's circle and letter.
        if let Some(parent) = v
            .parent
            .as_ref()
            .and_then(|p| plans.iter().find(|q| &q.name == p))
        {
            let letter = v.seam["cut"]["label"]
                .as_str()
                .or(v
                    .labels
                    .first()
                    .map(|l| l.0.rsplit(' ').next().unwrap_or("")))
                .unwrap_or("")
                .split('-')
                .next()
                .unwrap_or("")
                .to_owned();
            if v.kind == "section" {
                if let Some(at) = v.seam["cut"]["at"].as_f64() {
                    let toward = xy(&v.seam["cut"]["toward"]).unwrap_or([0.0, 1.0]);
                    let pg = parent.geom_box();
                    let y = parent.to_sheet([0.0, at])[1];
                    let marks = ctx.std.cutting_plane(
                        &v.id,
                        [pg.x0 - 8.0, y],
                        [pg.x1 + 8.0, y],
                        toward,
                        &letter,
                    );
                    ctx.occ.add_prims(&marks, &v.id);
                    out.extend(marks);
                }
            } else if let (Some((c, r)), "detail") = (detail, v.kind.as_str()) {
                let c = parent.to_sheet(c);
                let rr = r * parent.scale;
                out.push(prim(
                    &v.id,
                    "view",
                    "Views",
                    "dimink",
                    Shape::Circle {
                        x: c[0],
                        y: c[1],
                        r: rr,
                    },
                ));
                let size = ctx.std.text * 1.4;
                // The letter just outside the circle, where it overlaps nothing.
                let spots = [
                    [0.75, 0.75],
                    [-0.75, 0.75],
                    [0.75, -0.75],
                    [-0.75, -0.75],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ];
                let mut chosen = None;
                for s in spots {
                    let p = prim(
                        &v.id,
                        "label",
                        "Views",
                        "dimtext",
                        text(
                            c[0] + s[0] * (rr + 2.0),
                            c[1] + s[1] * (rr + 2.0) - size / 2.0,
                            &letter,
                            size,
                            "middle",
                        ),
                    );
                    let b = super::solver::text_box(&p).unwrap_or(BoxR::around(c, 1.0, 1.0));
                    if chosen.is_none() && ctx.occ.free(&b, &v.id, 0.5, &[&parent.id]) {
                        chosen = Some((p, b));
                    }
                }
                if let Some((p, b)) = chosen {
                    ctx.occ.add(b, &v.id, Class::Text);
                    out.push(p);
                }
            }
        }
        if v.kind == "detail" {
            if let Some((_, r)) = detail {
                out.push(prim(
                    &v.id,
                    "view",
                    "Views",
                    "dimink",
                    Shape::Circle {
                        x: v.origin[0],
                        y: v.origin[1],
                        r: r * v.scale,
                    },
                ));
            }
        }
        ctx.push(out);
    }
}

/// A flat pattern's outline as polylines in view millimetres.
pub fn flat_edges(flat: &Value) -> Vec<Vec<f64>> {
    let arc = |c: Xy, r: f64, a0: f64, a1: f64| -> Vec<f64> {
        let n = ((a1 - a0).abs() / (std::f64::consts::PI / 36.0))
            .ceil()
            .max(2.0) as usize;
        (0..=n)
            .flat_map(|k| {
                let a = a0 + (a1 - a0) * k as f64 / n as f64;
                [c[0] + r * a.cos(), c[1] + r * a.sin()]
            })
            .collect()
    };
    let mut out = Vec::new();
    for lp in flat["outline"].as_array().into_iter().flatten() {
        for c in lp.as_array().into_iter().flatten() {
            let p = |k: &str| xy(&c[k]).unwrap_or([0.0, 0.0]);
            match c["curve"].as_str() {
                Some("line") => {
                    out.push(vec![p("start")[0], p("start")[1], p("end")[0], p("end")[1]])
                }
                Some("circle") => out.push(arc(
                    p("center"),
                    c["radius"].as_f64().unwrap_or(0.0),
                    0.0,
                    2.0 * std::f64::consts::PI,
                )),
                Some("polyline") => out.push(
                    c["points"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(xy)
                        .flatten()
                        .collect(),
                ),
                _ => {
                    let a0 = c["start_angle"].as_f64().unwrap_or(0.0).to_radians();
                    let mut a1 = c["end_angle"].as_f64().unwrap_or(0.0).to_radians();
                    while a1 <= a0 {
                        a1 += 2.0 * std::f64::consts::PI;
                    }
                    out.push(arc(
                        p("center"),
                        c["radius"].as_f64().unwrap_or(0.0),
                        a0,
                        a1,
                    ));
                }
            }
        }
    }
    out
}

/// A flat pattern's bend lines (an up bend a chain line, a down bend dashed), running 2 mm past
/// the outline, with their notes along them.
fn bend_lines(id: &str, flat: &Value, at: &dyn Fn(Xy) -> Xy) -> Vec<Prim> {
    let mut out = Vec::new();
    for b in flat["bend_lines"].as_array().into_iter().flatten() {
        let (Some(s), Some(e)) = (xy(&b["start"]), xy(&b["end"])) else {
            continue;
        };
        let (p, q) = (at(s), at(e));
        let u = super::marks::unit(p, q);
        let style = if b["direction"] == "up" {
            "bendup"
        } else {
            "benddown"
        };
        out.push(prim(
            id,
            "view",
            "Views",
            style,
            line(vec![
                p[0] - u[0] * 2.0,
                p[1] - u[1] * 2.0,
                q[0] + u[0] * 2.0,
                q[1] + u[1] * 2.0,
            ]),
        ));
        let Some(note) = b["note"].as_str().filter(|n| !n.is_empty()) else {
            continue;
        };
        let mut angle = u[1].atan2(u[0]).to_degrees();
        if !(-89.9999..=90.0001).contains(&angle) {
            angle += if angle > 0.0 { -180.0 } else { 180.0 };
        }
        let r = angle.to_radians();
        let n = [-r.sin(), r.cos()];
        let mid = [
            (p[0] + q[0]) / 2.0 + n[0] * 1.2,
            (p[1] + q[1]) / 2.0 + n[1] * 1.2,
        ];
        out.push(prim(
            id,
            "label",
            "Views",
            "note",
            Shape::Text {
                x: mid[0],
                y: mid[1],
                text: note.to_owned(),
                size: 2.5,
                anchor: Some("middle"),
                rotation: Some(angle),
            },
        ));
    }
    out
}

#[cfg(test)]
mod title_tests {
    use super::{mm_text, view_title};

    #[test]
    fn findings_name_views_as_their_labels_read() {
        let l = |t: &str| vec![(t.to_owned(), 5.0)];
        assert_eq!(view_title("top", &l("TOP VIEW")), "Top view");
        assert_eq!(view_title("a_a", &l("SECTION A-A")), "Section A-A");
        assert_eq!(view_title("b", &l("DETAIL B")), "Detail B");
        assert_eq!(view_title("front", &[]), "Front view");
        assert_eq!(view_title("front", &l("SCALE: 2:1")), "Front view");
        assert_eq!(
            (mm_text(0.9), mm_text(2.6), mm_text(0.1)),
            ("1".into(), "2.5".into(), "0.5".into())
        );
    }
}
