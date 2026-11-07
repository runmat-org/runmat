//! Callouts that point at the model: feature control frames, datum symbols, finish marks, hole
//! callouts and diameter and radius dimensions, angular dimensions, weld symbols, datum targets,
//! leadered notes and balloons; and center marks, centerlines and notes.
//!
//! Each leadered callout takes the free spot the solver finds nearest what it points at
//! ([`super::solver::place`]): its label overlaps no other label, frame, table or view, and its
//! leader crosses no text. A hole's leader points at the hole's centre and ends on its rim. A
//! callout placed by hand (`at`) stays where it was put, and views keep clear of it. When no
//! free spot exists, the callout goes beside its view and the layout says so.

use super::marks::{line, mark, unit};
use super::solver::{nearest_free, place, BoxR, Class, Request, Spot};
use super::views::{ViewPlan, CENTER_OVERRUN};
use super::{floats, placed_at, xy, Ctx, Prim, Std, Xy};
use serde_json::Value;

/// A notes block's box with its top-left corner at `at`: the first line full height, the
/// rest 0.8 of it, 1.7 heights apart.
fn notes_box(st: &Std, at: Xy, t: &str) -> BoxR {
    let lines: Vec<&str> = t.split('\n').collect();
    let w = lines.iter().enumerate().map(|(i, l)| Std::text_width(l, if i == 0 { st.text } else { st.text * 0.8 })).fold(0.0, f64::max);
    BoxR::new(at[0], at[1] - lines.len() as f64 * st.text * 1.7 - st.text * 0.3, at[0] + w, at[1])
}

fn notes_block(st: &Std, id: &str, at: Xy, t: &str) -> Vec<Prim> {
    t.split('\n')
        .enumerate()
        .map(|(i, l)| mark(id, "note", super::marks::text(at[0], at[1] - (i + 1) as f64 * st.text * 1.7, l, if i == 0 { st.text } else { st.text * 0.8 }, "start")))
        .collect()
}

fn frame_size(st: &Std, cells: &[&str]) -> Xy {
    [st.frame_widths(cells).iter().sum(), 2.0 * st.text]
}

/// Keeps room for callouts placed by hand before the views are placed.
pub fn reserve_manual(ctx: &mut Ctx<'_>) {
    let views_here: Vec<String> = ctx.doc()["views"].as_array().into_iter().flatten().filter(|v| v["sheet"].as_str() == Some(ctx.sheet.as_str())).filter_map(|v| v["name"].as_str().map(str::to_owned)).collect();
    for a in ctx.doc()["annotations"].as_array().into_iter().flatten() {
        if !views_here.iter().any(|v| a["view"].as_str() == Some(v)) {
            continue;
        }
        let Some(at) = placed_at(a) else { continue };
        let id = format!("annotation:{}", a["id"].as_str().unwrap_or(""));
        let t = ctx.text_of(a["id"].as_str().unwrap_or(""));
        let b = match a["kind"]["annotation"].as_str() {
            Some("note") => notes_box(&ctx.std, at, &t),
            Some("gdt") => {
                let cells: Vec<&str> = t.split('|').collect();
                let [w, h] = frame_size(&ctx.std, &cells);
                BoxR::new(at[0], at[1], at[0] + w, at[1] + h)
            }
            _ => continue,
        };
        ctx.occ.add(b, &id, Class::Text);
    }
}

struct Frame {
    view: String,
    anchor: Xy,
    frame: BoxR,
}

/// Draws every callout on the sheet's views.
pub fn draw(ctx: &mut Ctx<'_>, views: &[ViewPlan]) {
    let annotations: Vec<Value> = ctx.doc()["annotations"].as_array().cloned().unwrap_or_default();
    let mut frames: Vec<Frame> = Vec::new();
    // Frames first, so a datum symbol on the same feature hangs on its frame.
    let (gdt, rest): (Vec<&Value>, Vec<&Value>) = annotations.iter().partition(|a| a["kind"]["annotation"] == "gdt");
    for a in gdt.into_iter().chain(rest) {
        let Some(v) = views.iter().find(|v| a["view"].as_str() == Some(v.name.as_str())) else { continue };
        let kind = &a["kind"];
        let style = kind["style"].as_str().unwrap_or("");
        let what = kind["annotation"].as_str().unwrap_or("");
        if what == "dimension" && matches!(style, "linear" | "ordinate" | "baseline" | "chain") {
            continue;
        }
        let aid = a["id"].as_str().unwrap_or("").to_owned();
        let id = format!("annotation:{aid}");
        let t = ctx.text_of(&aid);
        let anchors: Vec<Xy> = ctx.data["anchors"][&aid].as_array().into_iter().flatten().filter_map(|c| {
            let f = floats(c);
            (f.len() == 3).then(|| v.model_to_sheet([f[0], f[1], f[2]]))
        }).collect();
        let anchor = anchors.first().copied();
        let radius = ctx.data["radii"][&aid].as_f64().map(|r| r * v.scale).or_else(|| anchor.map(|p| hole_radius_at(v, p))).unwrap_or(0.0);
        let manual = placed_at(a);
        let st = ctx.std.clone();
        let on_rim = move |c: Xy, r: f64| move |elbow: Xy| if r > 0.0 { let u = unit(c, elbow); [c[0] + u[0] * r, c[1] + u[1] * r] } else { c };
        match what {
            "gdt" => {
                let Some(anchor) = anchor else { continue };
                let cells: Vec<&str> = t.split('|').collect();
                let [w, h] = frame_size(&st, &cells);
                let label = |e: Xy, right: bool| if right { BoxR::new(e[0], e[1] - h / 2.0, e[0] + w, e[1] + h / 2.0) } else { BoxR::new(e[0] - w, e[1] - h / 2.0, e[0], e[1] + h / 2.0) };
                let spot = match manual {
                    Some(at) => {
                        let b = BoxR::new(at[0], at[1], at[0] + w, at[1] + h);
                        let right = anchor[0] < b.center()[0];
                        Some(Spot { elbow: [if right { b.x0 } else { b.x1 }, b.center()[1]], right, label: b })
                    }
                    None => solve(ctx, v, &id, anchor, &label, &on_rim(anchor, radius)),
                };
                let Some(spot) = spot else { continue };
                let tip = on_rim(anchor, radius)(spot.elbow);
                let mut prims = st.control_frame(&id, [spot.label.x0, spot.label.y0], &cells);
                prims.extend(leader_line(&st, &id, tip, spot.elbow));
                ctx.out.placed.insert(id.clone(), [spot.label.x0, spot.label.y0]);
                register(ctx, &id, &prims, spot.label);
                ctx.push(prims);
                frames.push(Frame { view: v.name.clone(), anchor, frame: spot.label });
            }
            "datum_symbol" => {
                let Some(anchor) = anchor else { continue };
                let letter = kind["label"].as_str().map(str::to_owned).unwrap_or_else(|| t.trim_start_matches("Datum ").to_owned());
                let on_frame = frames.iter().find(|f| f.view == v.name && super::marks::len(f.anchor, anchor) < 0.05).map(|f| f.frame);
                let mut chosen = None;
                let g = v.geom_box();
                let candidates: Vec<(Xy, Xy)> = match on_frame {
                    Some(fr) => vec![([fr.x0 + st.text, fr.y0], [0.0, -1.0]), ([fr.x0 + st.text, fr.y1], [0.0, 1.0])],
                    None => {
                        let mut sides = vec![
                            (anchor[1] - g.y0, [0.0, -1.0]),
                            (g.y1 - anchor[1], [0.0, 1.0]),
                            (anchor[0] - g.x0, [-1.0, 0.0]),
                            (g.x1 - anchor[0], [1.0, 0.0]),
                        ];
                        sides.sort_by(|a, b| a.0.total_cmp(&b.0));
                        let mut c = Vec::new();
                        for (_, away) in sides {
                            for slide in [0.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0, 18.0, -18.0] {
                                let on = if away[1] != 0.0 { [(anchor[0] + slide).clamp(g.x0 + 2.0, g.x1 - 2.0), if away[1] < 0.0 { g.y0 } else { g.y1 }] } else { [if away[0] < 0.0 { g.x0 } else { g.x1 }, (anchor[1] + slide).clamp(g.y0 + 1.0, g.y1 - 1.0)] };
                                c.push((on, away));
                            }
                        }
                        c
                    }
                };
                for (on, away) in &candidates {
                    let (prims, b) = st.datum_symbol(&id, *on, *away, &letter);
                    let sq = b;
                    let probe = if away[1] < 0.0 { BoxR::new(sq.x0, sq.y0, sq.x1, on[1] - 2.0) } else if away[1] > 0.0 { BoxR::new(sq.x0, on[1] + 2.0, sq.x1, sq.y1) } else if away[0] < 0.0 { BoxR::new(sq.x0, sq.y0, on[0] - 2.0, sq.y1) } else { BoxR::new(on[0] + 2.0, sq.y0, sq.x1, sq.y1) };
                    if ctx.occ.free(&probe, &id, 0.5, &[]) {
                        chosen = Some((prims, probe));
                        break;
                    }
                }
                let (prims, b) = chosen.unwrap_or_else(|| {
                    let (on, away) = candidates[0];
                    let (p, b) = st.datum_symbol(&id, on, away, &letter);
                    ctx.finding("callout_no_room", &aid, format!("Datum {letter} has no room by its surface; it overlaps what is there"));
                    (p, b)
                });
                register(ctx, &id, &prims, b);
                ctx.push(prims);
            }
            "surface_finish" => {
                let Some(anchor) = anchor else { continue };
                let label = |e: Xy, _right: bool| st.finish_box(e, &t);
                let tip_of = on_rim(anchor, radius);
                let spot = match manual {
                    Some(at) => Some(Spot { elbow: at, right: true, label: label(at, true) }),
                    None => solve(ctx, v, &id, anchor, &label, &tip_of),
                };
                let Some(spot) = spot else { continue };
                ctx.out.placed.insert(id.clone(), spot.elbow);
                let prims = st.surface_finish(&id, tip_of(spot.elbow), spot.elbow, &t);
                register(ctx, &id, &prims, spot.label);
                ctx.push(prims);
            }
            "center_mark" => {
                let Some(p) = anchor else { continue };
                let r = radius + CENTER_OVERRUN;
                ctx.push(vec![
                    mark(&id, "center", line(vec![p[0] - r, p[1], p[0] + r, p[1]])),
                    mark(&id, "center", line(vec![p[0], p[1] - r, p[0], p[1] + r])),
                ]);
            }
            "centerline" => {
                let (Some(p), Some(q)) = (anchors.first(), anchors.get(1).or(anchors.first())) else { continue };
                let g = v.geom_box();
                let m = [(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0];
                let across = (q[0] - p[0]).abs() >= (q[1] - p[1]).abs();
                let pts = if across { vec![m[0], g.y0 - CENTER_OVERRUN, m[0], g.y1 + CENTER_OVERRUN] } else { vec![g.x0 - CENTER_OVERRUN, m[1], g.x1 + CENTER_OVERRUN, m[1]] };
                ctx.push(vec![mark(&id, "center", line(pts))]);
            }
            "balloon" => {
                let Some(part) = occurrence_box(ctx, v, &kind["occurrence"]) else { continue };
                let number = ctx.data["balloons"][&aid].as_str().unwrap_or("?").to_owned();
                let r = st.balloon_radius();
                // The leader ends a little inside the part, on the side nearest the view's edge.
                let g = v.geom_box();
                let inset = |s: f64| (s * 0.25).clamp(0.0, 3.0);
                let c = part.center();
                let mut sides = [(part.x0 - g.x0, [part.x0 + inset(part.w()), c[1]]), (g.x1 - part.x1, [part.x1 - inset(part.w()), c[1]]), (part.y0 - g.y0, [c[0], part.y0 + inset(part.h())]), (g.y1 - part.y1, [c[0], part.y1 - inset(part.h())])];
                sides.sort_by(|a, b| a.0.total_cmp(&b.0));
                let on = sides[0].1;
                let label = |e: Xy, right: bool| BoxR::around([e[0] + if right { r } else { -r }, e[1]], 2.0 * r, 2.0 * r);
                let tip = |_e: Xy| on;
                let Some(spot) = solve(ctx, v, &id, on, &label, &tip) else { continue };
                let prims = st.balloon(&id, on, spot.label.center(), &number);
                register(ctx, &id, &prims, spot.label);
                ctx.push(prims);
            }
            _ => {
                // Hole callouts, diameter and radius dimensions, angular dimensions, weld
                // symbols, datum targets and leadered notes read on a leader.
                let label_text = match what {
                    "datum_target" => kind["label"].as_str().map(str::to_owned).unwrap_or(t.clone()),
                    "weld" => weld_text(kind),
                    _ => t.replace('\n', " "),
                };
                if what == "note" {
                    if let Some(at) = manual {
                        ctx.out.placed.insert(id.clone(), at);
                        let prims = notes_block(&st, &id, at, &t);
                        register(ctx, &id, &prims, notes_box(&st, at, &t));
                        ctx.push(prims);
                        continue;
                    }
                }
                let leadered = matches!(what, "hole_callout" | "weld" | "datum_target" | "note") || what == "dimension";
                let tip_point = match (what, style) {
                    ("dimension", "angular") | ("weld", _) | ("datum_target", _) if !anchors.is_empty() => {
                        Some([anchors.iter().map(|p| p[0]).sum::<f64>() / anchors.len() as f64, anchors.iter().map(|p| p[1]).sum::<f64>() / anchors.len() as f64])
                    }
                    _ => anchor,
                };
                match tip_point.filter(|_| leadered) {
                    Some(c) => {
                        let rim = if matches!(what, "hole_callout") || matches!(style, "diameter" | "radius") { hole_radius_at(v, c).max(if radius > 0.0 { radius } else { 0.0 }) } else { 0.0 };
                        let tip_of = on_rim(c, rim);
                        let label = |e: Xy, right: bool| st.leader_box(e, right, &label_text);
                        let spot = match manual {
                            Some(at) => Some(Spot { elbow: at, right: at[0] >= c[0], label: st.leader_box(at, at[0] >= c[0], &label_text) }),
                            None => solve(ctx, v, &id, c, &label, &tip_of),
                        };
                        let Some(spot) = spot else { continue };
                        ctx.out.placed.insert(id.clone(), spot.elbow);
                        let prims = st.leader(&id, tip_of(spot.elbow), spot.elbow, spot.right, &label_text);
                        register(ctx, &id, &prims, spot.label);
                        ctx.push(prims);
                    }
                    None => free_note(ctx, v, &id, &t),
                }
            }
        }
    }
}

/// A leader from a frame or label at `elbow` to `tip`, with its arrowhead.
fn leader_line(st: &Std, id: &str, tip: Xy, elbow: Xy) -> Vec<Prim> {
    let d = unit(elbow, tip);
    vec![mark(id, "dimink", line(vec![elbow[0], elbow[1], tip[0] - d[0] * st.arrow * 0.5, tip[1] - d[1] * st.arrow * 0.5])), st.arrowhead(id, tip, d)]
}

fn register(ctx: &mut Ctx<'_>, id: &str, prims: &[Prim], label: BoxR) {
    ctx.occ.add_prims(prims, id);
    ctx.occ.add(label, id, Class::Text);
}

/// The solver's spot for a callout on view `v`, or a finding and none when the sheet has no
/// free room for it.
fn solve(ctx: &mut Ctx<'_>, v: &ViewPlan, id: &str, anchor: Xy, label: &dyn Fn(Xy, bool) -> BoxR, tip: &dyn Fn(Xy) -> Xy) -> Option<Spot> {
    let req = Request { owner: id, view: &v.id, view_box: v.geom_box(), anchor, label, tip };
    let spot = place(&ctx.occ, &req);
    if spot.is_none() {
        let name = id.trim_start_matches("annotation:").to_owned();
        let text = ctx.text_of(&name);
        ctx.finding("callout_no_room", &name, format!("{text} has no room near the {}; move it by hand or give the sheet more room", v.title().to_lowercase()));
        // The free spot nearest the anchor for the label alone; its leader may cross what is there.
        let probe = label([0.0, 0.0], true);
        let g = v.geom_box();
        let elbow = match nearest_free(&ctx.occ, id, probe.w(), probe.h(), anchor, 2.0, 1.0, &|b| !b.hits(&g)) {
            Some(b) => [b.x0 - probe.x0, b.y0 - probe.y0],
            None => [g.x1 + 6.0, anchor[1]],
        };
        return Some(Spot { elbow, right: true, label: label(elbow, true) });
    }
    spot
}

/// A note with no leader and no spot: in the free space nearest its view's lower left.
fn free_note(ctx: &mut Ctx<'_>, v: &ViewPlan, id: &str, t: &str) {
    let st = ctx.std.clone();
    let probe = notes_box(&st, [0.0, 0.0], t);
    let g = v.geom_box();
    let preferred = [g.x0 + probe.w() / 2.0, g.y0 - probe.h()];
    let b = nearest_free(&ctx.occ, id, probe.w(), probe.h(), preferred, 2.0, 1.0, &|_| true).unwrap_or(BoxR::new(g.x1 + 6.0, g.y0, g.x1 + 6.0 + probe.w(), g.y0 + probe.h()));
    ctx.out.placed.insert(id.to_owned(), [b.x0, b.y1]);
    let prims = notes_block(&st, id, [b.x0, b.y1], t);
    register(ctx, id, &prims, b);
    ctx.push(prims);
}

/// A hole's drawn radius (sheet mm) when `p` is on its axis in the view: the smallest circle
/// there (a bore inside its boss), else 0.
fn hole_radius_at(v: &ViewPlan, p: Xy) -> f64 {
    let mut best: f64 = 0.0;
    for c in v.seam["centers"].as_array().into_iter().flatten() {
        if c["kind"] != "mark" {
            continue;
        }
        let Some(at) = xy(&c["at"]) else { continue };
        let q = v.to_sheet(at);
        let r = c["radius"].as_f64().unwrap_or(0.0) * v.scale;
        if super::marks::len(p, q) < 0.5 && (best == 0.0 || r < best) {
            best = r;
        }
    }
    best
}

/// The box an assembly occurrence covers in a view (sheet mm), from its prototype's bounds.
fn occurrence_box(ctx: &Ctx<'_>, v: &ViewPlan, occurrence: &Value) -> Option<BoxR> {
    let path = match occurrence {
        Value::String(s) => s.clone(),
        Value::Array(steps) => steps.iter().map(|s| s["name"].as_str().map(str::to_owned).unwrap_or_else(|| s.as_str().unwrap_or("").to_owned())).collect::<Vec<_>>().join("/"),
        _ => return None,
    };
    let file = ctx.doc()["views"].as_array()?.iter().find(|x| x["name"] == v.name.as_str())?["model"]["file"].as_str()?.to_owned();
    let display = ctx.displays.get(&file)?;
    if display["kind"] != "assembly" {
        return None;
    }
    let data = &display["data"];
    let occ = data["occurrences"].as_array()?.iter().find(|o| o["path"] == path.as_str())?;
    let proto = &data["prototypes"][occ["prototype"].as_u64()? as usize];
    let bounds = if proto["shape"] == "mesh" { &proto["part"]["bounds"] } else { &proto["bounds"] };
    let (lo, hi) = (floats(&bounds["min"]), floats(&bounds["max"]));
    let t = floats(&occ["transform"]);
    if lo.len() < 3 || hi.len() < 3 || t.len() < 12 {
        return None;
    }
    let mut b: Option<BoxR> = None;
    for k in 0..8 {
        let l = [0, 1, 2].map(|a| if (k >> a) & 1 == 1 { hi[a] } else { lo[a] });
        let w = [0, 1, 2].map(|r| (t[3 * r] * l[0] + t[3 * r + 1] * l[1] + t[3 * r + 2] * l[2] + t[9 + r]) * 1000.0);
        let p = v.model_to_sheet(w);
        let pb = BoxR::around(p, 0.0, 0.0);
        b = Some(b.map_or(pb, |x| x.union(&pb)));
    }
    b
}

/// A weld symbol's reference text: its type, size and the all-around and field marks.
fn weld_text(kind: &Value) -> String {
    let size = kind["size"]["value"].as_f64().or(kind["size"].as_f64()).map(|s| runmat_cad::drawing::standard::metric_text(s, 3));
    [
        Some(kind["symbol"].as_str().map(str::to_uppercase).unwrap_or_else(|| "WELD".into())),
        size,
        kind["all_around"].as_bool().filter(|b| *b).map(|_| "ALL AROUND".into()),
        kind["field"].as_bool().filter(|b| *b).map(|_| "FIELD".into()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(" ")
}
