//! Convert to sheet metal: a thin solid (a shelled box, a bent bracket modelled solid) becomes a
//! sheet plate with a bend at each picked edge of its fixed face.
//!
//! The fixed face is flat, bounded by straight edges in one loop, with a parallel face one
//! sheet thickness behind it; the plate is that face's outline through the thickness. At each
//! bend edge a wall stands on the face across the edge (the edge's other face): its angle to
//! the plate is the bend's angle and its extent away from the edge is the wall's height. The
//! wall becomes a bend of the sheet's radius and a flat, placed so the face the wall stood on
//! stays where it was (the inner face for an edge on the inside of the bend, the outer face for
//! one on the outside), and the plate stops where the bend starts.
//!
//! Where two bent walls meet at a corner of the outline, each wall stops short of the corner
//! by the other's bend setback and the plate gives up the corner inside both setbacks: the
//! corner opens into a square relief through the sheet, so the walls are ripped apart there,
//! the bends never overlap and the flat blank has the same notch. Walls at edges that are not picked are left off;
//! with no edges picked, every edge with a wall on it bends.
//!
//! The converted plate records its outline and, per bend, the strip it develops to, so the
//! flat pattern, unfold and fold treat it as a plate with edge flanges; a fold rebuilds it
//! from the recorded tools.

use super::super::families::{labeled, line_seg, plane_normal, polygon};
use super::*;
use crate::regen::records::ConvertedPlate;

/// One bent wall, in world coordinates.
struct Wall {
    /// Index of its edge in the outline.
    k: usize,
    theta: f64,
    /// Setback of the bend from the face the wall stood on: `(r + c) tan(θ/2)`.
    setback: f64,
    flat: f64,
    /// `y`: the outward normal of the sheet face the wall turns towards.
    y: [f64; 3],
    /// How far the edge lies from the `y = 0` sheet face (0 or the thickness).
    c: f64,
}

impl<'a, 'b> Run<'a, 'b> {
    pub(in crate::regen::engine) fn convert_to_sheet(
        &mut self,
        i: usize,
        id: &FeatureId,
        fixed_face: &Reference,
        bend_edges: &[Reference],
        rules: &crate::model::sheet_metal::SheetMetalRules,
        simple: &NamingContext,
    ) -> Result<Compiled, Halt> {
        // A fold rebuilds the converted plate from its recorded tools.
        if self.doc.feature_index(id) != Some(i) {
            return match self.st.replays.get(id) {
                Some(made) => Ok(op(made.clone(), Vec::new(), simple.clone())),
                None => Err(self.not_lowered(id, "folding this sheet")),
            };
        }
        let rules = self.sheet_rules(id, rules)?;
        let (t, r) = (rules.thickness, rules.bend_radius);
        if r <= 0.0 {
            return Err(self.error(
                id,
                Some("rules.bend_radius"),
                DiagCode::RegenInvalid,
                "converted bends need a positive bend radius".into(),
                Vec::new(),
            ));
        }
        let h = self.face_handle(i, id, "fixed_face", fixed_face)?;
        let body = h.body as usize;
        let topo = self.st.bodies[body].topology.clone();
        let fi = h.index as usize;
        let gf = topo.faces[fi].geom;
        let refuse = |s: &mut Self, m: &str| {
            s.error(id, Some("fixed_face"), DiagCode::RegenInvalid, m.into(), Vec::new())
        };
        let Some(raw) = plane_normal(&gf) else {
            return Err(refuse(self, "the fixed face of a conversion is flat"));
        };
        // The face one thickness behind the fixed face gives the material's side.
        let tol = 1e-6 * (1.0 + (0..3).map(|k| gf.bbox_max[k] - gf.bbox_min[k]).fold(0.0, f64::max));
        // A planar face `j` (normal `nj`) is a sheet's face when another lies one thickness
        // behind it; answers the signed offset along `nj`.
        let behind_of = |j: usize, nj: [f64; 3]| {
            let cj = topo.faces[j].geom.centroid;
            topo.faces.iter().enumerate().find_map(|(q, f)| {
                let n = plane_normal(&f.geom)?;
                if q == j || dot(n, nj).abs() < 1.0 - 1e-9 {
                    return None;
                }
                let d = dot(sub(f.geom.centroid, cj), nj);
                ((d.abs() - t).abs() <= 1e3 * tol).then_some(d)
            })
        };
        let behind = behind_of(fi, raw);
        let Some(d) = behind else {
            return Err(refuse(
                self,
                &format!(
                    "no face lies {:.3} mm (the sheet's thickness) behind the fixed face; pick a face of a wall that thick, or set the thickness to the solid's",
                    t * 1e3
                ),
            ));
        };
        let n = if d < 0.0 { raw } else { scale(raw, -1.0) };
        // The fixed face's outline: one loop of straight edges.
        let mine: Vec<usize> = (0..topo.edges.len())
            .filter(|&k| topo.edges[k].faces.contains(&(fi as _)))
            .collect();
        let Some(loop_) = outline_loop(&topo, &mine, n, tol) else {
            return Err(self.not_lowered(
                id,
                "converting a face with holes or curved edges (the fixed face is one loop of straight edges)",
            ));
        };
        let m = loop_.len();
        let pts: Vec<[f64; 3]> = loop_.iter().map(|(_, p)| *p).collect();
        let dir = |k: usize| normalize(sub(pts[(k + 1) % m], pts[k]));
        let len = |k: usize| {
            let v = sub(pts[(k + 1) % m], pts[k]);
            dot(v, v).sqrt()
        };
        let out = |k: usize| normalize(cross(dir(k), n));
        // The walls on the outline's edges.
        let picked: Option<Vec<usize>> = if bend_edges.is_empty() {
            None
        } else {
            let set = self.list(i, id, "bend_edges", bend_edges, Some(TopoKind::Edge))?;
            let mut ks = Vec::new();
            for e in &set {
                match loop_.iter().position(|(edge, _)| *edge == e.index as usize && e.body as usize == body) {
                    Some(k) => ks.push(k),
                    None => {
                        return Err(self.error(
                            id,
                            Some("bend_edges"),
                            DiagCode::RefKindMismatch,
                            "each bend edge is an edge of the fixed face".into(),
                            Vec::new(),
                        ))
                    }
                }
            }
            Some(ks)
        };
        let mut walls: Vec<Option<Wall>> = (0..m).map(|_| None).collect();
        for (k, (edge, p)) in loop_.iter().enumerate() {
            if picked.as_ref().is_some_and(|ks| !ks.contains(&k)) {
                continue;
            }
            let e = &topo.edges[*edge];
            let g = if e.faces[0] as usize == fi { e.faces[1] } else { e.faces[0] } as usize;
            // A wall is a sheet as thick as the plate: an end cap or a block's side is not.
            let wall = plane_normal(&topo.faces[g].geom).filter(|&ng| behind_of(g, ng).is_some()).and_then(|ng| {
                let w0 = normalize(cross(ng, dir(k)));
                // The wall runs from the edge towards its far vertices.
                let far = |w: [f64; 3]| {
                    topo.edges
                        .iter()
                        .filter(|x| x.faces.contains(&(g as _)))
                        .flat_map(|x| [x.geom.start, x.geom.end])
                        .map(|v| dot(sub(v, *p), w))
                        .fold(0.0, f64::max)
                };
                let (a, b) = (far(w0), far(scale(w0, -1.0)));
                let (w, height) = if a >= b { (w0, a) } else { (scale(w0, -1.0), b) };
                let theta = dot(out(k), w).clamp(-1.0, 1.0).acos();
                // A thickness face (as tall as the sheet) or a face in the plate's plane is no wall.
                (height > t + 1e3 * tol && theta > 1e-6 && theta < std::f64::consts::PI - 1e-6)
                    .then(|| {
                        let (y, c) = if dot(w, n) > 0.0 { (n, 0.0) } else { (scale(n, -1.0), t) };
                        let setback = (r + c) * (theta / 2.0).tan();
                        Wall { k, theta, setback, flat: height - setback, y, c }
                    })
            });
            match wall {
                Some(w) if w.flat <= 0.0 => {
                    return Err(self.error(
                        id,
                        Some("bend_edges"),
                        DiagCode::RegenInvalid,
                        format!("the wall on outline edge {} is shorter than its bend; use a smaller bend radius", k + 1),
                        Vec::new(),
                    ))
                }
                Some(w) => walls[k] = Some(w),
                None if picked.is_some() => {
                    return Err(self.error(
                        id,
                        Some("bend_edges"),
                        DiagCode::RegenInvalid,
                        format!("outline edge {} of the fixed face has no wall standing on it to bend", k + 1),
                        Vec::new(),
                    ))
                }
                None => {}
            }
        }
        if walls.iter().all(Option::is_none) {
            return Err(refuse(self, "no wall stands on an edge of the fixed face, so nothing bends"));
        }
        // How far each wall stops short of a corner it shares with another wall: past the
        // other's bend zone, wherever across its own. `ua` and `ub` run from the corner along
        // the two edges; the corner must be convex (the outline turns left at it).
        let relief = |a: &Wall, b: &Wall, ua: [f64; 3], ub: [f64; 3], convex: bool| -> Option<f64> {
            let cos_phi = dot(ua, ub).clamp(-1.0, 1.0);
            let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
            (convex && sin_phi > 1e-6).then(|| (b.setback + a.setback * cos_phi.max(0.0)) / sin_phi)
        };
        let turns_left = |from: usize, to: usize| dot(cross(dir(from), dir(to)), n) > 1e-9;
        let mut spans = vec![(0.0, 0.0); m];
        for k in 0..m {
            let Some(a) = &walls[k] else { continue };
            let prev = (k + m - 1) % m;
            let next = (k + 1) % m;
            let concave = || "bends on both sides of a concave corner of the fixed face";
            let g0 = match &walls[prev] {
                Some(b) => match relief(a, b, dir(k), scale(dir(prev), -1.0), turns_left(prev, k)) {
                    Some(g) => g,
                    None => return Err(self.not_lowered(id, concave())),
                },
                None => 0.0,
            };
            let g1 = match &walls[next] {
                Some(b) => match relief(a, b, scale(dir(k), -1.0), dir(next), turns_left(k, next)) {
                    Some(g) => g,
                    None => return Err(self.not_lowered(id, concave())),
                },
                None => 0.0,
            };
            if len(k) - g0 - g1 <= 1e3 * tol {
                return Err(self.error(
                    id,
                    Some("bend_edges"),
                    DiagCode::RegenInvalid,
                    format!("outline edge {} is too short for its bend and its neighbours' reliefs", k + 1),
                    Vec::new(),
                ));
            }
            spans[k] = (g0, g1);
        }
        // The plate's frame: on the face behind the fixed face, z along n.
        let fr = super::super::families::frame_of(sub(pts[0], scale(n, t)), dir(0), cross(n, dir(0)));
        let to2d = |p: [f64; 3]| {
            let q = sub(p, fr.origin);
            [dot(q, fr.x), dot(q, fr.y)]
        };
        let key = |k: usize| format!("c{k}");
        let name = |k: usize, part: &str| format!("{}.{part}", key(k));
        // Each walled edge's stations: where its bend runs along the edge line (`a0`, `a1`) and
        // the same points a setback in, where the plate stops (`r0`, `r1`).
        let station = |k: usize| {
            let w = walls[k].as_ref()?;
            let (g0, g1) = spans[k];
            let p = to2d(pts[k]);
            let d2 = to2d(add(fr.origin, dir(k)));
            let o2 = to2d(add(fr.origin, out(k)));
            let at = |u: f64, v: f64| [p[0] + u * d2[0] + v * o2[0], p[1] + u * d2[1] + v * o2[1]];
            let l = len(k);
            Some([at(g0, 0.0), at(g0, -w.setback), at(l - g1, -w.setback), at(l - g1, 0.0)])
        };
        // A corner between two bent walls loses the plate inside both bends' setbacks: a square
        // relief through the sheet, folded and flat (each wall's bend zone would otherwise
        // hold the corner on by its end face alone).
        let cut = |k: usize| walls[k].is_some() && walls[(k + m - 1) % m].is_some();
        // The outline (the flat blank's plate) and the folded plate, each as the vertices it
        // passes with the name of the side that reaches each one.
        let mut flat_at: Vec<(String, [f64; 2])> = Vec::new();
        let mut fold_at: Vec<(String, [f64; 2])> = Vec::new();
        for k in 0..m {
            let q = to2d(pts[(k + 1) % m]);
            let next_cut = cut((k + 1) % m);
            let Some([a0, r0, r1, a1]) = station(k) else {
                flat_at.push((key(k), q));
                fold_at.push((key(k), q));
                continue;
            };
            if cut(k) {
                flat_at.push((name(k, "relief"), r0));
                flat_at.push((name(k, "s0"), a0));
                fold_at.push((name(k, "relief"), r0));
            } else {
                fold_at.push((name(k, "r0"), a0));
                fold_at.push((name(k, "s0"), r0));
            }
            fold_at.push((name(k, "root"), r1));
            if next_cut {
                flat_at.push((key(k), a1));
                flat_at.push((name(k, "s1"), r1));
            } else {
                flat_at.push((key(k), q));
                fold_at.push((name(k, "s1"), a1));
                fold_at.push((key(k), q));
            }
        }
        let sides = |at: &[(String, [f64; 2])]| {
            let mut segs = Vec::new();
            let mut from = at.last().expect("an outline vertex").1;
            for (side, to) in at {
                if (to[0] - from[0]).hypot(to[1] - from[1]) > tol {
                    segs.push(line_seg(side, from, *to));
                    from = *to;
                }
            }
            segs
        };
        let outline = CanonicalProfile {
            frame: fr,
            loops: vec![sides(&flat_at)],
            merge_collinear: false,
        };
        let notched = sides(&fold_at);
        let plate_profile = CanonicalProfile {
            frame: fr,
            loops: vec![notched],
            merge_collinear: false,
        };
        let mut plate = LabeledTool::new(ToolShape::prism(plate_profile.clone(), 0.0, t));
        for seg in plate_profile.loops.iter().flatten() {
            plate.sides.push((seg.curve.clone(), FaceLabel::key(Role::SheetEdge, seg.curve.0.clone())));
        }
        plate.start = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(0));
        plate.end = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(1));
        let mut tools = vec![plate];
        let mut developed = Vec::new();
        for w in walls.iter().flatten() {
            let k = w.k;
            let (g0, g1) = spans[k];
            let x = out(k);
            let on = add(pts[k], scale(w.y, w.c));
            let start = add(on, scale(dir(k), g0));
            let end = add(on, scale(dir(k), len(k) - g1));
            let shift = -w.setback;
            let frame = super::super::families::frame_of(add(start, scale(x, shift)), x, w.y);
            tools.extend(bend_and_flat(&frame, start, end, w.theta, r, t, w.flat, &key(k)));
            developed.push(FlangeEdgeIn { start, end, frame, shift, theta: w.theta, radius: r, flat: w.flat });
        }
        let edges = self.developed_flange_edges(i, id, body, id, &outline, &rules, developed)?;
        self.st.records.insert(
            id.clone(),
            FeatureRecord::Sheet {
                rules,
                section: None,
                depth: 0.0,
                plate: Some(Box::new(ConvertedPlate { outline, edges })),
            },
        );
        let made = labeled(tools, Vec::new(), Combine::New, false);
        self.st.replays.insert(id.clone(), made.clone());
        Ok(self.in_place(op(made, Vec::new(), simple.clone()), body))
    }
}

/// The edges `mine` (indices into `topo.edges`) as one closed loop of straight edges,
/// counter-clockwise about `n`, starting at the least vertex: each entry is the edge and its
/// first point along the loop.
fn outline_loop(
    topo: &crate::refs::history::Topology,
    mine: &[usize],
    n: [f64; 3],
    tol: f64,
) -> Option<Vec<(usize, [f64; 3])>> {
    if mine.len() < 3 {
        return None;
    }
    let close = |a: [f64; 3], b: [f64; 3]| {
        let d = sub(a, b);
        dot(d, d).sqrt() <= 1e3 * tol
    };
    for &k in mine {
        let g = topo.edges[k].geom;
        let chord = sub(g.end, g.start);
        if (dot(chord, chord).sqrt() - g.length).abs() > 1e3 * tol {
            return None;
        }
    }
    let least = |a: &[f64; 3], b: &[f64; 3]| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    let first = mine
        .iter()
        .flat_map(|&k| [topo.edges[k].geom.start, topo.edges[k].geom.end])
        .min_by(least)?;
    let mut out: Vec<(usize, [f64; 3])> = Vec::new();
    let mut at = first;
    let mut used = vec![false; mine.len()];
    // Two edges leave the first vertex; take either and fix the turn afterwards.
    loop {
        let next = (0..mine.len()).find(|&j| {
            let g = topo.edges[mine[j]].geom;
            !used[j] && (close(g.start, at) || close(g.end, at))
        });
        let Some(j) = next else { break };
        used[j] = true;
        let g = topo.edges[mine[j]].geom;
        out.push((mine[j], at));
        at = if close(g.start, at) { g.end } else { g.start };
        if close(at, first) {
            break;
        }
    }
    if out.len() != mine.len() || !close(at, first) {
        return None;
    }
    // Counter-clockwise about n: the shoelace area along n is positive.
    let area: f64 = (0..out.len())
        .map(|k| dot(cross(out[k].1, out[(k + 1) % out.len()].1), n))
        .sum();
    if area < 0.0 {
        // Reverse: edge k of the reversed loop is edge m-1-k, starting where it used to end.
        let m = out.len();
        let rev: Vec<(usize, [f64; 3])> = (0..m)
            .map(|k| (out[m - 1 - k].0, out[(m - k) % m].1))
            .collect();
        out = rev;
    }
    Some(out)
}

/// A bend (inner radius `r`, turning by `theta` towards the frame's y) and its flat, along an
/// edge from `start` to `end`, labelled by `key` (bend faces 0 inner and 1 outer, flat faces 0
/// and 1, its tip edge 1, the ends 2 to 5 as an edge flange numbers them).
#[allow(clippy::too_many_arguments)]
fn bend_and_flat(
    fr: &Frame,
    start: [f64; 3],
    end: [f64; 3],
    theta: f64,
    r: f64,
    t: f64,
    flat: f64,
    key: &str,
) -> Vec<LabeledTool> {
    let s0 = dot(sub(start, fr.origin), fr.z);
    let s1 = dot(sub(end, fr.origin), fr.z);
    let (z0, z1) = (s0.min(s1), s0.max(s1));
    let lab = |role: Role, o: u32| FaceLabel::key(role, key.to_string()).with_ordinal(o);
    let c = [0.0, r];
    let a0 = -std::f64::consts::FRAC_PI_2;
    let a1 = a0 + theta;
    let at = |rad: f64, a: f64| [c[0] + rad * a.cos(), c[1] + rad * a.sin()];
    let sid = |s: &str| crate::ids::SketchEntityId::new(s);
    let bend_profile = CanonicalProfile {
        frame: *fr,
        loops: vec![vec![
            ProfileSegment {
                curve: sid("in"),
                kind: SegmentKind::Arc { center: c, clockwise: false },
                start: at(r, a0),
                end: at(r, a1),
            },
            line_seg("e1", at(r, a1), at(r + t, a1)),
            ProfileSegment {
                curve: sid("out"),
                kind: SegmentKind::Arc { center: c, clockwise: true },
                start: at(r + t, a1),
                end: at(r + t, a0),
            },
            line_seg("e0", at(r + t, a0), at(r, a0)),
        ]],
        merge_collinear: false,
    };
    let mut bend = LabeledTool::new(ToolShape::prism(bend_profile, z0, z1));
    bend.sides.push((sid("in"), lab(Role::BendFace, 0)));
    bend.sides.push((sid("out"), lab(Role::BendFace, 1)));
    bend.sides.push((sid("e1"), lab(Role::SheetEdge, 6)));
    bend.sides.push((sid("e0"), lab(Role::SheetEdge, 7)));
    bend.start = Some(lab(Role::SheetEdge, 2));
    bend.end = Some(lab(Role::SheetEdge, 3));
    let d = [theta.cos(), theta.sin()];
    let (ci, co) = (at(r, a1), at(r + t, a1));
    let fwd = |p: [f64; 2]| [p[0] + flat * d[0], p[1] + flat * d[1]];
    let corners = vec![
        ("in".to_string(), ci),
        ("tip".to_string(), fwd(ci)),
        ("out".to_string(), fwd(co)),
        ("root".to_string(), co),
    ];
    let mut ft = LabeledTool::new(ToolShape::prism(polygon(*fr, &corners), z0, z1));
    ft.sides.push((sid("in"), lab(Role::SheetFace, 0)));
    ft.sides.push((sid("out"), lab(Role::SheetFace, 1)));
    ft.sides.push((sid("tip"), lab(Role::SheetEdge, 1)));
    ft.sides.push((sid("root"), lab(Role::SheetEdge, 8)));
    ft.start = Some(lab(Role::SheetEdge, 4));
    ft.end = Some(lab(Role::SheetEdge, 5));
    vec![bend, ft]
}
