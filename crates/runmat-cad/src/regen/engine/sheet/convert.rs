//! Convert to sheet metal: a thin solid (a shelled box, a bent bracket modelled solid) becomes a
//! sheet plate with a bend at each picked edge of its fixed face.
//!
//! The fixed face is flat, with a parallel face one sheet thickness behind it; the plate is
//! that face's outline through the thickness, holes and all. Its edges are lines or circular
//! arcs ([`super::convert_loops`]): an arc stays in the outline and never bends (a wall along a
//! curved edge, a rolled wall, stretches as it bends, so it is refused by name), and a hole
//! goes through the plate and the blank alike, refused by name when it reaches into a bend
//! zone or stops short of the face behind. At each
//! bend edge a wall stands on the face across the edge (the edge's other face): its angle to
//! the plate is the bend's angle and its extent away from the edge is the wall's height. The
//! wall becomes a bend of the sheet's radius and a flat, placed so the face the wall stood on
//! stays where it was (the inner face for an edge on the inside of the bend, the outer face for
//! one on the outside), and the plate stops where the bend starts.
//!
//! Where two bent walls meet at a convex corner of the outline, each wall stops short of the corner
//! by the other's bend setback and the plate gives up the corner inside both setbacks: the
//! corner opens into a square relief through the sheet, so the walls are ripped apart there,
//! the bends never overlap and the flat blank has the same notch. At a concave corner (or two
//! walls in line) the walls run on into the corner and meet on the plane through it that
//! bisects them, as a miter flange's sides meet, so the converted part keeps the walls whole;
//! its blank needs the same sheet twice there, and the flat pattern refuses it by the sides
//! that cross. Walls at edges that are not picked are left off;
//! with no edges picked, every edge with a wall on it bends.
//!
//! The converted plate records its outline and, per bend, the strip it develops to, so the
//! flat pattern, unfold and fold treat it as a plate with edge flanges; a fold rebuilds it
//! from the recorded tools.

use super::super::families::{labeled, line_seg, plane_normal, polygon};
use super::convert_loops::{face_loops, Piece};
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
            s.error(
                id,
                Some("fixed_face"),
                DiagCode::RegenInvalid,
                m.into(),
                Vec::new(),
            )
        };
        let Some(raw) = plane_normal(&gf) else {
            return Err(refuse(self, "the fixed face of a conversion is flat"));
        };
        // The face one thickness behind the fixed face gives the material's side.
        let tol = 1e-6
            * (1.0
                + (0..3)
                    .map(|k| gf.bbox_max[k] - gf.bbox_min[k])
                    .fold(0.0, f64::max));
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
                ((d.abs() - t).abs() <= 1e3 * tol).then_some((q, d))
            })
        };
        let behind = behind_of(fi, raw);
        let Some((back, d)) = behind else {
            return Err(refuse(
                self,
                &format!(
                    "no face lies {:.3} mm (the sheet's thickness) behind the fixed face; pick a face of a wall that thick, or set the thickness to the solid's",
                    t * 1e3
                ),
            ));
        };
        let n = if d < 0.0 { raw } else { scale(raw, -1.0) };
        // The fixed face's outline and holes, of lines and circular arcs.
        let mine: Vec<usize> = (0..topo.edges.len())
            .filter(|&k| topo.edges[k].faces.contains(&(fi as _)))
            .collect();
        let loops = match face_loops(&topo, fi, &mine, n, tol) {
            Ok(l) => l,
            Err(m) => return Err(refuse(self, &m)),
        };
        let loop_ = &loops.outer;
        let m = loop_.len();
        let pts: Vec<[f64; 3]> = loop_.iter().map(|e| e.from).collect();
        // The face across an edge of the fixed face.
        let across = |edge: usize| {
            let e = &topo.edges[edge];
            (if e.faces[0] as usize == fi {
                e.faces[1]
            } else {
                e.faces[0]
            }) as usize
        };
        // A hole must go through the sheet: its sides reach the face behind.
        for hole in &loops.holes {
            let through = hole.iter().all(|e| {
                let g = across(e.edge);
                topo.edges
                    .iter()
                    .any(|x| x.faces.contains(&(g as _)) && x.faces.contains(&(back as _)))
            });
            if !through {
                let c = scale(hole[0].from, 1e3);
                return Err(refuse(
                    self,
                    &format!(
                        "the recess in the fixed face at ({:.2}, {:.2}, {:.2}) mm stops short of the face behind; only holes through the sheet convert",
                        c[0], c[1], c[2]
                    ),
                ));
            }
        }
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
                match loop_
                    .iter()
                    .position(|le| le.edge == e.index as usize && e.body as usize == body)
                {
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
        for (k, le) in loop_.iter().enumerate() {
            if picked.as_ref().is_some_and(|ks| !ks.contains(&k)) {
                continue;
            }
            let (edge, p) = (&le.edge, &le.from);
            let g = across(*edge);
            // A wall on a curved edge (a rolled wall) stretches as it bends, so it has no flat
            // blank: refused by name when picked, or when it is a sheet and nothing is picked.
            if let Piece::Arc { radius, .. } = le.piece {
                let rolled = match topo.faces[g].geom.surface {
                    crate::refs::history::SurfaceKind::Cylinder {
                        axis_point,
                        axis_direction,
                        ..
                    } => topo.faces.iter().any(|f| match f.geom.surface {
                        crate::refs::history::SurfaceKind::Cylinder {
                            axis_point: q,
                            axis_direction: a2,
                            radius: r2,
                        } => {
                            let off = sub(q, axis_point);
                            let a = normalize(axis_direction);
                            let perp = sub(off, scale(a, dot(off, a)));
                            dot(normalize(a2), a).abs() > 1.0 - 1e-9
                                && dot(perp, perp).sqrt() <= 1e3 * tol
                                && ((r2 - radius).abs() - t).abs() <= 1e3 * tol
                        }
                        _ => false,
                    }),
                    _ => false,
                };
                if picked.is_some() || rolled {
                    return Err(self.error(
                        id,
                        Some("bend_edges"),
                        DiagCode::RegenInvalid,
                        format!(
                            "outline edge {} of the fixed face is a circular arc (radius {:.2} mm): a wall along a curved edge stretches as it bends, so it has no flat blank; {}",
                            k + 1,
                            radius * 1e3,
                            if picked.is_some() { "bend straight edges only" } else { "pick the straight edges to bend and the rolled wall is left off" }
                        ),
                        Vec::new(),
                    ));
                }
                continue;
            }
            // A wall is a sheet as thick as the plate: an end cap or a block's side is not.
            let wall = plane_normal(&topo.faces[g].geom)
                .filter(|&ng| behind_of(g, ng).is_some())
                .and_then(|ng| {
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
                    let (w, height) = if a >= b {
                        (w0, a)
                    } else {
                        (scale(w0, -1.0), b)
                    };
                    let theta = dot(out(k), w).clamp(-1.0, 1.0).acos();
                    // A thickness face (as tall as the sheet) or a face in the plate's plane is no wall.
                    (height > t + 1e3 * tol && theta > 1e-6 && theta < std::f64::consts::PI - 1e-6)
                        .then(|| {
                            let (y, c) = if dot(w, n) > 0.0 {
                                (n, 0.0)
                            } else {
                                (scale(n, -1.0), t)
                            };
                            let setback = (r + c) * (theta / 2.0).tan();
                            Wall {
                                k,
                                theta,
                                setback,
                                flat: height - setback,
                                y,
                                c,
                            }
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
            return Err(refuse(
                self,
                "no wall stands on an edge of the fixed face, so nothing bends",
            ));
        }
        // How far each wall stops short of a corner it shares with another wall: past the
        // other's bend zone, wherever across its own. `ua` and `ub` run from the corner along
        // the two edges; the corner must be convex (the outline turns left at it).
        let relief =
            |a: &Wall, b: &Wall, ua: [f64; 3], ub: [f64; 3], convex: bool| -> Option<f64> {
                let cos_phi = dot(ua, ub).clamp(-1.0, 1.0);
                let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
                (convex && sin_phi > 1e-6)
                    .then(|| (b.setback + a.setback * cos_phi.max(0.0)) / sin_phi)
            };
        let turns_left = |from: usize, to: usize| dot(cross(dir(from), dir(to)), n) > 1e-9;
        // A corner between two bent walls that is not convex (concave, or straight on): the
        // walls run to the corner and meet on the plane through it that bisects them, as a
        // miter flange's sides do, and the plate stops where both setback lines cross.
        let mitred = |k: usize| {
            let prev = (k + m - 1) % m;
            walls[k].is_some() && walls[prev].is_some() && !turns_left(prev, k)
        };
        let mut spans = vec![(0.0, 0.0); m];
        for k in 0..m {
            let Some(a) = &walls[k] else { continue };
            let prev = (k + m - 1) % m;
            let next = (k + 1) % m;
            let g0 = match &walls[prev] {
                Some(b) => {
                    relief(a, b, dir(k), scale(dir(prev), -1.0), turns_left(prev, k)).unwrap_or(0.0)
                }
                None => 0.0,
            };
            let g1 = match &walls[next] {
                Some(b) => {
                    relief(a, b, scale(dir(k), -1.0), dir(next), turns_left(k, next)).unwrap_or(0.0)
                }
                None => 0.0,
            };
            if len(k) - g0 - g1 <= 1e3 * tol {
                return Err(self.error(
                    id,
                    Some("bend_edges"),
                    DiagCode::RegenInvalid,
                    format!(
                        "outline edge {} is too short for its bend and its neighbours' reliefs",
                        k + 1
                    ),
                    Vec::new(),
                ));
            }
            spans[k] = (g0, g1);
        }
        // The plate's frame: on the face behind the fixed face, z along n.
        let fr =
            super::super::families::frame_of(sub(pts[0], scale(n, t)), dir(0), cross(n, dir(0)));
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
            Some([
                at(g0, 0.0),
                at(g0, -w.setback),
                at(l - g1, -w.setback),
                at(l - g1, 0.0),
            ])
        };
        // A corner between two bent walls loses the plate inside both bends' setbacks: a square
        // relief through the sheet, folded and flat (each wall's bend zone would otherwise
        // hold the corner on by its end face alone).
        let cut = |k: usize| walls[k].is_some() && walls[(k + m - 1) % m].is_some() && !mitred(k);
        // Where the setback lines of the walls meeting at mitred corner `k` cross.
        let meet = |k: usize| -> [f64; 2] {
            let prev = (k + m - 1) % m;
            let (a, b) = (
                walls[prev].as_ref().expect("a wall"),
                walls[k].as_ref().expect("a wall"),
            );
            let (oa, ob) = (
                to2d(add(fr.origin, out(prev))),
                to2d(add(fr.origin, out(k))),
            );
            let p = to2d(pts[k]);
            // (x − p)·oa = −sa and (x − p)·ob = −sb.
            let det = oa[0] * ob[1] - oa[1] * ob[0];
            if det.abs() < 1e-12 {
                // Straight on: both lines are one.
                return [p[0] - a.setback * oa[0], p[1] - a.setback * oa[1]];
            }
            let (ra, rb) = (-a.setback, -b.setback);
            [
                p[0] + (ra * ob[1] - rb * oa[1]) / det,
                p[1] + (oa[0] * rb - ob[0] * ra) / det,
            ]
        };
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
            } else if mitred(k) {
                fold_at.push((name(k, "s0"), meet(k)));
            } else {
                fold_at.push((name(k, "r0"), a0));
                fold_at.push((name(k, "s0"), r0));
            }
            let next_mitred = mitred((k + 1) % m);
            fold_at.push((
                name(k, "root"),
                if next_mitred { meet((k + 1) % m) } else { r1 },
            ));
            if next_cut {
                flat_at.push((key(k), a1));
                flat_at.push((name(k, "s1"), r1));
            } else if next_mitred {
                flat_at.push((key(k), q));
            } else {
                flat_at.push((key(k), q));
                fold_at.push((name(k, "s1"), a1));
                fold_at.push((key(k), q));
            }
        }
        // The outline's circular arcs, by the name of the side that reaches their end.
        let arcs: std::collections::HashMap<String, SegmentKind> = loop_
            .iter()
            .enumerate()
            .filter_map(|(k, e)| match e.piece {
                Piece::Arc {
                    center, clockwise, ..
                } => Some((
                    key(k),
                    SegmentKind::Arc {
                        center: to2d(center),
                        clockwise,
                    },
                )),
                _ => None,
            })
            .collect();
        let sides = |at: &[(String, [f64; 2])]| {
            let mut segs = Vec::new();
            let mut from = at.last().expect("an outline vertex").1;
            for (side, to) in at {
                if (to[0] - from[0]).hypot(to[1] - from[1]) > tol {
                    let mut seg = line_seg(side, from, *to);
                    if let Some(kind) = arcs.get(side) {
                        seg.kind = kind.clone();
                    }
                    segs.push(seg);
                    from = *to;
                }
            }
            segs
        };
        let mut outline = CanonicalProfile {
            frame: fr,
            loops: vec![sides(&flat_at)],
            merge_collinear: false,
        };
        let notched = sides(&fold_at);
        let mut plate_profile = CanonicalProfile {
            frame: fr,
            loops: vec![notched],
            merge_collinear: false,
        };
        // The holes, through the plate and the blank alike. Each stays clear of every bend: a
        // hole reaching past a wall's setback line would cut into its bend zone.
        for (j, hole) in loops.holes.iter().enumerate() {
            let hn = hole.len();
            let segs: Vec<ProfileSegment> = (0..hn)
                .map(|q| {
                    let e = hole[q];
                    let curve = format!("h{j}.{q}");
                    let (a, b) = (to2d(e.from), to2d(hole[(q + 1) % hn].from));
                    let mut seg = line_seg(&curve, a, b);
                    match e.piece {
                        Piece::Line => {}
                        Piece::Arc {
                            center, clockwise, ..
                        } => {
                            seg.kind = SegmentKind::Arc {
                                center: to2d(center),
                                clockwise,
                            }
                        }
                        Piece::Circle { center, radius } => {
                            let c = to2d(center);
                            seg.kind = SegmentKind::Circle { center: c, radius };
                            seg.start = [c[0] + radius, c[1]];
                            seg.end = seg.start;
                        }
                    }
                    seg
                })
                .collect();
            // The hole's reach: its vertices, and each arc's or circle's whole circle.
            let reach: Vec<([f64; 3], f64)> = hole
                .iter()
                .map(|e| match e.piece {
                    Piece::Line => (e.from, 0.0),
                    Piece::Arc { center, radius, .. } | Piece::Circle { center, radius } => {
                        (center, radius)
                    }
                })
                .collect();
            for w in walls.iter().flatten() {
                let k = w.k;
                let clear = reach.iter().all(|(p, rad)| {
                    let v = sub(*p, pts[k]);
                    let (u, inward) = (dot(v, dir(k)), -dot(v, out(k)));
                    u + rad < 0.0 || u - rad > len(k) || inward - rad >= w.setback - 10.0 * tol
                });
                if !clear {
                    let c = scale(reach[0].0, 1e3);
                    return Err(self.error(
                        id,
                        Some("bend_edges"),
                        DiagCode::RegenInvalid,
                        format!(
                            "the hole at ({:.2}, {:.2}, {:.2}) mm reaches into the bend at outline edge {}: keep it at least the bend's setback, {:.3} mm, inside the edge",
                            c[0], c[1], c[2], k + 1, w.setback * 1e3
                        ),
                        Vec::new(),
                    ));
                }
            }
            outline.loops.push(segs.clone());
            plate_profile.loops.push(segs);
        }
        let mut plate = LabeledTool::new(ToolShape::prism(plate_profile.clone(), 0.0, t));
        for seg in plate_profile.loops.iter().flatten() {
            plate.sides.push((
                seg.curve.clone(),
                FaceLabel::key(Role::SheetEdge, seg.curve.0.clone()),
            ));
        }
        plate.start = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(0));
        plate.end = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(1));
        let mut tools = vec![plate];
        let mut developed = Vec::new();
        // The plane two walls meet on at mitred corner `k`, its normal along the earlier
        // wall's edge (that wall keeps the side behind it).
        let bisector = |k: usize| {
            let prev = (k + m - 1) % m;
            let mut nrm = normalize(cross(add(out(prev), out(k)), n));
            if dot(nrm, dir(prev)) < 0.0 {
                nrm = scale(nrm, -1.0);
            }
            nrm
        };
        for w in walls.iter().flatten() {
            let k = w.k;
            let next = (k + 1) % m;
            let (g0, g1) = spans[k];
            let x = out(k);
            let on = add(pts[k], scale(w.y, w.c));
            let start = add(on, scale(dir(k), g0));
            let end = add(on, scale(dir(k), len(k) - g1));
            let shift = -w.setback;
            let frame = super::super::families::frame_of(add(start, scale(x, shift)), x, w.y);
            // At a mitred corner the bend and flat run past the corner and are clipped on the
            // bisecting plane.
            let past = |other: &Option<Wall>| {
                other.as_ref().map_or(0.0, |o| o.setback) + w.setback + r + t
            };
            let mut clip = Vec::new();
            let mut from = start;
            let mut to = end;
            if mitred(k) {
                from = sub(start, scale(dir(k), past(&walls[(k + m - 1) % m])));
                let nrm = scale(bisector(k), -1.0);
                clip.push(ClipPlane {
                    origin: pts[k],
                    normal: nrm,
                    label: FaceLabel::key(Role::SheetEdge, key(k)).with_ordinal(9),
                });
            }
            if mitred(next) {
                to = add(end, scale(dir(k), past(&walls[next])));
                clip.push(ClipPlane {
                    origin: pts[next],
                    normal: bisector(next),
                    label: FaceLabel::key(Role::SheetEdge, key(k)).with_ordinal(10),
                });
            }
            let mut made = bend_and_flat(&frame, from, to, w.theta, r, t, w.flat, &key(k));
            for tool in &mut made {
                tool.clip = clip.clone();
            }
            tools.extend(made);
            developed.push(FlangeEdgeIn {
                start,
                end,
                frame,
                shift,
                theta: w.theta,
                radius: r,
                flat: w.flat,
            });
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
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: false,
                },
                start: at(r, a0),
                end: at(r, a1),
            },
            line_seg("e1", at(r, a1), at(r + t, a1)),
            ProfileSegment {
                curve: sid("out"),
                kind: SegmentKind::Arc {
                    center: c,
                    clockwise: true,
                },
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
