//! A sampled tool's labels and exact surfaces, carried through relabelling and rigid moves.

use super::*;

impl Sampled {
    pub(in crate::regen::fake) fn map_labels(&mut self, fix: &mut dyn FnMut(&mut HistorySource)) {
        for s in &mut self.surfaces {
            fix(&mut s.label);
        }
    }

    pub(in crate::regen::fake) fn relabel(&mut self, seed: u32, instance: &InstanceKey) {
        for s in &mut self.surfaces {
            if let HistorySource::Generated {
                seed: sd,
                instance: inst,
                ..
            } = &mut s.label
            {
                *sd = Some(seed);
                *inst = Some(instance.clone());
            }
        }
    }

    /// The same tool moved by `t` (applied after the tool's own placement).
    pub(in crate::regen::fake) fn transformed(&self, t: &Transform) -> Sampled {
        let mut out = self.clone();
        out.to_local = self.to_local.compose(&t.inverse());
        let (lo, hi) = moved_box(self.lo, self.hi, t);
        out.lo = lo;
        out.hi = hi;
        out.exact = Vec::new();
        for s in &mut out.surfaces {
            s.hint = move_hint(&s.hint, t);
            s.plane = None;
        }
        out.recompute_exact();
        out
    }

    /// Exact planes: surfaces whose plane is normal to a world axis. Found from the hint's
    /// normal and a point on the surface.
    pub(super) fn recompute_exact(&mut self) {
        let world = self.to_local.inverse();
        let mut exact = Vec::new();
        let anchors = self.plane_anchors();
        for (i, s) in self.surfaces.iter_mut().enumerate() {
            if let (SurfaceKind::Plane { normal }, Some(anchor)) = (s.hint, anchors[i]) {
                if let Some(a) = axis_along(normal) {
                    let w = world.apply(anchor);
                    s.plane = Some((a, w[a]));
                    exact.push((a, w[a]));
                    continue;
                }
            }
            s.plane = None;
        }
        self.exact = exact;
    }

    /// The exact surface of surface `i`, in world coordinates, where the shape has a chart
    /// for it (see [`crate::regen::fake::exact`]). Lofts and drafted or thin extrusion walls have none.
    pub(in crate::regen::fake) fn exact_of(&self, i: usize) -> Option<Exact> {
        let world = self.to_local.inverse();
        let plane_at = |o: [f64; 3], e1: [f64; 3], e2: [f64; 3]| {
            Some(Exact::plane(
                world.apply(o),
                world.apply_vector(e1),
                world.apply_vector(e2),
            ))
        };
        let (x, y) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        match &self.shape {
            Shape::Prism {
                loops,
                z0,
                z1,
                caps,
                ..
            } => {
                if i == caps[0] {
                    return plane_at([0.0, 0.0, *z0], x, y);
                }
                if i == caps[1] {
                    return plane_at([0.0, 0.0, *z1], x, y);
                }
                let (c, _) = loops.iter().flatten().find(|(_, si)| *si == i)?;
                Exact::extruded(world, c.clone(), [0.0; 3], [0.0, 0.0, 1.0])
            }
            Shape::Revolve {
                loops,
                ap,
                ad,
                e1,
                sense,
                angle,
                caps,
                ..
            } => {
                if let Some([s, e]) = caps {
                    if i == *s || i == *e {
                        let th = if i == *s { 0.0 } else { *angle };
                        let ad3 = [ad[0], ad[1], 0.0];
                        let e13 = [e1[0], e1[1], 0.0];
                        let e23 = {
                            let c = [
                                ad3[1] * e13[2] - ad3[2] * e13[1],
                                ad3[2] * e13[0] - ad3[0] * e13[2],
                                ad3[0] * e13[1] - ad3[1] * e13[0],
                            ];
                            [c[0] * sense, c[1] * sense, c[2] * sense]
                        };
                        let radial = [
                            e13[0] * th.cos() + e23[0] * th.sin(),
                            e13[1] * th.cos() + e23[1] * th.sin(),
                            e13[2] * th.cos() + e23[2] * th.sin(),
                        ];
                        return plane_at([ap[0], ap[1], 0.0], ad3, radial);
                    }
                }
                let (c, _) = loops
                    .iter()
                    .flatten()
                    .find(|(c, si)| *si == i && !self.on_axis(c))?;
                Exact::revolved(world, c.clone(), *ap, *ad, *e1, *sense)
            }
            Shape::Sweep {
                loops,
                offsets,
                sides,
                caps,
                ..
            } => {
                if i == caps[0] {
                    return plane_at([0.0; 3], x, y);
                }
                if i == caps[1] {
                    return plane_at(offsets[offsets.len() - 1], x, y);
                }
                for (li, l) in loops.iter().enumerate() {
                    for (ci, c) in l.iter().enumerate() {
                        if let Some(j) = sides[li][ci].iter().position(|s| *s == i) {
                            // One side surface per path segment; a curve reused across
                            // segments of one direction is still one extrusion.
                            let segs: Vec<usize> = sides[li][ci]
                                .iter()
                                .enumerate()
                                .filter(|(_, s)| **s == i)
                                .map(|(k, _)| k)
                                .collect();
                            let d = sub3(offsets[j + 1], offsets[j]);
                            let same = segs.iter().all(|&k| {
                                let e = sub3(offsets[k + 1], offsets[k]);
                                let (a, b) = (norm3(d), norm3(e));
                                a > 1e-300
                                    && b > 1e-300
                                    && (dot3(d, e) / (a * b) - 1.0).abs() < 1e-9
                            });
                            if !same {
                                return None;
                            }
                            return Exact::extruded(world, c.clone(), offsets[j], d);
                        }
                    }
                }
                None
            }
            Shape::Loft { .. } => None,
            Shape::Image { solid } => {
                let f = solid.faces.get(i)?;
                let local = match (&f.exact, f.plane) {
                    (Some(e), _) => (**e).clone(),
                    (None, Some(p)) => crate::regen::fake::mesh::grid_plane(&p),
                    _ => return None,
                };
                Some(local.transformed(&world))
            }
            Shape::Extrusion(x) => {
                if x.draft.is_some() {
                    return None;
                }
                if i == x.caps[0] {
                    return x
                        .start
                        .fixed()
                        .and_then(|z| plane_at([0.0, 0.0, z], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
                }
                if i == x.caps[1] {
                    return x
                        .end
                        .fixed()
                        .and_then(|z| plane_at([0.0, 0.0, z], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
                }
                let outside = x.thin.map(|t| t.1).unwrap_or(0.0);
                let (c, _, _) = x.loops.iter().flatten().find(|(_, o, _)| *o == i)?;
                if outside.abs() > 1e-12 {
                    return None;
                }
                Exact::extruded(world, c.clone(), [0.0; 3], [0.0, 0.0, 1.0])
            }
        }
    }

    /// A local point on each planar surface (where the shape knows one).
    pub(super) fn plane_anchors(&self) -> Vec<Option<[f64; 3]>> {
        let mut out = vec![None; self.surfaces.len()];
        match &self.shape {
            Shape::Prism {
                loops,
                z0,
                z1,
                caps,
                ..
            } => {
                for l in loops {
                    for (c, si) in l {
                        if let Some((a, _)) = c.line() {
                            out[*si] = Some([a[0], a[1], *z0]);
                        }
                    }
                }
                out[caps[0]] = Some([0.0, 0.0, *z0]);
                out[caps[1]] = Some([0.0, 0.0, *z1]);
            }
            Shape::Revolve {
                loops, ap, caps, ..
            } => {
                // A profile line perpendicular to the axis sweeps a plane through its points.
                for l in loops {
                    for (c, si) in l {
                        if let Some((a, _)) = c.line() {
                            out[*si] = Some([a[0], a[1], 0.0]);
                        }
                    }
                }
                if let Some([s, e]) = caps {
                    let a = [ap[0], ap[1], 0.0];
                    out[*s] = Some(a);
                    out[*e] = Some(a);
                }
            }
            Shape::Sweep {
                offsets,
                caps,
                loops,
                sides,
                ..
            } => {
                out[caps[0]] = Some([0.0, 0.0, 0.0]);
                out[caps[1]] = Some(offsets[offsets.len() - 1]);
                for (li, l) in loops.iter().enumerate() {
                    for (ci, c) in l.iter().enumerate() {
                        if let Some((a, _)) = c.line() {
                            for (j, si) in sides[li][ci].iter().enumerate() {
                                let o = offsets[j];
                                out[*si] = Some([a[0] + o[0], a[1] + o[1], o[2]]);
                            }
                        }
                    }
                }
            }
            Shape::Loft { z, caps, .. } => {
                out[caps[0]] = Some([0.0, 0.0, z[0]]);
                out[caps[1]] = Some([0.0, 0.0, z[z.len() - 1]]);
            }
            Shape::Image { solid } => {
                for (i, f) in solid.faces.iter().enumerate() {
                    if let Some(r) = f.rects.first() {
                        out[i] = Some(r.lo);
                    }
                }
            }
            Shape::Extrusion(x) => {
                let hinge = x.draft.map(|d| d.0);
                for (c, o, inn) in x.loops.iter().flatten() {
                    if let Some((a, _)) = c.line() {
                        let z = hinge.or(x.start.fixed()).unwrap_or(0.0);
                        out[*o] = Some([a[0], a[1], z]);
                        if let (Some(k), Some((i, _, _)), None) = (inn, x.thin, x.draft) {
                            if let Some(b) = inner_line(c, i) {
                                out[*k] = Some([b[0], b[1], z]);
                            }
                        }
                    }
                }
                if let Some(z) = x.start.fixed() {
                    out[x.caps[0]] = Some([0.0, 0.0, z]);
                }
                if let Some(z) = x.end.fixed() {
                    out[x.caps[1]] = Some([0.0, 0.0, z]);
                }
            }
        }
        out
    }
}
