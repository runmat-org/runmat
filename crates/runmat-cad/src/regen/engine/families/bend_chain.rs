//! Sheet-metal bends along a chain of edges: sketched bends, jogs and the bend tools they need.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    /// A sketched bend (one bend) or a jog (a bend, a flat and the opposite bend) of a flat
    /// plate along a sketch line.
    ///
    /// The line is the first bend's centre line in the flat, so the plate's developed length
    /// is kept: the strip each bend consumes is its bend allowance, centred on the line for
    /// the first bend and following it for the rest. The side `fixed` (by the line's left
    /// normal; the jog keeps the negative side) stays; the rest turns by `angle` towards the
    /// face's outward normal (away from it when negative). A jog's offset is how far the
    /// moved flat's faces move along the sheet normal; its middle flat is what is left of
    /// the offset after the two bends.
    ///
    /// The plate is the base flange's closed profile, unchanged by later features; its
    /// edges cross the bend strip square to the line, and holes stay clear of the strip.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bend_chain(
        &mut self,
        i: usize,
        id: &FeatureId,
        face: &Reference,
        line: &(FeatureId, crate::ids::SketchEntityId),
        theta: f64,
        radius: Option<f64>,
        jog: Option<f64>,
        fixed: Option<crate::refs::name::PlaneSide>,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        use crate::refs::name::PlaneSide;
        let h = self.face_handle(i, id, "face", face)?;
        let body = h.body as usize;
        let invalid = |me: &mut Self, param: &str, message: String| {
            me.error(id, Some(param), DiagCode::RegenInvalid, message, Vec::new())
        };
        let Some((maker, rules, None, _)) = self.sheet_of(body) else {
            return Err(self.not_lowered(
                id,
                "bends of a sheet made by a bent base flange (bend a flat plate, or add the bend to the base flange's chain)",
            ));
        };
        // The flat being bent: the plate's unmoved part, or a flat an earlier bend moved
        // (its parent), whose own outline and place that bend recorded.
        let face_names = self.st.entity_names(&h);
        let parent: Option<(FeatureId, CanonicalProfile, Transform)> = if face_names
            .iter()
            .any(|n| n.maker() == &maker)
        {
            None
        } else {
            let found = face_names.iter().find_map(|n| match n {
                EntityName::Generated(g) if g.role == Role::SheetFace => {
                    match (&g.tag, self.st.records.get(&g.feature)) {
                        (
                            crate::refs::name::SourceTag::Key { key },
                            Some(FeatureRecord::PlateBend {
                                base, flat, motion, ..
                            }),
                        ) if base == &maker && key.ends_with(".flat") => {
                            Some((g.feature.clone(), flat.clone(), *motion))
                        }
                        _ => None,
                    }
                }
                _ => None,
            });
            match found {
                    Some(p) => Some(p),
                    None => {
                        return Err(self.not_lowered(
                            id,
                            "bends of a bend face or of a jog's middle flat (bend the plate or a flat a bend moved)",
                        ))
                    }
                }
        };
        let Some(n_face) = self.face_normal(&h) else {
            return Err(invalid(self, "face", "the bend face is not flat".into()));
        };
        let t = rules.thickness;
        let r = radius.unwrap_or(rules.bend_radius);
        if r <= 0.0 {
            return Err(invalid(
                self,
                "radius",
                "a bend needs a positive radius".into(),
            ));
        }
        let a = theta.abs();
        if a <= 1e-9 || a > std::f64::consts::PI + 1e-12 {
            return Err(invalid(
                self,
                "angle",
                "a bend angle lies in (0°, 180°]".into(),
            ));
        }
        let ba = crate::model::sheet_metal::bend_allowance(&rules.rule, t, r, a)
            .map_err(|m| invalid(self, "angle", m))?;
        // The middle flat of a jog: the rise left after two bends.
        let middle = match jog {
            None => None,
            Some(rise) => {
                if a >= std::f64::consts::PI - 1e-9 {
                    return Err(invalid(
                        self,
                        "angle",
                        "a jog's angle lies in (0°, 180°)".into(),
                    ));
                }
                let flat = (rise.abs() - (2.0 * r + t) * (1.0 - a.cos())) / a.sin();
                if flat < -1e-12 {
                    return Err(invalid(
                        self,
                        "offset",
                        format!(
                            "the offset is less than the two bends rise ({:.3} mm at this angle and radius)",
                            (2.0 * r + t) * (1.0 - a.cos()) * 1e3
                        ),
                    ));
                }
                Some(flat.max(0.0))
            }
        };
        // The plate: the base flange's profile, between z = 0 and z = t of its frame.
        let profile_ref = match self.doc.feature(&maker).map(|f| &f.kind) {
            Some(FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { profile, .. })) => {
                profile.clone()
            }
            _ => return Err(self.not_lowered(id, "bends of this sheet")),
        };
        let base_plate = self.profile(i, id, "face", &profile_ref)?;
        let plate = match &parent {
            Some((_, flat, _)) => flat.clone(),
            None => base_plate.clone(),
        };
        // World to the developed plate: undo the parent's motion.
        let back = parent.as_ref().map(|(_, _, m)| inverse(m));
        let pf = plate.frame;
        let n = pf.z;
        // The bend turns towards b; the face whose outward normal is b is the inner face.
        let toward = if theta > 0.0 {
            n_face
        } else {
            scale(n_face, -1.0)
        };
        let b = if dot(toward, n) >= 0.0 {
            n
        } else {
            scale(n, -1.0)
        };
        let inner_height = if dot(b, n) > 0.0 { t } else { 0.0 };
        // The line, in world coordinates on the inner face.
        self.gate_feature(i, id, &line.0, "line")?;
        let Some((solved, sframe)) = self.st.sketches.get(&line.0).cloned() else {
            return Err(self.error(
                id,
                Some("line"),
                DiagCode::RefKindMismatch,
                format!("`{}` is not a sketch", line.0),
                Vec::new(),
            ));
        };
        use crate::model::sketch::PointRole;
        let (Some(p0), Some(p1)) = (
            solved
                .points
                .get(&(line.1.clone(), PointRole::Start))
                .copied(),
            solved
                .points
                .get(&(line.1.clone(), PointRole::End))
                .copied(),
        ) else {
            return Err(self.error(
                id,
                Some("line"),
                DiagCode::RefKindMismatch,
                format!("`{}` has no line `{}`", line.0, line.1),
                Vec::new(),
            ));
        };
        let (w0, w1) = (sframe.to_world(p0), sframe.to_world(p1));
        let on_inner = |p: [f64; 3]| {
            let h = dot(sub(p, pf.origin), n);
            add(p, scale(n, inner_height - h))
        };
        let l0 = on_inner(w0);
        let u = sub(on_inner(w1), l0);
        if dot(u, u).sqrt() < 1e-9 {
            return Err(invalid(
                self,
                "line",
                "the bend line does not lie across the plate".into(),
            ));
        }
        let u = normalize(u);
        // Left normal of the line in the sketch, in world coordinates; `fixed` keeps a side.
        let d2 = [p1[0] - p0[0], p1[1] - p0[1]];
        let left = normalize(add(scale(sframe.x, -d2[1]), scale(sframe.y, d2[0])));
        let left = normalize(sub(left, scale(n, dot(left, n))));
        let m = match fixed.unwrap_or(PlaneSide::Negative) {
            PlaneSide::Positive => scale(left, -1.0),
            PlaneSide::Negative => left,
        };
        let sw = |p: [f64; 3]| {
            let q = sub(on_inner(p), l0);
            [dot(q, m), dot(q, u)]
        };
        // The strip the bends consume, in developed length along m.
        let s0 = -ba / 2.0;
        let s1 = s0 + ba + middle.map_or(0.0, |f| f + ba);
        let tol = 1e-9;
        let to_sw = |seg: &ProfileSegment| (sw(pf.to_world(seg.start)), sw(pf.to_world(seg.end)));
        let reach = |seg: &ProfileSegment| -> (f64, f64) {
            let (a, b) = to_sw(seg);
            match seg.kind {
                SegmentKind::Line => (a[0].min(b[0]), a[0].max(b[0])),
                SegmentKind::Arc { center, .. } => {
                    let c = sw(pf.to_world(center));
                    let rad = ((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2)).sqrt();
                    (c[0] - rad, c[0] + rad)
                }
                SegmentKind::Circle { center, radius } => {
                    let c = sw(pf.to_world(center));
                    (c[0] - radius, c[0] + radius)
                }
                // Ellipses and splines: anywhere, so they count as crossing the strip.
                _ => (f64::NEG_INFINITY, f64::INFINITY),
            }
        };
        let Some(outer) = plate.loops.first() else {
            return Err(invalid(self, "face", "the plate has no outline".into()));
        };
        // Faces other features made (earlier bends, flanges, cuts) must lie wholly on the
        // fixed side: the moved part is rebuilt from the plate's outline.
        let own = |n: &EntityName| match &parent {
            None => n.maker() == &maker,
            // The parent's moved flat: its two faces and its sides.
            Some((p, _, _)) => match n {
                EntityName::Generated(g) if &g.feature == p => match (&g.role, &g.tag) {
                    (Role::SheetFace, crate::refs::name::SourceTag::Key { key }) => {
                        key.ends_with(".flat")
                    }
                    (Role::SheetEdge, _) => matches!(g.ordinal, Some(0) | Some(9)),
                    _ => false,
                },
                _ => false,
            },
        };
        let foreign = {
            let live = &self.st.bodies[body];
            live.names.faces.iter().enumerate().any(|(index, names)| {
                if names.iter().any(own) {
                    return false;
                }
                let g = &live.topology.faces[index].geom;
                (0..8).any(|k| {
                    let c = [
                        if k & 1 == 0 {
                            g.bbox_min[0]
                        } else {
                            g.bbox_max[0]
                        },
                        if k & 2 == 0 {
                            g.bbox_min[1]
                        } else {
                            g.bbox_max[1]
                        },
                        if k & 4 == 0 {
                            g.bbox_min[2]
                        } else {
                            g.bbox_max[2]
                        },
                    ];
                    dot(sub(c, l0), m) > s0 - tol
                })
            })
        };
        if foreign {
            return Err(self.not_lowered(
                id,
                "bends whose moving side holds faces of other features (bend the plate before adding them there)",
            ));
        }
        let big = 2.0
            * outer
                .iter()
                .map(|g| {
                    let (a2, b2) = to_sw(g);
                    a2[0]
                        .abs()
                        .max(a2[1].abs())
                        .max(b2[0].abs())
                        .max(b2[1].abs())
                })
                .fold(0.0, f64::max)
            + 10.0 * t;
        // Square crossings: every outline segment over the strip is a line of constant w.
        let mut chord: Vec<f64> = Vec::new();
        for seg in outer {
            let (lo, hi) = reach(seg);
            if hi <= s0 + tol || lo >= s1 - tol {
                continue;
            }
            let (a2, b2) = to_sw(seg);
            if !matches!(seg.kind, SegmentKind::Line) || (a2[1] - b2[1]).abs() > tol {
                return Err(self.not_lowered(
                    id,
                    "bends whose strip the plate's edges cross at an angle or along a curve (edges must cross the bend zone square to the line)",
                ));
            }
            if !chord.iter().any(|w| (w - a2[1]).abs() <= tol) {
                chord.push(a2[1]);
            }
        }
        if chord.len() != 2 {
            return Err(invalid(
                self,
                "line",
                "the bend line must cross the plate once, from edge to edge".into(),
            ));
        }
        let (wa, wb) = (chord[0].min(chord[1]), chord[0].max(chord[1]));
        for inner in plate.loops.iter().skip(1) {
            let lo = inner
                .iter()
                .map(|g| reach(g).0)
                .fold(f64::INFINITY, f64::min);
            let hi = inner
                .iter()
                .map(|g| reach(g).1)
                .fold(f64::NEG_INFINITY, f64::max);
            if !(hi < s0 - tol || lo > s1 + tol) {
                return Err(self.not_lowered(id, "bends through a hole or cut-out of the plate"));
            }
        }
        // The moved flat: the outline beyond s1 (shifted to start at 0) and the holes there.
        let mut moved: Vec<ProfileSegment> = Vec::new();
        let n_out = outer.len();
        let start_at = (0..n_out).find(|k| {
            let (a2, b2) = to_sw(&outer[*k]);
            a2[0] < s1 - tol && b2[0] >= s1 - tol
        });
        let Some(start_at) = start_at else {
            return Err(invalid(
                self,
                "line",
                "nothing of the plate lies past the bend".into(),
            ));
        };
        let cut = |a2: [f64; 2], b2: [f64; 2]| {
            let f = (s1 - a2[0]) / (b2[0] - a2[0]);
            [s1, a2[1] + f * (b2[1] - a2[1])]
        };
        let local = |p: [f64; 2]| [p[0] - s1, p[1]];
        let mut entry = None;
        let mut exit = None;
        for step in 0..n_out {
            let seg = &outer[(start_at + step) % n_out];
            let (a2, b2) = to_sw(seg);
            let (ina, inb) = (a2[0] >= s1 - tol, b2[0] >= s1 - tol);
            if step == 0 {
                let e = cut(a2, b2);
                entry = Some(e);
                moved.push(line_seg(&seg.curve.0, local(e), local(b2)));
                continue;
            }
            match (ina, inb) {
                (true, true) => {
                    let mut g = seg.clone();
                    g.start = local(a2);
                    g.end = local(b2);
                    if let SegmentKind::Arc { center, clockwise } = seg.kind {
                        // Developed coordinates keep the sketch's handedness only when (m, u)
                        // is right-handed with the plate normal.
                        let c = local(sw(pf.to_world(center)));
                        let flip = dot(cross(m, u), n) < 0.0;
                        g.kind = SegmentKind::Arc {
                            center: c,
                            clockwise: clockwise ^ flip,
                        };
                    }
                    moved.push(g);
                }
                (true, false) => {
                    if !matches!(seg.kind, SegmentKind::Line) {
                        return Err(self.not_lowered(id, "bends whose strip a curved edge crosses"));
                    }
                    let e = cut(a2, b2);
                    exit = Some(e);
                    moved.push(line_seg(&seg.curve.0, local(a2), local(e)));
                    break;
                }
                _ => {
                    return Err(invalid(
                        self,
                        "line",
                        "the bend line must cross the plate once, from edge to edge".into(),
                    ))
                }
            }
        }
        let (Some(entry), Some(exit)) = (entry, exit) else {
            return Err(invalid(
                self,
                "line",
                "the plate does not cross the bend line".into(),
            ));
        };
        let close_id = format!("{}.root", line.1 .0);
        moved.push(line_seg(&close_id, local(exit), local(entry)));
        let mut loops = vec![moved];
        for inner in plate.loops.iter().skip(1) {
            let lo = inner
                .iter()
                .map(|g| reach(g).0)
                .fold(f64::INFINITY, f64::min);
            if lo > s1 + tol {
                let flip = dot(cross(m, u), n) < 0.0;
                loops.push(
                    inner
                        .iter()
                        .map(|seg| {
                            let (a2, b2) = to_sw(seg);
                            let mut g = seg.clone();
                            g.start = local(a2);
                            g.end = local(b2);
                            if let SegmentKind::Arc { center, clockwise } = seg.kind {
                                g.kind = SegmentKind::Arc {
                                    center: local(sw(pf.to_world(center))),
                                    clockwise: clockwise ^ flip,
                                };
                            }
                            g
                        })
                        .collect(),
                );
            }
        }
        // Walk the section: a cursor on the inner (b-side) face, x along the developed
        // direction, y the face normal; z is the line direction for every local frame.
        let key = line.1 .0.clone();
        let mut o = add(l0, scale(m, s0));
        let (mut x, mut y) = (m, b);
        let mut tools = Vec::new();
        let span = (wa, wb);
        let mut turns = vec![(true, key.clone())];
        if middle.is_some() {
            turns.push((false, format!("{key}.2")));
        }
        for (k, (up, bend_key)) in turns.iter().enumerate() {
            let fr = Frame {
                origin: o,
                x,
                y,
                z: cross(x, y),
            };
            let (wz0, wz1) = {
                let along = dot(u, fr.z);
                let (p, q) = (span.0 * along, span.1 * along);
                let base = dot(sub(l0, o), fr.z);
                (base + p.min(q), base + p.max(q))
            };
            tools.push(sheet_bend_tool(fr, *up, r, t, a, bend_key, (wz0, wz1)));
            // The cursor after the bend.
            let (c, s) = (a.cos(), a.sin());
            let (dx, dy, nx, ny) = if *up {
                (r * s, r - r * c, [c, s], [-s, c])
            } else {
                ((r + t) * s, -t - r + (r + t) * c, [c, -s], [s, c])
            };
            o = add(o, add(scale(x, dx), scale(y, dy)));
            let (x2, y2) = (
                add(scale(x, nx[0]), scale(y, nx[1])),
                add(scale(x, ny[0]), scale(y, ny[1])),
            );
            x = normalize(x2);
            y = normalize(y2);
            if k == 0 {
                if let Some(flat) = middle {
                    if flat > 1e-12 {
                        let fr = frame_of(o, x, u);
                        let sign = if dot(fr.z, y) >= 0.0 { 1.0 } else { -1.0 };
                        let corners = vec![
                            ("y0".to_string(), [0.0, wa]),
                            ("x1".to_string(), [flat, wa]),
                            ("y1".to_string(), [flat, wb]),
                            ("x0".to_string(), [0.0, wb]),
                        ];
                        let mut tool = LabeledTool::new(ToolShape::prism(
                            polygon(fr, &corners),
                            0.0,
                            -t * sign,
                        ));
                        let mid = format!("{key}.mid");
                        for (side, o) in [("y0", 2), ("y1", 3), ("x0", 4), ("x1", 5)] {
                            tool.sides.push((
                                crate::ids::SketchEntityId::new(side),
                                FaceLabel::key(Role::SheetEdge, mid.clone()).with_ordinal(o),
                            ));
                        }
                        tool.start =
                            Some(FaceLabel::key(Role::SheetFace, mid.clone()).with_ordinal(0));
                        tool.end = Some(FaceLabel::key(Role::SheetFace, mid).with_ordinal(1));
                        tools.push(tool);
                        o = add(o, scale(x, flat));
                    }
                }
            }
        }
        // The moved flat, from the last cursor.
        let fr = frame_of(o, x, u);
        let sign = if dot(fr.z, y) >= 0.0 { 1.0 } else { -1.0 };
        let profile = CanonicalProfile {
            frame: fr,
            loops,
            merge_collinear: false,
        };
        let mut flat = LabeledTool::new(ToolShape::prism(profile.clone(), 0.0, -t * sign));
        for lp in &profile.loops {
            for seg in lp {
                let label = if seg.curve.0 == close_id {
                    FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(9)
                } else {
                    FaceLabel::key(Role::SheetEdge, seg.curve.0.clone()).with_ordinal(0)
                };
                flat.sides.push((seg.curve.clone(), label));
            }
        }
        let moved_key = format!("{key}.flat");
        flat.start = Some(FaceLabel::key(Role::SheetFace, moved_key.clone()).with_ordinal(0));
        flat.end = Some(FaceLabel::key(Role::SheetFace, moved_key).with_ordinal(1));
        tools.push(flat);
        // The bend lines on the developed plate, for its flat pattern.
        let bf = base_plate.frame;
        let to2d = |p: [f64; 3]| {
            let p = back.as_ref().map_or(p, |b| b.apply(p));
            let q = sub(p, bf.origin);
            [dot(q, bf.x), dot(q, bf.y)]
        };
        let dir2d = |v: [f64; 3]| {
            let v = back.as_ref().map_or(v, |b| apply_rotation(b, v));
            [dot(v, bf.x), dot(v, bf.y)]
        };
        let bd = crate::model::sheet_metal::bend_deduction(&rules.rule, t, r, a)
            .map_err(|m| invalid(self, "angle", m))?;
        let first_up = dot(b, n) > 0.0;
        let mut lines = Vec::new();
        for (k, at) in std::iter::once(0.0)
            .chain(middle.map(|f| ba + f))
            .enumerate()
        {
            let up = first_up == (k == 0);
            lines.push(crate::model::sheet_metal::FlatBendLine {
                key: if k == 0 {
                    key.clone()
                } else {
                    format!("{key}.2")
                },
                feature: id.clone(),
                start: to2d(add(l0, add(scale(m, at), scale(u, wa)))),
                end: to2d(add(l0, add(scale(m, at), scale(u, wb)))),
                angle: a,
                inner_radius: r,
                allowance: ba,
                deduction: bd,
                direction: if up {
                    crate::model::sheet_metal::BendDirection::Up
                } else {
                    crate::model::sheet_metal::BendDirection::Down
                },
            });
        }
        let moved = (to2d(add(l0, scale(m, s0))), dir2d(m));
        // Developed to bent: m, u and b (the inner face's normal) turn to x, u and y, and the
        // developed point where the moved flat starts goes to the last cursor.
        let motion = {
            let mut rotation = [[0.0; 3]; 3];
            for (r, row) in rotation.iter_mut().enumerate() {
                for (c, v) in row.iter_mut().enumerate() {
                    *v = x[r] * m[c] + u[r] * u[c] + y[r] * b[c];
                }
            }
            let start = add(l0, scale(m, s1));
            let moved_start = Transform {
                rotation,
                translation: [0.0; 3],
            }
            .apply(start);
            let local = Transform {
                rotation,
                translation: sub(o, moved_start),
            };
            // After the parent's motion when this bend bends a moved flat.
            match &parent {
                Some((_, _, pm)) => compose(&local, pm),
                None => local,
            }
        };
        // The moved flat, with the sheet between z = 0 and z = t of its frame.
        let mut moved_flat = profile.clone();
        if sign > 0.0 {
            moved_flat.frame.origin = add(moved_flat.frame.origin, scale(moved_flat.frame.z, -t));
        }
        self.record_plate_bend(
            id,
            body,
            &maker,
            &base_plate,
            lines,
            moved,
            motion,
            moved_flat,
            parent.map(|p| p.0),
        )?;
        // Carve everything of the plate past the fixed flat.
        let cf = frame_of(add(l0, scale(m, s0)), m, u);
        let csign = if dot(cf.z, b) >= 0.0 { 1.0 } else { -1.0 };
        let corners = vec![
            ("c0".to_string(), [0.0, -big]),
            ("c1".to_string(), [big, -big]),
            ("c2".to_string(), [big, big]),
            ("c3".to_string(), [0.0, big]),
        ];
        let mut carve = LabeledTool::new(ToolShape::prism(
            polygon(cf, &corners),
            1e-6 * csign,
            -(t + 1e-6) * csign,
        ));
        carve.sides.push((
            crate::ids::SketchEntityId::new("c0"),
            FaceLabel::key(Role::Relief, key).with_ordinal(0),
        ));
        let bend = labeled(tools, vec![carve], Combine::Add, false);
        self.st.replays.insert(id.clone(), bend.clone());
        Ok(op(bend, vec![body], naming.clone()))
    }

    // --------------------------------------------------------------------- weldments
}
