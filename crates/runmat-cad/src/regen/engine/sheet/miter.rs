//! The miter flange: an open profile swept along a chain of sheet edges (see the module docs).

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(in crate::regen::engine) fn miter_flange(
        &mut self,
        i: usize,
        id: &FeatureId,
        profile: &ProfileRef,
        edges: &[Reference],
        gap: &ValueExpr,
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let invalid = |me: &mut Self, param: &str, message: String| {
            me.error(id, Some(param), DiagCode::RegenInvalid, message, Vec::new())
        };
        let mismatch = |me: &mut Self, param: &str, message: &str| {
            me.error(
                id,
                Some(param),
                DiagCode::RefKindMismatch,
                message.into(),
                Vec::new(),
            )
        };
        let gap = self.value(id, "gap", gap, Dimension::LENGTH)?;
        if gap < 0.0 {
            return Err(invalid(
                self,
                "gap",
                "a miter gap cannot be negative".into(),
            ));
        }
        let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
        let operands = Self::bodies_of(&set);
        if operands.len() != 1 {
            return Err(mismatch(
                self,
                "edges",
                "miter flange edges must be on one sheet",
            ));
        }
        let body = operands[0];
        let Some((maker, rules, sheet_section, _)) = self.sheet_of(body) else {
            return Err(mismatch(
                self,
                "edges",
                "the edges are not on a sheet metal body",
            ));
        };
        let (t, r) = (rules.thickness, rules.bend_radius);
        if r <= 0.0 {
            return Err(invalid(
                self,
                "edges",
                "a miter flange needs a sheet with a positive bend radius".into(),
            ));
        }
        // Each edge: straight, between a sheet face and a thickness face.
        let mut sheet_normal: Option<[f64; 3]> = None;
        let mut loose: Vec<ChainEdge> = Vec::new();
        for h in &set {
            let topo = &self.st.bodies[body].topology;
            let e = topo.edges[h.index as usize].clone();
            let chord = sub(e.geom.end, e.geom.start);
            let len = dot(chord, chord).sqrt();
            if len < 1e-12 || (e.geom.length - len).abs() > 1e-9 * len.max(1e-3) {
                return Err(mismatch(
                    self,
                    "edges",
                    "a miter flange runs along straight sheet edges",
                ));
            }
            let (ga, gb) = (
                topo.faces[e.faces[0] as usize].geom,
                topo.faces[e.faces[1] as usize].geom,
            );
            let (Some(na), Some(nb)) = (plane_normal(&ga), plane_normal(&gb)) else {
                return Err(mismatch(
                    self,
                    "edges",
                    "a miter flange edge lies between two flat faces",
                ));
            };
            let dir = scale(chord, 1.0 / len);
            let width = |g: &FaceGeom| {
                let ext = sub(g.bbox_max, g.bbox_min);
                let across = sub(ext, scale(dir, dot(ext, dir)));
                dot(across, across).sqrt()
            };
            let (n_sheet, out) = if width(&ga) >= width(&gb) {
                (na, nb)
            } else {
                (nb, na)
            };
            match sheet_normal {
                None => sheet_normal = Some(n_sheet),
                Some(n) if dot(n, n_sheet) > 1.0 - 1e-9 => {}
                Some(_) => {
                    return Err(mismatch(
                        self,
                        "edges",
                        "a miter flange's edges lie on one face of the sheet",
                    ))
                }
            }
            loose.push(ChainEdge {
                index: h.index,
                start: e.geom.start,
                end: e.geom.end,
                out,
            });
        }
        let Some(n_sheet) = sheet_normal else {
            return Err(mismatch(self, "edges", "a miter flange needs edges"));
        };
        let Some(chain) = order_chain(loose, 1e-7) else {
            return Err(invalid(
                self,
                "edges",
                "a miter flange's edges must join end to end in one chain".into(),
            ));
        };
        let closed = chain.len() > 2 && near(chain[0].start, chain[chain.len() - 1].end, 1e-7);
        // The profile, read in the section plane of the edge it starts on.
        let (sframe, segs) = self.chain(i, id, "profile", profile)?;
        if segs.iter().any(|s| !matches!(s.kind, SegmentKind::Line)) {
            return Err(self.error(
                id,
                Some("profile"),
                DiagCode::SketchOpenProfile,
                "a miter flange profile is drawn with lines; bends come from its corners".into(),
                Vec::new(),
            ));
        }
        let p0 = sframe.to_world(segs[0].start);
        let anchor = chain.iter().position(|e| {
            let d = normalize(sub(e.end, e.start));
            let along = dot(sub(p0, e.start), d);
            let off = sub(sub(p0, e.start), scale(d, along));
            let len = dot(sub(e.end, e.start), d);
            dot(off, off).sqrt() < 1e-7
                && along > -1e-7
                && along < len + 1e-7
                && dot(sframe.z, d).abs() > 1.0 - 1e-9
        });
        let Some(anchor) = anchor else {
            return Err(invalid(
                self,
                "profile",
                "the profile must start on one of the edges, drawn on a plane normal to it".into(),
            ));
        };
        let out0 = chain[anchor].out;
        let local = |p: [f64; 2]| {
            let w = sub(sframe.to_world(p), p0);
            [dot(w, out0), dot(w, n_sheet)]
        };
        // The section: the sheet's mold line, then the profile; material on the sheet's side.
        let span: f64 = segs
            .iter()
            .map(|s| (s.end[0] - s.start[0]).hypot(s.end[1] - s.start[1]))
            .sum();
        let lead = 1.0 + span + 1e3 * (r + t);
        let mut pts = vec![[-lead, 0.0], [0.0, 0.0]];
        pts.extend(segs.iter().map(|s| local(s.end)));
        let plate_curve = crate::ids::SketchEntityId::new(format!("{}.sheet", id.0));
        let mut curves = vec![plate_curve];
        curves.extend(segs.iter().map(|s| s.curve.clone()));
        let mut sec = section(&pts, &curves, MaterialSide::Right, t, r)
            .map_err(|m| invalid(self, "profile", m))?;
        if sec.bends.len() + 1 != sec.flats.len() {
            return Err(invalid(
                self,
                "profile",
                "the profile must turn at the sheet edge and at each of its corners".into(),
            ));
        }
        // Start the first bend at the sheet edge.
        let shift = -sec.flats[0].end[0];
        for f in &mut sec.flats {
            f.start[0] += shift;
            f.end[0] += shift;
        }
        for b in &mut sec.bends {
            b.center[0] += shift;
        }
        // How far the section reaches out of the sheet (for the mitred extensions).
        let reach = sec
            .flats
            .iter()
            .skip(1)
            .flat_map(|f| {
                [f.start, f.end]
                    .into_iter()
                    .flat_map(move |p| [f.near, f.far].map(|o| (p[0] + o * f.normal[0]).abs()))
            })
            .chain(
                sec.bends
                    .iter()
                    .map(|b| b.center[0].abs() + b.inner_radius + b.thickness),
            )
            .fold(0.0, f64::max);
        // The clip plane at each corner, as the side before it keeps it.
        let corners = chain.len() - if closed { 0 } else { 1 };
        let mut planes: Vec<([f64; 3], [f64; 3], f64)> = Vec::new();
        for k in 0..corners {
            let (a, b) = (&chain[k], &chain[(k + 1) % chain.len()]);
            let (da, db) = (
                normalize(sub(a.end, a.start)),
                normalize(sub(b.end, b.start)),
            );
            let turn = dot(da, db).clamp(-1.0, 1.0).acos();
            if turn > 170f64.to_radians() {
                return Err(invalid(
                    self,
                    "edges",
                    "a miter flange's edges turn by at most 170° at a corner".into(),
                ));
            }
            let bisector = add(a.out, b.out);
            let mut normal = normalize(cross(bisector, n_sheet));
            if dot(normal, da) < 0.0 {
                normal = scale(normal, -1.0);
            }
            // The distance along the edge the farthest section point needs past the corner.
            let ext = reach * (turn / 2.0).tan() + t;
            planes.push((a.end, normal, ext));
        }
        let mut tools = Vec::new();
        let n = chain.len();
        for (k, e) in chain.iter().enumerate() {
            let before = if k > 0 || closed {
                Some(&planes[(k + n - 1) % n])
            } else {
                None
            };
            let after = if k + 1 < n || closed {
                Some(&planes[k])
            } else {
                None
            };
            let fr = frame_of(e.start, e.out, n_sheet);
            let (s_a, s_b) = (0.0, dot(sub(e.end, e.start), fr.z));
            let sign = if s_b >= s_a { 1.0 } else { -1.0 };
            let za = s_a - sign * before.map_or(0.0, |p| p.2);
            let zb = s_b + sign * after.map_or(0.0, |p| p.2);
            let (z0, z1) = (za.min(zb), za.max(zb));
            let edge_tag = KernelTag::InputEdge {
                edge: OperandEntity {
                    operand: 0,
                    entity: e.index,
                },
            };
            let lab = |role: Role, o: u32| FaceLabel {
                role,
                tag: edge_tag.clone(),
                ordinal: Some(o),
            };
            let clips = |base: u32| {
                let mut v = Vec::new();
                if let Some((o, nrm, _)) = before {
                    // The side after the corner keeps the far half: the plane's normal turned.
                    let m = scale(*nrm, -1.0);
                    v.push(ClipPlane {
                        origin: sub(*o, scale(m, gap / 2.0)),
                        normal: m,
                        label: lab(Role::SheetEdge, base + 6),
                    });
                }
                if let Some((o, nrm, _)) = after {
                    v.push(ClipPlane {
                        origin: sub(*o, scale(*nrm, gap / 2.0)),
                        normal: *nrm,
                        label: lab(Role::SheetEdge, base + 7),
                    });
                }
                v
            };
            let last = sec.flats.len() - 1;
            for (m, f) in sec.flats.iter().enumerate().skip(1) {
                let j = (m - 1) as u32;
                let base = 16 * j;
                let off = |p: [f64; 2], o: f64| [p[0] + o * f.normal[0], p[1] + o * f.normal[1]];
                let corners = vec![
                    ("near".to_string(), off(f.start, f.near)),
                    ("b".to_string(), off(f.end, f.near)),
                    ("far".to_string(), off(f.end, f.far)),
                    ("a".to_string(), off(f.start, f.far)),
                ];
                let mut tool = LabeledTool::new(ToolShape::prism(polygon(fr, &corners), z0, z1));
                let sid = crate::ids::SketchEntityId::new;
                // Material lies right of the profile, so `far` (offset 0) is the face on the
                // profile's line.
                tool.sides
                    .push((sid("near"), lab(Role::SheetFace, base + 1)));
                tool.sides.push((sid("far"), lab(Role::SheetFace, base)));
                tool.sides.push((sid("a"), lab(Role::SheetEdge, base + 4)));
                tool.sides.push((
                    sid("b"),
                    lab(Role::SheetEdge, base + if m == last { 1 } else { 5 }),
                ));
                tool.start = Some(lab(Role::SheetEdge, base + 2));
                tool.end = Some(lab(Role::SheetEdge, base + 3));
                tool.clip = clips(base);
                tools.push(tool);
            }
            for (m, b) in sec.bends.iter().enumerate() {
                let base = 16 * m as u32 + 8;
                let mut tool = bend_tool(fr, b, 1.0, None);
                if let ToolShape::Prism { start, end, .. } = &mut tool.shape {
                    *start = z0;
                    *end = z1;
                }
                for (_, l) in tool.sides.iter_mut() {
                    *l = lab(l.role, base + l.ordinal.unwrap_or(0));
                }
                tool.start = Some(lab(Role::SheetEdge, base + 2));
                tool.end = Some(lab(Role::SheetEdge, base + 3));
                tool.clip = clips(base);
                tools.push(tool);
            }
        }
        let made = labeled(tools, Vec::new(), Combine::Add, false);
        // On a plate, a one-line profile turned towards the sheet's face develops as an edge
        // flange strip per edge with mitred ends, so the plate's flat pattern, unfold and fold
        // take it; a fold replays it.
        let up = sec.bends.len() == 1 && sec.bends[0].angle > 0.0;
        if sheet_section.is_none() && up {
            let (b, flat) = (&sec.bends[0], &sec.flats[1]);
            let developed = chain
                .iter()
                .map(|e| FlangeEdgeIn {
                    start: e.start,
                    end: e.end,
                    frame: frame_of(e.start, e.out, n_sheet),
                    shift: 0.0,
                    theta: b.angle,
                    radius: r,
                    flat: flat.length(),
                })
                .collect();
            self.record_plate_flange(i, id, body, &maker, &rules, developed)?;
            // The mitre: a point of the strip at its section distance x out of the sheet edge
            // (on the bend's inner face) lies on the clip plane where its offset along the edge
            // is (±gap/2 − x n·out) / n·d.
            let allowance = match self.st.records.get(id) {
                Some(FeatureRecord::PlateFlange { edges, .. }) => edges.first().map_or(0.0, |e| e.allowance),
                _ => 0.0,
            };
            let stations = vec![0.0, allowance, allowance + flat.length()];
            let xs = [0.0, flat.start[0], flat.end[0]];
            let n = chain.len();
            let mitres: Vec<crate::regen::records::MitreEnds> = chain
                .iter()
                .enumerate()
                .map(|(k, e)| {
                    let d = normalize(sub(e.end, e.start));
                    let offsets = |plane: Option<&([f64; 3], [f64; 3], f64)>, sign: f64| -> Vec<f64> {
                        xs.iter()
                            .map(|x| match plane {
                                Some((_, nrm, _)) => (sign * gap / 2.0 - x * dot(*nrm, e.out)) / dot(*nrm, d),
                                None => 0.0,
                            })
                            .collect()
                    };
                    let before = (k > 0 || closed).then(|| &planes[(k + n - 1) % n]);
                    let after = (k + 1 < n || closed).then(|| &planes[k]);
                    crate::regen::records::MitreEnds {
                        stations: stations.clone(),
                        start: offsets(before, 1.0),
                        end: offsets(after, -1.0),
                    }
                })
                .collect();
            if let Some(FeatureRecord::PlateFlange { edges, .. }) = self.st.records.get_mut(id) {
                for (e, m) in edges.iter_mut().zip(mitres) {
                    e.mitre = Some(m);
                }
            }
            self.st.replays.insert(id.clone(), made.clone());
        }
        Ok(op(made, vec![body], naming.clone()))
    }
}
