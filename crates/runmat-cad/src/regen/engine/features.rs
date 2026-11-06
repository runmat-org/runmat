//! Lowering of the Features-tab kinds with options: extrude end conditions (both directions,
//! from a surface, offset or vertex), draft and thin; holes from standard tables (tapped, pipe
//! tap, slot, drill points, series); variable and setback fillets, face fillets and full
//! rounds; chamfer forms and vertex chamfers; multi-thickness and outward shells; parting-line
//! and step drafts.

use super::*;
use crate::drawing::hole_table::{HoleInstance, ThreadCallout};
use crate::model::hole_tables::{self, HoleStandard, ThreadData};
use crate::regen::records::FeatureRecord;

/// An extrude end resolved: the signed distance from the profile plane and its bound.
struct End {
    at: f64,
    bound: PrismBound,
}

/// Operands of a tool feature: the bodies it combines, then bodies read by end conditions.
struct Operands {
    combine: Vec<usize>,
    refs: Vec<usize>,
}

impl Operands {
    fn index(&mut self, body: usize) -> u32 {
        if let Some(k) = self.combine.iter().position(|b| *b == body) {
            return k as u32;
        }
        if let Some(k) = self.refs.iter().position(|b| *b == body) {
            return (self.combine.len() + k) as u32;
        }
        self.refs.push(body);
        (self.combine.len() + self.refs.len() - 1) as u32
    }
    fn all(&self) -> Vec<usize> {
        self.combine.iter().chain(&self.refs).copied().collect()
    }
}

impl<'a, 'b> Run<'a, 'b> {
    /// A face as a plane parallel to `frame` (its signed height along `frame.z`), when it is
    /// one.
    fn parallel_height(&self, h: &EntityHandle, frame: &Frame) -> Option<f64> {
        let g = &self.st.bodies[h.body as usize].topology.faces[h.index as usize].geom;
        let SurfaceKind::Plane { normal } = g.surface else {
            return None;
        };
        let n = normalize(normal);
        (dot(n, frame.z).abs() > 1.0 - 1e-9).then(|| dot(sub(g.centroid, frame.origin), frame.z))
    }

    fn face_entity(
        &mut self,
        i: usize,
        id: &FeatureId,
        param: &str,
        r: &Reference,
    ) -> Result<EntityHandle, Halt> {
        let res = self.single(i, id, param, r, Some(TopoKind::Face))?;
        Ok(*res.entities().iter().next().expect("one"))
    }

    #[allow(clippy::too_many_arguments)]
    fn extrude_end(
        &mut self,
        i: usize,
        id: &FeatureId,
        prefix: &str,
        e: &Extent,
        frame: &Frame,
        base: f64,
        dir: f64,
        ops: &mut Operands,
    ) -> Result<End, Halt> {
        let p = |s: &str| format!("{}{}", prefix, s);
        Ok(match e {
            Extent::Blind { distance } => {
                let d = self.value(id, &p("distance"), distance, Dimension::LENGTH)?;
                End {
                    at: base + dir * d,
                    bound: PrismBound::Blind,
                }
            }
            Extent::Symmetric { .. } => {
                return Err(self.error(
                    id,
                    Some(&p("extent")),
                    DiagCode::RegenInvalid,
                    "mid-plane already extrudes both ways; it cannot be combined with a \
                     second direction"
                        .into(),
                    Vec::new(),
                ))
            }
            Extent::ThroughAll => End {
                at: base + dir,
                bound: PrismBound::ThroughAll,
            },
            Extent::UpToNext => End {
                at: base + dir * 1e-6,
                bound: PrismBound::UpToNext,
            },
            Extent::UpTo { face } => {
                self.up_to_face(i, id, &p("up_to"), face, 0.0, frame, dir, ops)?
            }
            Extent::OffsetFromFace {
                face,
                offset,
                beyond,
            } => {
                let o = self.value(id, &p("offset"), offset, Dimension::LENGTH)?;
                let o = if *beyond { o } else { -o };
                self.up_to_face(i, id, &p("up_to"), face, o, frame, dir, ops)?
            }
            Extent::UpToVertex { vertex } => {
                let w = self.point(i, id, &p("up_to"), vertex)?;
                let at = dot(sub(w, frame.origin), frame.z);
                if (at - base) * dir <= 0.0 {
                    return Err(self.error(
                        id,
                        Some(&p("up_to")),
                        DiagCode::RegenInvalid,
                        "the vertex is behind the start of the extrude in this direction".into(),
                        Vec::new(),
                    ));
                }
                End {
                    at,
                    bound: PrismBound::Blind,
                }
            }
            Extent::UpToBody { body, offset } => {
                let r = self.single(i, id, &p("up_to"), body, Some(TopoKind::Body))?;
                let b = r.entities().iter().next().expect("one").body as usize;
                let o = match offset {
                    Some(o) => self.value(id, &p("offset"), o, Dimension::LENGTH)?,
                    None => 0.0,
                };
                End {
                    at: base + dir * 1e-6,
                    bound: PrismBound::UpToBody {
                        operand: ops.index(b),
                        offset: o,
                    },
                }
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn up_to_face(
        &mut self,
        i: usize,
        id: &FeatureId,
        param: &str,
        face: &Reference,
        offset: f64,
        frame: &Frame,
        dir: f64,
        ops: &mut Operands,
    ) -> Result<End, Halt> {
        let h = self.face_entity(i, id, param, face)?;
        // A plane parallel to the profile is an exact distance; any other face is an end
        // condition the kernel evaluates per point of the profile.
        if let Some(z) = self.parallel_height(&h, frame) {
            return Ok(End {
                at: z + dir * offset,
                bound: PrismBound::Blind,
            });
        }
        Ok(End {
            at: dir * 1e-6,
            bound: PrismBound::UpToFace {
                face: OperandEntity {
                    operand: ops.index(h.body as usize),
                    entity: h.index,
                },
                offset,
            },
        })
    }

    pub(super) fn extrude(
        &mut self,
        i: usize,
        id: &FeatureId,
        e: &Extrude,
    ) -> Result<Compiled, Halt> {
        let mut profile = self.profile(i, id, "profile", &e.profile)?;
        let dir = if e.reverse { -1.0 } else { 1.0 };
        let (combine, combine_ops) = self.mode_operands(i, id, &e.mode)?;
        let mut ops = Operands {
            combine: combine_ops,
            refs: Vec::new(),
        };
        // Where it starts: the profile moves to the start plane, so every distance and end
        // condition is measured from the profile plane.
        let sketch_frame = profile.frame;
        let mut start_bound = PrismBound::Blind;
        let shift = match &e.from {
            ExtrudeFrom::SketchPlane => 0.0,
            ExtrudeFrom::Offset { distance } => {
                self.value(id, "from.distance", distance, Dimension::LENGTH)?
            }
            ExtrudeFrom::Vertex { vertex } => {
                let w = self.point(i, id, "from", vertex)?;
                dot(sub(w, sketch_frame.origin), sketch_frame.z)
            }
            ExtrudeFrom::Surface { face } => {
                let h = self.face_entity(i, id, "from", face)?;
                match self.parallel_height(&h, &sketch_frame) {
                    Some(z) => z,
                    None => {
                        start_bound = PrismBound::FromFace {
                            face: OperandEntity {
                                operand: ops.index(h.body as usize),
                                entity: h.index,
                            },
                        };
                        0.0
                    }
                }
            }
        };
        profile.frame.origin = add(sketch_frame.origin, scale(sketch_frame.z, shift));
        let frame = profile.frame;
        let base = 0.0;
        let (start, end, start_bound, end_bound) = match (&e.extent, &e.second) {
            (Extent::Symmetric { distance }, None) => {
                let d = self.value(id, "distance", distance, Dimension::LENGTH)?;
                (
                    base - d / 2.0,
                    base + d / 2.0,
                    start_bound,
                    PrismBound::Blind,
                )
            }
            (first, second) => {
                let end = self.extrude_end(i, id, "", first, &frame, base, dir, &mut ops)?;
                match second {
                    None => (base, end.at, start_bound, end.bound),
                    Some(s) => {
                        if !start_bound.is_blind() {
                            return Err(self.error(
                                id,
                                Some("second"),
                                DiagCode::RegenInvalid,
                                "an extrude from a surface goes one way; the second \
                                 direction starts at a plane"
                                    .into(),
                                Vec::new(),
                            ));
                        }
                        let st =
                            self.extrude_end(i, id, "second.", s, &frame, base, -dir, &mut ops)?;
                        (st.at, end.at, st.bound, end.bound)
                    }
                }
            }
        };
        let draft = match &e.draft {
            None => None,
            Some(d) => {
                let a = self.draft_angle(id, "draft.angle", &d.angle, d.outward)?;
                let b = match &d.second_angle {
                    Some(x) => self.draft_angle(id, "draft.second_angle", x, d.second_outward)?,
                    None => a,
                };
                Some(PrismDraft { end: a, start: b })
            }
        };
        let thin = match &e.thin {
            None => None,
            Some(t) => Some(self.thin_wall(id, t)?),
        };
        let reads = start_bound.reads_operands() || end_bound.reads_operands();
        let addressed = matches!(
            (&start_bound, &end_bound),
            (PrismBound::FromFace { .. }, _)
                | (_, PrismBound::UpToFace { .. } | PrismBound::UpToBody { .. })
                | (PrismBound::UpToFace { .. } | PrismBound::UpToBody { .. }, _)
        );
        if combine == Combine::New && reads && ops.all().is_empty() {
            // Up to next on a new body reads every body there is.
            ops.refs = (0..self.st.bodies.len()).collect();
        }
        let references = ops.refs.len() as u32;
        let tool = ToolOp {
            shape: ToolShape::Prism {
                profile,
                start,
                end,
                start_bound,
                end_bound,
                draft,
                thin,
            },
            combine,
            references,
        };
        // A tool addressing operand faces or bodies cannot be re-applied elsewhere.
        let reusable = !addressed;
        Ok(Compiled::Op {
            op: KernelOp::Tool(tool.clone()),
            operands: ops.all(),
            naming: NamingContext::simple(id, vec![e.profile.sketch.clone()]),
            import: false,
            instances: None,
            tools: reusable.then(|| {
                let mut t = tool;
                t.references = 0;
                vec![t]
            }),
        })
    }

    fn draft_angle(
        &mut self,
        id: &FeatureId,
        param: &str,
        v: &ValueExpr,
        outward: bool,
    ) -> Result<f64, Halt> {
        let a = self.value(id, param, v, Dimension::ANGLE)?;
        if a.abs() >= 89f64.to_radians() {
            return Err(self.error(
                id,
                Some(param),
                DiagCode::ValueFailed,
                format!(
                    "a draft angle must be under 89°, got {:.3}°",
                    a.to_degrees()
                ),
                Vec::new(),
            ));
        }
        Ok(if outward { -a } else { a })
    }

    fn thin_wall(&mut self, id: &FeatureId, t: &ThinWall) -> Result<ThinWallSpec, Halt> {
        let a = self.value(id, "thin.thickness", &t.thickness, Dimension::LENGTH)?;
        let (inside, outside) = match t.side {
            ThinSide::One => (a, 0.0),
            ThinSide::Other => (0.0, a),
            ThinSide::Mid => (a / 2.0, a / 2.0),
            ThinSide::Two => {
                let Some(b) = &t.second_thickness else {
                    return Err(self.error(
                        id,
                        Some("thin.second_thickness"),
                        DiagCode::RegenInvalid,
                        "a two-sided thin feature needs its second thickness".into(),
                        Vec::new(),
                    ));
                };
                (
                    a,
                    self.value(id, "thin.second_thickness", b, Dimension::LENGTH)?,
                )
            }
        };
        if inside + outside <= 0.0 {
            return Err(self.error(
                id,
                Some("thin.thickness"),
                DiagCode::ValueFailed,
                "a thin wall needs a positive thickness".into(),
                Vec::new(),
            ));
        }
        let cap = match &t.cap_ends {
            Some(c) => Some(self.value(id, "thin.cap_ends", c, Dimension::LENGTH)?),
            None => None,
        };
        Ok(ThinWallSpec {
            inside,
            outside,
            cap,
        })
    }

    // ------------------------------------------------------------------------------ holes

    fn hole_spec_error(&mut self, id: &FeatureId, msg: String) -> Halt {
        self.error(id, Some("spec"), DiagCode::ValueFailed, msg, Vec::new())
    }

    fn opt_len(
        &mut self,
        id: &FeatureId,
        param: &str,
        v: &Option<ValueExpr>,
    ) -> Result<Option<f64>, Halt> {
        match v {
            Some(x) => Ok(Some(self.value(id, param, x, Dimension::LENGTH)?)),
            None => Ok(None),
        }
    }

    /// The thread a tapped or pipe-tapped form needs, from the spec.
    fn hole_thread(&mut self, id: &FeatureId, h: &Hole, pipe: bool) -> Result<ThreadData, Halt> {
        let Some(spec) = &h.spec else {
            return Err(self.error(
                id,
                Some("spec"),
                DiagCode::RegenInvalid,
                "a tapped hole needs its standard size (`spec`) for the thread".into(),
                Vec::new(),
            ));
        };
        let is_pipe = matches!(spec.standard, HoleStandard::Npt | HoleStandard::Bspt);
        if is_pipe != pipe {
            return Err(self.hole_spec_error(
                id,
                if pipe {
                    format!("a pipe tap needs an NPT or BSPT size, not `{}`", spec.size)
                } else {
                    format!("`{}` is a pipe size; use a pipe tap", spec.size)
                },
            ));
        }
        hole_tables::thread(spec).map_err(|m| self.hole_spec_error(id, m))
    }

    pub(super) fn hole(&mut self, i: usize, id: &FeatureId, h: &Hole) -> Result<Compiled, Halt> {
        let (bodies, frame) = self.hole_faces(i, id, &h.face)?;
        let mut positions = Vec::new();
        for (k, p) in h.positions.iter().enumerate() {
            let w = self.point(i, id, &format!("positions[{}]", k), &p.at)?;
            let d = sub(w, frame.origin);
            positions.push((p.key.clone(), [dot(d, frame.x), dot(d, frame.y)]));
        }
        let fastener = match &h.spec {
            Some(s) if !matches!(s.standard, HoleStandard::Npt | HoleStandard::Bspt) => {
                Some(hole_tables::fastener(s).map_err(|m| self.hole_spec_error(id, m))?)
            }
            _ => None,
        };
        let fit = h.spec.as_ref().map(|s| s.fit).unwrap_or_default();
        let inch = matches!(
            h.spec.as_ref().map(|s| s.standard),
            Some(HoleStandard::AnsiInch)
        );
        let mut thread: Option<ThreadData> = None;
        let mut taper = None;
        let mut slot = None;
        let mut counterbore = None;
        let mut countersink = None;
        let mut modelled = false;
        let mut thread_depth = None;
        let table_diameter = match &h.form {
            HoleForm::Tapped {
                thread_depth: td,
                modelled: m,
                near_countersink,
            } => {
                let t = self.hole_thread(id, h, false)?;
                modelled = *m;
                thread_depth = self.opt_len(id, "form.thread_depth", td)?;
                if let Some(a) = near_countersink {
                    let a = self.value(id, "form.near_countersink", a, Dimension::ANGLE)?;
                    countersink = Some((t.major * 1.1, a));
                }
                let d = t.tap_drill;
                thread = Some(t);
                Some(d)
            }
            HoleForm::PipeTap {
                thread_depth: td,
                modelled: m,
            } => {
                let t = self.hole_thread(id, h, true)?;
                modelled = *m;
                thread_depth = self.opt_len(id, "form.thread_depth", td)?.or(t.length);
                taper = t.taper;
                let d = t.tap_drill;
                thread = Some(t);
                Some(d)
            }
            HoleForm::Simple | HoleForm::Slot { .. } => fastener.as_ref().map(|f| f.clearance(fit)),
            HoleForm::Counterbore { diameter, depth } => {
                let table = fastener.as_ref().and_then(|f| f.counterbore);
                let d = self
                    .opt_len(id, "form.diameter", diameter)?
                    .or(table.map(|t| t.0));
                let k = self
                    .opt_len(id, "form.depth", depth)?
                    .or(table.map(|t| t.1));
                match (d, k) {
                    (Some(d), Some(k)) => counterbore = Some((d, k)),
                    _ => {
                        return Err(self.error(
                            id,
                            Some("form"),
                            DiagCode::RegenInvalid,
                            "a counterbore needs its diameter and depth, or a standard size".into(),
                            Vec::new(),
                        ))
                    }
                }
                fastener.as_ref().map(|f| f.clearance(fit))
            }
            HoleForm::Countersink { diameter, angle } => {
                let table = fastener.as_ref().and_then(|f| f.countersink);
                let d = self
                    .opt_len(id, "form.diameter", diameter)?
                    .or(table.map(|t| t.0));
                let a = match angle {
                    Some(a) => Some(self.value(id, "form.angle", a, Dimension::ANGLE)?),
                    None => table.map(|t| t.1).or(Some(if inch {
                        82f64.to_radians()
                    } else {
                        std::f64::consts::FRAC_PI_2
                    })),
                };
                match (d, a) {
                    (Some(d), Some(a)) => countersink = Some((d, a)),
                    _ => {
                        return Err(self.error(
                            id,
                            Some("form"),
                            DiagCode::RegenInvalid,
                            "a countersink needs its diameter, or a standard size".into(),
                            Vec::new(),
                        ))
                    }
                }
                fastener.as_ref().map(|f| f.clearance(fit))
            }
        };
        if let HoleForm::Slot {
            length,
            angle,
            counterbore: cb,
        } = &h.form
        {
            let l = self.value(id, "form.length", length, Dimension::LENGTH)?;
            let a = match angle {
                Some(a) => self.value(id, "form.angle", a, Dimension::ANGLE)?,
                None => 0.0,
            };
            slot = Some((l, a));
            if let Some((d, k)) = cb {
                counterbore = Some((
                    self.value(id, "form.counterbore.diameter", d, Dimension::LENGTH)?,
                    self.value(id, "form.counterbore.depth", k, Dimension::LENGTH)?,
                ));
            }
        }
        let diameter = match self
            .opt_len(id, "diameter", &h.diameter)?
            .or(table_diameter)
        {
            Some(d) if d > 0.0 => d,
            _ => {
                return Err(self.error(
                    id,
                    Some("diameter"),
                    DiagCode::RegenInvalid,
                    "a hole needs a diameter or a standard size".into(),
                    Vec::new(),
                ))
            }
        };
        let mut ops = Operands {
            combine: bodies,
            refs: Vec::new(),
        };
        let mut stages = Vec::new();
        if let Some(series) = &h.series {
            stages = self.hole_series(i, id, h, series, &frame, &positions, diameter, &mut ops)?;
        }
        let (end, depth) = match &h.end {
            HoleEnd::Blind { depth } => {
                let d = self.value(id, "end.depth", depth, Dimension::LENGTH)?;
                (HoleDepth::Blind { depth: d }, Some(d))
            }
            HoleEnd::ThroughAll => (HoleDepth::ThroughAll, None),
            HoleEnd::UpToNext => (HoleDepth::UpToNext, None),
            HoleEnd::UpToFace { face: f, offset } => {
                let fh = self.face_entity(i, id, "up_to", f)?;
                let o = self.opt_len(id, "end.offset", offset)?.unwrap_or(0.0);
                if let Some(z) = self.parallel_height(&fh, &frame) {
                    let d = -z + o;
                    (HoleDepth::Blind { depth: d }, Some(d))
                } else {
                    (
                        HoleDepth::UpToFace {
                            face: OperandEntity {
                                operand: ops.index(fh.body as usize),
                                entity: fh.index,
                            },
                            offset: o,
                        },
                        None,
                    )
                }
            }
        };
        let drill_point = match (&h.drill_point, &end) {
            (Some(a), HoleDepth::Blind { .. }) => {
                Some(self.value(id, "drill_point", a, Dimension::ANGLE)?)
            }
            _ => None,
        };
        // Thread length: as given; blind holes stop one and a half pitches short of the drill
        // depth; through holes thread their whole wall.
        let thread_len = thread
            .as_ref()
            .map(|t| thread_depth.or(depth.map(|d| (d - 1.5 * t.pitch).max(t.pitch))));
        let tool = ToolOp {
            shape: ToolShape::Holes {
                frame,
                positions: positions.clone(),
                diameter,
                end,
                counterbore,
                countersink,
                drill_point,
                taper,
                thread: thread_len
                    .filter(|_| modelled)
                    .map(|length| HoleThread { length }),
                slot,
                stages,
            },
            combine: Combine::Cut,
            references: ops.refs.len() as u32,
        };
        // The holes, for drawings and the hole table.
        let instances = positions
            .iter()
            .map(|(key, at)| HoleInstance {
                feature: id.clone(),
                key: key.clone(),
                position: frame.to_world(*at),
                axis: scale(frame.z, -1.0),
                diameter,
                depth,
                thread: thread.as_ref().map(|t| ThreadCallout {
                    designation: t.designation.clone(),
                    class: (!t.class.is_empty()).then(|| t.class.clone()),
                    depth: thread_len.flatten(),
                }),
                counterbore,
                countersink,
            })
            .collect();
        self.pending_record = Some(FeatureRecord::Holes { holes: instances });
        let reusable = ops.refs.is_empty();
        Ok(Compiled::Op {
            op: KernelOp::Tool(tool.clone()),
            operands: ops.all(),
            naming: NamingContext::simple(id, Vec::new()),
            import: false,
            instances: None,
            tools: reusable.then(|| vec![tool]),
        })
    }

    /// The bodies of a hole series in order from the start face, with each one's diameter.
    #[allow(clippy::too_many_arguments)]
    fn hole_series(
        &mut self,
        i: usize,
        id: &FeatureId,
        h: &Hole,
        series: &HoleSeries,
        frame: &Frame,
        positions: &[(String, [f64; 2])],
        start_diameter: f64,
        ops: &mut Operands,
    ) -> Result<Vec<HoleStage>, Halt> {
        let start = ops.combine[0];
        let mut bodies: Vec<usize> = Vec::new();
        if series.bodies.is_empty() {
            // Every body a hole axis crosses below the start face, nearest first.
            let mut found: Vec<(f64, usize)> = Vec::new();
            for (b, body) in self.st.bodies.iter().enumerate() {
                let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
                for f in &body.topology.faces {
                    for a in 0..3 {
                        lo[a] = lo[a].min(f.geom.bbox_min[a]);
                        hi[a] = hi[a].max(f.geom.bbox_max[a]);
                    }
                }
                let crossed = positions.iter().any(|(_, at)| {
                    let o = frame.to_world(*at);
                    // The axis runs along -z; test the box in the frame.
                    let mut tmin = f64::NEG_INFINITY;
                    let mut tmax = f64::INFINITY;
                    for a in 0..3 {
                        let d = -frame.z[a];
                        if d.abs() < 1e-12 {
                            if o[a] < lo[a] - 1e-9 || o[a] > hi[a] + 1e-9 {
                                return false;
                            }
                        } else {
                            let (t0, t1) = ((lo[a] - o[a]) / d, (hi[a] - o[a]) / d);
                            tmin = tmin.max(t0.min(t1));
                            tmax = tmax.min(t0.max(t1));
                        }
                    }
                    tmax > tmin + 1e-9 && tmax > 1e-9
                });
                if crossed {
                    let c = [
                        (lo[0] + hi[0]) / 2.0,
                        (lo[1] + hi[1]) / 2.0,
                        (lo[2] + hi[2]) / 2.0,
                    ];
                    found.push((-dot(sub(c, frame.origin), frame.z), b));
                }
            }
            found.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
            bodies = found.into_iter().map(|x| x.1).collect();
            bodies.retain(|b| *b != start);
            bodies.insert(0, start);
        } else {
            for (k, r) in series.bodies.iter().enumerate() {
                let res = self.single(
                    i,
                    id,
                    &format!("series.bodies[{}]", k),
                    r,
                    Some(TopoKind::Body),
                )?;
                bodies.push(res.entities().iter().next().expect("one").body as usize);
            }
            if bodies.first() != Some(&start) {
                return Err(self.error(
                    id,
                    Some("series.bodies"),
                    DiagCode::RegenInvalid,
                    "a hole series starts in the body of its start face".into(),
                    Vec::new(),
                ));
            }
        }
        if bodies.len() < 2 {
            return Err(self.error(
                id,
                Some("series"),
                DiagCode::RefEmpty,
                "a hole series needs the hole axis to cross at least two bodies".into(),
                Vec::new(),
            ));
        }
        let middle = self
            .opt_len(id, "series.middle_diameter", &series.middle_diameter)?
            .unwrap_or(start_diameter);
        let (end_diameter, end_thread) = match &series.end {
            HoleForm::Tapped { modelled, .. } => {
                (self.hole_thread(id, h, false)?.tap_drill, *modelled)
            }
            _ => (middle, false),
        };
        ops.combine = bodies.clone();
        let n = bodies.len();
        Ok((0..n)
            .map(|k| HoleStage {
                operand: k as u32,
                diameter: if k == 0 {
                    start_diameter
                } else if k == n - 1 {
                    end_diameter
                } else {
                    middle
                },
                thread: k == n - 1 && end_thread,
            })
            .collect())
    }

    // ---------------------------------------------------------------------------- blends

    fn blend_profile(&mut self, id: &FeatureId, p: &FilletProfile) -> Result<BlendProfile, Halt> {
        Ok(match p {
            FilletProfile::Circular => BlendProfile::Circular,
            FilletProfile::Curvature => BlendProfile::Curvature,
            FilletProfile::Conic { rho } => {
                let r = self.value(id, "profile.rho", rho, Dimension::DIMENSIONLESS)?;
                if !(0.0..1.0).contains(&r) || r == 0.0 {
                    return Err(self.error(
                        id,
                        Some("profile.rho"),
                        DiagCode::ValueFailed,
                        format!("a conic blend's rho must be between 0 and 1, got {}", r),
                        Vec::new(),
                    ));
                }
                BlendProfile::Conic { rho: r }
            }
        })
    }

    pub(super) fn fillet(
        &mut self,
        i: usize,
        id: &FeatureId,
        fl: &Fillet,
    ) -> Result<Compiled, Halt> {
        let set = self.list(i, id, "edges", &fl.edges, Some(TopoKind::Edge))?;
        let radius = self.value(id, "radius", &fl.radius, Dimension::LENGTH)?;
        let mut variable = Vec::new();
        for (k, r) in fl.variable.iter().enumerate() {
            let t = self.value(
                id,
                &format!("variable[{}].at", k),
                &r.at,
                Dimension::DIMENSIONLESS,
            )?;
            let v = self.value(
                id,
                &format!("variable[{}].radius", k),
                &r.radius,
                Dimension::LENGTH,
            )?;
            if !(0.0..=1.0).contains(&t) || v <= 0.0 {
                return Err(self.error(
                    id,
                    Some(&format!("variable[{}]", k)),
                    DiagCode::ValueFailed,
                    "a variable radius sits at 0 to 1 along the edge and is positive".into(),
                    Vec::new(),
                ));
            }
            variable.push((t, v));
        }
        variable.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
        let profile = self.blend_profile(id, &fl.profile)?;
        let mut operands = Self::bodies_of(&set);
        let mut distances = Vec::new();
        for (k, s) in fl.setbacks.iter().enumerate() {
            let r = self.single(
                i,
                id,
                &format!("setbacks[{}]", k),
                &s.vertex,
                Some(TopoKind::Vertex),
            )?;
            let h = *r.entities().iter().next().expect("one");
            distances.push((
                h,
                self.value(
                    id,
                    &format!("setbacks[{}].distance", k),
                    &s.distance,
                    Dimension::LENGTH,
                )?,
            ));
        }
        for (h, _) in &distances {
            if !operands.contains(&(h.body as usize)) {
                operands.push(h.body as usize);
                operands.sort();
            }
        }
        let edges = Self::operand_entities(&set, &operands);
        let setbacks = distances
            .iter()
            .map(|(h, d)| {
                let one: BTreeSet<EntityHandle> = [*h].into_iter().collect();
                (Self::operand_entities(&one, &operands)[0], *d)
            })
            .collect();
        Ok(Compiled::Op {
            op: KernelOp::Fillet {
                edges,
                radius,
                variable,
                profile,
                setbacks,
            },
            operands,
            naming: NamingContext::simple(id, Vec::new()),
            import: false,
            instances: None,
            tools: None,
        })
    }

    /// Face sets of one body, as operand entities of it.
    fn face_sets(
        &mut self,
        i: usize,
        id: &FeatureId,
        sets: &[(&str, &[Reference])],
    ) -> Result<(Vec<usize>, Vec<Vec<OperandEntity>>), Halt> {
        let mut resolved = Vec::new();
        for (param, refs) in sets {
            resolved.push(self.list(i, id, param, refs, Some(TopoKind::Face))?);
        }
        let mut bodies: Vec<usize> = resolved.iter().flat_map(Self::bodies_of).collect();
        bodies.sort();
        bodies.dedup();
        if bodies.len() != 1 {
            return Err(self.error(
                id,
                Some(sets[0].0),
                DiagCode::RefKindMismatch,
                "the faces of a face blend must all be on one body".into(),
                Vec::new(),
            ));
        }
        let out = resolved
            .iter()
            .map(|s| Self::operand_entities(s, &bodies))
            .collect();
        Ok((bodies, out))
    }

    pub(super) fn face_fillet(
        &mut self,
        i: usize,
        id: &FeatureId,
        f: &FaceFillet,
    ) -> Result<Compiled, Halt> {
        let (operands, mut sets) =
            self.face_sets(i, id, &[("first", &f.first), ("second", &f.second)])?;
        let radius = self.value(id, "radius", &f.radius, Dimension::LENGTH)?;
        let profile = self.blend_profile(id, &f.profile)?;
        let second = sets.pop().expect("two");
        let first = sets.pop().expect("two");
        Ok(Compiled::Op {
            op: KernelOp::FaceBlend {
                first,
                second,
                radius,
                profile,
            },
            operands,
            naming: NamingContext::simple(id, Vec::new()),
            import: false,
            instances: None,
            tools: None,
        })
    }

    pub(super) fn full_round(
        &mut self,
        i: usize,
        id: &FeatureId,
        f: &FullRound,
    ) -> Result<Compiled, Halt> {
        let (operands, mut sets) = self.face_sets(
            i,
            id,
            &[
                ("first", &f.first),
                ("center", &f.center),
                ("second", &f.second),
            ],
        )?;
        let second = sets.pop().expect("three");
        let center = sets.pop().expect("three");
        let first = sets.pop().expect("three");
        Ok(Compiled::Op {
            op: KernelOp::FullRound {
                first,
                center,
                second,
            },
            operands,
            naming: NamingContext::simple(id, Vec::new()),
            import: false,
            instances: None,
            tools: None,
        })
    }

    pub(super) fn chamfer(
        &mut self,
        i: usize,
        id: &FeatureId,
        c: &Chamfer,
    ) -> Result<Compiled, Halt> {
        let len = |s: &mut Self, p: &str, v: &ValueExpr| s.value(id, p, v, Dimension::LENGTH);
        if let ChamferForm::Vertex { distances } = &c.form {
            let set = self.list(i, id, "edges", &c.edges, Some(TopoKind::Vertex))?;
            let mut d = [0.0; 3];
            for k in 0..3 {
                d[k] = len(self, &format!("form.distances[{}]", k), &distances[k])?;
            }
            let operands = Self::bodies_of(&set);
            let vertices = Self::operand_entities(&set, &operands);
            return Ok(Compiled::Op {
                op: KernelOp::VertexChamfer {
                    vertices,
                    distances: d,
                },
                operands,
                naming: NamingContext::simple(id, Vec::new()),
                import: false,
                instances: None,
                tools: None,
            });
        }
        let set = self.list(i, id, "edges", &c.edges, Some(TopoKind::Edge))?;
        let size = match &c.form {
            ChamferForm::Equal { distance } => ChamferSize::Equal {
                distance: len(self, "form.distance", distance)?,
            },
            ChamferForm::Distances {
                distance,
                distance2,
            } => {
                let (a, b) = (
                    len(self, "form.distance", distance)?,
                    len(self, "form.distance2", distance2)?,
                );
                let (first, second) = if c.flip { (b, a) } else { (a, b) };
                ChamferSize::Distances { first, second }
            }
            ChamferForm::DistanceAngle { distance, angle } => {
                let d = len(self, "form.distance", distance)?;
                let a = self.value(id, "form.angle", angle, Dimension::ANGLE)?;
                if a <= 0.0 || a >= std::f64::consts::FRAC_PI_2 {
                    return Err(self.error(
                        id,
                        Some("form.angle"),
                        DiagCode::ValueFailed,
                        "a chamfer angle lies strictly between 0° and 90°".into(),
                        Vec::new(),
                    ));
                }
                ChamferSize::DistanceAngle {
                    distance: d,
                    angle: a,
                    on_second: c.flip,
                }
            }
            ChamferForm::Vertex { .. } => unreachable!("handled above"),
        };
        let operands = Self::bodies_of(&set);
        let edges = Self::operand_entities(&set, &operands);
        Ok(Compiled::Op {
            op: KernelOp::Chamfer { edges, size },
            operands,
            naming: NamingContext::simple(id, Vec::new()),
            import: false,
            instances: None,
            tools: None,
        })
    }

    pub(super) fn shell(&mut self, i: usize, id: &FeatureId, s: &Shell) -> Result<Compiled, Halt> {
        let b = self.single(i, id, "body", &s.body, Some(TopoKind::Body))?;
        let body = b.entities().iter().next().expect("one").body as usize;
        let operands = vec![body];
        let on_body = |s: &mut Self, set: &BTreeSet<EntityHandle>, param: &str| {
            if set.iter().any(|h| h.body as usize != body) {
                Err(s.error(
                    id,
                    Some(param),
                    DiagCode::RefKindMismatch,
                    "faces of a shell must be on the shelled body".into(),
                    Vec::new(),
                ))
            } else {
                Ok(())
            }
        };
        let faces = if s.open_faces.is_empty() {
            BTreeSet::new()
        } else {
            self.list(i, id, "open_faces", &s.open_faces, Some(TopoKind::Face))?
        };
        on_body(self, &faces, "open_faces")?;
        let thickness = self.value(id, "thickness", &s.thickness, Dimension::LENGTH)?;
        let mut per_face = Vec::new();
        for (k, ft) in s.faces.iter().enumerate() {
            let p = format!("faces[{}].faces", k);
            let set = self.list(i, id, &p, &ft.faces, Some(TopoKind::Face))?;
            on_body(self, &set, &p)?;
            let t = self.value(
                id,
                &format!("faces[{}].thickness", k),
                &ft.thickness,
                Dimension::LENGTH,
            )?;
            if t <= 0.0 {
                return Err(self.error(
                    id,
                    Some(&format!("faces[{}].thickness", k)),
                    DiagCode::ValueFailed,
                    "shell thickness must be positive".into(),
                    Vec::new(),
                ));
            }
            for e in Self::operand_entities(&set, &operands) {
                per_face.push((e, t));
            }
        }
        let open_faces = Self::operand_entities(&faces, &operands);
        Ok(Compiled::Op {
            op: KernelOp::Shell {
                open_faces,
                thickness,
                faces: per_face,
                outward: s.outward,
            },
            operands,
            naming: NamingContext::simple(id, Vec::new()),
            import: false,
            instances: None,
            tools: None,
        })
    }

    pub(super) fn draft(&mut self, i: usize, id: &FeatureId, d: &Draft) -> Result<Compiled, Halt> {
        let set = self.list(i, id, "faces", &d.faces, Some(TopoKind::Face))?;
        let mut neutral = self.plane(i, id, "neutral", &d.neutral)?;
        if d.flip {
            neutral.z = scale(neutral.z, -1.0);
            neutral.x = scale(neutral.x, -1.0);
        }
        let angle = self.value(id, "angle", &d.angle, Dimension::ANGLE)?;
        let operands = Self::bodies_of(&set);
        let faces = Self::operand_entities(&set, &operands);
        let edges = d.kind.edges();
        let parting = if edges.is_empty() {
            Vec::new()
        } else {
            let es = self.list(i, id, "parting", edges, Some(TopoKind::Edge))?;
            if es.iter().any(|h| !operands.contains(&(h.body as usize))) {
                return Err(self.error(
                    id,
                    Some("parting"),
                    DiagCode::RefKindMismatch,
                    "parting-line edges must be on the drafted bodies".into(),
                    Vec::new(),
                ));
            }
            Self::operand_entities(&es, &operands)
        };
        if !matches!(d.kind, DraftKind::NeutralPlane) && parting.is_empty() {
            return Err(self.error(
                id,
                Some("parting"),
                DiagCode::RefEmpty,
                "a parting-line draft needs its parting-line edges".into(),
                Vec::new(),
            ));
        }
        Ok(Compiled::Op {
            op: KernelOp::Draft {
                faces,
                neutral,
                angle,
                parting,
                step: matches!(d.kind, DraftKind::Step { .. }),
            },
            operands,
            naming: NamingContext::simple(id, Vec::new()),
            import: false,
            instances: None,
            tools: None,
        })
    }
}
