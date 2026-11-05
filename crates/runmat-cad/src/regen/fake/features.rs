//! Fake-kernel support for the Features-tab options: extrusions with end conditions, draft and
//! thin walls; holes with end conditions, drill points, tapers, modelled threads, slots and
//! series; tools with reference-only operands; face blends, full rounds and vertex chamfers.

use super::sampled::{self, Curve2, ZBound};
use super::*;
use std::sync::Arc;

pub(super) type RectSet = Vec<(usize, f64, Rect)>;

pub(super) fn rects_of_face(s: &FakeSolid, face: u32) -> RectSet {
    s.faces
        .get(face as usize)
        .map(|f| {
            f.rect_planes()
                .into_iter()
                .map(|(p, r)| (p.axis, p.offset, r))
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn rects_of_solid(s: &FakeSolid) -> RectSet {
    (0..s.faces.len() as u32)
        .flat_map(|f| rects_of_face(s, f))
        .collect()
}

/// Every parameter `t` where the line `o + t d` crosses a rect, sorted and distinct.
fn line_hits(rects: &RectSet, o: [f64; 3], d: [f64; 3]) -> Vec<f64> {
    let mut out = Vec::new();
    for (axis, off, r) in rects {
        if d[*axis].abs() < 1e-12 {
            continue;
        }
        let t = (off - o[*axis]) / d[*axis];
        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        if (0..3).all(|a| a == *axis || (p[a] >= r.lo[a] - 1e-9 && p[a] <= r.hi[a] + 1e-9)) {
            out.push(t);
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    out
}

/// Outputs of separate runs as one, with body indices offset.
pub(super) fn concat(outs: Vec<KernelOutput>) -> KernelOutput {
    let mut bodies = Vec::new();
    let mut history = OpHistory::default();
    for mut out in outs {
        let base = bodies.len() as u32;
        for r in &mut out.history.faces {
            r.body += base;
        }
        for r in &mut out.history.bodies {
            r.body += base;
        }
        for r in &mut out.history.boundaries {
            r.body += base;
        }
        history.faces.extend(out.history.faces);
        history.bodies.extend(out.history.bodies);
        history.boundaries.extend(out.history.boundaries);
        history.deleted_faces.extend(out.history.deleted_faces);
        history.deleted_bodies.extend(out.history.deleted_bodies);
        bodies.extend(out.bodies);
    }
    history.deleted_faces.sort();
    history.deleted_bodies.sort();
    KernelOutput { bodies, history }
}

pub(super) fn keep_job<'a>(operands: Vec<&'a FakeSolid>, ids: Vec<u32>) -> GridJob<'a> {
    GridJob {
        operands,
        operand_ids: ids,
        tools: Vec::new(),
        mode: Mode::Keep,
        split: None,
        separate_tools: false,
        body_label: None,
        modifier: None,
        merge_collinear: false,
        carve: Vec::new(),
    }
}

type Reach<'r> = &'r dyn Fn([f64; 3], [f64; 3]) -> (f64, f64);

impl FakeKernel {
    /// Run a tool operation: tools may read every operand (end conditions); the last
    /// `references` operands come back unchanged after the combined result.
    pub(super) fn tool_op(
        &self,
        t: &ToolOp,
        solids: &[&FakeSolid],
    ) -> Result<KernelOutput, KernelError> {
        let n_ref = (t.references as usize).min(solids.len());
        let n_comb = solids.len() - n_ref;
        let tools = self.tools(t, solids, 0)?;
        let mode = Self::mode(t.combine);
        let (ops, ids): (Vec<&FakeSolid>, Vec<u32>) = if mode == Mode::New {
            (Vec::new(), Vec::new())
        } else {
            (solids[..n_comb].to_vec(), (0..n_comb as u32).collect())
        };
        let res = run_grid(GridJob {
            operands: ops.clone(),
            operand_ids: ids.clone(),
            tools,
            mode,
            split: None,
            separate_tools: false,
            body_label: None,
            modifier: None,
            merge_collinear: t.shape.merge_collinear(),
            carve: Vec::new(),
        })?;
        let mut outs = vec![self.finish(&ops, &ids, res, Vec::new(), Vec::new())?];
        // Operands not combined (references, or all of them for a new body) are kept.
        let kept: Vec<usize> = if mode == Mode::New {
            (0..solids.len()).collect()
        } else {
            (n_comb..solids.len()).collect()
        };
        for k in kept {
            let res = run_grid(keep_job(vec![solids[k]], vec![k as u32]))?;
            let mut out = self.finish(&[solids[k]], &[k as u32], res, Vec::new(), Vec::new())?;
            for r in &mut out.history.faces {
                r.evolution = Evolution::Kept;
            }
            outs.push(out);
        }
        Ok(concat(outs))
    }

    fn bound(
        &self,
        b: &PrismBound,
        at: f64,
        sense: f64,
        frame: &Frame,
        solids: &[&FakeSolid],
        reach: Reach<'_>,
    ) -> Result<ZBound, KernelError> {
        let solid = |k: u32| {
            solids.get(k as usize).copied().ok_or(KernelError::Failed {
                message: "end condition operand out of range".into(),
            })
        };
        Ok(match b {
            PrismBound::Blind => ZBound::At(at),
            PrismBound::ThroughAll => {
                let (lo, hi) = reach(frame.origin, frame.z);
                ZBound::At(if sense > 0.0 { hi } else { lo })
            }
            PrismBound::UpToNext => ZBound::Hits {
                rects: Arc::new(solids.iter().flat_map(|s| rects_of_solid(s)).collect()),
                from: 0.0,
                sense,
                offset: 0.0,
            },
            PrismBound::UpToFace { face, offset } => ZBound::Hits {
                rects: Arc::new(rects_of_face(solid(face.operand)?, face.entity)),
                from: 0.0,
                sense,
                offset: *offset,
            },
            PrismBound::UpToBody { operand, offset } => ZBound::Hits {
                rects: Arc::new(rects_of_solid(solid(*operand)?)),
                from: 0.0,
                sense,
                offset: *offset,
            },
            PrismBound::FromFace { face } => ZBound::Hits {
                rects: Arc::new(rects_of_face(solid(face.operand)?, face.entity)),
                from: 0.0,
                sense: 0.0,
                offset: 0.0,
            },
        })
    }

    pub(super) fn extrusion(
        &self,
        op: &ToolOp,
        solids: &[&FakeSolid],
        profile_index: u32,
        reach: Reach<'_>,
    ) -> Result<Vec<Tool>, KernelError> {
        let ToolShape::Prism {
            profile,
            start,
            end,
            start_bound,
            end_bound,
            draft,
            thin,
        } = &op.shape
        else {
            unreachable!("prisms only")
        };
        let frame = profile.frame;
        let label = |role: Role, curve: &crate::ids::SketchEntityId| {
            gen(
                role,
                KernelTag::Curve {
                    profile: profile_index,
                    curve: curve.clone(),
                },
                None,
            )
        };
        let loops: Vec<Vec<(Curve2, HistorySource)>> = profile
            .loops
            .iter()
            .map(|l| {
                l.iter()
                    .map(|seg| (Curve2::from_segment(seg), label(Role::Side, &seg.curve)))
                    .collect()
            })
            .collect();
        let inner: Vec<Vec<HistorySource>> = profile
            .loops
            .iter()
            .map(|l| {
                l.iter()
                    .map(|seg| label(Role::ThinWall, &seg.curve))
                    .collect()
            })
            .collect();
        let end_sense = if *end != 0.0 {
            end.signum()
        } else {
            -start.signum()
        };
        let start_sense = if *start != 0.0 {
            start.signum()
        } else {
            -end_sense
        };
        let zs = self.bound(start_bound, *start, start_sense, &frame, solids, reach)?;
        let ze = match (start_bound, end_bound) {
            (PrismBound::FromFace { .. }, PrismBound::Blind) => ZBound::Past {
                base: Box::new(zs.clone()),
                distance: *end,
            },
            _ => self.bound(end_bound, *end, end_sense, &frame, solids, reach)?,
        };
        let tool = sampled::extrusion(
            Transform::from_frame(&frame),
            &loops,
            thin.is_some().then_some(inner.as_slice()),
            zs,
            ze,
            [
                gen(Role::StartCap, KernelTag::None, None),
                gen(Role::EndCap, KernelTag::None, None),
            ],
            draft.map(|d| (0.0, d.end.tan(), d.start.tan())),
            thin.map(|t| {
                (
                    t.inside,
                    t.outside,
                    t.cap.map(|c| {
                        (
                            c,
                            [
                                gen(Role::ThinWall, KernelTag::None, Some(0)),
                                gen(Role::ThinWall, KernelTag::None, Some(1)),
                            ],
                        )
                    }),
                )
            }),
        )?;
        Ok(vec![Tool::Sampled(Box::new(tool))])
    }

    pub(super) fn holes(
        &self,
        shape: &ToolShape,
        solids: &[&FakeSolid],
        reach: Reach<'_>,
    ) -> Result<Vec<Tool>, KernelError> {
        let ToolShape::Holes {
            frame,
            positions,
            diameter,
            end,
            counterbore,
            countersink,
            drill_point,
            taper,
            thread,
            slot,
            stages,
        } = shape
        else {
            unreachable!("holes only")
        };
        let down = [-frame.z[0], -frame.z[1], -frame.z[2]];
        let all: RectSet = solids.iter().flat_map(|s| rects_of_solid(s)).collect();
        let mut out = Vec::new();
        for (key, at) in positions {
            let origin = frame.to_world(*at);
            // How deep (positive, along -z) the hole's full diameter goes.
            let depth = match end {
                HoleDepth::Blind { depth } => *depth,
                HoleDepth::ThroughAll => -reach(origin, frame.z).0,
                HoleDepth::UpToNext => line_hits(&all, origin, down)
                    .into_iter()
                    .find(|t| *t > 1e-9)
                    .ok_or_else(|| KernelError::Failed {
                        message: format!("hole `{}` meets no face below its start", key),
                    })?,
                HoleDepth::UpToFace { face, offset } => {
                    let s = solids
                        .get(face.operand as usize)
                        .ok_or(KernelError::Failed {
                            message: "hole end face out of range".into(),
                        })?;
                    line_hits(&rects_of_face(s, face.entity), origin, down)
                        .into_iter()
                        .find(|t| *t > 1e-9)
                        .map(|t| t + offset)
                        .ok_or_else(|| KernelError::Failed {
                            message: format!("hole `{}` does not reach its end face", key),
                        })?
                }
            };
            if depth <= 0.0 {
                return Err(KernelError::Failed {
                    message: format!("hole `{}` has no depth", key),
                });
            }
            if let Some((length, angle)) = slot {
                out.extend(self.slot(
                    key,
                    frame,
                    *at,
                    *diameter,
                    *length,
                    *angle,
                    depth,
                    *counterbore,
                )?);
                continue;
            }
            // Series stages: each body's depth range along the axis.
            let mut spans: Vec<(f64, f64, bool, Option<u32>)> = Vec::new();
            for st in stages {
                let s = solids.get(st.operand as usize).ok_or(KernelError::Failed {
                    message: "hole series operand out of range".into(),
                })?;
                let hits = line_hits(&rects_of_solid(s), origin, down);
                let probe = |t: f64| {
                    let p = [
                        origin[0] + down[0] * t,
                        origin[1] + down[1] * t,
                        origin[2] + down[2] * t,
                    ];
                    s.contains(p)
                };
                let mut exit = None;
                for w in hits.windows(2) {
                    if w[1] > 1e-9 && probe((w[0].max(0.0) + w[1]) / 2.0) {
                        exit = Some(w[1]);
                    }
                }
                let Some(exit) = exit else {
                    return Err(KernelError::Failed {
                        message: format!(
                            "hole `{}` does not cross series body {}",
                            key, st.operand
                        ),
                    });
                };
                spans.push((
                    exit.min(depth),
                    st.diameter / 2.0,
                    st.thread,
                    Some(st.operand),
                ));
            }
            let section_frame = Frame {
                origin,
                x: frame.x,
                y: frame.z,
                z: [-frame.y[0], -frame.y[1], -frame.y[2]],
            };
            let section = hole_section(
                key,
                diameter / 2.0,
                -depth,
                *counterbore,
                *countersink,
                *drill_point,
                *taper,
                thread.map(|t| t.length),
                &spans,
            );
            out.push(Tool::Sampled(Box::new(sampled::revolve(
                Transform::from_frame(&section_frame),
                &[section],
                [0.0, 0.0],
                [0.0, 1.0],
                std::f64::consts::TAU,
                None,
            )?)));
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn slot(
        &self,
        key: &str,
        frame: &Frame,
        at: [f64; 2],
        diameter: f64,
        length: f64,
        angle: f64,
        depth: f64,
        counterbore: Option<(f64, f64)>,
    ) -> Result<Vec<Tool>, KernelError> {
        let (s, c) = angle.sin_cos();
        let u = [c, s];
        let obround = |w: f64| -> Vec<Curve2> {
            let a = [at[0] - u[0] * length / 2.0, at[1] - u[1] * length / 2.0];
            let b = [at[0] + u[0] * length / 2.0, at[1] + u[1] * length / 2.0];
            let v = [-u[1] * w, u[0] * w];
            let a0 = u[1].atan2(u[0]) - std::f64::consts::FRAC_PI_2;
            vec![
                Curve2::Line {
                    a: [a[0] - v[0], a[1] - v[1]],
                    b: [b[0] - v[0], b[1] - v[1]],
                },
                Curve2::Arc {
                    c: b,
                    r: w,
                    a0,
                    sweep: std::f64::consts::PI,
                },
                Curve2::Line {
                    a: [b[0] + v[0], b[1] + v[1]],
                    b: [a[0] + v[0], a[1] + v[1]],
                },
                Curve2::Arc {
                    c: a,
                    r: w,
                    a0: a0 + std::f64::consts::PI,
                    sweep: std::f64::consts::PI,
                },
            ]
        };
        let k = || KernelTag::Key { key: key.into() };
        let labelled = |w: f64, role: Role| -> Vec<Vec<(Curve2, HistorySource)>> {
            vec![obround(w)
                .into_iter()
                .enumerate()
                .map(|(i, c)| (c, gen(role, k(), Some(i as u32))))
                .collect()]
        };
        let to_world = Transform::from_frame(frame);
        let mut out = vec![Tool::Sampled(Box::new(sampled::prism(
            to_world,
            &labelled(diameter / 2.0, Role::HoleWall),
            0.0,
            -depth,
            [
                gen(Role::HoleBottom, k(), Some(1)),
                gen(Role::HoleBottom, k(), None),
            ],
        )))];
        if let Some((d, h)) = counterbore {
            out.push(Tool::Sampled(Box::new(sampled::prism(
                to_world,
                &labelled(d / 2.0, Role::HoleCounterboreWall),
                0.0,
                -h,
                [
                    gen(Role::HoleBottom, k(), Some(1)),
                    gen(Role::HoleCounterbore, k(), None),
                ],
            ))));
        }
        Ok(out)
    }

    /// Face blend: blend every edge shared by a face of each set.
    pub(super) fn face_blend(
        &self,
        first: &[OperandEntity],
        second: &[OperandEntity],
        radius: f64,
        solids: &[&FakeSolid],
    ) -> Result<Vec<OperandEntity>, KernelError> {
        let s = solids.first().ok_or(KernelError::Failed {
            message: "a face blend needs its body".into(),
        })?;
        let a: BTreeSet<u32> = first.iter().map(|e| e.entity).collect();
        let b: BTreeSet<u32> = second.iter().map(|e| e.entity).collect();
        let mut edges = Vec::new();
        for (i, e) in s.topology.edges.iter().enumerate() {
            let [f, g] = e.faces;
            if (a.contains(&f) && b.contains(&g)) || (a.contains(&g) && b.contains(&f)) {
                edges.push(OperandEntity {
                    operand: 0,
                    entity: i as u32,
                });
            }
        }
        if edges.is_empty() {
            return Err(KernelError::NotSupported {
                what: "fake kernel: face fillets between faces that do not meet".into(),
            });
        }
        if radius <= 0.0 {
            return Err(KernelError::Failed {
                message: "a fillet radius must be positive".into(),
            });
        }
        Ok(edges)
    }

    /// Full round: the centre faces become one blend face tangent to both sides.
    pub(super) fn full_round(
        &self,
        first: &[OperandEntity],
        center: &[OperandEntity],
        second: &[OperandEntity],
        solids: &[&FakeSolid],
    ) -> Result<KernelOutput, KernelError> {
        let s = *solids.first().ok_or(KernelError::Failed {
            message: "a full round needs its body".into(),
        })?;
        let plane = |e: &OperandEntity| s.faces.get(e.entity as usize).and_then(|f| f.plane);
        let (Some(pa), Some(pb)) = (
            first.first().and_then(plane),
            second.first().and_then(plane),
        ) else {
            return Err(KernelError::NotSupported {
                what: "fake kernel: full rounds between curved side faces".into(),
            });
        };
        if pa.axis != pb.axis || pa.dir == pb.dir {
            return Err(KernelError::Failed {
                message: "a full round's side faces must face away from each other".into(),
            });
        }
        let radius = (pa.offset - pb.offset).abs() / 2.0;
        let res = run_grid(keep_job(vec![s], vec![0]))?;
        let mut out = self.finish(&[s], &[0], res, Vec::new(), Vec::new())?;
        let centre: BTreeSet<u32> = center.iter().map(|e| e.entity).collect();
        let mut axis_dir = [0.0; 3];
        let mut replaced = BTreeSet::new();
        for r in &mut out.history.faces {
            let hit = r.sources.iter().find_map(|x| match x {
                HistorySource::Input { from } if centre.contains(&from.entity) => Some(*from),
                _ => None,
            });
            if let Some(from) = hit {
                r.sources = vec![gen(Role::Blend, KernelTag::InputFace { face: from }, None)];
                r.evolution = Evolution::Generated;
                replaced.insert(r.face);
            }
        }
        for c in center {
            out.history.deleted_faces.push(*c);
        }
        out.history.deleted_faces.sort();
        out.history.deleted_faces.dedup();
        // The blend's geometry: a half cylinder between the sides.
        if let Some(body) = out.bodies.first_mut() {
            for f in &replaced {
                let g = &mut body.topology.faces[*f as usize].geom;
                let SurfaceKind::Plane { normal } = g.surface else {
                    continue;
                };
                for a in 0..3 {
                    axis_dir[a] = if a != pa.axis && normal[a].abs() < 0.5 {
                        1.0
                    } else {
                        0.0
                    };
                }
                if axis_dir.iter().filter(|x| **x > 0.0).count() == 1 {
                    g.surface = SurfaceKind::Cylinder {
                        axis_point: g.centroid,
                        axis_direction: axis_dir,
                        radius,
                    };
                } else {
                    g.surface = SurfaceKind::Other;
                }
            }
        }
        Ok(out)
    }

    /// Vertex chamfer on corners whose three edges run along the part axes.
    pub(super) fn vertex_chamfer(
        &self,
        vertices: &[OperandEntity],
        distances: [f64; 3],
        solids: &[&FakeSolid],
    ) -> Result<KernelOutput, KernelError> {
        let mut outs = Vec::new();
        for (k, s) in solids.iter().enumerate() {
            let mut tools = Vec::new();
            for v in vertices.iter().filter(|v| v.operand as usize == k) {
                let vt = s
                    .topology
                    .vertices
                    .get(v.entity as usize)
                    .ok_or(KernelError::Failed {
                        message: "vertex out of range".into(),
                    })?;
                if vt.edges.len() != 3 {
                    return Err(KernelError::Failed {
                        message: "a vertex chamfer needs a corner of three edges".into(),
                    });
                }
                let p = vt.point;
                let mut corner = [p, p];
                let mut pts = Vec::new();
                for (i, e) in vt.edges.iter().enumerate() {
                    let g = s.topology.edges[*e as usize].geom;
                    let other = if dist(g.start, p) < 1e-9 {
                        g.end
                    } else {
                        g.start
                    };
                    let d = sub3(other, p);
                    let Some((axis, sign)) = axis_of(normalize3(d)) else {
                        return Err(KernelError::NotSupported {
                            what: "fake kernel: vertex chamfers on corners off the part axes"
                                .into(),
                        });
                    };
                    let mut q = p;
                    q[axis] += sign * distances[i];
                    pts.push(q);
                    corner[0][axis] = corner[0][axis].min(q[axis]);
                    corner[1][axis] = corner[1][axis].max(q[axis]);
                }
                // The plane through the three points, its normal towards the vertex: the
                // material on the vertex side is cut.
                let n = cross3(sub3(pts[1], pts[0]), sub3(pts[2], pts[0]));
                let n = normalize3(n);
                let n = if dot3(sub3(p, pts[0]), n) < 0.0 {
                    [-n[0], -n[1], -n[2]]
                } else {
                    n
                };
                let label = gen(Role::Chamfer, KernelTag::InputVertex { vertex: *v }, None);
                let boxed = Prism {
                    k: 2,
                    u: 0,
                    v: 1,
                    loops: vec![vec![
                        [corner[0][0], corner[0][1]],
                        [corner[1][0], corner[0][1]],
                        [corner[1][0], corner[1][1]],
                        [corner[0][0], corner[1][1]],
                    ]],
                    edge_labels: vec![vec![label.clone(); 4]],
                    lo: corner[0][2],
                    hi: corner[1][2],
                    lo_label: label.clone(),
                    hi_label: label.clone(),
                };
                // The tool is the box clipped to (x − pts[0])·(−n) < 0: the vertex side.
                tools.push(Tool::Clipped(
                    Box::new(Tool::Exact(boxed)),
                    vec![(pts[0], [-n[0], -n[1], -n[2]], label)],
                ));
            }
            let res = run_grid(GridJob {
                operands: vec![s],
                operand_ids: vec![k as u32],
                tools,
                mode: Mode::Cut,
                split: None,
                separate_tools: false,
                body_label: None,
                modifier: None,
                merge_collinear: false,
                carve: Vec::new(),
            })?;
            outs.push(self.finish(&[s], &[k as u32], res, Vec::new(), Vec::new())?);
        }
        Ok(concat(outs))
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub3(a, b);
    dot3(d, d).sqrt()
}
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let l = dot3(a, a).sqrt().max(1e-300);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// A hole's section about its axis, from the kernel contract's one definition
/// ([`runmat_geometry_kernel::sections::hole_section`]), as grid curves.
#[allow(clippy::too_many_arguments)]
fn hole_section(
    key: &str,
    radius: f64,
    bottom: f64,
    counterbore: Option<(f64, f64)>,
    countersink: Option<(f64, f64)>,
    drill_point: Option<f64>,
    taper: Option<f64>,
    thread: Option<Option<f64>>,
    stages: &[(f64, f64, bool, Option<u32>)],
) -> Vec<(Curve2, HistorySource)> {
    let stages: Vec<runmat_geometry_kernel::sections::StageSpan> = stages
        .iter()
        .map(
            |(exit, radius, threaded, _)| runmat_geometry_kernel::sections::StageSpan {
                exit: *exit,
                radius: *radius,
                threaded: *threaded,
            },
        )
        .collect();
    runmat_geometry_kernel::sections::hole_section(
        key,
        radius,
        bottom,
        counterbore,
        countersink,
        drill_point,
        taper,
        thread,
        &stages,
    )
    .into_iter()
    .map(|s| (Curve2::Line { a: s.a, b: s.b }, s.label))
    .collect()
}
