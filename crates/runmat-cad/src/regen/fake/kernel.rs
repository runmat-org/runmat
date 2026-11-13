//! The fake kernel's own steps: making solids from grid results, imports and the shared
//! operation plumbing.

use super::*;

// ------------------------------------------------------------------------------- the kernel

impl FakeKernel {
    pub fn new() -> Self {
        Self::default()
    }

    pub(super) fn id_static() -> KernelId {
        KernelId {
            name: "fake-grid".into(),
            version: "1".into(),
        }
    }

    pub(super) fn solid<'a>(
        operands: &[&'a BodyHandle],
        i: usize,
    ) -> Result<&'a FakeSolid, KernelError> {
        operands[i]
            .downcast::<FakeSolid>()
            .ok_or(KernelError::ForeignOperand { operand: i as u32 })
    }

    pub(super) fn bounds(solids: &[&FakeSolid]) -> (f64, f64, [f64; 3], [f64; 3]) {
        let mut lo = [-1.0f64; 3];
        let mut hi = [1.0f64; 3];
        let mut first = true;
        for s in solids {
            if let Some((l, h)) = s.bounds() {
                for a in 0..3 {
                    if first {
                        lo[a] = l[a];
                        hi[a] = h[a];
                    } else {
                        lo[a] = lo[a].min(l[a]);
                        hi[a] = hi[a].max(h[a]);
                    }
                }
                first = false;
            }
        }
        let m = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max).max(1e-3);
        (
            lo.iter().cloned().fold(f64::INFINITY, f64::min) - m,
            hi.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + m,
            lo,
            hi,
        )
    }

    /// Finish a grid result: carry operand decorations, add fresh ones, build topology, and
    /// write history.
    pub(super) fn finish(
        &self,
        operands: &[&FakeSolid],
        operand_ids: &[u32],
        mut res: GridResult,
        fresh: Vec<Decoration>,
        extra: Vec<(FakeSolid, Vec<Vec<HistorySource>>, Vec<HistorySource>)>,
    ) -> Result<KernelOutput, KernelError> {
        // Carry decorations: every operand decoration goes to each output (only bodies with a
        // matching edge keep it).
        let mut carried: Vec<(Decoration, u32)> = Vec::new();
        for (k, o) in operands.iter().enumerate() {
            for dec in &o.decorations {
                carried.push((dec.clone(), operand_ids[k]));
            }
        }
        for (s, srcs, b) in extra {
            res.solids.push(s);
            res.face_sources.push(srcs);
            res.body_sources.push(b);
        }
        let mut bodies = Vec::new();
        let mut history = OpHistory::default();
        for (bi, solid) in res.solids.iter_mut().enumerate() {
            let grid_faces = solid.faces.len();
            {
                let mut decs: Vec<Decoration> = carried.iter().map(|(d, _)| d.clone()).collect();
                decs.extend(fresh.iter().cloned());
                let keep_old = solid.decorations.clone();
                solid.faces.truncate(grid_faces);
                solid.decorations = if decs.is_empty() { keep_old } else { decs };
                build_topology(solid);
            }
            for f in 0..solid.faces.len() {
                let sources = if f < grid_faces {
                    res.face_sources[bi][f].clone()
                } else {
                    // A blend face: find its decoration and source.
                    let dec = solid
                        .decorations
                        .iter()
                        .find(|d| d.pieces.iter().any(|p| p.2 == f as u32))
                        .expect("blend face belongs to a decoration");
                    let piece = dec.pieces.iter().find(|p| p.2 == f as u32).expect("piece");
                    match &dec.fresh {
                        Some(label) => vec![label.clone()],
                        None => {
                            // Previous faces of this decoration in its operand, by overlap.
                            let mut v = Vec::new();
                            for (k, o) in operands.iter().enumerate() {
                                for od in &o.decorations {
                                    if od.along == dec.along
                                        && od.planes[0].same(&dec.planes[0])
                                        && od.planes[1].same(&dec.planes[1])
                                        && (od.lo - dec.lo).abs() < EPS
                                    {
                                        for (lo, hi, fid) in &od.pieces {
                                            if piece.0 < hi - EPS && piece.1 > lo + EPS {
                                                v.push(HistorySource::Input {
                                                    from: OperandEntity {
                                                        operand: operand_ids[k],
                                                        entity: *fid,
                                                    },
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                            if v.is_empty() {
                                return Err(KernelError::HistoryIncomplete {
                                    message: "fake kernel: blend face lost its source".into(),
                                });
                            }
                            v.sort();
                            v.dedup();
                            v
                        }
                    }
                };
                history.faces.push(FaceRecord {
                    body: bi as u32,
                    face: f as u32,
                    evolution: Evolution::Modified,
                    sources,
                });
            }
            for d in &mut solid.decorations {
                d.fresh = None;
            }
            history.bodies.push(BodyRecord {
                body: bi as u32,
                sources: res.body_sources[bi].clone(),
            });
        }
        // Evolution and deletions.
        let mut uses: BTreeMap<OperandEntity, u32> = BTreeMap::new();
        for r in &history.faces {
            for s in &r.sources {
                if let HistorySource::Input { from } | HistorySource::Copy { from, .. } = s {
                    *uses.entry(*from).or_default() += 1;
                }
            }
        }
        for r in &mut history.faces {
            let inputs: Vec<&OperandEntity> = r
                .sources
                .iter()
                .filter_map(|s| match s {
                    HistorySource::Input { from } => Some(from),
                    _ => None,
                })
                .collect();
            r.evolution = if inputs.is_empty() {
                Evolution::Generated
            } else if inputs.len() > 1 || r.sources.len() > inputs.len() {
                Evolution::Merged
            } else if uses.get(inputs[0]).copied().unwrap_or(0) > 1 {
                Evolution::Split
            } else {
                Evolution::Modified
            };
        }
        for (k, o) in operands.iter().enumerate() {
            for f in 0..o.faces.len() {
                let oe = OperandEntity {
                    operand: operand_ids[k],
                    entity: f as u32,
                };
                if !uses.contains_key(&oe) {
                    history.deleted_faces.push(oe);
                }
            }
            let used = history.bodies.iter().any(|b| {
                b.sources.iter().any(|s| matches!(s, HistorySource::Input { from } | HistorySource::Copy { from, .. } if from.operand == operand_ids[k]))
            });
            if !used {
                history.deleted_bodies.push(operand_ids[k]);
            }
        }
        history.deleted_faces.sort();
        history.deleted_bodies.sort();
        for s in res.solids {
            let topology = s.topology.clone();
            bodies.push(KernelBody {
                handle: BodyHandle::new(Self::id_static(), s),
                topology,
            });
        }
        Ok(KernelOutput { bodies, history })
    }

    pub(super) fn tools(
        &self,
        op: &ToolOp,
        solids: &[&FakeSolid],
        profile_index: u32,
    ) -> Result<Vec<Tool>, KernelError> {
        let (blo, bhi, lo, hi) = Self::bounds(solids);
        let corners: Vec<[f64; 3]> = (0..8)
            .map(|i| {
                [
                    if i & 1 == 0 { lo[0] } else { hi[0] },
                    if i & 2 == 0 { lo[1] } else { hi[1] },
                    if i & 4 == 0 { lo[2] } else { hi[2] },
                ]
            })
            .collect();
        // How far along `dir` from `origin` the operands reach, either way, with margin.
        let reach = |origin: [f64; 3], dir: [f64; 3]| -> (f64, f64) {
            let ds: Vec<f64> = corners
                .iter()
                .map(|c| (0..3).map(|a| (c[a] - origin[a]) * dir[a]).sum::<f64>())
                .collect();
            let m = (bhi - blo).abs().max(1.0) * 0.05;
            (
                ds.iter().cloned().fold(f64::INFINITY, f64::min) - m,
                ds.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + m,
            )
        };
        let side = |curve: &crate::ids::SketchEntityId| {
            gen(
                Role::Side,
                KernelTag::Curve {
                    profile: profile_index,
                    curve: curve.clone(),
                },
                None,
            )
        };
        let curves = |p: &CanonicalProfile| -> Vec<Vec<(sampled::Curve2, HistorySource)>> {
            p.loops
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|seg| (sampled::Curve2::from_segment(seg), side(&seg.curve)))
                        .collect()
                })
                .collect()
        };
        let caps = || {
            [
                gen(Role::StartCap, KernelTag::None, None),
                gen(Role::EndCap, KernelTag::None, None),
            ]
        };
        match &op.shape {
            ToolShape::Prism {
                profile,
                start,
                end,
                start_bound,
                end_bound,
                draft,
                thin,
            } => {
                let through_all = matches!(end_bound, PrismBound::ThroughAll);
                let plain = start_bound.is_blind()
                    && (end_bound.is_blind() || through_all)
                    && draft.is_none()
                    && thin.is_none();
                if !plain {
                    return self.extrusion(op, solids, profile_index, &reach);
                }
                let exact = prism_from_profile(
                    profile,
                    *start,
                    *end,
                    through_all.then_some((blo, bhi)),
                    profile_index,
                );
                match exact {
                    Ok(p) => Ok(vec![Tool::Exact(p)]),
                    Err(KernelError::NotSupported { .. }) => {
                        let mut e = *end;
                        if through_all {
                            let (a, b) = reach(profile.frame.origin, profile.frame.z);
                            e = if end - start >= 0.0 { b } else { a };
                        }
                        Ok(vec![Tool::Sampled(Box::new(sampled::prism(
                            Transform::from_frame(&profile.frame),
                            &curves(profile),
                            *start,
                            e,
                            caps(),
                        )))])
                    }
                    Err(other) => Err(other),
                }
            }
            ToolShape::Revolve {
                profile,
                axis_point,
                axis_direction,
                angle,
            } => {
                let to_world = Transform::from_frame(&profile.frame);
                let inv = to_world.inverse();
                let ap = inv.apply(*axis_point);
                let ad = inv.apply_vector(*axis_direction);
                if ap[2].abs() > 1e-7 || ad[2].abs() > 1e-7 {
                    return Err(KernelError::Failed {
                        message: "the revolve axis must lie in the profile plane".into(),
                    });
                }
                Ok(vec![Tool::Sampled(Box::new(sampled::revolve(
                    to_world,
                    &curves(profile),
                    [ap[0], ap[1]],
                    [ad[0], ad[1]],
                    *angle,
                    Some(caps()),
                )?))])
            }
            ToolShape::Sweep {
                profile,
                path,
                path_curves,
                follow_path,
            } => {
                // A profile that follows a straight path is the translated profile; the grid
                // kernel builds no profile turning through a bend.
                if *follow_path && super::sheet::turns(path) {
                    return Err(KernelError::NotSupported {
                        what: "a sweep that follows a turning path".into(),
                    });
                }
                let to_world = Transform::from_frame(&profile.frame);
                let inv = to_world.inverse();
                let local: Vec<[f64; 3]> = path.iter().map(|p| inv.apply(*p)).collect();
                Ok(vec![Tool::Sampled(Box::new(sampled::sweep(
                    to_world,
                    &curves(profile),
                    &local,
                    path_curves,
                    caps(),
                )?))])
            }
            ToolShape::Loft { profiles } => {
                let first = profiles.first().ok_or_else(|| KernelError::Failed {
                    message: "a loft needs profiles".into(),
                })?;
                let to_world = Transform::from_frame(&first.frame);
                let inv = to_world.inverse();
                let tilted = |p: &CanonicalProfile| {
                    let zl = inv.apply_vector(p.frame.z);
                    zl[0].abs() > 1e-7 || zl[1].abs() > 1e-7
                };
                if profiles.iter().any(tilted) {
                    // Sections in any orientation: a closed polyhedron through them.
                    return Ok(vec![Tool::Polys(Box::new(sheet::loft_tool(
                        profiles,
                        &side,
                        caps(),
                    )?))]);
                }
                let mut sections = Vec::new();
                for p in profiles {
                    let z = inv.apply(p.frame.origin)[2];
                    let map = |q: [f64; 2]| {
                        let l = inv.apply(p.frame.to_world(q));
                        [l[0], l[1]]
                    };
                    let loops: Vec<Vec<sampled::Curve2>> = p
                        .loops
                        .iter()
                        .map(|l| {
                            l.iter()
                                .map(|seg| {
                                    let mut m = seg.clone();
                                    m.start = map(seg.start);
                                    m.end = map(seg.end);
                                    m.kind = seg.kind.mapped(&map);
                                    sampled::Curve2::from_segment(&m)
                                })
                                .collect()
                        })
                        .collect();
                    sections.push((z, loops));
                }
                // Sides are tagged by the first section's curves.
                let labels: Vec<Vec<HistorySource>> = first
                    .loops
                    .iter()
                    .map(|l| l.iter().map(|seg| side(&seg.curve)).collect())
                    .collect();
                Ok(vec![Tool::Sampled(Box::new(sampled::loft(
                    to_world,
                    sections,
                    &labels,
                    caps(),
                )?))])
            }
            ToolShape::Holes { .. } => self.holes(&op.shape, solids, &reach),
        }
    }

    pub(super) fn mode(c: Combine) -> Mode {
        match c {
            Combine::New => Mode::New,
            Combine::Add => Mode::Add,
            Combine::Cut => Mode::Cut,
            Combine::Intersect => Mode::Intersect,
        }
    }
}
