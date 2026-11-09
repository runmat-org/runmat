//! Fake-kernel coverage for labelled tools (sheet metal, weldments, ribs, embosses, form
//! tools), direct edits (move, offset, rotate, delete and replace faces), scaling and surface
//! bodies.

mod direct;
mod surface;
mod analyzer;

use super::sheet::{self, Facet, FakeSheet, PolyTool, SheetBuild};
use super::*;

fn grid_job<'a>(
    operands: Vec<&'a FakeSolid>,
    operand_ids: Vec<u32>,
    tools: Vec<Tool>,
    mode: Mode,
) -> GridJob<'a> {
    GridJob {
        operands,
        operand_ids,
        tools,
        mode,
        split: None,
        separate_tools: false,
        body_label: None,
        modifier: None,
        merge_collinear: false,
        carve: Vec::new(),
    }
}

/// A solid operand passed through an operation unchanged.
fn kept_solid(operands: &[&BodyHandle], i: usize) -> Result<sheet::ExtraBody, KernelError> {
    let so = operands[i]
        .downcast::<FakeSolid>()
        .ok_or(KernelError::ForeignOperand { operand: i as u32 })?;
    let input = |entity: u32| {
        vec![HistorySource::Input {
            from: OperandEntity {
                operand: i as u32,
                entity,
            },
        }]
    };
    Ok((
        KernelBody {
            handle: operands[i].clone(),
            topology: so.topology.clone(),
        },
        (0..so.faces.len() as u32).map(input).collect(),
        input(0),
    ))
}

impl FakeKernel {
    /// Operations handled here; `None` for the rest.
    pub(super) fn execute_extra(
        &self,
        op: &KernelOp,
        operands: &[&BodyHandle],
    ) -> Option<Result<KernelOutput, KernelError>> {
        Some(match op {
            KernelOp::Labeled {
                tools,
                carve,
                combine,
                separate,
            } => self.labeled(tools, carve, *combine, *separate, operands),
            KernelOp::MoveFaces { faces, motion } => self.move_faces(faces, motion, operands),
            KernelOp::DeleteFaces { faces, heal } => self.delete_faces(faces, *heal, operands),
            KernelOp::ReplaceFaces { faces, with } => self.replace_faces(faces, *with, operands),
            KernelOp::Scale { center, factors } => self.scale(*center, *factors, operands),
            KernelOp::Surface(s) => self.surface(s, operands),
            _ => return None,
        })
    }

    fn solids<'a>(operands: &[&'a BodyHandle]) -> Result<Vec<&'a FakeSolid>, KernelError> {
        (0..operands.len())
            .map(|i| Self::solid(operands, i))
            .collect()
    }

    fn labeled(
        &self,
        tools: &[LabeledTool],
        carve: &[LabeledTool],
        combine: Combine,
        separate: bool,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        let solids = Self::solids(operands)?;
        let ids: Vec<u32> = (0..solids.len() as u32).collect();
        let build = |lt: &LabeledTool| -> Result<Vec<Tool>, KernelError> {
            let base = self.tools(&ToolOp::new(lt.shape.clone(), combine), &solids, 0)?;
            let remap = |l: &HistorySource| -> Option<HistorySource> {
                let HistorySource::Generated { role, tag, .. } = l else {
                    return None;
                };
                let found = match (role, tag) {
                    (Role::Side, KernelTag::Curve { curve, .. }) => lt
                        .sides
                        .iter()
                        .find(|(c, _)| c == curve)
                        .map(|(_, f)| f.clone()),
                    (Role::StartCap, _) => lt.start.clone(),
                    (Role::EndCap, _) => lt.end.clone(),
                    _ => None,
                }?;
                Some(gen(found.role, found.tag, found.ordinal))
            };
            Ok(base
                .into_iter()
                .map(|mut t| {
                    t.map_labels(&remap);
                    if lt.clip.is_empty() {
                        t
                    } else {
                        Tool::Clipped(
                            Box::new(t),
                            lt.clip
                                .iter()
                                .map(|c| {
                                    (
                                        c.origin,
                                        c.normal,
                                        gen(c.label.role, c.label.tag.clone(), c.label.ordinal),
                                    )
                                })
                                .collect(),
                        )
                    }
                })
                .collect())
        };
        let mut built = Vec::new();
        let mut body_labels = Vec::new();
        for lt in tools {
            for t in build(lt)? {
                built.push(t);
                body_labels.push(
                    lt.body
                        .as_ref()
                        .map(|b| gen(b.role, b.tag.clone(), b.ordinal))
                        .unwrap_or_else(|| gen(Role::Body, KernelTag::None, None)),
                );
            }
        }
        let mut carved = Vec::new();
        for lt in carve {
            carved.extend(build(lt)?);
        }
        let mode = Self::mode(combine);
        let (ops, op_ids) = if mode == Mode::New {
            (Vec::new(), Vec::new())
        } else {
            (solids.clone(), ids.clone())
        };
        let mut job = grid_job(ops.clone(), op_ids.clone(), built, mode);
        job.separate_tools = separate;
        job.body_label = Some(body_labels);
        job.carve = carved;
        job.merge_collinear = tools.iter().all(|t| t.shape.merge_collinear());
        let res = run_grid(job)?;
        self.finish(&ops, &op_ids, res, Vec::new(), Vec::new())
    }
}

fn sheet_rect(pk: PlaneKey, r: Rect) -> Facet {
    let a = pk.axis;
    let (u, v) = ((a + 1) % 3, (a + 2) % 3);
    let mk = |x: f64, y: f64| {
        let mut p = [0.0; 3];
        p[a] = pk.offset;
        p[u] = x;
        p[v] = y;
        p
    };
    let mut pts = vec![
        mk(r.lo[u], r.lo[v]),
        mk(r.hi[u], r.lo[v]),
        mk(r.hi[u], r.hi[v]),
        mk(r.lo[u], r.hi[v]),
    ];
    if pk.dir < 0 {
        pts.reverse();
    }
    let mut n = [0.0; 3];
    n[a] = pk.dir as f64;
    Facet {
        loops: vec![pts],
        normal: n,
        labels: vec![vec![gen(Role::Boundary, KernelTag::None, None); 4]],
    }
}

/// Offset-surface boundaries: each continues the input boundary it offsets.
fn relabel_boundaries(f: &Facet, _role: Role) -> Facet {
    f.clone()
}

/// Input faces of sheet operands with no image are deleted.
fn fill_deleted(out: &mut KernelOutput, operands: &[&BodyHandle]) {
    let used: BTreeSet<(u32, u32)> = out
        .history
        .faces
        .iter()
        .flat_map(|r| r.sources.iter())
        .filter_map(|s| match s {
            HistorySource::Input { from } => Some((from.operand, from.entity)),
            _ => None,
        })
        .collect();
    for (i, h) in operands.iter().enumerate() {
        if let Some(s) = h.downcast::<FakeSheet>() {
            for f in 0..s.topology.faces.len() as u32 {
                if !used.contains(&(i as u32, f)) {
                    out.history.deleted_faces.push(OperandEntity {
                        operand: i as u32,
                        entity: f,
                    });
                }
            }
        }
    }
}

fn face_bounds(f: &FFace) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for r in &f.rects {
        for a in 0..3 {
            lo[a] = lo[a].min(r.lo[a]);
            hi[a] = hi[a].max(r.hi[a]);
        }
    }
    (lo, hi)
}

/// Scale a solid's grid and faces about `c` by per-axis factors (positive).
fn scale_solid(s: &mut FakeSolid, c: [f64; 3], k: [f64; 3]) {
    let m = |a: usize, x: f64| c[a] + k[a] * (x - c[a]);
    for a in 0..3 {
        for g in &mut s.grid[a] {
            *g = m(a, *g);
        }
    }
    let uniform = (k[0] - k[1]).abs() < 1e-12 && (k[1] - k[2]).abs() < 1e-12;
    for f in &mut s.faces {
        if let Some(p) = &mut f.plane {
            p.offset = m(p.axis, p.offset);
        }
        for p in &mut f.facet_planes {
            p.offset = m(p.axis, p.offset);
        }
        if let Some(h) = &mut f.surface {
            *h = match *h {
                SurfaceKind::Cylinder {
                    axis_point,
                    axis_direction,
                    radius,
                } if uniform => SurfaceKind::Cylinder {
                    axis_point: [
                        m(0, axis_point[0]),
                        m(1, axis_point[1]),
                        m(2, axis_point[2]),
                    ],
                    axis_direction,
                    radius: radius * k[0],
                },
                SurfaceKind::Plane { normal } => {
                    let n = [normal[0] / k[0], normal[1] / k[1], normal[2] / k[2]];
                    SurfaceKind::Plane {
                        normal: crate::model::datum::normalize(n),
                    }
                }
                _ => SurfaceKind::Other,
            };
        }
        // A scaled chart is not rigid; the face draws on its facets.
        if (0..3).any(|a| (k[a] - 1.0).abs() > 1e-12) {
            f.exact = None;
        }
        for r in &mut f.rects {
            for a in 0..3 {
                r.lo[a] = m(a, r.lo[a]);
                r.hi[a] = m(a, r.hi[a]);
            }
        }
    }
}
