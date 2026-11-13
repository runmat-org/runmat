//! Splits by a curved face or a body, and patterns of fillets and chamfers: the seed's edges
//! moved to each instance and found on the body by their geometry.

use super::*;
use crate::refs::history::EdgeGeom;

/// One micrometre: edges this close end to end are the same edge.
const SAME: f64 = 1e-6;

fn near(a: [f64; 3], b: [f64; 3]) -> bool {
    let d = sub(a, b);
    dot(d, d).sqrt() < SAME
}

fn moved(g: &EdgeGeom, t: &Transform) -> EdgeGeom {
    let curve = match g.curve {
        EdgeCurve::Circle {
            center,
            normal,
            radius,
        } => EdgeCurve::Circle {
            center: t.apply(center),
            normal: t.apply_vector(normal),
            radius,
        },
        other => other,
    };
    EdgeGeom {
        start: t.apply(g.start),
        end: t.apply(g.end),
        length: g.length,
        curve,
    }
}

fn same_edge(a: &EdgeGeom, b: &EdgeGeom) -> bool {
    let ends = (near(a.start, b.start) && near(a.end, b.end))
        || (near(a.start, b.end) && near(a.end, b.start));
    let curve = match (a.curve, b.curve) {
        (EdgeCurve::Line, EdgeCurve::Line) | (EdgeCurve::Other, EdgeCurve::Other) => true,
        (
            EdgeCurve::Circle {
                center: c1,
                radius: r1,
                ..
            },
            EdgeCurve::Circle {
                center: c2,
                radius: r2,
                ..
            },
        ) => near(c1, c2) && (r1 - r2).abs() < SAME,
        _ => false,
    };
    ends && curve && (a.length - b.length).abs() < SAME
}

fn with_edges(op: &KernelOp, edges: Vec<OperandEntity>) -> KernelOp {
    let mut op = op.clone();
    match &mut op {
        KernelOp::Fillet { edges: e, .. } | KernelOp::Chamfer { edges: e, .. } => *e = edges,
        _ => {}
    }
    op
}

impl<'a, 'b> Run<'a, 'b> {
    /// A fillet's or chamfer's operation with its edges' geometry, before it runs.
    pub(super) fn dressup_of(
        &self,
        op: &KernelOp,
        operands: &[usize],
    ) -> Option<(KernelOp, Vec<EdgeGeom>)> {
        let edges = match op {
            KernelOp::Fillet { edges, .. } | KernelOp::Chamfer { edges, .. } => edges,
            _ => return None,
        };
        let geoms = edges
            .iter()
            .map(|e| {
                let b = &self.st.bodies[*operands.get(e.operand as usize)?];
                b.topology.edges.get(e.entity as usize).map(|t| t.geom)
            })
            .collect::<Option<Vec<_>>>()?;
        Some((with_edges(op, Vec::new()), geoms))
    }

    /// A split by a curved face or by a body; `None` when the tool is a plane.
    pub(super) fn split_with(
        &mut self,
        i: usize,
        id: &FeatureId,
        body: usize,
        tool: &PlaneRef,
    ) -> Result<Option<Compiled>, Halt> {
        let PlaneRef::Face { face } = tool else {
            return Ok(None);
        };
        let r = self.single(i, id, "tool", face, None)?;
        let Some(h) = r.entities().iter().next().copied() else {
            return Ok(None);
        };
        let other = h.body as usize;
        let (operands, faces) = match h.kind {
            TopoKind::Face if self.face_frame(&h).is_some() => return Ok(None),
            TopoKind::Face if other == body => (
                vec![body],
                vec![OperandEntity {
                    operand: 0,
                    entity: h.index,
                }],
            ),
            TopoKind::Face => (
                vec![body, other],
                vec![OperandEntity {
                    operand: 1,
                    entity: h.index,
                }],
            ),
            TopoKind::Body if other != body => (vec![body, other], Vec::new()),
            TopoKind::Body => {
                return Err(self.error(
                    id,
                    Some("tool"),
                    DiagCode::RegenInvalid,
                    "a body can't split itself; split it with a plane, a face or another body"
                        .into(),
                    Vec::new(),
                ))
            }
            _ => {
                return Err(self.error(
                    id,
                    Some("tool"),
                    DiagCode::RefKindMismatch,
                    format!("`{}` is not a plane, a face or a body", face),
                    Vec::new(),
                ))
            }
        };
        Ok(Some(Compiled::Op {
            op: KernelOp::SplitWith { faces },
            operands,
            naming: NamingContext::simple(id, Vec::new()),
            import: false,
            instances: None,
            tools: None,
        }))
    }

    /// A pattern or mirror of fillets and chamfers: each seed's edges, moved to each instance,
    /// blended with the seed's size in one operation.
    pub(super) fn dressup_pattern(
        &mut self,
        i: usize,
        id: &FeatureId,
        features: &[FeatureId],
        instances: &[Instance],
    ) -> Result<Compiled, Halt> {
        let mut template: Option<KernelOp> = None;
        let mut seeds = Vec::new();
        for (k, s) in features.iter().enumerate() {
            self.gate_feature(i, id, s, &format!("seeds[{}]", k))?;
            let (op, geoms) = self
                .st
                .dressups
                .get(s)
                .cloned()
                .expect("checked by the caller");
            match &template {
                Some(t) if *t != op => {
                    return Err(self.error(
                        id,
                        Some(&format!("seeds[{}]", k)),
                        DiagCode::RegenInvalid,
                        "fillets and chamfers patterned together need the same size; pattern \
                         each in its own pattern"
                            .into(),
                        Vec::new(),
                    ))
                }
                Some(_) => {}
                None => template = Some(op),
            }
            seeds.extend(geoms);
        }
        let mut found: BTreeSet<(usize, u32)> = BTreeSet::new();
        for inst in instances {
            for g in &seeds {
                let want = moved(g, &inst.transform);
                let hit = self.st.bodies.iter().enumerate().find_map(|(b, body)| {
                    body.topology
                        .edges
                        .iter()
                        .position(|e| same_edge(&e.geom, &want))
                        .map(|e| (b, e as u32))
                });
                match hit {
                    Some(x) => {
                        found.insert(x);
                    }
                    None => {
                        return Err(self.error(
                            id,
                            Some("layout"),
                            DiagCode::RegenInvalid,
                            format!(
                                "instance {} has no edge where the seed's blended edge would \
                                 be; pattern the feature that makes those edges first",
                                inst.key
                            ),
                            Vec::new(),
                        ))
                    }
                }
            }
        }
        let mut operands: Vec<usize> = found.iter().map(|x| x.0).collect();
        operands.dedup();
        let edges = found
            .iter()
            .map(|(b, e)| OperandEntity {
                operand: operands.iter().position(|o| o == b).expect("listed") as u32,
                entity: *e,
            })
            .collect();
        Ok(Compiled::Op {
            op: with_edges(&template.expect("one seed or more"), edges),
            operands,
            naming: NamingContext::simple(id, Vec::new()),
            import: false,
            instances: None,
            tools: None,
        })
    }
}
