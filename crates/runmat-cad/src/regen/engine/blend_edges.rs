//! The edges a fillet or chamfer rounds. A picked edge is itself; a picked face stands for its
//! boundary edges; a picked feature stands for the edges of the faces it made (every edge of its
//! body when it made the whole body).

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn blend_edges(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        refs: &[Reference],
    ) -> Result<BTreeSet<EntityHandle>, Halt> {
        let set = self.list(i, f, param, refs, None)?;
        // The features a body pick names (`feature.<id>.body`).
        let features: Vec<FeatureId> = refs
            .iter()
            .filter_map(|r| match r {
                Reference::Name {
                    kind: TopoKind::Body,
                    name,
                } => Some(name.maker().clone()),
                _ => None,
            })
            .collect();
        let mut out = BTreeSet::new();
        for h in &set {
            let b = &self.st.bodies[h.body as usize];
            let edge = |index: usize| EntityHandle {
                body: h.body,
                kind: TopoKind::Edge,
                index: index as u32,
            };
            match h.kind {
                TopoKind::Edge => {
                    out.insert(*h);
                }
                TopoKind::Face => out.extend(
                    b.topology
                        .edges
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| !e.laminar && e.faces.contains(&h.index))
                        .map(|(k, _)| edge(k)),
                ),
                TopoKind::Body => {
                    let made: BTreeSet<u32> = (0..b.topology.faces.len() as u32)
                        .filter(|k| {
                            b.names.faces[*k as usize]
                                .iter()
                                .any(|n| features.contains(n.maker()))
                        })
                        .collect();
                    out.extend(
                        b.topology
                            .edges
                            .iter()
                            .enumerate()
                            .filter(|(_, e)| {
                                !e.laminar
                                    && (made.is_empty() || e.faces.iter().any(|x| made.contains(x)))
                            })
                            .map(|(k, _)| edge(k)),
                    );
                }
                TopoKind::Vertex => {
                    return Err(self.error(
                        f,
                        Some(param),
                        DiagCode::RefKindMismatch,
                        "a vertex has no edges to round here; pick edges, faces or a feature"
                            .into(),
                        Vec::new(),
                    ))
                }
            }
        }
        if out.is_empty() {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefEmpty,
                "the picks have no edges to round".into(),
                Vec::new(),
            ));
        }
        Ok(out)
    }
}
