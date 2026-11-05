//! The occurrence tree during resolution: conditions, value environments, definitions and
//! their configured summaries, prototypes, and paths and transforms between nodes.

use super::*;

impl<'a, 'b> Resolver<'a, 'b> {
    /// Evaluate a `when` at the configuration point. Problems are reported once per text.
    pub(in crate::assembly) fn holds(
        &mut self,
        c: Option<&Condition>,
        file: &FilePath,
        site: &Site,
    ) -> bool {
        let Some(c) = c else { return true };
        let Some(point) = &self.point else {
            return false;
        };
        match self.space.eval(c, point) {
            Ok(b) => b,
            Err(p) => {
                if self.reported.insert((file.0.clone(), c.to_string())) {
                    self.diag(
                        option_code(&p),
                        Severity::Error,
                        site.clone(),
                        format!("`when {}`: {}", c, p),
                        Vec::new(),
                    );
                }
                false
            }
        }
    }

    pub(in crate::assembly) fn ev(&self) -> Evaluator<'a> {
        Evaluator {
            units: self.ctx.units,
            dynamic: self.ctx.dynamic,
        }
    }

    pub(in crate::assembly) fn env_for(
        &mut self,
        doc: &AssemblyDocument,
        overrides: &BTreeMap<InputName, ValueExpr>,
    ) -> ValueEnv {
        let env = self.ev().evaluate_env(&doc.inputs, &doc.lets, overrides);
        for (name, e) in &env.failed {
            self.diag(
                DiagCode::ValueFailed,
                Severity::Error,
                Site::with_param(Subject::Value { name: name.clone() }, doc.path.as_str()),
                format!("`{}`: {}", doc.path, e),
                Vec::new(),
            );
        }
        env
    }

    pub(super) fn local_uuid(
        &mut self,
        file: &FilePath,
        kind: &str,
        name: &str,
    ) -> (ItemUuid, bool) {
        let p = item_path(file, kind, name);
        // Only a current entry counts: a path that is now an alias names the renamed item,
        // and a new item authored under the old name is a different item.
        if let Some(l) = self
            .ctx
            .lock
            .and_then(|l| l.lookup(&p))
            .filter(|l| !l.via_alias)
        {
            return (l.uuid, true);
        }
        *self.unlocked_files.entry(file.clone()).or_insert(0) += 1;
        self.stats.unlocked += 1;
        (
            hash16(&[
                b"runmat-cad/occurrence/v1/unlocked",
                p.to_string().as_bytes(),
            ]),
            false,
        )
    }

    pub(in crate::assembly) fn definition(
        &mut self,
        d: &DefinitionRef,
        rev: Option<&RevisionId>,
    ) -> Result<Arc<DefinitionSummary>, SourceError> {
        let key = (d.clone(), rev.cloned());
        if let Some(v) = self.defs.get(&key) {
            return v.clone();
        }
        let v = self.ctx.source.definition(d, rev);
        self.defs.insert(key, v.clone());
        v
    }

    pub(in crate::assembly) fn configured_summary(
        &mut self,
        d: &DefinitionRef,
        rev: Option<&RevisionId>,
        c: &ConfigKey,
    ) -> Result<Arc<ConfiguredSummary>, SourceError> {
        let key = (d.clone(), rev.cloned(), c.digest());
        if let Some(v) = self.configured.get(&key) {
            return v.clone();
        }
        let v = self.ctx.source.configured(d, rev, c);
        self.configured.insert(key, v.clone());
        v
    }

    pub(super) fn prototype(
        &mut self,
        d: &DefinitionRef,
        rev: Option<&RevisionId>,
        c: &ConfigKey,
        rep: Representation,
    ) -> u32 {
        let key = format!(
            "{}|{}|{}|{:?}",
            d,
            rev.map(|r| r.0.as_str()).unwrap_or(""),
            c.digest(),
            rep
        );
        if let Some(&i) = self.proto_index.get(&key) {
            self.prototypes[i as usize].instances += 1;
            return i;
        }
        let i = self.prototypes.len() as u32;
        self.prototypes.push(Prototype {
            definition: d.clone(),
            revision: rev.cloned(),
            configuration: c.clone(),
            representation: rep,
            instances: 1,
        });
        self.proto_index.insert(key, i);
        i
    }

    pub(in crate::assembly) fn display_path(&self, id: NodeId) -> String {
        let mut segs = Vec::new();
        let mut cur = Some(id);
        while let Some(c) = cur {
            let n = &self.nodes[c as usize];
            if n.parent.is_some() {
                segs.push(n.segment.to_string());
            }
            cur = n.parent;
        }
        segs.reverse();
        segs.join("/")
    }

    pub(in crate::assembly) fn occ_site(&self, id: NodeId) -> Site {
        Site::of(Subject::Occurrence {
            path: self.display_path(id),
        })
    }

    pub(in crate::assembly) fn child(&self, parent: NodeId, seg: &PathSegment) -> Option<NodeId> {
        self.nodes[parent as usize]
            .children
            .iter()
            .copied()
            .find(|c| &self.nodes[*c as usize].segment == seg)
    }

    /// Walk a relative path; `Err(i)` names the first segment that is missing.
    pub(in crate::assembly) fn walk(&self, from: NodeId, path: &RelPath) -> Result<NodeId, usize> {
        let mut cur = from;
        for (i, s) in path.0.iter().enumerate() {
            cur = self.child(cur, s).ok_or(i)?;
        }
        Ok(cur)
    }

    /// `from`-coordinates of the descendant `to`'s frame.
    pub(in crate::assembly) fn chain(&self, from: NodeId, to: NodeId) -> Transform {
        let mut stack = Vec::new();
        let mut cur = to;
        while cur != from {
            stack.push(cur);
            match self.nodes[cur as usize].parent {
                Some(p) => cur = p,
                None => break,
            }
        }
        let mut t = Transform::IDENTITY;
        for n in stack.iter().rev() {
            t = t.compose(&self.nodes[*n as usize].local);
        }
        t
    }

    // ------------------------------------------------------------------ values to transforms
}
