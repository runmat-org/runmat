//! Static dependency analysis over the feature list.
//!
//! Without regenerating, work out which inputs and configurations affect which features, part
//! outputs, regions and publish entries. A feature depends on:
//! - the inputs and lets its parameters read (lets transitively);
//! - every feature its references, profiles, seeds and datums name;
//! - the earlier features that shaped the bodies it acts on. Statically this is a body group:
//!   a new-body feature starts a group, a feature naming another group's entities joins it,
//!   and a feature with implicit targets (every body) joins all groups so far.

use crate::ids::{
    ConfigurationName, FeatureId, InputName, LetName, PartName, PublishName, RegionName,
};
use crate::model::items::PublishTarget;
use crate::model::{BodyMode, FeatureKind, PartDocument};
use crate::refs::Reference;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyGraph {
    pub order: Vec<FeatureId>,
    /// Inputs each feature reads, directly or through lets.
    pub feature_inputs: BTreeMap<FeatureId, BTreeSet<InputName>>,
    /// Direct feature-to-feature dependencies (each points backwards).
    pub feature_deps: BTreeMap<FeatureId, BTreeSet<FeatureId>>,
    /// Features each part output, region and publish entry needs.
    pub part_deps: BTreeMap<PartName, BTreeSet<FeatureId>>,
    pub region_deps: BTreeMap<RegionName, BTreeSet<FeatureId>>,
    pub publish_deps: BTreeMap<PublishName, BTreeSet<FeatureId>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Affected {
    pub features: BTreeSet<FeatureId>,
    pub parts: BTreeSet<PartName>,
    pub regions: BTreeSet<RegionName>,
    pub publishes: BTreeSet<PublishName>,
}

fn reference_makers(refs: &[&Reference]) -> BTreeSet<FeatureId> {
    refs.iter().flat_map(|r| r.makers()).collect()
}

impl DependencyGraph {
    pub fn analyse(doc: &PartDocument) -> DependencyGraph {
        let mut g = DependencyGraph {
            order: doc.features.iter().map(|f| f.id.clone()).collect(),
            ..Default::default()
        };

        // Let → inputs, transitively.
        let mut let_inputs: BTreeMap<LetName, BTreeSet<InputName>> = BTreeMap::new();
        let lets: BTreeMap<&LetName, &crate::value::LetDecl> =
            doc.lets.iter().map(|l| (&l.name, l)).collect();
        fn let_closure(
            name: &LetName,
            lets: &BTreeMap<&LetName, &crate::value::LetDecl>,
            memo: &mut BTreeMap<LetName, BTreeSet<InputName>>,
            stack: &mut BTreeSet<LetName>,
        ) -> BTreeSet<InputName> {
            if let Some(v) = memo.get(name) {
                return v.clone();
            }
            if !stack.insert(name.clone()) {
                return BTreeSet::new();
            }
            let mut ins = BTreeSet::new();
            let mut ls = BTreeSet::new();
            if let Some(d) = lets.get(name) {
                d.value.collect_names(&mut ins, &mut ls);
            }
            for l in ls {
                ins.extend(let_closure(&l, lets, memo, stack));
            }
            stack.remove(name);
            memo.insert(name.clone(), ins.clone());
            ins
        }
        for l in &doc.lets {
            let mut stack = BTreeSet::new();
            let_closure(&l.name, &lets, &mut let_inputs, &mut stack);
        }
        // Inputs whose defaults read other inputs.
        let input_inputs: BTreeMap<InputName, BTreeSet<InputName>> = doc
            .inputs
            .iter()
            .map(|i| {
                let mut ins = BTreeSet::new();
                let mut ls = BTreeSet::new();
                i.default.collect_names(&mut ins, &mut ls);
                (i.name.clone(), ins)
            })
            .collect();

        // Body groups via union-find over feature indices.
        let n = doc.features.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            if p[x] != x {
                let r = find(p, p[x]);
                p[x] = r;
            }
            p[x]
        }
        let index: BTreeMap<&FeatureId, usize> = doc
            .features
            .iter()
            .enumerate()
            .map(|(i, f)| (&f.id, i))
            .collect();
        let mut body_features: Vec<usize> = Vec::new();

        for (i, f) in doc.features.iter().enumerate() {
            let mut ins = BTreeSet::new();
            let mut ls = BTreeSet::new();
            for v in f.kind.values() {
                v.collect_names(&mut ins, &mut ls);
            }
            for l in &ls {
                ins.extend(let_inputs.get(l).cloned().unwrap_or_default());
            }
            let direct: Vec<InputName> = ins.iter().cloned().collect();
            for d in direct {
                ins.extend(input_inputs.get(&d).cloned().unwrap_or_default());
            }
            g.feature_inputs.insert(f.id.clone(), ins);

            let refs: Vec<&Reference> = f.kind.references().into_iter().map(|(_, r)| r).collect();
            let mut deps = reference_makers(&refs);
            deps.extend(f.kind.feature_dependencies());
            deps.remove(&f.id);
            deps.retain(|d| index.get(d).map(|j| *j < i).unwrap_or(false));

            if f.kind.is_body_feature() {
                let new_body = matches!(
                    &f.kind,
                    FeatureKind::Extrude(e) if e.mode == BodyMode::New
                ) || matches!(&f.kind, FeatureKind::Revolve(r) if r.mode == BodyMode::New)
                    || matches!(&f.kind, FeatureKind::Sweep(s) if s.mode == BodyMode::New)
                    || matches!(&f.kind, FeatureKind::Loft(l) if l.mode == BodyMode::New)
                    || matches!(&f.kind, FeatureKind::Import(_));
                if f.kind.has_implicit_targets() {
                    for &j in &body_features {
                        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                        parent[a] = b;
                    }
                } else if !new_body || !deps.is_empty() {
                    for d in &deps {
                        if let Some(&j) = index.get(d) {
                            if doc.features[j].kind.is_body_feature() {
                                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                                parent[a] = b;
                            }
                        }
                    }
                }
                // Earlier members of the group shape the bodies this feature acts on.
                if !new_body || f.kind.has_implicit_targets() {
                    let root = find(&mut parent, i);
                    for &j in &body_features {
                        if find(&mut parent, j) == root {
                            deps.insert(doc.features[j].id.clone());
                        }
                    }
                }
                body_features.push(i);
            }
            g.feature_deps.insert(f.id.clone(), deps);
        }

        // Items after the history depend on the makers they name plus those makers' groups.
        let group_closure =
            |makers: BTreeSet<FeatureId>, parent: &mut Vec<usize>| -> BTreeSet<FeatureId> {
                let mut out = makers.clone();
                for m in &makers {
                    if let Some(&j) = index.get(m) {
                        let root = find(parent, j);
                        for &k in &body_features {
                            if find(parent, k) == root {
                                out.insert(doc.features[k].id.clone());
                            }
                        }
                    }
                }
                out
            };
        for p in &doc.parts {
            let m = reference_makers(&[&p.body]);
            let c = group_closure(m, &mut parent);
            g.part_deps.insert(p.name.clone(), c);
        }
        for r in &doc.regions {
            let refs: Vec<&Reference> = r.select.iter().collect();
            let c = group_closure(reference_makers(&refs), &mut parent);
            g.region_deps.insert(r.name.clone(), c);
        }
        for p in &doc.publishes {
            let mut m = BTreeSet::new();
            match &p.target {
                PublishTarget::Entities { references, .. } => {
                    let refs: Vec<&Reference> = references.iter().collect();
                    m = reference_makers(&refs);
                }
                PublishTarget::Datum {
                    datum: crate::model::DatumRef::Feature { feature },
                } => {
                    m.insert(feature.clone());
                }
                PublishTarget::Region { region } => {
                    m = g.region_deps.get(region).cloned().unwrap_or_default();
                }
                PublishTarget::Part { part } => {
                    m = g.part_deps.get(part).cloned().unwrap_or_default();
                }
                _ => {}
            }
            let c = group_closure(m, &mut parent);
            g.publish_deps.insert(p.name.clone(), c);
        }
        g
    }

    /// Every feature downstream of `roots` (including them).
    pub fn downstream(&self, roots: &BTreeSet<FeatureId>) -> BTreeSet<FeatureId> {
        let mut out = roots.clone();
        for f in &self.order {
            if out.contains(f) {
                continue;
            }
            if self
                .feature_deps
                .get(f)
                .map(|d| d.iter().any(|x| out.contains(x)))
                .unwrap_or(false)
            {
                out.insert(f.clone());
            }
        }
        out
    }

    fn items(&self, features: BTreeSet<FeatureId>) -> Affected {
        let hit = |deps: &BTreeSet<FeatureId>| deps.iter().any(|d| features.contains(d));
        Affected {
            parts: self
                .part_deps
                .iter()
                .filter(|(_, d)| hit(d))
                .map(|(k, _)| k.clone())
                .collect(),
            regions: self
                .region_deps
                .iter()
                .filter(|(_, d)| hit(d))
                .map(|(k, _)| k.clone())
                .collect(),
            publishes: self
                .publish_deps
                .iter()
                .filter(|(_, d)| hit(d))
                .map(|(k, _)| k.clone())
                .collect(),
            features,
        }
    }

    /// What changing these inputs can change.
    pub fn affected_by_inputs(&self, inputs: &[InputName]) -> Affected {
        let set: BTreeSet<InputName> = inputs.iter().cloned().collect();
        let roots: BTreeSet<FeatureId> = self
            .feature_inputs
            .iter()
            .filter(|(_, ins)| ins.iter().any(|i| set.contains(i)))
            .map(|(f, _)| f.clone())
            .collect();
        let features = self.downstream(&roots);
        self.items(features)
    }

    /// What selecting a configuration changes relative to the defaults.
    pub fn affected_by_configuration(
        &self,
        doc: &PartDocument,
        name: &ConfigurationName,
    ) -> Option<Affected> {
        let c = doc.configuration(name)?;
        let inputs: Vec<InputName> = c.overrides.keys().cloned().collect();
        let mut a = self.affected_by_inputs(&inputs);
        let extra = self.downstream(&c.suppress);
        a.features.extend(extra);
        let features = a.features.clone();
        Some(self.items(features))
    }
}
