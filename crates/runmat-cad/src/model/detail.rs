//! Detail features: threads (cosmetic or modelled), emboss and wrap, ribs, and library
//! features (`feature def`): pure, parameterised templates with no triggers.

use crate::ids::{FeatureId, InputName};
use crate::model::feature::{BodyMode, Feature, ProfileRef};
use crate::model::visit::{Params, Visit};
use crate::refs::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A thread's specification (a standard designation, or explicit pitch and depth).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadSpec {
    /// `"M6x1"`, `"1/4-20 UNC"`.
    pub designation: String,
    pub pitch: ValueExpr,
    /// Thread depth (radial).
    pub depth: ValueExpr,
    #[serde(default)]
    pub left_hand: bool,
    /// `6H`, `6g`, `2B`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Thread {
    /// The cylindrical face threaded.
    pub face: Reference,
    pub spec: ThreadSpec,
    /// `None`: the whole face.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<ValueExpr>,
    /// Modelled threads change geometry (a helical cut or add); cosmetic threads are
    /// attributes on the face for drawings, PMI and manufacturing.
    #[serde(default)]
    pub modelled: bool,
}

impl Params for Thread {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.reference("face", &self.face);
        v.value(&self.spec.pitch);
        v.value(&self.spec.depth);
        v.opt_value(&self.length);
    }
    fn kind_name(&self) -> &'static str {
        if self.modelled {
            "thread.modelled"
        } else {
            "thread.cosmetic"
        }
    }
    fn body_feature(&self) -> bool {
        self.modelled
    }
}

/// Raise or sink a sketch profile on a face; `wrap` lays it onto a curved face first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Emboss {
    pub profile: ProfileRef,
    pub face: Reference,
    /// Positive raises (emboss), negative sinks (deboss).
    pub height: ValueExpr,
    #[serde(default)]
    pub wrap: bool,
}

impl Params for Emboss {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.profile(&self.profile);
        v.reference("face", &self.face);
        v.value(&self.height);
    }
    fn kind_name(&self) -> &'static str {
        if self.wrap {
            "emboss.wrap"
        } else {
            "emboss"
        }
    }
}

/// A rib: an open sketch chain thickened in its plane and extended to the body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rib {
    pub profile: ProfileRef,
    pub thickness: ValueExpr,
    /// How far the rib extends along the sketch normal (into the body).
    pub depth: ValueExpr,
    #[serde(default)]
    pub reverse: bool,
}

impl Params for Rib {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.profile(&self.profile);
        v.value(&self.thickness);
        v.value(&self.depth);
    }
    fn kind_name(&self) -> &'static str {
        "rib"
    }
}

/// A `feature def`: a named, pure template. Its body is ordinary features over its own
/// parameters and placeholder references; an instance binds them. No triggers, no state:
/// the same arguments give the same features.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureDef {
    pub name: String,
    pub params: Vec<crate::value::InputDecl>,
    /// Named reference slots the template's features use as `Reference::Published { file:
    /// None, name }` placeholders.
    #[serde(default)]
    pub slots: Vec<String>,
    pub features: Vec<Feature>,
}

/// One use of a library feature.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LibraryInstance {
    /// `library/bosses.part#feature_def.mount_boss`, or a def in this file.
    pub def: String,
    #[serde(default)]
    pub args: BTreeMap<InputName, ValueExpr>,
    #[serde(default)]
    pub bindings: BTreeMap<String, Reference>,
    #[serde(default)]
    pub mode: BodyMode,
}

impl Params for LibraryInstance {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        for a in self.args.values() {
            v.value(a);
        }
        for (k, r) in &self.bindings {
            v.reference(format!("bindings.{}", k), r);
        }
    }
    fn kind_name(&self) -> &'static str {
        "library"
    }
}

#[derive(Clone, Debug, PartialEq, thiserror::Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error")]
pub enum ExpandError {
    #[error("no argument for parameter `{param}` and no default")]
    MissingArgument { param: String },
    #[error("unknown parameter `{param}`")]
    UnknownParameter { param: String },
    #[error("slot `{slot}` is not bound")]
    UnboundSlot { slot: String },
}

impl FeatureDef {
    /// Expand one instance into ordinary features: ids prefixed `<instance>.`, parameter reads
    /// replaced by the arguments (or defaults), slot placeholders by the bound references.
    /// Pure: the same instance always gives the same features.
    pub fn expand(
        &self,
        instance: &FeatureId,
        use_: &LibraryInstance,
    ) -> Result<Vec<Feature>, ExpandError> {
        for k in use_.args.keys() {
            if !self.params.iter().any(|p| &p.name == k) {
                return Err(ExpandError::UnknownParameter { param: k.0.clone() });
            }
        }
        let mut values: BTreeMap<InputName, ValueExpr> = BTreeMap::new();
        for p in &self.params {
            values.insert(
                p.name.clone(),
                use_.args.get(&p.name).cloned().unwrap_or(p.default.clone()),
            );
        }
        for s in &self.slots {
            if !use_.bindings.contains_key(s) {
                return Err(ExpandError::UnboundSlot { slot: s.clone() });
            }
        }
        let prefix = |id: &FeatureId| FeatureId::new(format!("{}.{}", instance, id));
        let ids: Vec<FeatureId> = self.features.iter().map(|f| f.id.clone()).collect();
        let mut out = Vec::new();
        for f in &self.features {
            let mut json = serde_json::to_value(f).expect("feature serialises");
            rewrite(&mut json, &values, &use_.bindings, &ids, &prefix);
            let mut g: Feature = serde_json::from_value(json).expect("rewritten feature");
            g.id = prefix(&f.id);
            out.push(g);
        }
        Ok(out)
    }
}

/// The feature with these inputs' reads replaced by the expressions given (a variable
/// pattern's per-instance values).
pub fn substitute_inputs(f: &Feature, values: &BTreeMap<InputName, ValueExpr>) -> Feature {
    let mut json = serde_json::to_value(f).expect("feature serialises");
    rewrite(&mut json, values, &BTreeMap::new(), &[], &|id| id.clone());
    serde_json::from_value(json).expect("rewritten feature")
}

/// Rewrite a serialised feature: input reads → argument expressions, slot placeholders →
/// bindings, template feature ids → prefixed ids.
fn rewrite(
    v: &mut serde_json::Value,
    values: &BTreeMap<InputName, ValueExpr>,
    bindings: &BTreeMap<String, Reference>,
    ids: &[FeatureId],
    prefix: &dyn Fn(&FeatureId) -> FeatureId,
) {
    use serde_json::Value as J;
    match v {
        J::Object(map) => {
            // An input read.
            if map.get("form").and_then(|x| x.as_str()) == Some("input") {
                if let Some(name) = map.get("name").and_then(|x| x.as_str()) {
                    if let Some(val) = values.get(&InputName::new(name)) {
                        *v = serde_json::to_value(val).expect("value serialises");
                        return;
                    }
                }
            }
            // A slot placeholder: a local published name.
            if map.get("ref").and_then(|x| x.as_str()) == Some("published")
                && map.get("file").is_none()
            {
                if let Some(name) = map.get("name").and_then(|x| x.as_str()) {
                    if let Some(r) = bindings.get(name) {
                        *v = serde_json::to_value(r).expect("reference serialises");
                        return;
                    }
                }
            }
            let fix = |s: &mut String| {
                if let Some(id) = ids.iter().find(|i| i.0 == *s) {
                    *s = prefix(id).0;
                }
            };
            for (k, x) in map.iter_mut() {
                match (k.as_str(), &mut *x) {
                    ("sketch" | "feature", J::String(s)) => fix(s),
                    ("features", J::Array(a)) => {
                        for e in a.iter_mut() {
                            if let J::String(s) = e {
                                fix(s);
                            }
                        }
                    }
                    _ => rewrite(x, values, bindings, ids, prefix),
                }
            }
        }
        J::Array(a) => a
            .iter_mut()
            .for_each(|x| rewrite(x, values, bindings, ids, prefix)),
        _ => {}
    }
}
