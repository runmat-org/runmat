//! One visitor for what a feature's parameters hold: value expressions (for dependency
//! analysis and the cascade check), topological references with the parameter they sit in,
//! and features named directly (profiles, datums, seeds). Feature families implement
//! [`Params`] once and every analysis reads it.

use crate::ids::FeatureId;
use crate::model::datum::{AxisRef, DatumRef, PlaneRef, PointRef};
use crate::model::feature::ProfileRef;
use crate::refs::Reference;
use crate::value::ValueExpr;

#[derive(Default)]
pub struct Visit<'a> {
    pub values: Vec<&'a ValueExpr>,
    pub references: Vec<(String, &'a Reference)>,
    pub features: Vec<FeatureId>,
}

impl<'a> Visit<'a> {
    pub fn value(&mut self, v: &'a ValueExpr) {
        self.values.push(v);
    }
    pub fn opt_value(&mut self, v: &'a Option<ValueExpr>) {
        if let Some(v) = v {
            self.values.push(v);
        }
    }
    pub fn reference(&mut self, param: impl Into<String>, r: &'a Reference) {
        self.references.push((param.into(), r));
    }
    pub fn references(&mut self, param: &str, rs: &'a [Reference]) {
        for (i, r) in rs.iter().enumerate() {
            self.references.push((format!("{}[{}]", param, i), r));
        }
    }
    pub fn feature(&mut self, f: &FeatureId) {
        self.features.push(f.clone());
    }
    pub fn profile(&mut self, p: &ProfileRef) {
        self.features.push(p.sketch.clone());
    }
    fn datum(&mut self, d: &DatumRef) {
        if let DatumRef::Feature { feature } = d {
            self.features.push(feature.clone());
        }
    }
    pub fn plane(&mut self, param: &str, p: &'a PlaneRef) {
        match p {
            PlaneRef::Face { face } => self.reference(param, face),
            PlaneRef::Datum { datum } => self.datum(datum),
        }
    }
    pub fn axis(&mut self, param: &str, a: &'a AxisRef) {
        match a {
            AxisRef::Edge { edge } => self.reference(param, edge),
            AxisRef::Datum { datum } => self.datum(datum),
        }
    }
    pub fn point(&mut self, param: &str, p: &'a PointRef) {
        match p {
            PointRef::Vertex { vertex } => self.reference(param, vertex),
            PointRef::Datum { datum } => self.datum(datum),
            PointRef::Coords { x, y, z } => {
                self.values.push(x);
                self.values.push(y);
                self.values.push(z);
            }
            PointRef::Sketch { sketch, .. } => self.features.push(sketch.clone()),
        }
    }
}

/// What every feature family reports about its parameters.
pub trait Params {
    fn visit<'a>(&'a self, v: &mut Visit<'a>);
    /// A short kind name for outcomes and the index (`sheet_metal.edge_flange`).
    fn kind_name(&self) -> &'static str;
    /// Whether the feature makes a new body rather than changing existing ones.
    fn new_body(&self) -> bool {
        false
    }
    /// Whether the feature acts on every body when it names none.
    fn implicit_targets(&self) -> bool {
        false
    }
    /// Whether the feature changes bodies (as opposed to curves, datums and connectors).
    fn body_feature(&self) -> bool {
        true
    }
}
