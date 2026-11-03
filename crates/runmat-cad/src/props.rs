//! Properties, checks and links.
//!
//! - **Schemas** (`[properties]` in `runmat.toml` or a company package): each field gives a
//!   type (text, enumeration, boolean, or a quantity of a kind), the scopes it may be set at
//!   (definition, configuration, occurrence), the item kinds it applies to, and whether it is
//!   required at release. [`validate_properties`] checks every assignment against its field.
//! - **Checks** (`check` blocks; packs enabled in `[checks]`): a `runmat-rules` selector picks
//!   items, an expectation tests a property or a measurement against a value with units, and
//!   a gate says whether a failure blocks publish or release. The item graph is the resolved
//!   assembly ([`AssemblyGraph`]); measurements (mass, box size, counts, results) come through
//!   the [`Measurements`] seam. Selectors take their units from the one unit authority through
//!   [`AuthorityUnits`].
//! - **Links** (`derived_from`, `verifies`, `supersedes`, `supplier_doc`): [`check_links`]
//!   resolves targets and, at release, requires every `verifies` link to point at a check that
//!   passed.
//! - **Gate**: [`release_gate`] folds property, check and link results into one decision with
//!   reasons (zero errors by default).

use crate::assembly::model::DefinitionKind;
use crate::assembly::resolve::ResolvedAssembly;
use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Severity, Site, Subject};
use crate::ids::PropertyName;
use crate::model::items::{Link, LinkKind, Property, PropertyScope, PropertyValue};
use crate::units::{Dimension, QuantityKindId, UnitAuthority};
use crate::value::{Evaluator, ValueEnv};
use runmat_rules as rules;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// ------------------------------------------------------------------------------ schemas

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum PropertyType {
    Text,
    Boolean,
    Enum { values: Vec<String> },
    Quantity { kind: QuantityKindId },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyField {
    pub name: PropertyName,
    pub ty: PropertyType,
    pub scopes: BTreeSet<PropertyScope>,
    /// Item kinds it applies to (`part`, `assembly`, `occurrence`, `circuit`); empty: any.
    #[serde(default)]
    pub applies_to: BTreeSet<String>,
    #[serde(default)]
    pub required_at_release: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PropertySchema {
    pub fields: Vec<PropertyField>,
}

impl PropertySchema {
    pub fn field(&self, name: &PropertyName) -> Option<&PropertyField> {
        self.fields.iter().find(|f| &f.name == name)
    }
}

/// One item's assignments, for validation.
pub struct PropertySite<'a> {
    pub label: String,
    pub kind: &'a str,
    pub properties: &'a [Property],
}

pub fn validate_properties(
    schema: &PropertySchema,
    sites: &[PropertySite<'_>],
    units: &dyn UnitAuthority,
    ev: &Evaluator<'_>,
    release: bool,
) -> DiagnosticReport {
    let mut out = DiagnosticReport::default();
    let mut push = |code: DiagCode, site: &str, msg: String| {
        out.push(
            code,
            Severity::Error,
            Site::of(Subject::Occurrence { path: site.into() }),
            msg,
            Cause::Root,
            Vec::new(),
        );
    };
    for s in sites {
        for p in s.properties {
            let Some(f) = schema.field(&p.name) else {
                push(
                    DiagCode::PropertyInvalid,
                    &s.label,
                    format!("`{}`: no property `{}` in the schema", s.label, p.name),
                );
                continue;
            };
            if !f.applies_to.is_empty() && !f.applies_to.contains(s.kind) {
                push(
                    DiagCode::PropertyInvalid,
                    &s.label,
                    format!("`{}` does not apply to a {}", p.name, s.kind),
                );
            }
            if !f.scopes.contains(&p.scope) {
                push(
                    DiagCode::PropertyInvalid,
                    &s.label,
                    format!("`{}` cannot be set at {:?} scope", p.name, p.scope),
                );
            }
            match (&f.ty, &p.value) {
                (PropertyType::Text, PropertyValue::Text(_))
                | (PropertyType::Boolean, PropertyValue::Boolean(_)) => {}
                (PropertyType::Enum { values }, PropertyValue::Text(t)) => {
                    if !values.contains(t) {
                        push(
                            DiagCode::PropertyInvalid,
                            &s.label,
                            format!("`{}` = {:?} is not one of {:?}", p.name, t, values),
                        );
                    }
                }
                (PropertyType::Quantity { kind }, PropertyValue::Quantity(e)) => {
                    let want = units.kind(kind).map(|k| k.dimension);
                    match (ev.eval(e, &ValueEnv::default()), want) {
                        (Ok(q), Some(d)) if q.dimension == d => {}
                        (Ok(q), Some(d)) => push(
                            DiagCode::PropertyInvalid,
                            &s.label,
                            format!(
                                "`{}` must be a {} ({}), found {}",
                                p.name, kind, d, q.dimension
                            ),
                        ),
                        (Ok(_), None) => push(
                            DiagCode::PropertyInvalid,
                            &s.label,
                            format!("`{}` has an unknown kind `{}`", p.name, kind),
                        ),
                        (Err(e), _) => push(
                            DiagCode::PropertyInvalid,
                            &s.label,
                            format!("`{}`: {}", p.name, e),
                        ),
                    }
                }
                _ => push(
                    DiagCode::PropertyInvalid,
                    &s.label,
                    format!("`{}` has the wrong type for {:?}", p.name, f.ty),
                ),
            }
        }
        if release {
            for f in schema.fields.iter().filter(|f| f.required_at_release) {
                if (f.applies_to.is_empty() || f.applies_to.contains(s.kind))
                    && !s.properties.iter().any(|p| p.name == f.name)
                {
                    push(
                        DiagCode::PropertyMissing,
                        &s.label,
                        format!("`{}` needs `{}` before release", s.label, f.name),
                    );
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------ units

/// Selector units from the one unit authority.
pub struct AuthorityUnits<'a>(pub &'a dyn UnitAuthority);

pub fn to_rules_dim(d: Dimension) -> rules::Dim {
    let mut b = [0i8; 7];
    b.copy_from_slice(&d.base[..7]);
    rules::Dim(b)
}

impl rules::UnitResolver for AuthorityUnits<'_> {
    fn resolve(&self, symbol: &str) -> Option<rules::UnitDef> {
        self.0.unit(symbol).map(|u| rules::UnitDef {
            scale: u.scale,
            offset: 0.0,
            dim: to_rules_dim(u.dimension),
        })
    }
}

// ------------------------------------------------------------------------------ the graph

/// Measured values (mass, box extent, counts, simulation results) by item and name.
pub trait Measurements {
    fn measure(&self, tree: &ResolvedAssembly, node: u32, name: &str) -> Option<rules::Quantity>;
}

/// No measurements.
pub struct NoMeasurements;

impl Measurements for NoMeasurements {
    fn measure(&self, _: &ResolvedAssembly, _: u32, _: &str) -> Option<rules::Quantity> {
        None
    }
}

/// A resolved assembly as a `runmat-rules` item graph. Kinds: `part`, `assembly`, `circuit`
/// (every item also matches `occurrence`). `#name` is the occurrence name. Classes come from a
/// `class` text property (space-separated) and the representation. Attributes: occurrence
/// properties, then definition-scope ones given in `definition_properties`, then
/// `part_number`, `material`, `definition`, `depth`, `configuration`, then measurements.
pub struct AssemblyGraph<'a> {
    pub tree: &'a ResolvedAssembly,
    pub definition_properties: BTreeMap<String, Vec<Property>>,
    pub measurements: &'a dyn Measurements,
    pub units: &'a dyn UnitAuthority,
    pub ev: Evaluator<'a>,
    kinds: Vec<&'static str>,
}

impl<'a> AssemblyGraph<'a> {
    pub fn new(
        tree: &'a ResolvedAssembly,
        measurements: &'a dyn Measurements,
        ev: Evaluator<'a>,
    ) -> Self {
        let kinds = tree
            .nodes
            .iter()
            .map(|n| match n.definition.kind {
                DefinitionKind::Part => "part",
                DefinitionKind::Assembly => "assembly",
                DefinitionKind::Circuit => "circuit",
            })
            .collect();
        Self {
            tree,
            definition_properties: BTreeMap::new(),
            measurements,
            units: ev.units,
            ev,
            kinds,
        }
    }

    fn property(&self, id: u32, name: &str) -> Option<PropertyValue> {
        let n = self.tree.node(id);
        n.properties
            .iter()
            .find(|p| p.name.0 == name)
            .map(|p| p.value.clone())
            .or_else(|| {
                self.definition_properties
                    .get(&n.definition.to_string())
                    .and_then(|ps| {
                        ps.iter()
                            .find(|p| p.name.0 == name)
                            .map(|p| p.value.clone())
                    })
            })
    }
}

impl rules::ItemGraph for AssemblyGraph<'_> {
    type Id = u32;

    fn items(&self) -> Vec<u32> {
        (0..self.tree.nodes.len() as u32).collect()
    }
    fn kind(&self, id: u32) -> &str {
        self.kinds[id as usize]
    }
    fn is_kind(&self, id: u32, kind: &str) -> bool {
        kind == "occurrence" || self.kinds[id as usize] == kind
    }
    fn name(&self, id: u32) -> Option<&str> {
        Some(self.tree.node(id).segment.name.as_str())
    }
    fn has_class(&self, id: u32, class: &str) -> bool {
        let rep = format!("{:?}", self.tree.node(id).representation).to_lowercase();
        if rep == class {
            return true;
        }
        matches!(self.property(id, "class"), Some(PropertyValue::Text(t)) if t.split_whitespace().any(|c| c == class))
    }
    fn attr(&self, id: u32, name: &str) -> Option<rules::AttrValue> {
        if let Some(v) = self.property(id, name) {
            return Some(match v {
                PropertyValue::Text(t) => rules::AttrValue::Str(t),
                PropertyValue::Boolean(b) => rules::AttrValue::Bool(b),
                PropertyValue::Quantity(e) => {
                    let q = self.ev.eval(&e, &ValueEnv::default()).ok()?;
                    rules::AttrValue::Num(rules::Quantity::new(q.si, to_rules_dim(q.dimension)))
                }
            });
        }
        let n = self.tree.node(id);
        match name {
            "part_number" => n.part_number.clone().map(rules::AttrValue::Str),
            "material" => n
                .material
                .as_ref()
                .map(|m| rules::AttrValue::Str(m.0.clone())),
            "definition" => Some(rules::AttrValue::Str(n.definition.to_string())),
            "configuration" => Some(rules::AttrValue::Str(n.configuration.label())),
            "depth" => Some(rules::AttrValue::Num(rules::Quantity::new(
                n.depth as f64,
                rules::Dim::NONE,
            ))),
            _ => self
                .measurements
                .measure(self.tree, id, name)
                .map(rules::AttrValue::Num),
        }
    }
    fn parent(&self, id: u32) -> Option<u32> {
        self.tree.node(id).parent
    }
    fn children(&self, id: u32) -> Vec<u32> {
        self.tree.node(id).children.clone()
    }
    fn pseudo(&self, id: u32, name: &str, args: &[rules::PseudoArg]) -> Option<bool> {
        let _ = args;
        match name {
            "root" => Some(self.tree.node(id).parent.is_none()),
            "leaf" => Some(self.tree.node(id).children.is_empty()),
            "fixed" => Some(self.tree.node(id).fixed),
            _ => None,
        }
    }
}

// ------------------------------------------------------------------------------ checks

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    /// The attribute is present.
    Present,
}

/// A value to compare with, written with units (`4.2 kg`) or as text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "is")]
pub enum Expected {
    Quantity { text: String },
    Text { value: String },
    Boolean { value: bool },
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gate {
    /// Report only.
    None,
    Publish,
    Release,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckDef {
    pub name: String,
    /// A `runmat-rules` selector over the item graph.
    pub select: String,
    pub attribute: String,
    pub op: CompareOp,
    pub expected: Expected,
    #[serde(default = "release")]
    pub gate: Gate,
    /// An item the selector matches but which lacks the attribute fails (`true`) or is
    /// reported as unknown (`false`).
    #[serde(default = "yes")]
    pub missing_fails: bool,
}

fn release() -> Gate {
    Gate::Release
}
fn yes() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckPack {
    pub name: String,
    pub version: String,
    pub checks: Vec<CheckDef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum ItemOutcome {
    Pass,
    Fail { found: String },
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckResult {
    pub pack: String,
    pub check: String,
    pub gate: Gate,
    pub items: Vec<(String, ItemOutcome)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl CheckResult {
    pub fn passed(&self) -> bool {
        self.error.is_none()
            && self
                .items
                .iter()
                .all(|(_, o)| matches!(o, ItemOutcome::Pass))
    }
}

fn compare(op: CompareOp, a: &rules::AttrValue, b: &rules::AttrValue) -> Option<bool> {
    use rules::AttrValue as A;
    Some(match (a, b) {
        (A::Num(x), A::Num(y)) => {
            if x.dim != y.dim {
                return None;
            }
            let (x, y) = (x.value, y.value);
            match op {
                CompareOp::Lt => x < y,
                CompareOp::Le => x <= y,
                CompareOp::Gt => x > y,
                CompareOp::Ge => x >= y,
                CompareOp::Eq => (x - y).abs() <= 1e-12 * y.abs().max(1.0),
                CompareOp::Ne => (x - y).abs() > 1e-12 * y.abs().max(1.0),
                CompareOp::Present => true,
            }
        }
        (A::Str(x), A::Str(y)) => match op {
            CompareOp::Eq => x == y,
            CompareOp::Ne => x != y,
            CompareOp::Present => true,
            _ => return None,
        },
        (A::Bool(x), A::Bool(y)) => match op {
            CompareOp::Eq => x == y,
            CompareOp::Ne => x != y,
            _ => return None,
        },
        _ => return None,
    })
}

fn show(v: &rules::AttrValue) -> String {
    match v {
        rules::AttrValue::Str(s) => format!("{:?}", s),
        rules::AttrValue::Num(q) => format!("{}", q),
        rules::AttrValue::Bool(b) => b.to_string(),
        rules::AttrValue::List(l) => format!("{} values", l.len()),
    }
}

pub fn run_checks(
    packs: &[CheckPack],
    graph: &AssemblyGraph<'_>,
) -> (Vec<CheckResult>, DiagnosticReport) {
    let units = AuthorityUnits(graph.units);
    let mut results = Vec::new();
    let mut diags = DiagnosticReport::default();
    for pack in packs {
        for c in &pack.checks {
            let mut r = CheckResult {
                pack: pack.name.clone(),
                check: c.name.clone(),
                gate: c.gate,
                items: Vec::new(),
                error: None,
            };
            let site = Site::of(Subject::Value {
                name: format!("check.{}.{}", pack.name, c.name),
            });
            let sel = match rules::Selector::parse(&c.select, &units) {
                Ok(s) => s,
                Err(e) => {
                    r.error = Some(format!("selector: {:?}", e));
                    diags.push(
                        DiagCode::CheckInvalid,
                        Severity::Error,
                        site,
                        format!("check `{}`: selector {:?}", c.name, e),
                        Cause::Root,
                        Vec::new(),
                    );
                    results.push(r);
                    continue;
                }
            };
            let expected = match &c.expected {
                Expected::Quantity { text } => match rules::parse_quantity(text, &units) {
                    Ok(q) => Some(rules::AttrValue::Num(q)),
                    Err(e) => {
                        r.error = Some(format!("value {:?}: {:?}", text, e));
                        diags.push(
                            DiagCode::CheckInvalid,
                            Severity::Error,
                            site,
                            format!("check `{}`: value {:?}", c.name, text),
                            Cause::Root,
                            Vec::new(),
                        );
                        results.push(r);
                        continue;
                    }
                },
                Expected::Text { value } => Some(rules::AttrValue::Str(value.clone())),
                Expected::Boolean { value } => Some(rules::AttrValue::Bool(*value)),
                Expected::None => None,
            };
            for (id, _) in sel.select(graph) {
                let label = graph.tree.path(id);
                let found = rules::ItemGraph::attr(graph, id, &c.attribute);
                let outcome = match (&found, &expected, c.op) {
                    (Some(_), _, CompareOp::Present) => ItemOutcome::Pass,
                    (None, _, _) if c.missing_fails => ItemOutcome::Fail {
                        found: "(missing)".into(),
                    },
                    (None, _, _) => ItemOutcome::Unknown,
                    (Some(f), Some(e), op) => match compare(op, f, e) {
                        Some(true) => ItemOutcome::Pass,
                        Some(false) => ItemOutcome::Fail { found: show(f) },
                        None => ItemOutcome::Fail {
                            found: format!("{} (not comparable)", show(f)),
                        },
                    },
                    (Some(_), None, _) => ItemOutcome::Unknown,
                };
                if let ItemOutcome::Fail { found } = &outcome {
                    diags.push(
                        DiagCode::AsmCheckFailed,
                        if c.gate == Gate::None {
                            Severity::Warning
                        } else {
                            Severity::Error
                        },
                        Site::of(Subject::Occurrence {
                            path: label.clone(),
                        }),
                        format!(
                            "check `{}` ({}): `{}` {:?} expected, found {}",
                            c.name, pack.name, c.attribute, c.op, found
                        ),
                        Cause::Root,
                        Vec::new(),
                    );
                }
                r.items.push((label, outcome));
            }
            results.push(r);
        }
    }
    (results, diags)
}

// ------------------------------------------------------------------------------ links

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "link")]
pub enum LinkProblem {
    Unresolved { to: String },
    VerifiesNotACheck { to: String },
    VerifiesFailing { to: String },
}

/// Resolve link targets; at release every `verifies` link must reach a passing check
/// (`<pack>.<check>` or `check.<pack>.<check>`).
pub fn check_links(
    links: &[Link],
    resolves: &dyn Fn(&str) -> bool,
    checks: &[CheckResult],
    release: bool,
) -> Vec<(Link, LinkProblem)> {
    let mut out = Vec::new();
    for l in links {
        if l.kind == LinkKind::Verifies {
            let key = l.to.strip_prefix("check.").unwrap_or(&l.to);
            match checks
                .iter()
                .find(|c| format!("{}.{}", c.pack, c.check) == key)
            {
                None => out.push((
                    l.clone(),
                    LinkProblem::VerifiesNotACheck { to: l.to.clone() },
                )),
                Some(c) if release && !c.passed() => {
                    out.push((l.clone(), LinkProblem::VerifiesFailing { to: l.to.clone() }))
                }
                _ => {}
            }
        } else if !resolves(&l.to) {
            out.push((l.clone(), LinkProblem::Unresolved { to: l.to.clone() }));
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateDecision {
    pub open: bool,
    pub reasons: Vec<String>,
}

/// The release (or publish) gate: zero errors in properties, failing gated checks and link
/// problems.
pub fn release_gate(
    gate: Gate,
    properties: &DiagnosticReport,
    checks: &[CheckResult],
    links: &[(Link, LinkProblem)],
) -> GateDecision {
    let mut reasons = Vec::new();
    for d in properties.errors() {
        reasons.push(format!("{} {}", d.code, d.message));
    }
    for c in checks {
        if c.gate != Gate::None && c.gate <= gate && !c.passed() {
            let failing = c
                .items
                .iter()
                .filter(|(_, o)| !matches!(o, ItemOutcome::Pass))
                .count();
            reasons.push(format!(
                "check {}.{}: {} item(s) fail{}",
                c.pack,
                c.check,
                failing,
                c.error
                    .as_ref()
                    .map(|e| format!(" ({})", e))
                    .unwrap_or_default()
            ));
        }
    }
    for (l, p) in links {
        reasons.push(format!("link {:?} → {}: {:?}", l.kind, l.to, p));
    }
    GateDecision {
        open: reasons.is_empty(),
        reasons,
    }
}
