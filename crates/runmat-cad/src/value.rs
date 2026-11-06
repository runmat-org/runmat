//! The value model used by inputs, lets, feature parameters and configuration overrides.
//!
//! Follows `research/12-values-and-resolution.md` design A in outline: a value is static (a
//! quantity literal, a reference to an input or let, an expression, a tolerance) or dynamic (a
//! pure function call evaluated through [`DynamicValueEvaluator`]). Units go through the
//! [`UnitAuthority`] seam.
//!
//! The owner questions in report 12 §6 are bookmarked; each default is one field of
//! [`VALUE_CHOICES`], so a different answer is a one-line change that every consumer
//! (evaluator, scope resolver, syntax layer) reads from one place.

use crate::ids::{FilePath, InputName, LetName, PublishName};
use crate::refs::publish::XrefProblem;
use crate::refs::reference::XrefContext;
use crate::units::{Dimension, QuantityKindId, UnitAuthority};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub mod m_functions;

/// The bookmarked value-model decisions (report 12 §6), with the defaults in force.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueChoices {
    /// Q1: declarations are referenced kind-qualified (`material.aluminum`); bare names are
    /// locals and functions only.
    pub kind_qualified_references: bool,
    /// Q2: a bare double from `.m` for a dimensioned target is an error unless the design
    /// file writes `as <unit>`.
    pub as_unit_escape: bool,
    /// Q3: the one comment form.
    pub comment: &'static str,
    /// Q4: same-folder files cannot shadow the dialect's vocabulary.
    pub vocabulary_shadowable: bool,
    /// Q5: `volatile` values are allowed and snapshotted into `runmat.lock` at release.
    pub allow_volatile: bool,
    /// Q6: `+pkg` folders and `use <folder>` reach other folders; plain subfolders are not
    /// in scope.
    pub packages_and_use: bool,
    /// Q8: randomness seeded by an argument counts as deterministic.
    pub seeded_random_is_deterministic: bool,
}

pub const VALUE_CHOICES: ValueChoices = ValueChoices {
    kind_qualified_references: true,
    as_unit_escape: true,
    comment: "//",
    vocabulary_shadowable: false,
    allow_volatile: true,
    packages_and_use: true,
    seeded_random_is_deterministic: true,
};

/// Serialised with a `form` tag. Deserialisation also takes a bare JSON number as
/// [`ValueExpr::Number`], which is how the syntax layer writes a dimensionless value.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "form")]
pub enum ValueExpr {
    /// A number with a unit as written (`80 mm`).
    Quantity {
        value: f64,
        unit: String,
    },
    /// A bare number: dimensionless.
    Number {
        value: f64,
    },
    /// `input.<name>`.
    Input {
        name: InputName,
    },
    /// A `let` value.
    Let {
        name: LetName,
    },
    Neg {
        operand: Box<ValueExpr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<ValueExpr>,
        rhs: Box<ValueExpr>,
    },
    /// `nominal +plus -minus`. Kernels and solvers consume the nominal value.
    Toleranced {
        nominal: Box<ValueExpr>,
        minus: Box<ValueExpr>,
        plus: Box<ValueExpr>,
    },
    /// A dynamic value: a pure function call, evaluated in the value phase.
    Dynamic {
        call: DynamicCall,
    },
    /// `<expr> as <unit>`: states the unit of a bare double (an `.m` result). Reviewable and
    /// greppable in the design file.
    Coerce {
        expr: Box<ValueExpr>,
        unit: String,
    },
    /// A kind-qualified reference to another declaration's value (`part.bracket.mass`,
    /// `material.al_7075.density`), resolved by the scope ([`DynamicValueEvaluator::reference`]).
    /// Derived values are read-only.
    Reference {
        path: Vec<String>,
    },
    /// A plain value another file publishes (`part.skeleton.publish.station_pitch`, after the
    /// scope has bound `part.skeleton` to its file and `use … @ "R3"` to a pin). Read through
    /// the project catalog by the same rules as entity cross-file references: live within a
    /// partition, pinned to a released revision across partitions, renames through the lock
    /// ([`DynamicValueEvaluator::published`], [`crate::refs::values::PublishedValues`]).
    Published(Box<ValueXref>),
}

/// The tagged form of [`ValueExpr`], derived from the enum itself (the compiler checks the two
/// agree field for field).
#[derive(Deserialize)]
#[serde(remote = "ValueExpr", rename_all = "snake_case", tag = "form")]
enum ValueExprForm {
    Quantity {
        value: f64,
        unit: String,
    },
    Number {
        value: f64,
    },
    Input {
        name: InputName,
    },
    Let {
        name: LetName,
    },
    Neg {
        operand: Box<ValueExpr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<ValueExpr>,
        rhs: Box<ValueExpr>,
    },
    Toleranced {
        nominal: Box<ValueExpr>,
        minus: Box<ValueExpr>,
        plus: Box<ValueExpr>,
    },
    Dynamic {
        call: DynamicCall,
    },
    Coerce {
        expr: Box<ValueExpr>,
        unit: String,
    },
    Reference {
        path: Vec<String>,
    },
    Published(Box<ValueXref>),
}

impl<'de> Deserialize<'de> for ValueExpr {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(number) = value.as_f64().filter(|_| value.is_number()) {
            return Ok(ValueExpr::Number { value: number });
        }
        ValueExprForm::deserialize(value).map_err(serde::de::Error::custom)
    }
}

/// The target of a [`ValueExpr::Published`] read: a publish name in another file, live or
/// pinned.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ValueXref {
    pub file: FilePath,
    pub name: PublishName,
    #[serde(default = "live_context")]
    pub context: XrefContext,
}

fn live_context() -> XrefContext {
    XrefContext::Live
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicCall {
    pub function: String,
    pub arguments: Vec<(String, ValueExpr)>,
    /// `volatile` was written at the use site.
    #[serde(default)]
    pub volatile: bool,
}

impl ValueExpr {
    pub fn qty(value: f64, unit: &str) -> Self {
        ValueExpr::Quantity {
            value,
            unit: unit.to_owned(),
        }
    }
    pub fn mm(value: f64) -> Self {
        Self::qty(value, "mm")
    }
    pub fn deg(value: f64) -> Self {
        Self::qty(value, "deg")
    }
    pub fn number(value: f64) -> Self {
        ValueExpr::Number { value }
    }
    pub fn input(name: &str) -> Self {
        ValueExpr::Input {
            name: InputName::new(name),
        }
    }
    pub fn let_(name: &str) -> Self {
        ValueExpr::Let {
            name: LetName::new(name),
        }
    }
    pub fn binary(op: BinaryOp, lhs: ValueExpr, rhs: ValueExpr) -> Self {
        ValueExpr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Every input and let this expression reads directly.
    pub fn collect_names(&self, inputs: &mut BTreeSet<InputName>, lets: &mut BTreeSet<LetName>) {
        match self {
            ValueExpr::Quantity { .. } | ValueExpr::Number { .. } => {}
            ValueExpr::Input { name } => {
                inputs.insert(name.clone());
            }
            ValueExpr::Let { name } => {
                lets.insert(name.clone());
            }
            ValueExpr::Neg { operand } => operand.collect_names(inputs, lets),
            ValueExpr::Binary { lhs, rhs, .. } => {
                lhs.collect_names(inputs, lets);
                rhs.collect_names(inputs, lets);
            }
            ValueExpr::Toleranced {
                nominal,
                minus,
                plus,
            } => {
                nominal.collect_names(inputs, lets);
                minus.collect_names(inputs, lets);
                plus.collect_names(inputs, lets);
            }
            ValueExpr::Dynamic { call } => {
                for (_, arg) in &call.arguments {
                    arg.collect_names(inputs, lets);
                }
            }
            ValueExpr::Coerce { expr, .. } => expr.collect_names(inputs, lets),
            ValueExpr::Reference { .. } | ValueExpr::Published(_) => {}
        }
    }

    pub fn published(file: &str, name: &str, context: XrefContext) -> Self {
        ValueExpr::Published(Box::new(ValueXref {
            file: FilePath::new(file),
            name: PublishName::new(name),
            context,
        }))
    }

    /// This expression with every read of input `name` replaced by `with`.
    pub fn substitute_input(&self, name: &InputName, with: &ValueExpr) -> ValueExpr {
        let mut j = serde_json::to_value(self).expect("value serialises");
        fn walk(v: &mut serde_json::Value, name: &str, with: &serde_json::Value) {
            match v {
                serde_json::Value::Object(m) => {
                    if m.get("form").and_then(|x| x.as_str()) == Some("input")
                        && m.get("name").and_then(|x| x.as_str()) == Some(name)
                    {
                        *v = with.clone();
                        return;
                    }
                    m.values_mut().for_each(|x| walk(x, name, with));
                }
                serde_json::Value::Array(a) => a.iter_mut().for_each(|x| walk(x, name, with)),
                _ => {}
            }
        }
        let w = serde_json::to_value(with).expect("value serialises");
        walk(&mut j, name.as_str(), &w);
        serde_json::from_value(j).expect("substituted value")
    }

    /// Every published value this expression reads, in order.
    pub fn collect_published<'a>(&'a self, out: &mut Vec<&'a ValueXref>) {
        match self {
            ValueExpr::Published(x) => out.push(x),
            ValueExpr::Quantity { .. }
            | ValueExpr::Number { .. }
            | ValueExpr::Input { .. }
            | ValueExpr::Let { .. }
            | ValueExpr::Reference { .. } => {}
            ValueExpr::Neg { operand } => operand.collect_published(out),
            ValueExpr::Binary { lhs, rhs, .. } => {
                lhs.collect_published(out);
                rhs.collect_published(out);
            }
            ValueExpr::Toleranced {
                nominal,
                minus,
                plus,
            } => {
                nominal.collect_published(out);
                minus.collect_published(out);
                plus.collect_published(out);
            }
            ValueExpr::Dynamic { call } => {
                for (_, a) in &call.arguments {
                    a.collect_published(out);
                }
            }
            ValueExpr::Coerce { expr, .. } => expr.collect_published(out),
        }
    }

    pub fn coerce(expr: ValueExpr, unit: &str) -> Self {
        ValueExpr::Coerce {
            expr: Box::new(expr),
            unit: unit.to_owned(),
        }
    }
}

/// An evaluated value: canonical SI magnitude plus its dimension.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quantity {
    pub si: f64,
    pub dimension: Dimension,
}

impl Quantity {
    pub fn new(si: f64, dimension: Dimension) -> Self {
        Self { si, dimension }
    }
    pub fn dimensionless(value: f64) -> Self {
        Self {
            si: value,
            dimension: Dimension::DIMENSIONLESS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputDecl {
    pub name: InputName,
    pub kind: QuantityKindId,
    pub default: ValueExpr,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<ValueExpr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<ValueExpr>,
}

impl InputDecl {
    pub fn new(name: &str, kind: &str, default: ValueExpr) -> Self {
        Self {
            name: InputName::new(name),
            kind: QuantityKindId::new(kind),
            default,
            minimum: None,
            maximum: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LetDecl {
    pub name: LetName,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<QuantityKindId>,
    pub value: ValueExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error")]
pub enum ValueError {
    #[error("unknown unit `{unit}`")]
    UnknownUnit { unit: String },
    #[error("unknown quantity kind `{kind}`")]
    UnknownKind { kind: String },
    #[error("unresolved input `{name}`")]
    UnresolvedInput { name: String },
    #[error("unresolved value `{name}`")]
    UnresolvedLet { name: String },
    #[error("unit mismatch: expected {expected}, found {found}")]
    UnitMismatch { expected: String, found: String },
    #[error("`{name}` = {value} is outside its declared range")]
    OutOfRange { name: String, value: String },
    #[error("dynamic value `{function}` failed: {message}")]
    DynamicFailed { function: String, message: String },
    #[error("values depend on each other: {cycle:?}")]
    Cycle { cycle: Vec<String> },
    #[error("`{function}` returned a plain number for a {expected} value; write `as {si}` at the call or return a quantity")]
    UnitlessResult {
        function: String,
        expected: String,
        si: String,
    },
    #[error("`{function}` is not deterministic ({reason}); mark the use `volatile` or remove the effect")]
    NotDeterministic { function: String, reason: String },
    #[error("`volatile` values are not allowed in this project")]
    VolatileNotAllowed { function: String },
    #[error("`as {unit}` applies to a plain number, found {found}")]
    CoerceDimensioned { unit: String, found: String },
    #[error("unresolved reference `{path}`")]
    UnresolvedReference { path: String },
    /// A published value could not be read across files.
    #[error("`{}` from `{}`: {problem}", xref.name, xref.file)]
    Xref {
        xref: Box<ValueXref>,
        problem: Box<XrefProblem>,
    },
    /// The published entry exists but is geometry or a datum, not a plain value.
    #[error("`{file}` publishes `{name}` as {found}, not as a value")]
    NotAValue {
        file: FilePath,
        name: PublishName,
        found: String,
    },
}

/// What a dynamic value function returned.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "output")]
pub enum DynamicOutput {
    Quantity {
        value: Quantity,
    },
    /// A bare double (the `.m` returned no unit).
    Double {
        value: f64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "randomness")]
pub enum Randomness {
    None,
    /// Seeded by this argument.
    Seeded {
        argument: String,
    },
    Unseeded,
}

/// The determinism contract the value phase checks, and the inputs of the cache key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicContract {
    /// The compiled function's digest.
    pub function_digest: String,
    /// Transitive callee digests, sorted.
    #[serde(default)]
    pub callees: Vec<String>,
    /// Files the function declares it reads (literal paths), with their content digests.
    #[serde(default)]
    pub reads: Vec<(String, String)>,
    pub runtime_version: String,
    /// Static effects other than declared reads (I/O, clock, network): the reason, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<String>,
    pub randomness: Randomness,
}

impl DynamicContract {
    pub fn deterministic(&self) -> Result<(), String> {
        if let Some(e) = &self.effect {
            return Err(e.clone());
        }
        match &self.randomness {
            Randomness::None => Ok(()),
            Randomness::Seeded { .. } if VALUE_CHOICES.seeded_random_is_deterministic => Ok(()),
            Randomness::Seeded { argument } => Err(format!("randomness seeded by `{}`", argument)),
            Randomness::Unseeded => Err("unseeded randomness".into()),
        }
    }
}

/// The value cache key: function digest, callees, canonical SI arguments (bit exact), declared
/// reads and the runtime version.
pub fn dynamic_cache_key(
    call: &DynamicCall,
    args: &[(String, Quantity)],
    contract: &DynamicContract,
) -> String {
    let mut h = Sha256::new();
    let mut part = |b: &[u8]| {
        h.update((b.len() as u64).to_be_bytes());
        h.update(b);
    };
    part(b"runmat-cad/value/v1");
    part(call.function.as_bytes());
    part(contract.function_digest.as_bytes());
    for c in &contract.callees {
        part(c.as_bytes());
    }
    for (n, q) in args {
        part(n.as_bytes());
        part(&q.si.to_bits().to_be_bytes());
        part(&serde_json::to_vec(&q.dimension).expect("dimension serialises"));
    }
    for (path, digest) in &contract.reads {
        part(path.as_bytes());
        part(digest.as_bytes());
    }
    part(contract.runtime_version.as_bytes());
    format!("sha256:{}", hex::encode(h.finalize()))
}

/// A volatile value as evaluated (or as replayed from a release snapshot).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VolatileValue {
    pub function: String,
    pub value: Quantity,
    /// Replayed from `runmat.lock [values]` rather than evaluated.
    pub from_snapshot: bool,
}

/// Evaluates dynamic values. The real implementation runs `.m` functions in the value phase;
/// this crate only needs the seam.
pub trait DynamicValueEvaluator {
    fn evaluate(
        &self,
        call: &DynamicCall,
        arguments: &[(String, Quantity)],
    ) -> Result<Quantity, String>;
    /// The function's digest for cache keys (compiled digest plus callees and declared reads).
    fn digest(&self, call: &DynamicCall) -> String;

    /// The determinism contract. The default is a pure function with only its digest.
    fn contract(&self, call: &DynamicCall) -> DynamicContract {
        DynamicContract {
            function_digest: self.digest(call),
            callees: Vec::new(),
            reads: Vec::new(),
            runtime_version: String::new(),
            effect: None,
            randomness: Randomness::None,
        }
    }

    /// The raw result, which may be a bare double. The default wraps [`Self::evaluate`].
    fn evaluate_output(
        &self,
        call: &DynamicCall,
        arguments: &[(String, Quantity)],
    ) -> Result<DynamicOutput, String> {
        self.evaluate(call, arguments)
            .map(|value| DynamicOutput::Quantity { value })
    }

    /// A released snapshot of a volatile value, by cache key (`runmat.lock [values]`).
    fn snapshot(&self, _key: &str) -> Option<Quantity> {
        None
    }

    /// Resolve a kind-qualified reference to another declaration's value.
    fn reference(&self, _path: &[String]) -> Option<Result<Quantity, String>> {
        None
    }

    /// Read a plain value another file publishes. `None`: this evaluator has no project
    /// catalog (the read fails with `cad.xref.no_catalog`).
    fn published(&self, _xref: &ValueXref) -> Option<Result<Quantity, ValueError>> {
        None
    }
}

/// Rejects every dynamic value. Use until the value phase lands.
#[derive(Clone, Debug, Default)]
pub struct NoDynamicValues;

impl DynamicValueEvaluator for NoDynamicValues {
    fn evaluate(
        &self,
        call: &DynamicCall,
        _arguments: &[(String, Quantity)],
    ) -> Result<Quantity, String> {
        Err(format!(
            "no dynamic value evaluator is configured for `{}`",
            call.function
        ))
    }
    fn digest(&self, call: &DynamicCall) -> String {
        format!("unevaluated:{}", call.function)
    }
}

/// Evaluated inputs and lets for one configuration.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ValueEnv {
    pub inputs: BTreeMap<InputName, Quantity>,
    pub lets: BTreeMap<LetName, Quantity>,
    /// Values that failed, with the error. Their dependents fail with a cascade.
    pub failed: BTreeMap<String, ValueError>,
    /// Volatile values used, by cache key: what a release snapshots.
    #[serde(default)]
    pub volatile: RefCell<BTreeMap<String, VolatileValue>>,
}

pub struct Evaluator<'a> {
    pub units: &'a dyn UnitAuthority,
    pub dynamic: &'a dyn DynamicValueEvaluator,
}

impl<'a> Evaluator<'a> {
    pub fn eval(&self, expr: &ValueExpr, env: &ValueEnv) -> Result<Quantity, ValueError> {
        match expr {
            ValueExpr::Quantity { value, unit } => {
                let info = self
                    .units
                    .unit(unit)
                    .ok_or_else(|| ValueError::UnknownUnit { unit: unit.clone() })?;
                Ok(Quantity::new(value * info.scale, info.dimension))
            }
            ValueExpr::Number { value } => Ok(Quantity::dimensionless(*value)),
            ValueExpr::Input { name } => {
                env.inputs
                    .get(name)
                    .copied()
                    .ok_or_else(|| ValueError::UnresolvedInput {
                        name: name.0.clone(),
                    })
            }
            ValueExpr::Let { name } => {
                env.lets
                    .get(name)
                    .copied()
                    .ok_or_else(|| ValueError::UnresolvedLet {
                        name: name.0.clone(),
                    })
            }
            ValueExpr::Neg { operand } => {
                let v = self.eval(operand, env)?;
                Ok(Quantity::new(-v.si, v.dimension))
            }
            ValueExpr::Binary { op, lhs, rhs } => {
                let a = self.eval(lhs, env)?;
                let b = self.eval(rhs, env)?;
                match op {
                    BinaryOp::Add | BinaryOp::Sub => {
                        if a.dimension != b.dimension {
                            return Err(ValueError::UnitMismatch {
                                expected: a.dimension.to_string(),
                                found: b.dimension.to_string(),
                            });
                        }
                        let si = if *op == BinaryOp::Add {
                            a.si + b.si
                        } else {
                            a.si - b.si
                        };
                        Ok(Quantity::new(si, a.dimension))
                    }
                    BinaryOp::Mul => Ok(Quantity::new(
                        a.si * b.si,
                        a.dimension.multiply(b.dimension),
                    )),
                    BinaryOp::Div => {
                        Ok(Quantity::new(a.si / b.si, a.dimension.divide(b.dimension)))
                    }
                }
            }
            ValueExpr::Toleranced {
                nominal,
                minus,
                plus,
            } => {
                let n = self.eval(nominal, env)?;
                for bound in [minus, plus] {
                    let b = self.eval(bound, env)?;
                    if b.dimension != n.dimension && !b.dimension.is_dimensionless() {
                        return Err(ValueError::UnitMismatch {
                            expected: n.dimension.to_string(),
                            found: b.dimension.to_string(),
                        });
                    }
                }
                Ok(n)
            }
            ValueExpr::Dynamic { call } => {
                let mut args = Vec::with_capacity(call.arguments.len());
                for (name, arg) in &call.arguments {
                    args.push((name.clone(), self.eval(arg, env)?));
                }
                let contract = self.dynamic.contract(call);
                if call.volatile {
                    if !VALUE_CHOICES.allow_volatile {
                        return Err(ValueError::VolatileNotAllowed {
                            function: call.function.clone(),
                        });
                    }
                } else if let Err(reason) = contract.deterministic() {
                    return Err(ValueError::NotDeterministic {
                        function: call.function.clone(),
                        reason,
                    });
                }
                let key = dynamic_cache_key(call, &args, &contract);
                if call.volatile {
                    if let Some(v) = self.dynamic.snapshot(&key) {
                        env.volatile.borrow_mut().insert(
                            key,
                            VolatileValue {
                                function: call.function.clone(),
                                value: v,
                                from_snapshot: true,
                            },
                        );
                        return Ok(v);
                    }
                }
                let out = self
                    .dynamic
                    .evaluate_output(call, &args)
                    .map_err(|message| ValueError::DynamicFailed {
                        function: call.function.clone(),
                        message,
                    })?;
                let q = match out {
                    DynamicOutput::Quantity { value } => value,
                    DynamicOutput::Double { value } => Quantity::dimensionless(value),
                };
                if call.volatile {
                    env.volatile.borrow_mut().insert(
                        key,
                        VolatileValue {
                            function: call.function.clone(),
                            value: q,
                            from_snapshot: false,
                        },
                    );
                }
                Ok(q)
            }
            ValueExpr::Coerce { expr, unit } => {
                let info = self
                    .units
                    .unit(unit)
                    .ok_or_else(|| ValueError::UnknownUnit { unit: unit.clone() })?;
                let q = self.eval(expr, env)?;
                if q.dimension.is_dimensionless() {
                    Ok(Quantity::new(q.si * info.scale, info.dimension))
                } else if q.dimension == info.dimension {
                    Ok(q)
                } else {
                    Err(ValueError::CoerceDimensioned {
                        unit: unit.clone(),
                        found: q.dimension.to_string(),
                    })
                }
            }
            ValueExpr::Published(x) => match self.dynamic.published(x) {
                Some(r) => r,
                None => Err(ValueError::Xref {
                    xref: x.clone(),
                    problem: Box::new(XrefProblem::NoCatalog),
                }),
            },
            ValueExpr::Reference { path } => match self.dynamic.reference(path) {
                Some(Ok(q)) => Ok(q),
                Some(Err(message)) => Err(ValueError::DynamicFailed {
                    function: path.join("."),
                    message,
                }),
                None => Err(ValueError::UnresolvedReference {
                    path: path.join("."),
                }),
            },
        }
    }

    /// A bare `.m` call whose plain-number result meets a dimensioned target: the specific
    /// error with its fix-it, instead of a generic mismatch.
    fn unitless(
        &self,
        expr: &ValueExpr,
        env: &ValueEnv,
        expected: Dimension,
        si: &str,
    ) -> Option<ValueError> {
        if let ValueExpr::Dynamic { call } = expr {
            if expected.is_dimensionless() {
                return None;
            }
            let mut args = Vec::new();
            for (n, a) in &call.arguments {
                args.push((n.clone(), self.eval(a, env).ok()?));
            }
            if let Ok(DynamicOutput::Double { .. }) = self.dynamic.evaluate_output(call, &args) {
                return Some(ValueError::UnitlessResult {
                    function: call.function.clone(),
                    expected: expected.to_string(),
                    si: si.to_owned(),
                });
            }
        }
        None
    }

    /// Evaluate and check against an expected dimension.
    pub fn eval_as(
        &self,
        expr: &ValueExpr,
        env: &ValueEnv,
        expected: Dimension,
    ) -> Result<f64, ValueError> {
        let q = self.eval(expr, env)?;
        if q.dimension != expected {
            if let Some(e) = self.unitless(expr, env, expected, &expected.to_string()) {
                return Err(e);
            }
            return Err(ValueError::UnitMismatch {
                expected: expected.to_string(),
                found: q.dimension.to_string(),
            });
        }
        Ok(q.si)
    }

    /// Evaluate inputs (with overrides) and lets in dependency order. Failures are recorded in
    /// `ValueEnv::failed` and the rest still evaluate.
    pub fn evaluate_env(
        &self,
        inputs: &[InputDecl],
        lets: &[LetDecl],
        overrides: &BTreeMap<InputName, ValueExpr>,
    ) -> ValueEnv {
        let mut env = ValueEnv::default();
        // Inputs may refer to other inputs only through their defaults; evaluate in declaration
        // order and retry once to admit forward references.
        let mut pending: Vec<&InputDecl> = inputs.iter().collect();
        for _pass in 0..2 {
            let mut next = Vec::new();
            for decl in pending {
                let expr = overrides.get(&decl.name).unwrap_or(&decl.default);
                match self.eval_input(decl, expr, &env) {
                    Ok(q) => {
                        env.inputs.insert(decl.name.clone(), q);
                    }
                    Err(ValueError::UnresolvedInput { .. }) => next.push(decl),
                    Err(e) => {
                        env.failed.insert(format!("input.{}", decl.name), e);
                    }
                }
            }
            pending = next;
        }
        for decl in pending {
            let expr = overrides.get(&decl.name).unwrap_or(&decl.default);
            if let Err(e) = self.eval_input(decl, expr, &env) {
                env.failed.insert(format!("input.{}", decl.name), e);
            }
        }

        // Lets: topological order over let-to-let references.
        let by_name: BTreeMap<&LetName, &LetDecl> = lets.iter().map(|l| (&l.name, l)).collect();
        let mut state: BTreeMap<&LetName, u8> = BTreeMap::new(); // 1 = visiting, 2 = done
        fn visit<'b>(
            ev: &Evaluator<'_>,
            name: &'b LetName,
            by_name: &BTreeMap<&'b LetName, &'b LetDecl>,
            state: &mut BTreeMap<&'b LetName, u8>,
            stack: &mut Vec<String>,
            env: &mut ValueEnv,
        ) {
            match state.get(name) {
                Some(2) => return,
                Some(1) => {
                    let mut cycle = stack.clone();
                    cycle.push(name.0.clone());
                    env.failed
                        .insert(format!("let.{}", name), ValueError::Cycle { cycle });
                    return;
                }
                _ => {}
            }
            let Some(decl) = by_name.get(name) else {
                return;
            };
            state.insert(name, 1);
            stack.push(name.0.clone());
            let mut ins = BTreeSet::new();
            let mut ls = BTreeSet::new();
            decl.value.collect_names(&mut ins, &mut ls);
            for dep in &ls {
                if let Some((k, _)) = by_name.get_key_value(dep) {
                    visit(ev, k, by_name, state, stack, env);
                }
            }
            stack.pop();
            state.insert(name, 2);
            if env.failed.contains_key(&format!("let.{}", name)) {
                return;
            }
            match ev.eval(&decl.value, env) {
                Ok(q) => {
                    if let Some(kind) = &decl.kind {
                        match ev.units.kind(kind) {
                            Some(info) if info.dimension != q.dimension => {
                                env.failed.insert(
                                    format!("let.{}", name),
                                    ValueError::UnitMismatch {
                                        expected: info.dimension.to_string(),
                                        found: q.dimension.to_string(),
                                    },
                                );
                                return;
                            }
                            None => {
                                env.failed.insert(
                                    format!("let.{}", name),
                                    ValueError::UnknownKind {
                                        kind: kind.0.clone(),
                                    },
                                );
                                return;
                            }
                            _ => {}
                        }
                    }
                    env.lets.insert(decl.name.clone(), q);
                }
                Err(e) => {
                    env.failed.insert(format!("let.{}", name), e);
                }
            }
        }
        let mut stack = Vec::new();
        for decl in lets {
            visit(self, &decl.name, &by_name, &mut state, &mut stack, &mut env);
        }
        env
    }

    fn eval_input(
        &self,
        decl: &InputDecl,
        expr: &ValueExpr,
        env: &ValueEnv,
    ) -> Result<Quantity, ValueError> {
        let info = self
            .units
            .kind(&decl.kind)
            .ok_or_else(|| ValueError::UnknownKind {
                kind: decl.kind.0.clone(),
            })?;
        let q = self.eval(expr, env)?;
        if q.dimension != info.dimension {
            if let Some(e) = self.unitless(expr, env, info.dimension, &info.si_symbol) {
                return Err(e);
            }
            return Err(ValueError::UnitMismatch {
                expected: format!("{} ({})", info.id, info.si_symbol),
                found: q.dimension.to_string(),
            });
        }
        for (bound, is_min) in [(&decl.minimum, true), (&decl.maximum, false)] {
            if let Some(b) = bound {
                let b = self.eval(b, env)?;
                if (is_min && q.si < b.si) || (!is_min && q.si > b.si) {
                    return Err(ValueError::OutOfRange {
                        name: decl.name.0.clone(),
                        value: format!("{} {}", q.si, info.si_symbol),
                    });
                }
            }
        }
        Ok(q)
    }
}

/// One value of each [`ValueExpr`] form, as the Desktop's value editor reads and writes them
/// (`desktop/src/generated/value-expr-examples.json`). The Desktop's TypeScript mirror parses,
/// prints and round-trips each one, so the two sides can't drift: a new form fails to compile
/// here until it has an example, and the Desktop's test fails until it reads it.
pub fn value_expr_examples() -> Vec<(&'static str, ValueExpr)> {
    let examples = vec![
        ("8 mm", ValueExpr::mm(8.0)),
        ("2.5", ValueExpr::number(2.5)),
        ("= width", ValueExpr::input("width")),
        ("= let.half", ValueExpr::let_("half")),
        (
            "= -width",
            ValueExpr::Neg {
                operand: Box::new(ValueExpr::input("width")),
            },
        ),
        (
            "= width / 2",
            ValueExpr::binary(
                BinaryOp::Div,
                ValueExpr::input("width"),
                ValueExpr::number(2.0),
            ),
        ),
        (
            "10 kohm \u{b1} 1 %",
            ValueExpr::Toleranced {
                nominal: Box::new(ValueExpr::qty(10.0, "kohm")),
                minus: Box::new(ValueExpr::qty(1.0, "%")),
                plus: Box::new(ValueExpr::qty(1.0, "%")),
            },
        ),
        (
            "= airDensity(altitude = 10 km)",
            ValueExpr::Dynamic {
                call: DynamicCall {
                    function: "airDensity".into(),
                    arguments: vec![("altitude".into(), ValueExpr::qty(10.0, "km"))],
                    volatile: false,
                },
            },
        ),
        (
            "= rho(2) as kg/m^3",
            ValueExpr::coerce(
                ValueExpr::Dynamic {
                    call: DynamicCall {
                        function: "rho".into(),
                        arguments: vec![(String::new(), ValueExpr::number(2.0))],
                        volatile: false,
                    },
                },
                "kg/m^3",
            ),
        ),
        (
            "= feature.boss.depth",
            ValueExpr::Reference {
                path: vec!["feature".into(), "boss".into(), "depth".into()],
            },
        ),
        (
            "= part.bracket.mass",
            ValueExpr::published("bracket.part", "mass", XrefContext::Live),
        ),
    ];
    let mut forms: Vec<&str> = examples.iter().map(|(_, e)| value_form(e)).collect();
    forms.sort_unstable();
    forms.dedup();
    assert_eq!(forms.len(), 11, "one example per form");
    examples
}

/// The serde tag of a form; exhaustive, so a new form needs an example above.
fn value_form(expr: &ValueExpr) -> &'static str {
    match expr {
        ValueExpr::Quantity { .. } => "quantity",
        ValueExpr::Number { .. } => "number",
        ValueExpr::Input { .. } => "input",
        ValueExpr::Let { .. } => "let",
        ValueExpr::Neg { .. } => "neg",
        ValueExpr::Binary { .. } => "binary",
        ValueExpr::Toleranced { .. } => "toleranced",
        ValueExpr::Dynamic { .. } => "dynamic",
        ValueExpr::Coerce { .. } => "coerce",
        ValueExpr::Reference { .. } => "reference",
        ValueExpr::Published(_) => "published",
    }
}

#[cfg(test)]
mod value_expr_example_tests {
    use super::*;

    /// Writes or checks the Desktop's copy. Set `RUNMAT_VALUE_EXPR_OUT` to the Desktop's
    /// `src/generated/value-expr-examples.json` to write it.
    #[test]
    fn value_expr_examples_json_is_current() {
        let examples: Vec<serde_json::Value> = value_expr_examples()
            .into_iter()
            .map(|(text, expr)| {
                let json = serde_json::to_value(&expr).expect("value serialises");
                assert_eq!(
                    serde_json::from_value::<ValueExpr>(json.clone()).unwrap(),
                    expr
                );
                serde_json::json!({ "text": text, "form": value_form(&expr), "value": json })
            })
            .collect();
        let mut json = serde_json::to_string_pretty(&serde_json::json!({
            "format": "runmat.value-expr-examples",
            "binary_ops": ["add", "sub", "mul", "div"],
            "examples": examples,
        }))
        .unwrap();
        json.push('\n');
        if let Ok(out) = std::env::var("RUNMAT_VALUE_EXPR_OUT") {
            std::fs::write(out, &json).expect("write value-expr-examples.json");
            return;
        }
        let sibling = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../runmat-private-cad/desktop/src/generated/value-expr-examples.json");
        if let Ok(checked_in) = std::fs::read_to_string(&sibling) {
            assert!(
                checked_in == json,
                "{} is stale; regenerate with RUNMAT_VALUE_EXPR_OUT",
                sibling.display()
            );
        }
    }
}
