//! The value model used by inputs, lets, feature parameters and configuration overrides.
//!
//! Follows `research/12-values-and-resolution.md` design A in outline: a value is static (a
//! quantity literal, a reference to an input or let, an expression, a tolerance) or dynamic (a
//! pure function call evaluated through [`DynamicValueEvaluator`]). Units go through the
//! [`UnitAuthority`] seam. Open questions bookmarked in that report (tables, ranges, `.m` typing)
//! are not modelled yet.

use crate::ids::{InputName, LetName};
use crate::units::{Dimension, QuantityKindId, UnitAuthority};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
                self.dynamic
                    .evaluate(call, &args)
                    .map_err(|message| ValueError::DynamicFailed {
                        function: call.function.clone(),
                        message,
                    })
            }
        }
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
