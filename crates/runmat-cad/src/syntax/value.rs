//! [`ValueExpr`] in design-language text: the CAD dialects' value forms.
//!
//! | Text | Value |
//! |---|---|
//! | `80 mm`, `2.5` | a quantity, a dimensionless number |
//! | `input.width`, `let.half` (or `half` inside an expression) | an input, a `let` |
//! | `-x`, `a + b`, `a * b` | arithmetic |
//! | `10 mm ± 0.1 mm`, `10 mm ± [0.02 mm, 0 mm]` | a tolerance, symmetric or `[plus, minus]` |
//! | `airDensity(altitude = 10 km)`, `volatile sensor(2)` | a dynamic value |
//! | `rho(2) as kg/m^3` | a coerced bare double |
//! | `part.bracket.mass`, `part.bracket.mass @ "R3"` | a value another file publishes |
//! | `feature.boss.depth` | a reference to another declaration's value |
//!
//! The printer writes each value so that it reads back as the same [`ValueExpr`]; a value it
//! cannot write that way (a reference whose path looks like an input) is left to the generic
//! record form, which also reads back exactly.

use super::refs;
use crate::ids::{FilePath, InputName, LetName, PublishName};
use crate::model::datum::{AxisRef, PlaneRef, PointRef};
use crate::refs::reference::XrefContext;
use crate::refs::Reference;
use crate::value::{BinaryOp, DynamicCall, ValueExpr, ValueXref};
use runmat_design_syntax::ast::{self, Argument, Expr};
use runmat_design_syntax::document::EntryForms;
use runmat_design_syntax::json::{
    format_f64, is_bare_key, is_bare_word, is_identifier, quote, Json, LowerError, ValueSyntax,
};
use runmat_design_syntax::lexical::FeaTokenKind;

/// File kinds whose published values a value can read (`part.bracket.mass`).
pub const PUBLISHING_KINDS: &[&str] = &["part", "assembly", "circuit", "drawing"];

/// The CAD dialects' value forms.
pub struct CadValues;

impl ValueSyntax for CadValues {
    fn lower(&self, expr: &Expr) -> Option<Result<Json, LowerError>> {
        let value = match expr {
            Expr::Number(_)
            | Expr::String(_)
            | Expr::List(_)
            | Expr::Record(_)
            | Expr::Group(_) => return None,
            Expr::Unary(unary)
                if unary.is_negative() && matches!(unary.operand(), Some(Expr::Number(_))) =>
            {
                return None
            }
            Expr::Reference(reference) if reference.segments().len() == 1 => return None,
            Expr::PrefixApplication(application) if !is_volatile(application) => return None,
            Expr::Reference(_) | Expr::Call(_) => {
                if let Some(reference) = refs::lower_reference(expr) {
                    return Some(reference.and_then(|reference| {
                        Json::from_serialize(&reference)
                            .map_err(|error| LowerError::new(expr.span(), error.to_string()))
                    }));
                }
                if let Some(datum) = refs::lower_datum(expr, &lower_value) {
                    return Some(datum);
                }
                lower_value(expr)
            }
            _ => lower_value(expr),
        };
        Some(value.and_then(|value| {
            Json::from_serialize(&value)
                .map_err(|error| LowerError::new(expr.span(), error.to_string()))
        }))
    }

    fn print(&self, value: &Json) -> Option<String> {
        let Json::Object(entries) = value else {
            return None;
        };
        let (first, _) = entries.first()?;
        match first.as_str() {
            "ref" => refs::print_reference(&exactly::<Reference>(value)?),
            "on" => refs::print_plane(&exactly::<PlaneRef>(value)?),
            "along" => refs::print_axis(&exactly::<AxisRef>(value)?),
            "point" => refs::print_point(&exactly::<PointRef>(value)?, &print_value),
            "sketch" if entries.len() == 1 => match &entries[0].1 {
                Json::String(name) if is_identifier(name) => Some(format!("sketch.{name}")),
                _ => None,
            },
            _ => {
                value.get("form")?;
                print_value(&exactly::<ValueExpr>(value)?)
            }
        }
    }

    fn tag_field(&self, scope: Option<&str>, key: &str) -> Option<&'static str> {
        super::forms::tag_field(scope, key)
    }

    fn lower_keyed(&self, key: &str, expr: &Expr) -> Option<Result<Json, LowerError>> {
        if !PLANE_FIELDS.contains(&key) {
            return None;
        }
        let reference = refs::lower_reference(expr)?;
        Some(reference.and_then(|face| {
            Json::from_serialize(&PlaneRef::Face { face })
                .map_err(|error| LowerError::new(expr.span(), error.to_string()))
        }))
    }

    fn print_keyed(&self, key: &str, value: &Json) -> Option<String> {
        if !PLANE_FIELDS.contains(&key) {
            return None;
        }
        match exactly::<PlaneRef>(value)? {
            PlaneRef::Face { face } => refs::print_reference(&face),
            _ => None,
        }
    }

    fn entry_forms(&self) -> Option<&dyn EntryForms> {
        Some(&super::forms::CadForms)
    }
}

/// Fields that hold only planes, so a face reference there is the plane of that face
/// (`neutral = feature.body.face.end`).
pub const PLANE_FIELDS: &[&str] = &["neutral", "tool", "parting"];

/// `value` as a `T` when it is exactly `T`'s serialised form.
pub fn exactly<T: serde::Serialize + serde::de::DeserializeOwned>(value: &Json) -> Option<T> {
    let decoded: T = value.decode().ok()?;
    (Json::from_serialize(&decoded).ok()?.to_value() == value.to_value()).then_some(decoded)
}

fn is_volatile(application: &ast::PrefixApplication) -> bool {
    matches!(application.callee(), Some(Expr::Reference(r)) if r.segments() == ["volatile"])
}

fn error(expr: &Expr, message: impl Into<String>) -> LowerError {
    LowerError::new(expr.span(), message)
}

fn number(expr: &Expr, text: &str) -> Result<f64, LowerError> {
    text.parse::<f64>()
        .map_err(|_| error(expr, format!("`{text}` is not a number")))
}

/// Lowers an expression in a value position.
pub fn lower_value(expr: &Expr) -> Result<ValueExpr, LowerError> {
    Ok(match expr {
        Expr::Number(literal) => ValueExpr::Number {
            value: number(expr, &literal.text())?,
        },
        Expr::Quantity(quantity) => {
            let unit = quantity.unit();
            match quantity.value() {
                Some(Expr::Number(literal)) => ValueExpr::Quantity {
                    value: number(expr, &literal.text())?,
                    unit,
                },
                _ => {
                    return Err(error(
                        expr,
                        "a vector quantity is not one value; give each component its unit",
                    ))
                }
            }
        }
        Expr::Unary(unary) => {
            let operand = unary
                .operand()
                .ok_or_else(|| error(expr, "a sign needs a value"))?;
            if !unary.is_negative() {
                return lower_value(&operand);
            }
            match &operand {
                Expr::Number(literal) => ValueExpr::Number {
                    value: -number(expr, &literal.text())?,
                },
                Expr::Quantity(quantity) if matches!(quantity.value(), Some(Expr::Number(_))) => {
                    match lower_value(&operand)? {
                        ValueExpr::Quantity { value, unit } => ValueExpr::Quantity {
                            value: -value,
                            unit,
                        },
                        other => other,
                    }
                }
                _ => ValueExpr::Neg {
                    operand: Box::new(lower_value(&operand)?),
                },
            }
        }
        Expr::Group(group) => {
            lower_value(&group.inner().ok_or_else(|| error(expr, "an empty group"))?)?
        }
        Expr::Binary(binary) => {
            let op = match binary.operator() {
                Some(FeaTokenKind::Plus) => BinaryOp::Add,
                Some(FeaTokenKind::Minus) => BinaryOp::Sub,
                Some(FeaTokenKind::Star) => BinaryOp::Mul,
                Some(FeaTokenKind::Slash) => BinaryOp::Div,
                _ => return Err(error(expr, "a value takes `+`, `-`, `*` and `/`")),
            };
            let lhs = binary.lhs().ok_or_else(|| error(expr, "missing operand"))?;
            let rhs = binary.rhs().ok_or_else(|| error(expr, "missing operand"))?;
            ValueExpr::Binary {
                op,
                lhs: Box::new(lower_value(&lhs)?),
                rhs: Box::new(lower_value(&rhs)?),
            }
        }
        Expr::Toleranced(toleranced) => {
            let nominal = toleranced
                .nominal()
                .ok_or_else(|| error(expr, "missing nominal"))?;
            let tolerance = toleranced
                .tolerance()
                .ok_or_else(|| error(expr, "missing tolerance"))?;
            let (plus, minus) = match &tolerance {
                Expr::List(list) => match list.items().as_slice() {
                    [plus, minus] => (lower_value(plus)?, lower_value(minus)?),
                    _ => {
                        return Err(error(
                            &tolerance,
                            "an asymmetric tolerance is `[plus, minus]`",
                        ))
                    }
                },
                other => {
                    let both = lower_value(other)?;
                    (both.clone(), both)
                }
            };
            ValueExpr::Toleranced {
                nominal: Box::new(lower_value(&nominal)?),
                minus: Box::new(minus),
                plus: Box::new(plus),
            }
        }
        Expr::Coerce(coerce) => ValueExpr::Coerce {
            expr: Box::new(lower_value(
                &coerce.expr().ok_or_else(|| error(expr, "missing value"))?,
            )?),
            unit: coerce.unit(),
        },
        Expr::Pinned(pinned) => {
            let inner = pinned.expr().ok_or_else(|| error(expr, "missing value"))?;
            let revision = pinned
                .revision()
                .ok_or_else(|| error(expr, "a pin is `@ \"<revision>\"`"))?;
            match &inner {
                Expr::Reference(reference) => match published(&reference.segments()) {
                    Some(xref) => ValueExpr::Published(Box::new(ValueXref {
                        context: XrefContext::Pinned {
                            revision: crate::ids::RevisionId::new(revision),
                        },
                        ..xref
                    })),
                    None => {
                        return Err(error(
                            expr,
                            "only another file's published value takes a pin",
                        ))
                    }
                },
                _ => {
                    return Err(error(
                        expr,
                        "only another file's published value takes a pin",
                    ))
                }
            }
        }
        Expr::Call(call) => ValueExpr::Dynamic {
            call: dynamic_call(expr, call, false)?,
        },
        Expr::PrefixApplication(application) if is_volatile(application) => {
            match application.argument() {
                Some(Expr::Call(call)) => ValueExpr::Dynamic {
                    call: dynamic_call(expr, &call, true)?,
                },
                _ => return Err(error(expr, "`volatile` marks a function call")),
            }
        }
        Expr::Reference(reference) => {
            let segments = reference.segments();
            match segments.as_slice() {
                [name] => ValueExpr::Let {
                    name: LetName::new(name),
                },
                [head, name] if head == "input" => ValueExpr::Input {
                    name: InputName::new(name),
                },
                [head, name] if head == "let" => ValueExpr::Let {
                    name: LetName::new(name),
                },
                _ => match published(&segments) {
                    Some(xref) => ValueExpr::Published(Box::new(xref)),
                    None => ValueExpr::Reference { path: segments },
                },
            }
        }
        _ => return Err(error(expr, "this expression is not a value")),
    })
}

/// `part.bracket.mass` or `part.bracket.publish.mass`: a live read of `bracket.part`.
fn published(segments: &[String]) -> Option<ValueXref> {
    match segments {
        [kind, stem, name] | [kind, stem, _, name]
            if PUBLISHING_KINDS.contains(&kind.as_str())
                && (segments.len() == 3 || segments[2] == "publish") =>
        {
            Some(ValueXref {
                file: FilePath::new(format!("{stem}.{kind}")),
                name: PublishName::new(name),
                context: XrefContext::Live,
            })
        }
        _ => None,
    }
}

fn dynamic_call(expr: &Expr, call: &ast::Call, volatile: bool) -> Result<DynamicCall, LowerError> {
    let function = match call.callee() {
        Some(Expr::Reference(reference)) if reference.segments().len() == 1 => {
            reference.segments().remove(0)
        }
        _ => return Err(error(expr, "a value calls a function by its name")),
    };
    let mut arguments = Vec::new();
    for argument in call.arguments() {
        match argument {
            Argument::Positional(value) => arguments.push((String::new(), lower_value(&value)?)),
            Argument::Named(named) => {
                let name = named.name().map(|n| n.text()).unwrap_or_default();
                let value = named
                    .value()
                    .ok_or_else(|| error(expr, format!("`{name}` needs a value")))?;
                arguments.push((name, lower_value(&value)?));
            }
        }
    }
    Ok(DynamicCall {
        function,
        arguments,
        volatile,
    })
}

// ------------------------------------------------------------------------------------ printing

/// Precedence of a printed form (higher binds tighter), matching the parser: `as` and `@`
/// 0, `±` 2, `+ -` 3, `* /` 4, a sign 7, atoms 8.
fn precedence(expr: &ValueExpr) -> u8 {
    match expr {
        ValueExpr::Coerce { .. } => 0,
        ValueExpr::Published(xref) if matches!(xref.context, XrefContext::Pinned { .. }) => 0,
        ValueExpr::Toleranced { .. } => 2,
        ValueExpr::Binary {
            op: BinaryOp::Add | BinaryOp::Sub,
            ..
        } => 3,
        ValueExpr::Binary { .. } => 4,
        ValueExpr::Neg { .. } => 7,
        ValueExpr::Number { value } | ValueExpr::Quantity { value, .. }
            if value.is_sign_negative() =>
        {
            7
        }
        _ => 8,
    }
}

/// Prints a value, or `None` when it has no text form that reads back as itself.
pub fn print_value(expr: &ValueExpr) -> Option<String> {
    print_at(expr, 0)
}

fn print_at(expr: &ValueExpr, context: u8) -> Option<String> {
    let text = print_bare(expr)?;
    Some(if precedence(expr) < context {
        format!("({text})")
    } else {
        text
    })
}

fn unit_ok(unit: &str) -> bool {
    // The unit must read back as the same compact text.
    let source = format!("part p {{\n    x = 1 {unit}\n}}\n");
    let syntax = runmat_design_syntax::parse_source(runmat_design_syntax::Dialect::Part, &source);
    if !syntax.diagnostics.is_empty() {
        return false;
    }
    let found = syntax
        .syntax()
        .descendants()
        .into_iter()
        .find(|node| node.kind() == runmat_design_syntax::tree::NodeKind::Unit)
        .map(|node| ast::compact_text(&node));
    found.as_deref() == Some(unit)
}

fn print_bare(expr: &ValueExpr) -> Option<String> {
    Some(match expr {
        ValueExpr::Quantity { value, unit } => {
            if !value.is_finite() || !unit_ok(unit) {
                return None;
            }
            format!("{} {unit}", format_f64(*value))
        }
        ValueExpr::Number { value } => {
            if !value.is_finite() {
                return None;
            }
            format_f64(*value)
        }
        ValueExpr::Input { name } if is_bare_key(&name.0) => format!("input.{}", name.0),
        ValueExpr::Let { name } if is_bare_key(&name.0) => format!("let.{}", name.0),
        ValueExpr::Input { .. } | ValueExpr::Let { .. } => return None,
        ValueExpr::Neg { operand } => match operand.as_ref() {
            // A sign on a literal would read back as a negative literal.
            ValueExpr::Number { .. } | ValueExpr::Quantity { .. } => {
                format!("-({})", print_at(operand, 0)?)
            }
            _ => format!("-{}", print_at(operand, 7)?),
        },
        ValueExpr::Binary { op, lhs, rhs } => {
            let (symbol, own) = match op {
                BinaryOp::Add => ("+", 3),
                BinaryOp::Sub => ("-", 3),
                BinaryOp::Mul => ("*", 4),
                BinaryOp::Div => ("/", 4),
            };
            let mut left = print_at(lhs, own)?;
            // `5 mm * 2` reads as the unit `mm*2`: a quantity before `*` or `/` takes parentheses.
            if own == 4
                && matches!(lhs.as_ref(), ValueExpr::Quantity { .. })
                && !left.starts_with('(')
            {
                left = format!("({left})");
            }
            format!("{left} {symbol} {}", print_at(rhs, own + 1)?)
        }
        ValueExpr::Toleranced {
            nominal,
            minus,
            plus,
        } => {
            let nominal = print_at(nominal, 2)?;
            if minus == plus {
                format!("{nominal} ± {}", print_at(plus, 3)?)
            } else {
                format!(
                    "{nominal} ± [{}, {}]",
                    print_at(plus, 0)?,
                    print_at(minus, 0)?
                )
            }
        }
        ValueExpr::Dynamic { call } => {
            if !is_bare_word(&call.function)
                || refs::REFERENCE_FUNCTIONS.contains(&call.function.as_str())
            {
                return None;
            }
            let mut arguments = Vec::new();
            for (name, value) in &call.arguments {
                let value = print_at(value, 0)?;
                arguments.push(if name.is_empty() {
                    value
                } else if is_bare_key(name) {
                    format!("{name} = {value}")
                } else {
                    return None;
                });
            }
            format!(
                "{}{}({})",
                if call.volatile { "volatile " } else { "" },
                call.function,
                arguments.join(", ")
            )
        }
        ValueExpr::Coerce { expr, unit } => {
            if !unit_ok(unit) {
                return None;
            }
            format!("{} as {unit}", print_at(expr, 1)?)
        }
        ValueExpr::Reference { path } => {
            let reads_back = match path.as_slice() {
                [] | [_] => false,
                [head, _] if head == "input" || head == "let" => false,
                _ => published(path).is_none() && !refs::is_reserved_path(path),
            };
            if !reads_back || !path.iter().all(|segment| is_bare_key(segment)) {
                return None;
            }
            path.join(".")
        }
        ValueExpr::Published(xref) => {
            let (stem, kind) = xref.file.0.rsplit_once('.')?;
            if !PUBLISHING_KINDS.contains(&kind) || !is_bare_key(stem) || !is_bare_key(&xref.name.0)
            {
                return None;
            }
            let pin = match &xref.context {
                XrefContext::Live => String::new(),
                XrefContext::Pinned { revision } => format!(" @ {}", quote(&revision.0)),
            };
            format!("{kind}.{stem}.{}{pin}", xref.name.0)
        }
    })
}
