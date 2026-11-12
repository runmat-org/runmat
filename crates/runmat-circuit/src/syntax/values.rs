//! Values in `.circuit.layout` statements: exact lengths, points, rings and records read
//! straight from the shared grammar's tree, and their printed forms.
//!
//! Lengths are integer nanometres. A bare number is millimetres; any length may carry a unit
//! (`50 mil`, `0.5 mm`, `25 um`, `0.1 in`). Decimal text converts exactly: `0.000001` is 1 nm,
//! and a length always prints in millimetres with at most six decimals, so a board reads back
//! bit for bit. Angles are degrees.

use crate::geom::{Coord, Point};
use runmat_design_syntax::ast::{Expr, RecordEntry};
use runmat_design_syntax::json::{format_f64, is_bare_key, is_bare_word, quote};
use runmat_design_syntax::lexical::FeaKeyword;
use runmat_design_syntax::{FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};
use runmat_types::physical::{
    parse_unit_expression, unit_mismatch_message, DynamicDimension, FoundUnit,
};
use runmat_types::Span;

pub(crate) fn diagnostic(span: Span, message: impl Into<String>) -> FeaSyntaxDiagnostic {
    FeaSyntaxDiagnostic {
        kind: FeaSyntaxDiagnosticKind::InvalidExpression,
        span,
        message: message.into(),
    }
}

pub(crate) type Read<T> = Result<T, FeaSyntaxDiagnostic>;

/// `span` without the whitespace and comments at its ends.
pub(crate) fn trim_span(source: &str, span: Span) -> Span {
    let (mut start, mut end) = (span.start.min(source.len()), span.end.min(source.len()));
    loop {
        let rest = &source[start..end];
        let trimmed = rest.trim_start();
        start += rest.len() - trimmed.len();
        if trimmed.starts_with("//") {
            start += trimmed.find('\n').unwrap_or(trimmed.len());
        } else {
            break;
        }
    }
    while end > start && source.as_bytes()[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    Span { start, end }
}

/// A value as written.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum V {
    /// Number text with its sign (`-0.5`).
    Num(String),
    /// Number text and unit (`50`, `mil`).
    Qty(String, String),
    Str(String),
    /// A word or dotted reference (`GND`, `F.Cu`, `true`).
    Word(String),
    List(Vec<Val>),
    Rec(Vec<(String, Val)>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Val {
    pub v: V,
    pub span: Span,
}

/// Reads an expression of the shared grammar.
pub(crate) fn val(expr: &Expr) -> Read<Val> {
    // The untrimmed span is O(1); diagnostics are trimmed once at the end ([`trim_span`]).
    let span = expr.syntax().span();
    let v = match expr {
        Expr::Number(n) => V::Num(n.text()),
        Expr::String(s) => V::Str(s.value()),
        Expr::Reference(r) => V::Word(r.segments().join(".")),
        Expr::Group(g) => {
            return g
                .inner()
                .ok_or_else(|| diagnostic(span, "expected a value inside `( )`"))
                .and_then(|e| val(&e))
        }
        Expr::Unary(u) => {
            let operand = u
                .operand()
                .ok_or_else(|| diagnostic(span, "expected a number after the sign"))?;
            let inner = val(&operand)?;
            let neg = |t: String| {
                if u.is_negative() {
                    match t.strip_prefix('-') {
                        Some(rest) => rest.to_owned(),
                        None => format!("-{t}"),
                    }
                } else {
                    t
                }
            };
            match inner.v {
                V::Num(t) => V::Num(neg(t)),
                V::Qty(t, unit) => V::Qty(neg(t), unit),
                _ => return Err(diagnostic(span, "a sign goes before a number")),
            }
        }
        Expr::Quantity(q) => {
            let value = q
                .value()
                .ok_or_else(|| diagnostic(span, "expected a number before the unit"))?;
            match val(&value)?.v {
                V::Num(t) => V::Qty(t, q.unit()),
                _ => return Err(diagnostic(span, "a unit follows a number")),
            }
        }
        Expr::List(l) => V::List(l.items().iter().map(val).collect::<Read<_>>()?),
        Expr::Record(r) => V::Rec(
            r.entries()
                .iter()
                .map(entry)
                .collect::<Read<Vec<(String, Val)>>>()?,
        ),
        _ => {
            return Err(diagnostic(
                span,
                "expected a number, length, name, quoted text, list or record",
            ))
        }
    };
    Ok(Val { v, span })
}

fn entry(e: &RecordEntry) -> Read<(String, Val)> {
    let span = e.syntax().span();
    let key = e
        .name()
        .map(|n| n.text())
        .ok_or_else(|| diagnostic(span, "expected `<key> = <value>`"))?;
    let value = e
        .value()
        .ok_or_else(|| diagnostic(span, format!("`{key}` needs a value")))?;
    Ok((key, val(&value)?))
}

/// The subject a value reader names before [`Fields`] knows the field.
const UNNAMED: &str = "the value";

/// The shared wrong-unit diagnostic ([`runmat_types::physical::unit_mismatch_message`]) at the
/// value, for a unit that measures something other than `expected`.
fn unit_mismatch(v: &Val, expected: DynamicDimension, unit: &str) -> FeaSyntaxDiagnostic {
    let found = match parse_unit_expression(unit) {
        Ok(definition) => FoundUnit::Authored(unit, DynamicDimension::from_unit(definition)),
        Err(_) => {
            let kind = runmat_types::physical::unit_kind(expected.dimension(), expected.angle_power())
                .map(|row| (row.kind, row.suggestions.join(", ")))
                .unwrap_or(("value", String::new()));
            return diagnostic(
                v.span,
                format!("unknown unit `{unit}`; a {} takes {}", kind.0, kind.1),
            );
        }
    };
    FeaSyntaxDiagnostic {
        kind: FeaSyntaxDiagnosticKind::UnitMismatch,
        span: v.span,
        message: unit_mismatch_message(UNNAMED, expected, found).message,
    }
}

/// `d` with its unit-mismatch subject named after the field `key` reads.
pub(crate) fn named(mut d: FeaSyntaxDiagnostic, key: &str) -> FeaSyntaxDiagnostic {
    if d.kind == FeaSyntaxDiagnosticKind::UnitMismatch {
        if let Some(rest) = d.message.strip_prefix(UNNAMED) {
            d.message = format!("{}{rest}", key.replace('_', " "));
        }
    }
    d
}

/// nm per unit for a length unit of the one registry (`mm`, `mil`, `in`, `um`), or the shared
/// wrong-unit diagnostic. A board's lengths are whole nanometres, so the unit's size must be one.
fn length_scale(v: &Val, unit: &str) -> Read<i128> {
    match parse_unit_expression(unit) {
        Ok(definition) if DynamicDimension::from_unit(definition) == DynamicDimension::LENGTH => {
            let nm = definition.scale() * 1e9;
            let whole = nm.round();
            if whole >= 1.0 && (nm - whole).abs() <= 1e-6 * whole {
                Ok(whole as i128)
            } else {
                Err(diagnostic(
                    v.span,
                    format!("`{unit}` is finer than a board's 1 nm grid; use nm, um, mm or mil"),
                ))
            }
        }
        _ => Err(unit_mismatch(v, DynamicDimension::LENGTH, unit)),
    }
}

/// Decimal text (`-12.5`, `3`, `1e-3`) times `scale`, rounded to the nearest integer; exact
/// for every text with at most 30 significant digits.
pub(crate) fn scaled(text: &str, scale: i128) -> Option<i64> {
    let (negative, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let (digits, exponent) = match body.find(['e', 'E']) {
        Some(at) => (&body[..at], body[at + 1..].parse::<i32>().ok()?),
        None => (body, 0),
    };
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    if whole.is_empty() && fraction.is_empty() {
        return None;
    }
    let mut mantissa: i128 = 0;
    for c in whole.chars().chain(fraction.chars()) {
        let d = c.to_digit(10)? as i128;
        mantissa = mantissa.checked_mul(10)?.checked_add(d)?;
    }
    let exponent = exponent - fraction.len() as i32;
    let mut value = mantissa.checked_mul(scale)?;
    if exponent >= 0 {
        value = value.checked_mul(10i128.checked_pow(exponent as u32)?)?;
    } else {
        let divisor = 10i128.checked_pow((-exponent) as u32).unwrap_or(0);
        if divisor == 0 {
            value = 0;
        } else {
            value = (value + divisor / 2) / divisor;
        }
    }
    let value = if negative { -value } else { value };
    i64::try_from(value).ok()
}

pub(crate) fn len(v: &Val) -> Read<Coord> {
    let bad = || {
        diagnostic(
            v.span,
            "expected a length: a number in mm, or `50 mil`, `0.2 mm`",
        )
    };
    match &v.v {
        V::Num(t) => scaled(t, 1_000_000).ok_or_else(bad),
        V::Qty(t, unit) => scaled(t, length_scale(v, unit)?).ok_or_else(bad),
        _ => Err(bad()),
    }
}

pub(crate) fn num(v: &Val) -> Read<f64> {
    match &v.v {
        V::Num(t) => t
            .parse::<f64>()
            .map_err(|_| diagnostic(v.span, "expected a number")),
        _ => Err(diagnostic(v.span, "expected a number")),
    }
}

/// Degrees: a number, or a quantity in an angle unit (`45 deg`, `0.5 rad`).
pub(crate) fn angle(v: &Val) -> Read<f64> {
    match &v.v {
        V::Qty(t, unit) => {
            let value = t
                .parse::<f64>()
                .map_err(|_| diagnostic(v.span, "expected an angle in degrees"))?;
            match parse_unit_expression(unit) {
                Ok(definition)
                    if DynamicDimension::from_unit(definition) == DynamicDimension::ANGLE =>
                {
                    Ok((value * definition.scale()).to_degrees())
                }
                _ => Err(unit_mismatch(v, DynamicDimension::ANGLE, unit)),
            }
        }
        _ => num(v),
    }
}

pub(crate) fn whole(v: &Val) -> Read<i64> {
    match &v.v {
        V::Num(t) => t
            .parse::<i64>()
            .map_err(|_| diagnostic(v.span, "expected a whole number")),
        _ => Err(diagnostic(v.span, "expected a whole number")),
    }
}

pub(crate) fn boolean(v: &Val) -> Read<bool> {
    match &v.v {
        V::Word(w) if w == "true" => Ok(true),
        V::Word(w) if w == "false" => Ok(false),
        _ => Err(diagnostic(v.span, "expected `true` or `false`")),
    }
}

/// A name: a word, a dotted reference or quoted text.
pub(crate) fn text(v: &Val) -> Read<String> {
    match &v.v {
        V::Str(s) | V::Word(s) => Ok(s.clone()),
        _ => Err(diagnostic(v.span, "expected a name or quoted text")),
    }
}

pub(crate) fn list(v: &Val) -> Read<&[Val]> {
    match &v.v {
        V::List(items) => Ok(items),
        _ => Err(diagnostic(v.span, "expected a list `[ … ]`")),
    }
}

pub(crate) fn point(v: &Val) -> Read<Point> {
    match list(v)? {
        [x, y] => Ok(Point::new(len(x)?, len(y)?)),
        _ => Err(diagnostic(v.span, "a point is `[x, y]`")),
    }
}

pub(crate) fn ring(v: &Val) -> Read<Vec<Point>> {
    list(v)?.iter().map(point).collect()
}

pub(crate) fn rings(v: &Val) -> Read<Vec<Vec<Point>>> {
    list(v)?.iter().map(ring).collect()
}

pub(crate) fn texts(v: &Val) -> Read<Vec<String>> {
    list(v)?.iter().map(text).collect()
}

/// A record's fields, taken one by one; [`Fields::finish`] reports the ones nobody took.
pub(crate) struct Fields<'a> {
    entries: Vec<(&'a str, &'a Val, bool)>,
    span: Span,
    what: &'static str,
}

impl<'a> Fields<'a> {
    pub fn of(v: &'a Val, what: &'static str) -> Read<Fields<'a>> {
        let V::Rec(entries) = &v.v else {
            return Err(diagnostic(
                v.span,
                format!("expected a record `{{ … }}` for the {what}"),
            ));
        };
        let mut seen: Vec<&str> = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            if seen.contains(&key.as_str()) {
                return Err(diagnostic(value.span, format!("`{key}` is given twice")));
            }
            seen.push(key);
        }
        Ok(Fields {
            entries: entries
                .iter()
                .map(|(k, v)| (k.as_str(), v, false))
                .collect(),
            span: v.span,
            what,
        })
    }

    pub fn get(&mut self, key: &str) -> Option<&'a Val> {
        self.entries
            .iter_mut()
            .find(|(k, _, _)| *k == key)
            .map(|entry| {
                entry.2 = true;
                entry.1
            })
    }

    pub fn need(&mut self, key: &str) -> Read<&'a Val> {
        let (span, what) = (self.span, self.what);
        self.get(key)
            .ok_or_else(|| diagnostic(span, format!("the {what} needs `{key}`")))
    }

    pub fn opt<T>(&mut self, key: &str, read: impl Fn(&Val) -> Read<T>) -> Read<Option<T>> {
        self.get(key)
            .map(|v| read(v).map_err(|d| named(d, key)))
            .transpose()
    }

    pub fn or<T>(&mut self, key: &str, default: T, read: impl Fn(&Val) -> Read<T>) -> Read<T> {
        Ok(self.opt(key, read)?.unwrap_or(default))
    }

    /// The fields nobody took, in order.
    pub fn rest(&mut self) -> Vec<(&'a str, &'a Val)> {
        self.entries
            .iter_mut()
            .filter(|(_, _, taken)| !taken)
            .map(|entry| {
                entry.2 = true;
                (entry.0, entry.1)
            })
            .collect()
    }

    pub fn finish(self) -> Read<()> {
        match self.entries.iter().find(|(_, _, taken)| !taken) {
            Some((key, value, _)) => Err(diagnostic(
                value.span,
                format!("the {} has no `{key}`", self.what),
            )),
            None => Ok(()),
        }
    }
}

/* ------------------------------------------------------------------------------------------------
 * Printing
 * --------------------------------------------------------------------------------------------- */

/// A length in millimetres, exact.
pub(crate) fn mm_text(v: Coord) -> String {
    let sign = if v < 0 { "-" } else { "" };
    let a = v.unsigned_abs();
    let (int, frac) = (a / 1_000_000, a % 1_000_000);
    if frac == 0 {
        format!("{sign}{int}")
    } else {
        let digits = format!("{frac:06}");
        format!("{sign}{int}.{}", digits.trim_end_matches('0'))
    }
}

pub(crate) fn pt_text(p: Point) -> String {
    format!("[{}, {}]", mm_text(p.x), mm_text(p.y))
}

pub(crate) fn ring_text(r: &[Point]) -> String {
    let mut out = String::with_capacity(r.len() * 16 + 2);
    out.push('[');
    for (i, p) in r.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&pt_text(*p));
    }
    out.push(']');
    out
}

pub(crate) fn rings_text(rs: &[Vec<Point>]) -> String {
    let inner: Vec<String> = rs.iter().map(|r| ring_text(r)).collect();
    format!("[{}]", inner.join(", "))
}

pub(crate) fn f64_text(v: f64) -> String {
    format_f64(v)
}

/// A name as a word when it reads back as one, else quoted.
pub(crate) fn name_text(text: &str) -> String {
    if is_simple_word(text) || is_bare_word(text) {
        text.to_owned()
    } else {
        quote(text)
    }
}

/// A record key, bare when it can be.
pub(crate) fn key_text(text: &str) -> String {
    if is_simple_word(text) || is_bare_key(text) {
        text.to_owned()
    } else {
        quote(text)
    }
}

/// Words of the lexer that do not read back as a name in value position.
const RESERVED: &[&str] = &[
    "true",
    "false",
    "none",
    "as",
    "volatile",
    "on",
    "of",
    "from",
    "over",
    "at",
    "in",
    "to",
    "for",
    "use",
    "import",
    "let",
    "input",
    "property",
    "and",
    "or",
    "not",
    "if",
    "else",
    "when",
    "check",
    "publish",
    "link",
    "solved",
    "constrain",
    "runmat",
    "std",
];

/// The fast check: `[A-Za-z_][A-Za-z0-9_]*` and not a word the lexer reserves. A `false`
/// answer falls back to the lexer's own check.
fn is_simple_word(text: &str) -> bool {
    let mut chars = text.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !RESERVED.contains(&text)
        && FeaKeyword::from_source(text).is_none()
}

/// A layer or other dotted name (`F.Cu`, `In1.Cu`) as a reference when every segment is a
/// word, else quoted.
pub(crate) fn dotted_text(text: &str) -> String {
    if !text.is_empty()
        && text
            .split('.')
            .all(|s| is_simple_word(s) || is_bare_word(s))
    {
        text.to_owned()
    } else {
        quote(text)
    }
}

/// `{ k = v, … }` built field by field.
#[derive(Default)]
pub(crate) struct Rec(Vec<(String, String)>);

impl Rec {
    pub fn new() -> Rec {
        Rec(Vec::new())
    }
    pub fn put(&mut self, key: &str, value: impl Into<String>) -> &mut Rec {
        self.0.push((key.to_owned(), value.into()));
        self
    }
    pub fn len(&mut self, key: &str, v: Coord) -> &mut Rec {
        self.put(key, mm_text(v))
    }
    pub fn pt(&mut self, key: &str, p: Point) -> &mut Rec {
        self.put(key, pt_text(p))
    }
    pub fn flag(&mut self, key: &str, on: bool) -> &mut Rec {
        if on {
            self.put(key, "true");
        }
        self
    }
    pub fn text(&self) -> String {
        if self.0.is_empty() {
            return "{}".into();
        }
        let mut out = String::from("{ ");
        for (i, (k, v)) in self.0.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&key_text(k));
            out.push_str(" = ");
            out.push_str(v);
        }
        out.push_str(" }");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_lengths_are_exact() {
        assert_eq!(scaled("0.000001", 1_000_000), Some(1));
        assert_eq!(scaled("-12.5", 1_000_000), Some(-12_500_000));
        assert_eq!(scaled("1e-3", 1_000_000), Some(1_000));
        assert_eq!(scaled("50", 25_400), Some(1_270_000));
        assert_eq!(scaled("0.0000004", 1_000_000), Some(0));
        assert_eq!(scaled("0.0000005", 1_000_000), Some(1));
        assert_eq!(scaled("x", 1), None);
        for v in [0, 1, -1, 999_999, 1_000_000, -123_456_789, 274_877_906_943] {
            assert_eq!(scaled(&mm_text(v), 1_000_000), Some(v), "{v}");
        }
        assert_eq!(mm_text(1_250_000), "1.25");
        assert_eq!(mm_text(-500), "-0.0005");
    }
}
