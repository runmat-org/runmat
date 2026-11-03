//! Selector syntax, specificity and matching.
//!
//! # Grammar
//!
//! ```text
//! expr      := inter ( ',' inter )*                    union (lowest precedence)
//! inter     := complex ( ( '&' | ' - ' ) complex )*     intersection / difference, left-assoc
//! complex   := compound ( combinator compound )*
//! combinator:= whitespace (descendant) | '>' (child)
//! compound  := ( type | '*' )? simple*                  at least one part
//! simple    := '#' name | '.' ident | attr | pseudo
//! attr      := '[' ident ( op value ( 'i' )? )? ']'
//! op        := '=' | '!=' | '^=' | '$=' | '*=' | '~=' | '<' | '<=' | '>' | '>='
//! value     := "quoted" | number unit? | bare-word
//! pseudo    := ':' ident ( '(' args ')' )?
//!              built-ins: :not(expr) :is(expr) :where(expr) :has(expr) :has(> expr)
//! ```
//!
//! `-` is the difference operator only with whitespace on both sides, so identifiers keep
//! their dashes (`:net-class`). `( expr )` groups. Names after `#` may be quoted to carry any
//! character (`#"/USB/D+"`).
//!
//! # Specificity
//!
//! A triple `(ids, classes, types)` compared lexicographically, as in CSS: `#id` counts in
//! the first place; `.class`, `[attr]` and host pseudo-classes in the second; type selectors
//! in the third. `:is`, `:not` and `:has` count as their most specific argument; `:where`
//! counts zero. For a union, an item gets the specificity of the most specific branch it
//! matches; an intersection `a & b` sums both sides; a difference `a - b` takes `a`'s.

use crate::graph::{AttrType, ItemGraph, Schema};
use crate::value::{parse_quantity, AttrValue, Quantity, UnitResolver};
use std::cmp::Ordering;
use std::fmt;

/// Byte range in the selector text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectorError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for SelectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}..{}: {}",
            self.span.start, self.span.end, self.message
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Specificity {
    pub ids: u32,
    pub classes: u32,
    pub types: u32,
}

impl Specificity {
    pub const ZERO: Specificity = Specificity {
        ids: 0,
        classes: 0,
        types: 0,
    };
}

impl std::ops::Add for Specificity {
    type Output = Specificity;
    fn add(self, o: Specificity) -> Specificity {
        Specificity {
            ids: self.ids + o.ids,
            classes: self.classes + o.classes,
            types: self.types + o.types,
        }
    }
}

impl fmt::Display for Specificity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{},{})", self.ids, self.classes, self.types)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttrOp {
    Eq,
    Ne,
    Prefix,
    Suffix,
    Contains,
    Word,
    Lt,
    Le,
    Gt,
    Ge,
}

impl AttrOp {
    fn text(self) -> &'static str {
        match self {
            AttrOp::Eq => "=",
            AttrOp::Ne => "!=",
            AttrOp::Prefix => "^=",
            AttrOp::Suffix => "$=",
            AttrOp::Contains => "*=",
            AttrOp::Word => "~=",
            AttrOp::Lt => "<",
            AttrOp::Le => "<=",
            AttrOp::Gt => ">",
            AttrOp::Ge => ">=",
        }
    }
    fn is_numeric(self) -> bool {
        matches!(self, AttrOp::Lt | AttrOp::Le | AttrOp::Gt | AttrOp::Ge)
    }
}

/// A literal on the right of an attribute operator: its source text, and its quantity when the
/// text is a number with a known unit.
#[derive(Clone, Debug, PartialEq)]
pub struct Literal {
    pub text: String,
    pub quantity: Option<Quantity>,
    pub quoted: bool,
}

/// An argument to a host pseudo-class: the raw text (`power`, `F.Cu`, `outer`) and its
/// quantity if it parses as one.
#[derive(Clone, Debug, PartialEq)]
pub struct PseudoArg {
    pub text: String,
    pub quantity: Option<Quantity>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combinator {
    Descendant,
    Child,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Simple {
    Id(String),
    Class(String),
    Attr {
        name: String,
        test: Option<(AttrOp, Literal, bool)>,
        span: Span,
    },
    Pseudo {
        name: String,
        args: Vec<PseudoArg>,
        span: Span,
    },
    Not(Box<Expr>),
    Is(Box<Expr>),
    Where(Box<Expr>),
    /// `:has(x)` (some descendant matches) or `:has(> x)` (some child matches).
    Has(Box<Expr>, Combinator),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Compound {
    /// `None` is the universal selector.
    pub kind: Option<String>,
    pub kind_span: Span,
    pub simples: Vec<Simple>,
}

/// A chain of compounds. `parts[0]` is the leftmost; `combinators[i]` joins `parts[i]` and
/// `parts[i + 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Complex {
    pub parts: Vec<Compound>,
    pub combinators: Vec<Combinator>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Complex(Complex),
    Union(Vec<Expr>),
    And(Box<Expr>, Box<Expr>),
    Diff(Box<Expr>, Box<Expr>),
}

/// A parsed selector with its source text.
#[derive(Clone, Debug, PartialEq)]
pub struct Selector {
    pub text: String,
    pub expr: Expr,
}

impl Selector {
    pub fn parse(text: &str, units: &dyn UnitResolver) -> Result<Selector, SelectorError> {
        let mut p = Parser {
            s: text,
            b: text.as_bytes(),
            i: 0,
            units,
        };
        p.ws();
        let expr = p.expr()?;
        p.ws();
        if p.i < p.b.len() {
            return Err(p.err(p.i, p.b.len(), "unexpected text after selector"));
        }
        Ok(Selector {
            text: text.to_string(),
            expr,
        })
    }

    /// Matches `id`, returning the specificity it matched with.
    pub fn matches<G: ItemGraph>(&self, g: &G, id: G::Id) -> Option<Specificity> {
        match_expr(&self.expr, g, id)
    }

    /// Every matching item with its specificity, in id order.
    pub fn select<G: ItemGraph>(&self, g: &G) -> Vec<(G::Id, Specificity)> {
        let mut items = g.items();
        items.sort();
        items.dedup();
        items
            .into_iter()
            .filter_map(|i| self.matches(g, i).map(|s| (i, s)))
            .collect()
    }

    /// The kinds an item must have to match: `Some(kinds)` when every branch names a type,
    /// used to bucket rules; `None` when any branch can match any kind.
    pub fn required_kinds(&self) -> Option<Vec<String>> {
        let mut out = Vec::new();
        if required_kinds(&self.expr, &mut out) {
            out.sort();
            out.dedup();
            Some(out)
        } else {
            None
        }
    }

    /// Whether matching depends on descendants (`:has`), so a change to an item can change
    /// its ancestors' matches.
    pub fn uses_has(&self) -> bool {
        uses_has(&self.expr)
    }

    /// Checks names against a schema. Returns every problem, not only the first.
    pub fn validate(&self, schema: &Schema) -> Vec<SelectorError> {
        let mut out = Vec::new();
        validate_expr(&self.expr, schema, &mut out);
        out
    }

    /// The highest specificity any match can have.
    pub fn max_specificity(&self) -> Specificity {
        static_spec(&self.expr)
    }
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

// ---------------------------------------------------------------- parser

struct Parser<'a> {
    s: &'a str,
    b: &'a [u8],
    i: usize,
    units: &'a dyn UnitResolver,
}

fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c >= 0x80
}

fn is_name_byte(c: u8) -> bool {
    is_ident_byte(c) || matches!(c, b'+' | b'/' | b'$' | b'~' | b'!' | b'@')
}

impl<'a> Parser<'a> {
    fn err(&self, start: usize, end: usize, msg: &str) -> SelectorError {
        SelectorError {
            span: Span {
                start: start as u32,
                end: end.max(start) as u32,
            },
            message: msg.to_string(),
        }
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn ws(&mut self) -> bool {
        let s = self.i;
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
        self.i > s
    }
    fn ident(&mut self) -> Option<String> {
        let s = self.i;
        while self.peek().is_some_and(is_ident_byte) {
            self.i += 1;
        }
        (self.i > s).then(|| self.s[s..self.i].to_string())
    }
    fn quoted(&mut self) -> Result<String, SelectorError> {
        let q = self.peek().unwrap();
        let start = self.i;
        self.i += 1;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err(self.err(start, self.i, "unterminated string")),
                Some(c) if c == q => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let ch = self.s[self.i..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.err(start, self.i, "unterminated string"))?;
                    out.push(ch);
                    self.i += ch.len_utf8();
                }
                Some(_) => {
                    let ch = self.s[self.i..].chars().next().unwrap();
                    out.push(ch);
                    self.i += ch.len_utf8();
                }
            }
        }
    }

    fn expr(&mut self) -> Result<Expr, SelectorError> {
        let mut branches = vec![self.inter()?];
        loop {
            let save = self.i;
            self.ws();
            if self.peek() == Some(b',') {
                self.i += 1;
                self.ws();
                branches.push(self.inter()?);
            } else {
                self.i = save;
                break;
            }
        }
        Ok(if branches.len() == 1 {
            branches.pop().unwrap()
        } else {
            Expr::Union(branches)
        })
    }

    fn inter(&mut self) -> Result<Expr, SelectorError> {
        let mut left = self.primary()?;
        loop {
            let save = self.i;
            let had_ws = self.ws();
            match self.peek() {
                Some(b'&') => {
                    self.i += 1;
                    self.ws();
                    let r = self.primary()?;
                    left = Expr::And(Box::new(left), Box::new(r));
                }
                Some(b'-')
                    if had_ws && matches!(self.b.get(self.i + 1), Some(b' ' | b'\t' | b'\n')) =>
                {
                    self.i += 1;
                    self.ws();
                    let r = self.primary()?;
                    left = Expr::Diff(Box::new(left), Box::new(r));
                }
                _ => {
                    self.i = save;
                    return Ok(left);
                }
            }
        }
    }

    fn primary(&mut self) -> Result<Expr, SelectorError> {
        if self.peek() == Some(b'(') {
            let start = self.i;
            self.i += 1;
            self.ws();
            let e = self.expr()?;
            self.ws();
            if self.peek() != Some(b')') {
                return Err(self.err(start, self.i, "expected `)`"));
            }
            self.i += 1;
            return Ok(e);
        }
        Ok(Expr::Complex(self.complex()?))
    }

    fn complex(&mut self) -> Result<Complex, SelectorError> {
        let mut parts = vec![self.compound()?];
        let mut combinators = Vec::new();
        loop {
            let save = self.i;
            let had_ws = self.ws();
            match self.peek() {
                Some(b'>') => {
                    self.i += 1;
                    self.ws();
                    combinators.push(Combinator::Child);
                    parts.push(self.compound()?);
                }
                Some(c)
                    if had_ws
                        && (is_ident_byte(c) && c != b'-'
                            || matches!(c, b'*' | b'#' | b'.' | b'[' | b':')) =>
                {
                    combinators.push(Combinator::Descendant);
                    parts.push(self.compound()?);
                }
                _ => {
                    self.i = save;
                    return Ok(Complex { parts, combinators });
                }
            }
        }
    }

    fn compound(&mut self) -> Result<Compound, SelectorError> {
        let start = self.i;
        let mut kind = None;
        let mut kind_span = Span::default();
        if self.peek() == Some(b'*') {
            self.i += 1;
        } else if self.peek().is_some_and(|c| is_ident_byte(c) && c != b'-') {
            kind = self.ident();
            kind_span = Span {
                start: start as u32,
                end: self.i as u32,
            };
        }
        let mut simples = Vec::new();
        loop {
            match self.peek() {
                Some(b'#') => {
                    self.i += 1;
                    let name = if matches!(self.peek(), Some(b'"' | b'\'')) {
                        self.quoted()?
                    } else {
                        let s = self.i;
                        while self.peek().is_some_and(is_name_byte) {
                            self.i += 1;
                        }
                        if self.i == s {
                            return Err(self.err(s - 1, s, "expected a name after `#`"));
                        }
                        self.s[s..self.i].to_string()
                    };
                    simples.push(Simple::Id(name));
                }
                Some(b'.') => {
                    self.i += 1;
                    let s = self.i;
                    let c = self
                        .ident()
                        .ok_or_else(|| self.err(s - 1, s, "expected a class name after `.`"))?;
                    simples.push(Simple::Class(c));
                }
                Some(b'[') => simples.push(self.attr()?),
                Some(b':') => simples.push(self.pseudo()?),
                _ => break,
            }
        }
        if kind.is_none() && simples.is_empty() && self.i == start {
            return Err(self.err(start, start + 1, "expected a selector"));
        }
        Ok(Compound {
            kind,
            kind_span,
            simples,
        })
    }

    fn attr(&mut self) -> Result<Simple, SelectorError> {
        let start = self.i;
        self.i += 1;
        self.ws();
        let s = self.i;
        let name = self
            .ident()
            .ok_or_else(|| self.err(s, s + 1, "expected an attribute name"))?;
        self.ws();
        let op = match (self.peek(), self.b.get(self.i + 1).copied()) {
            (Some(b']'), _) => None,
            (Some(b'='), _) => Some((AttrOp::Eq, 1)),
            (Some(b'!'), Some(b'=')) => Some((AttrOp::Ne, 2)),
            (Some(b'^'), Some(b'=')) => Some((AttrOp::Prefix, 2)),
            (Some(b'$'), Some(b'=')) => Some((AttrOp::Suffix, 2)),
            (Some(b'*'), Some(b'=')) => Some((AttrOp::Contains, 2)),
            (Some(b'~'), Some(b'=')) => Some((AttrOp::Word, 2)),
            (Some(b'<'), Some(b'=')) => Some((AttrOp::Le, 2)),
            (Some(b'>'), Some(b'=')) => Some((AttrOp::Ge, 2)),
            (Some(b'<'), _) => Some((AttrOp::Lt, 1)),
            (Some(b'>'), _) => Some((AttrOp::Gt, 1)),
            _ => return Err(self.err(self.i, self.i + 1, "expected an operator or `]`")),
        };
        let test = if let Some((op, n)) = op {
            self.i += n;
            self.ws();
            let vstart = self.i;
            let (text, quoted) = if matches!(self.peek(), Some(b'"' | b'\'')) {
                (self.quoted()?, true)
            } else {
                let mut depth = 0;
                while let Some(c) = self.peek() {
                    if c == b']' && depth == 0 || c == b' ' || c == b'\t' {
                        break;
                    }
                    if c == b'(' {
                        depth += 1;
                    }
                    if c == b')' {
                        depth -= 1;
                    }
                    self.i += 1;
                }
                (self.s[vstart..self.i].to_string(), false)
            };
            if text.is_empty() && !quoted {
                return Err(self.err(vstart, vstart + 1, "expected a value"));
            }
            let quantity = if quoted {
                None
            } else {
                parse_quantity(&text, self.units).ok()
            };
            if op.is_numeric() && quantity.is_none() {
                return Err(self.err(
                    vstart,
                    self.i,
                    &format!("`{}` needs a number with a known unit", op.text()),
                ));
            }
            self.ws();
            let mut ci = false;
            if matches!(self.peek(), Some(b'i' | b'I'))
                && matches!(self.b.get(self.i + 1), Some(b']' | b' '))
            {
                ci = true;
                self.i += 1;
                self.ws();
            }
            Some((
                op,
                Literal {
                    text,
                    quantity,
                    quoted,
                },
                ci,
            ))
        } else {
            None
        };
        if self.peek() != Some(b']') {
            return Err(self.err(start, self.i, "expected `]`"));
        }
        self.i += 1;
        Ok(Simple::Attr {
            name,
            test,
            span: Span {
                start: start as u32,
                end: self.i as u32,
            },
        })
    }

    fn pseudo(&mut self) -> Result<Simple, SelectorError> {
        let start = self.i;
        self.i += 1;
        let s = self.i;
        let name = self
            .ident()
            .ok_or_else(|| self.err(s - 1, s, "expected a pseudo-class name"))?;
        let structural = matches!(name.as_str(), "not" | "is" | "where" | "has");
        if self.peek() != Some(b'(') {
            if structural {
                return Err(self.err(start, self.i, &format!("`:{name}` needs an argument")));
            }
            return Ok(Simple::Pseudo {
                name,
                args: Vec::new(),
                span: Span {
                    start: start as u32,
                    end: self.i as u32,
                },
            });
        }
        self.i += 1;
        self.ws();
        if structural {
            let mut comb = Combinator::Descendant;
            if name == "has" && self.peek() == Some(b'>') {
                self.i += 1;
                self.ws();
                comb = Combinator::Child;
            }
            let e = self.expr()?;
            self.ws();
            if self.peek() != Some(b')') {
                return Err(self.err(start, self.i, "expected `)`"));
            }
            self.i += 1;
            let e = Box::new(e);
            return Ok(match name.as_str() {
                "not" => Simple::Not(e),
                "is" => Simple::Is(e),
                "where" => Simple::Where(e),
                _ => Simple::Has(e, comb),
            });
        }
        let mut args = Vec::new();
        loop {
            self.ws();
            if self.peek() == Some(b')') {
                self.i += 1;
                break;
            }
            let text = if matches!(self.peek(), Some(b'"' | b'\'')) {
                self.quoted()?
            } else {
                let a = self.i;
                while let Some(c) = self.peek() {
                    if c == b',' || c == b')' || c == b' ' {
                        break;
                    }
                    self.i += 1;
                }
                if a == self.i {
                    return Err(self.err(start, self.i, "expected `)`"));
                }
                self.s[a..self.i].to_string()
            };
            let quantity = parse_quantity(&text, self.units).ok();
            args.push(PseudoArg { text, quantity });
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b')') => {}
                _ => return Err(self.err(start, self.i, "expected `,` or `)`")),
            }
        }
        Ok(Simple::Pseudo {
            name,
            args,
            span: Span {
                start: start as u32,
                end: self.i as u32,
            },
        })
    }
}

// ---------------------------------------------------------------- specificity

fn static_spec(e: &Expr) -> Specificity {
    match e {
        Expr::Complex(c) => complex_spec(c),
        Expr::Union(bs) => bs.iter().map(static_spec).max().unwrap_or_default(),
        Expr::And(a, b) => static_spec(a) + static_spec(b),
        Expr::Diff(a, _) => static_spec(a),
    }
}

fn complex_spec(c: &Complex) -> Specificity {
    let mut s = Specificity::ZERO;
    for p in &c.parts {
        if p.kind.is_some() {
            s.types += 1;
        }
        for x in &p.simples {
            s = s
                + (match x {
                    Simple::Id(_) => Specificity {
                        ids: 1,
                        classes: 0,
                        types: 0,
                    },
                    Simple::Class(_) | Simple::Attr { .. } | Simple::Pseudo { .. } => Specificity {
                        ids: 0,
                        classes: 1,
                        types: 0,
                    },
                    Simple::Not(e) | Simple::Is(e) | Simple::Has(e, _) => static_spec(e),
                    Simple::Where(_) => Specificity::ZERO,
                });
        }
    }
    s
}

// ---------------------------------------------------------------- matching

fn match_expr<G: ItemGraph>(e: &Expr, g: &G, id: G::Id) -> Option<Specificity> {
    match e {
        Expr::Complex(c) => match_complex(c, c.parts.len() - 1, g, id).then(|| complex_spec(c)),
        Expr::Union(bs) => bs.iter().filter_map(|b| match_expr(b, g, id)).max(),
        Expr::And(a, b) => {
            let sa = match_expr(a, g, id)?;
            let sb = match_expr(b, g, id)?;
            Some(sa + sb)
        }
        Expr::Diff(a, b) => {
            let sa = match_expr(a, g, id)?;
            match_expr(b, g, id).is_none().then_some(sa)
        }
    }
}

fn match_complex<G: ItemGraph>(c: &Complex, idx: usize, g: &G, id: G::Id) -> bool {
    if !match_compound(&c.parts[idx], g, id) {
        return false;
    }
    if idx == 0 {
        return true;
    }
    match c.combinators[idx - 1] {
        Combinator::Child => g
            .parent(id)
            .is_some_and(|p| match_complex(c, idx - 1, g, p)),
        Combinator::Descendant => {
            let mut cur = g.parent(id);
            while let Some(p) = cur {
                if match_complex(c, idx - 1, g, p) {
                    return true;
                }
                cur = g.parent(p);
            }
            false
        }
    }
}

fn match_compound<G: ItemGraph>(c: &Compound, g: &G, id: G::Id) -> bool {
    if let Some(k) = &c.kind {
        if !g.is_kind(id, k) {
            return false;
        }
    }
    c.simples.iter().all(|s| match_simple(s, g, id))
}

fn match_simple<G: ItemGraph>(s: &Simple, g: &G, id: G::Id) -> bool {
    match s {
        Simple::Id(n) => g.name(id) == Some(n.as_str()),
        Simple::Class(c) => g.has_class(id, c),
        Simple::Attr { name, test, .. } => {
            let Some(v) = g.attr(id, name) else {
                return false;
            };
            match test {
                None => v.truthy(),
                Some((op, lit, ci)) => attr_test(&v, *op, lit, *ci),
            }
        }
        Simple::Pseudo { name, args, .. } => g.pseudo(id, name, args).unwrap_or(false),
        Simple::Not(e) => match_expr(e, g, id).is_none(),
        Simple::Is(e) | Simple::Where(e) => match_expr(e, g, id).is_some(),
        Simple::Has(e, comb) => has_match(e, *comb, g, id),
    }
}

fn has_match<G: ItemGraph>(e: &Expr, comb: Combinator, g: &G, id: G::Id) -> bool {
    let mut stack = g.children(id);
    stack.sort();
    while let Some(c) = stack.pop() {
        if match_expr(e, g, c).is_some() {
            return true;
        }
        if comb == Combinator::Descendant {
            let mut kids = g.children(c);
            kids.sort();
            stack.extend(kids);
        }
    }
    false
}

fn attr_test(v: &AttrValue, op: AttrOp, lit: &Literal, ci: bool) -> bool {
    if let AttrValue::List(items) = v {
        // A list matches when any member does; `!=` when none equals.
        return match op {
            AttrOp::Ne => !items.iter().any(|x| attr_test(x, AttrOp::Eq, lit, ci)),
            AttrOp::Word => items.iter().any(|x| attr_test(x, AttrOp::Eq, lit, ci)),
            _ => items.iter().any(|x| attr_test(x, op, lit, ci)),
        };
    }
    if let (AttrValue::Num(q), Some(l)) = (v, lit.quantity) {
        let ord = q.compare(&l);
        return match op {
            AttrOp::Eq => ord == Some(Ordering::Equal),
            AttrOp::Ne => ord != Some(Ordering::Equal),
            AttrOp::Lt => ord == Some(Ordering::Less),
            AttrOp::Le => matches!(ord, Some(Ordering::Less | Ordering::Equal)),
            AttrOp::Gt => ord == Some(Ordering::Greater),
            AttrOp::Ge => matches!(ord, Some(Ordering::Greater | Ordering::Equal)),
            _ => str_test(&v.to_text(), op, &lit.text, ci),
        };
    }
    if op.is_numeric() {
        return false;
    }
    if let AttrValue::Bool(b) = v {
        let want = match lit.text.as_str() {
            "true" | "yes" | "1" => true,
            "false" | "no" | "0" => false,
            _ => return false,
        };
        return match op {
            AttrOp::Eq => *b == want,
            AttrOp::Ne => *b != want,
            _ => false,
        };
    }
    str_test(&v.to_text(), op, &lit.text, ci)
}

fn str_test(v: &str, op: AttrOp, lit: &str, ci: bool) -> bool {
    let (v, lit) = if ci {
        (v.to_lowercase(), lit.to_lowercase())
    } else {
        (v.to_string(), lit.to_string())
    };
    match op {
        AttrOp::Eq => v == lit,
        AttrOp::Ne => v != lit,
        AttrOp::Prefix => v.starts_with(&lit),
        AttrOp::Suffix => v.ends_with(&lit),
        AttrOp::Contains => v.contains(&lit),
        AttrOp::Word => v.split_whitespace().any(|w| w == lit),
        _ => false,
    }
}

// ---------------------------------------------------------------- analysis

fn required_kinds(e: &Expr, out: &mut Vec<String>) -> bool {
    match e {
        Expr::Complex(c) => match &c.parts.last().unwrap().kind {
            Some(k) => {
                out.push(k.clone());
                true
            }
            None => false,
        },
        Expr::Union(bs) => bs.iter().all(|b| required_kinds(b, out)),
        Expr::And(a, b) => {
            // Either side's kinds are necessary; take the left when it names kinds.
            let mut l = Vec::new();
            if required_kinds(a, &mut l) {
                out.extend(l);
                true
            } else {
                required_kinds(b, out)
            }
        }
        Expr::Diff(a, _) => required_kinds(a, out),
    }
}

fn uses_has(e: &Expr) -> bool {
    match e {
        Expr::Complex(c) => c.parts.iter().any(|p| {
            p.simples.iter().any(|s| match s {
                Simple::Has(..) => true,
                Simple::Not(e) | Simple::Is(e) | Simple::Where(e) => uses_has(e),
                _ => false,
            })
        }),
        Expr::Union(bs) => bs.iter().any(uses_has),
        Expr::And(a, b) | Expr::Diff(a, b) => uses_has(a) || uses_has(b),
    }
}

fn validate_expr(e: &Expr, schema: &Schema, out: &mut Vec<SelectorError>) {
    match e {
        Expr::Complex(c) => {
            for p in &c.parts {
                if let Some(k) = &p.kind {
                    if !schema.kinds.is_empty() && !schema.kinds.contains_key(k) {
                        out.push(SelectorError {
                            span: p.kind_span,
                            message: format!("unknown item type `{k}`"),
                        });
                    }
                }
                for s in &p.simples {
                    validate_simple(s, schema, out);
                }
            }
        }
        Expr::Union(bs) => bs.iter().for_each(|b| validate_expr(b, schema, out)),
        Expr::And(a, b) | Expr::Diff(a, b) => {
            validate_expr(a, schema, out);
            validate_expr(b, schema, out);
        }
    }
}

fn validate_simple(s: &Simple, schema: &Schema, out: &mut Vec<SelectorError>) {
    match s {
        Simple::Attr { name, test, span } => {
            if schema.attrs.is_empty() {
                return;
            }
            let Some(ty) = schema.attrs.get(name) else {
                out.push(SelectorError {
                    span: *span,
                    message: format!("unknown attribute `{name}`"),
                });
                return;
            };
            if let Some((op, lit, _)) = test {
                match ty {
                    AttrType::Num(d) => match lit.quantity {
                        Some(q) if q.dim != *d => out.push(SelectorError {
                            span: *span,
                            message: format!(
                                "`{name}` has dimension {d}, but `{}` has dimension {}",
                                lit.text, q.dim
                            ),
                        }),
                        None if !matches!(op, AttrOp::Eq | AttrOp::Ne) || !lit.quoted => {
                            out.push(SelectorError {
                                span: *span,
                                message: format!(
                                    "`{name}` is a number; `{}` is not a number with a known unit",
                                    lit.text
                                ),
                            })
                        }
                        _ => {}
                    },
                    AttrType::Str | AttrType::Bool | AttrType::List if op.is_numeric() => {
                        out.push(SelectorError {
                            span: *span,
                            message: format!(
                                "`{}` compares numbers, but `{name}` is not numeric",
                                op.text()
                            ),
                        })
                    }
                    _ => {}
                }
            }
        }
        Simple::Pseudo { name, args, span } => {
            if schema.pseudos.is_empty() {
                return;
            }
            match schema.pseudos.get(name) {
                None => out.push(SelectorError {
                    span: *span,
                    message: format!("unknown pseudo-class `:{name}`"),
                }),
                Some(spec) if args.len() < spec.min_args || args.len() > spec.max_args => {
                    out.push(SelectorError {
                        span: *span,
                        message: format!(
                            "`:{name}` takes {}..={} arguments, got {}",
                            spec.min_args,
                            spec.max_args,
                            args.len()
                        ),
                    })
                }
                _ => {}
            }
        }
        Simple::Not(e) | Simple::Is(e) | Simple::Where(e) | Simple::Has(e, _) => {
            validate_expr(e, schema, out)
        }
        Simple::Id(_) | Simple::Class(_) => {}
    }
}
