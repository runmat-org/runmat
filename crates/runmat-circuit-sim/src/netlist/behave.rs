//! Behavioural sources: `B` (`V = …` or `I = …`), and `E`/`G` with `VALUE = {…}` or
//! `TABLE {…} = (x, y) …`.
//!
//! The expression is compiled once, at flattening, into an [`Ex`] tree:
//! - `V(a)`, `V(a, b)` and `I(vname)` become controls ([`Ctl`]), with the nodes and source
//!   names mapped through the subcircuit instance (ports to the caller's nodes, internal nodes and
//!   sources to `<instance>.<name>`), so a vendor model's expressions read its own nodes;
//! - parameters become numbers (the instance's scope), `time` stays a variable, and `.func`
//!   functions (the innermost definition first) are inlined with their arguments.
//!
//! The functions and operators are those of parameter expressions ([`super::expr`]) plus
//! `u(x)` (unit step), `uramp(x)`, `asin` and `acos`. [`Ex::eval`] gives the value and its
//! partials with respect to each control, so the Newton Jacobian and the AC small-signal gain
//! are exact.

use super::expr::{lex, Scope, T};
use super::value::parse_number;

/// A user function: `.func name(args) {body}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Func {
    pub name: String,
    pub args: Vec<String>,
    pub body: String,
}

/// A controlling quantity: a node-voltage difference or a source's branch current.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ctl {
    V(String, String),
    I(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bop {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fun {
    Sqrt,
    Exp,
    Ln,
    Log10,
    Abs,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Floor,
    Ceil,
    Int,
    Sgn,
    U,
    Uramp,
    Pow,
    Pwr,
    Atan2,
    Min,
    Max,
    If,
    Limit,
}

impl Fun {
    fn by_name(name: &str) -> Option<(Fun, usize)> {
        use Fun::*;
        Some(match name {
            "sqrt" => (Sqrt, 1),
            "exp" => (Exp, 1),
            "log" | "ln" => (Ln, 1),
            "log10" => (Log10, 1),
            "abs" => (Abs, 1),
            "sin" => (Sin, 1),
            "cos" => (Cos, 1),
            "tan" => (Tan, 1),
            "asin" => (Asin, 1),
            "acos" => (Acos, 1),
            "atan" => (Atan, 1),
            "sinh" => (Sinh, 1),
            "cosh" => (Cosh, 1),
            "tanh" => (Tanh, 1),
            "floor" => (Floor, 1),
            "ceil" => (Ceil, 1),
            "int" => (Int, 1),
            "sgn" | "sign" => (Sgn, 1),
            "u" | "stp" => (U, 1),
            "uramp" => (Uramp, 1),
            "pow" => (Pow, 2),
            "pwr" => (Pwr, 2),
            "atan2" => (Atan2, 2),
            "min" => (Min, 2),
            "max" => (Max, 2),
            "if" => (If, 3),
            "limit" => (Limit, 3),
            _ => return None,
        })
    }
}

/// A compiled expression. `Var(k)` is control `k`.
#[derive(Clone, Debug, PartialEq)]
pub enum Ex {
    Num(f64),
    Var(usize),
    Time,
    Neg(Box<Ex>),
    Bin(Bop, Box<Ex>, Box<Ex>),
    Call(Fun, Vec<Ex>),
    Cond(Box<Ex>, Box<Ex>, Box<Ex>),
    /// Piecewise-linear lookup over sorted `(x, y)` points, constant past the ends.
    Table(Box<Ex>, Vec<(f64, f64)>),
}

/// What a behavioural element drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drives {
    Voltage,
    Current,
}

/// A compiled behavioural element.
#[derive(Clone, Debug, PartialEq)]
pub struct Behave {
    pub drives: Drives,
    pub expr: Ex,
    pub ctl: Vec<Ctl>,
}

/// What the compiler needs from the enclosing instance.
pub struct Env<'a> {
    pub scope: &'a Scope,
    /// Visible `.func` definitions, innermost body last.
    pub funcs: &'a [&'a [Func]],
    /// Maps a node name as written to the flat node.
    pub node: &'a dyn Fn(&str) -> String,
    /// Maps a source name as written to the flat element name.
    pub source: &'a dyn Fn(&str) -> String,
}

/// Compiles the behavioural part of a card: `kind` is `b`, `e` or `g`; `raw` is the card's text
/// after its name.
pub fn compile_card(kind: char, raw: &str, env: &Env) -> Result<Behave, String> {
    let rest = skip_words(raw, 2);
    let lower = rest.to_ascii_lowercase();
    let (drives, body) = match kind {
        'b' => {
            let drives = match lower.chars().next() {
                Some('v') => Drives::Voltage,
                Some('i') => Drives::Current,
                _ => return Err("a B source needs `V = {…}` or `I = {…}`".into()),
            };
            let after = rest[1..].trim_start();
            let after = after.strip_prefix('=').ok_or("a B source needs `V = {…}` or `I = {…}`")?;
            (drives, after.trim())
        }
        _ => {
            let drives = if kind == 'e' { Drives::Voltage } else { Drives::Current };
            if lower.starts_with("value") {
                let after = rest[5..].trim_start();
                (drives, after.strip_prefix('=').unwrap_or(after).trim())
            } else if lower.starts_with("table") {
                let mut ctl = vec![];
                let expr = table(rest[5..].trim(), env, &mut ctl)?;
                return Ok(Behave { drives, expr, ctl });
            } else {
                return Err("expected VALUE or TABLE".into());
            }
        }
    };
    let mut ctl = vec![];
    let expr = compile(unwrap(first_value(body)), env, &mut ctl, &[], 0)?;
    Ok(Behave { drives, expr, ctl })
}

/// `{expr} = (x1, y1) (x2, y2) …` (or the pairs without parentheses).
fn table(text: &str, env: &Env, ctl: &mut Vec<Ctl>) -> Result<Ex, String> {
    let (expr, pairs) = if text.starts_with('{') {
        let end = matching(text, 0, '{', '}').ok_or("unclosed `{` in TABLE")?;
        (&text[..=end], text[end + 1..].trim_start())
    } else {
        let eq = text.find('=').ok_or("TABLE needs `= (x, y) …`")?;
        (&text[..eq], &text[eq..])
    };
    let pairs = pairs.strip_prefix('=').ok_or("TABLE needs `= (x, y) …`")?;
    let nums: Vec<f64> = pairs
        .split(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == ')')
        .filter(|s| !s.is_empty())
        .map(|s| parse_number(s).ok_or_else(|| format!("TABLE: `{s}` is not a number")))
        .collect::<Result<_, _>>()?;
    if nums.len() < 2 || !nums.len().is_multiple_of(2) {
        return Err("TABLE needs (x, y) pairs".into());
    }
    let points: Vec<(f64, f64)> = nums.chunks(2).map(|p| (p[0], p[1])).collect();
    if points.windows(2).any(|w| w[1].0 < w[0].0) {
        return Err("TABLE x values must increase".into());
    }
    let x = compile(unwrap(expr), env, ctl, &[], 0)?;
    Ok(Ex::Table(Box::new(x), points))
}

fn skip_words(s: &str, n: usize) -> &str {
    let mut rest = s.trim_start();
    for _ in 0..n {
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        rest = rest[end..].trim_start();
    }
    rest
}

/// The value up to the next top-level `name=` parameter (a B source's `tc1=…` and the like).
fn first_value(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('{') || s.starts_with('\'') {
        let close = if s.starts_with('{') { matching(s, 0, '{', '}') } else { s[1..].find('\'').map(|k| k + 1) };
        return close.map(|k| &s[..=k]).unwrap_or(s);
    }
    s
}

fn unwrap(s: &str) -> &str {
    let t = s.trim();
    t.strip_prefix('{')
        .and_then(|x| x.strip_suffix('}'))
        .or_else(|| t.strip_prefix('\'').and_then(|x| x.strip_suffix('\'')))
        .unwrap_or(t)
}

fn matching(s: &str, open_at: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0;
    for (k, c) in s.char_indices().skip(open_at) {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(k);
            }
        }
    }
    None
}

/// Replaces `V(…)` and `I(…)` with `__ctl<k>` placeholders, adding the controls to `ctl`.
fn controls(text: &str, env: &Env, ctl: &mut Vec<Ctl>) -> Result<String, String> {
    let b = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i].to_ascii_lowercase();
        let starts = (c == b'v' || c == b'i') && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.'));
        let paren = if starts { text[i + 1..].trim_start() } else { "" };
        if starts && paren.starts_with('(') {
            let open = text.len() - paren.len();
            let close = matching(text, open, '(', ')').ok_or("unclosed `(`")?;
            let args: Vec<String> = text[open + 1..close].split(',').map(|s| s.trim().to_ascii_lowercase()).collect();
            let control = match (c, args.as_slice()) {
                (b'v', [a]) => Ctl::V((env.node)(a), "0".into()),
                (b'v', [a, n]) => Ctl::V((env.node)(a), (env.node)(n)),
                (b'i', [s]) => Ctl::I((env.source)(s)),
                _ => return Err(format!("`{}` takes a node, two nodes or a source", &text[i..=close])),
            };
            let k = ctl.iter().position(|x| *x == control).unwrap_or_else(|| {
                ctl.push(control);
                ctl.len() - 1
            });
            out.extend_from_slice(format!(" __ctl{k} ").as_bytes());
            i = close + 1;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8(out).map_err(|_| "expression is not UTF-8".into())
}

/// Compiles expression text; `args` binds a `.func`'s parameters to compiled arguments.
fn compile(text: &str, env: &Env, ctl: &mut Vec<Ctl>, args: &[(String, Ex)], depth: usize) -> Result<Ex, String> {
    if depth > 32 {
        return Err(".func calls nest more than 32 deep".into());
    }
    let replaced = controls(text, env, ctl)?;
    let toks = lex(&replaced)?;
    let mut p = Parser { t: &toks, i: 0, env, ctl, args, depth };
    let e = p.ternary()?;
    if p.i != toks.len() {
        return Err(format!("unexpected `{}` in `{}`", toks[p.i].text(), text.trim()));
    }
    Ok(e)
}

struct Parser<'a, 'b> {
    t: &'a [T],
    i: usize,
    env: &'a Env<'b>,
    ctl: &'a mut Vec<Ctl>,
    args: &'a [(String, Ex)],
    depth: usize,
}

fn bin(op: Bop, a: Ex, b: Ex) -> Ex {
    Ex::Bin(op, Box::new(a), Box::new(b))
}

impl Parser<'_, '_> {
    fn eat(&mut self, o: &str) -> bool {
        if matches!(self.t.get(self.i), Some(T::Op(x)) if *x == o) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn ternary(&mut self) -> Result<Ex, String> {
        let c = self.binary(0)?;
        if self.eat("?") {
            let a = self.ternary()?;
            if !self.eat(":") {
                return Err("expected `:`".into());
            }
            let b = self.ternary()?;
            return Ok(Ex::Cond(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }
    /// Precedence levels, loosest first.
    fn binary(&mut self, level: usize) -> Result<Ex, String> {
        const LEVELS: &[&[(&str, Bop)]] = &[
            &[("||", Bop::Or)],
            &[("&&", Bop::And)],
            &[("<=", Bop::Le), (">=", Bop::Ge), ("==", Bop::Eq), ("!=", Bop::Ne), ("<", Bop::Lt), (">", Bop::Gt)],
            &[("+", Bop::Add), ("-", Bop::Sub)],
            &[("*", Bop::Mul), ("/", Bop::Div)],
        ];
        if level == LEVELS.len() {
            return self.unary();
        }
        let mut a = self.binary(level + 1)?;
        'outer: loop {
            for (o, op) in LEVELS[level] {
                if self.eat(o) {
                    let b = self.binary(level + 1)?;
                    a = bin(*op, a, b);
                    if level == 2 {
                        return Ok(a);
                    }
                    continue 'outer;
                }
            }
            return Ok(a);
        }
    }
    fn unary(&mut self) -> Result<Ex, String> {
        if self.eat("-") {
            return Ok(Ex::Neg(Box::new(self.unary()?)));
        }
        if self.eat("+") {
            return self.unary();
        }
        let a = self.atom()?;
        if self.eat("^") || self.eat("**") {
            let b = self.unary()?;
            return Ok(bin(Bop::Pow, a, b));
        }
        Ok(a)
    }
    fn atom(&mut self) -> Result<Ex, String> {
        match self.t.get(self.i).cloned() {
            Some(T::Num(v)) => {
                self.i += 1;
                Ok(Ex::Num(v))
            }
            Some(T::Op("(")) => {
                self.i += 1;
                let v = self.ternary()?;
                if !self.eat(")") {
                    return Err("expected `)`".into());
                }
                Ok(v)
            }
            Some(T::Id(name)) => {
                self.i += 1;
                if self.eat("(") {
                    let mut args = vec![];
                    if !self.eat(")") {
                        loop {
                            args.push(self.ternary()?);
                            if self.eat(")") {
                                break;
                            }
                            if !self.eat(",") {
                                return Err("expected `,` or `)`".into());
                            }
                        }
                    }
                    return self.call(&name, args);
                }
                if let Some(k) = name.strip_prefix("__ctl").and_then(|k| k.parse().ok()) {
                    return Ok(Ex::Var(k));
                }
                if let Some((_, e)) = self.args.iter().find(|(a, _)| *a == name) {
                    return Ok(e.clone());
                }
                if let Some(v) = self.env.scope.get(&name) {
                    return Ok(Ex::Num(*v));
                }
                match name.as_str() {
                    "time" => Ok(Ex::Time),
                    "pi" => Ok(Ex::Num(std::f64::consts::PI)),
                    "e" => Ok(Ex::Num(std::f64::consts::E)),
                    _ => Err(format!("unknown name `{name}`")),
                }
            }
            Some(t) => Err(format!("unexpected `{}`", t.text())),
            None => Err("unexpected end of expression".into()),
        }
    }
    fn call(&mut self, name: &str, args: Vec<Ex>) -> Result<Ex, String> {
        let func = self.env.funcs.iter().rev().find_map(|fs| fs.iter().find(|f| f.name == name));
        if let Some(f) = func {
            if f.args.len() != args.len() {
                return Err(format!("`{name}` takes {} argument(s)", f.args.len()));
            }
            let bound: Vec<(String, Ex)> = f.args.iter().cloned().zip(args).collect();
            return compile(&f.body, self.env, self.ctl, &bound, self.depth + 1).map_err(|e| format!("in `{name}`: {e}"));
        }
        let (fun, n) = Fun::by_name(name).ok_or_else(|| format!("unknown function `{name}`"))?;
        if args.len() != n {
            return Err(format!("`{name}` takes {n} argument(s)"));
        }
        Ok(Ex::Call(fun, args))
    }
}

/// Reads `.func name(a, b) {body}` (also `= {body}` and unbraced bodies) from the line after
/// `.func`.
pub fn read_func(raw: &str) -> Result<Func, String> {
    let open = raw.find('(').ok_or(".func needs `name(args) {body}`")?;
    let name = raw[..open].trim().to_ascii_lowercase();
    let close = matching(raw, open, '(', ')').ok_or(".func: unclosed `(`")?;
    let args: Vec<String> = raw[open + 1..close].split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect();
    let rest = raw[close + 1..].trim_start();
    let rest = rest.strip_prefix('=').unwrap_or(rest);
    let body = unwrap(rest).to_string();
    if name.is_empty() || body.is_empty() {
        return Err(".func needs `name(args) {body}`".into());
    }
    Ok(Func { name, args, body })
}
