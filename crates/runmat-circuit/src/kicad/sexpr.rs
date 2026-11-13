//! Lossless S-expressions for KiCad files.
//!
//! Atoms keep their exact source text (numbers are never re-formatted) and whether they were
//! quoted, so `write(parse(x))` reproduces every token of `x`. Formatting follows the KiCad
//! 8+ convention: one list per line when it contains lists, tab indentation, short lists of
//! atoms on one line. Whitespace and comments are not preserved; tokens are.

use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum SExpr {
    Atom(Atom),
    List(Vec<SExpr>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Atom {
    pub text: String,
    pub quoted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "byte {}: {}", self.offset, self.message)
    }
}

impl SExpr {
    pub fn sym(s: &str) -> SExpr {
        SExpr::Atom(Atom {
            text: s.into(),
            quoted: false,
        })
    }
    pub fn str(s: &str) -> SExpr {
        SExpr::Atom(Atom {
            text: s.into(),
            quoted: true,
        })
    }
    /// A number in KiCad's style: up to 6 decimals, trailing zeros trimmed.
    pub fn num(v: f64) -> SExpr {
        SExpr::sym(&fmt_num(v))
    }
    pub fn list(head: &str, items: Vec<SExpr>) -> SExpr {
        let mut v = vec![SExpr::sym(head)];
        v.extend(items);
        SExpr::List(v)
    }
    pub fn head(&self) -> Option<&str> {
        match self {
            SExpr::List(v) => match v.first() {
                Some(SExpr::Atom(a)) => Some(&a.text),
                _ => None,
            },
            _ => None,
        }
    }
    pub fn items(&self) -> &[SExpr] {
        match self {
            SExpr::List(v) => v,
            _ => &[],
        }
    }
    pub fn items_mut(&mut self) -> Option<&mut Vec<SExpr>> {
        match self {
            SExpr::List(v) => Some(v),
            _ => None,
        }
    }
    pub fn atom(&self) -> Option<&str> {
        match self {
            SExpr::Atom(a) => Some(&a.text),
            _ => None,
        }
    }
    /// The first child list with this head.
    pub fn child(&self, head: &str) -> Option<&SExpr> {
        self.items().iter().skip(1).find(|c| c.head() == Some(head))
    }
    pub fn child_mut(&mut self, head: &str) -> Option<&mut SExpr> {
        self.items_mut()?
            .iter_mut()
            .skip(1)
            .find(|c| c.head() == Some(head))
    }
    pub fn children<'a>(&'a self, head: &'a str) -> impl Iterator<Item = &'a SExpr> + 'a {
        self.items()
            .iter()
            .skip(1)
            .filter(move |c| c.head() == Some(head))
    }
    /// The `i`-th atom after the head (`(at 1 2 90)`.arg(1) == "2").
    pub fn arg(&self, i: usize) -> Option<&str> {
        self.items().get(i + 1).and_then(|x| x.atom())
    }
    pub fn arg_f(&self, i: usize) -> Option<f64> {
        self.arg(i).and_then(|s| s.parse().ok())
    }
    /// Atom argument of a child: `(layer "F.Cu")`.child_arg("layer") == "F.Cu".
    pub fn child_arg(&self, head: &str) -> Option<&str> {
        self.child(head).and_then(|c| c.arg(0))
    }
    pub fn child_f(&self, head: &str) -> Option<f64> {
        self.child(head).and_then(|c| c.arg_f(0))
    }
    /// Whether a bare symbol appears among the atoms (`(via blind ...)`).
    pub fn has_flag(&self, flag: &str) -> bool {
        self.items()
            .iter()
            .skip(1)
            .any(|x| matches!(x, SExpr::Atom(a) if !a.quoted && a.text == flag))
    }
    /// Replaces the first child with this head, or appends.
    pub fn set_child(&mut self, node: SExpr) {
        let head = node.head().map(|s| s.to_string());
        if let (Some(h), Some(v)) = (head, self.items_mut()) {
            if let Some(slot) = v.iter_mut().skip(1).find(|c| c.head() == Some(h.as_str())) {
                *slot = node;
                return;
            }
            v.push(node);
        }
    }
    pub fn remove_children(&mut self, head: &str) {
        if let Some(v) = self.items_mut() {
            let mut first = true;
            v.retain(|c| {
                let keep = first || c.head() != Some(head);
                first = false;
                keep
            });
        }
    }
}

/// KiCad-style number text.
pub fn fmt_num(v: f64) -> String {
    let v = if v == 0.0 { 0.0 } else { v };
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

/// Parses one or more top-level expressions.
pub fn parse_all(src: &str) -> Result<Vec<SExpr>, ParseError> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut stack: Vec<Vec<SExpr>> = Vec::new();
    let mut out = Vec::new();
    let err = |o: usize, m: &str| ParseError {
        offset: o,
        message: m.to_string(),
    };
    while i < b.len() {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            b'(' => {
                stack.push(Vec::new());
                i += 1;
            }
            b')' => {
                let l = stack.pop().ok_or_else(|| err(i, "unbalanced `)`"))?;
                let e = SExpr::List(l);
                match stack.last_mut() {
                    Some(p) => p.push(e),
                    None => out.push(e),
                }
                i += 1;
            }
            b'"' => {
                let start = i;
                i += 1;
                let mut s = Vec::new();
                loop {
                    if i >= b.len() {
                        return Err(err(start, "unterminated string"));
                    }
                    match b[i] {
                        b'"' => {
                            i += 1;
                            break;
                        }
                        b'\\' if i + 1 < b.len() => {
                            let n = b[i + 1];
                            s.push(match n {
                                b'n' => b'\n',
                                b't' => b'\t',
                                b'r' => b'\r',
                                other => other,
                            });
                            i += 2;
                        }
                        x => {
                            s.push(x);
                            i += 1;
                        }
                    }
                }
                let text = String::from_utf8(s).map_err(|_| err(start, "invalid UTF-8"))?;
                let a = SExpr::Atom(Atom { text, quoted: true });
                match stack.last_mut() {
                    Some(p) => p.push(a),
                    None => out.push(a),
                }
            }
            _ => {
                let start = i;
                while i < b.len()
                    && !matches!(b[i], b' ' | b'\t' | b'\n' | b'\r' | b'(' | b')' | b'"')
                {
                    i += 1;
                }
                let a = SExpr::Atom(Atom {
                    text: src[start..i].to_string(),
                    quoted: false,
                });
                match stack.last_mut() {
                    Some(p) => p.push(a),
                    None => out.push(a),
                }
            }
        }
    }
    if !stack.is_empty() {
        return Err(err(b.len(), "unclosed `(`"));
    }
    Ok(out)
}

pub fn parse(src: &str) -> Result<SExpr, ParseError> {
    let mut v = parse_all(src)?;
    if v.len() != 1 {
        return Err(ParseError {
            offset: 0,
            message: format!("expected one top-level expression, found {}", v.len()),
        });
    }
    Ok(v.pop().unwrap())
}

fn write_atom(out: &mut String, a: &Atom) {
    if a.quoted {
        out.push('"');
        for ch in a.text.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                c => out.push(c),
            }
        }
        out.push('"');
    } else {
        out.push_str(&a.text);
    }
}

/// A list is written on one line when it has no grandchildren and at most 8 children
/// (`(at 1 2 90)`, `(xy 0 0)`, `(stroke (width 0.1) (type solid))` stays nested).
fn inline(e: &SExpr) -> bool {
    match e {
        SExpr::Atom(_) => true,
        SExpr::List(v) => {
            v.len() <= 8
                && v.iter().all(|x| match x {
                    SExpr::Atom(_) => true,
                    SExpr::List(w) => w.iter().all(|y| matches!(y, SExpr::Atom(_))) && w.len() <= 5,
                })
                && v.iter().filter(|x| matches!(x, SExpr::List(_))).count() <= 3
        }
    }
}

fn write_expr(out: &mut String, e: &SExpr, indent: usize) {
    match e {
        SExpr::Atom(a) => write_atom(out, a),
        SExpr::List(v) => {
            out.push('(');
            if inline(e) {
                for (k, x) in v.iter().enumerate() {
                    if k > 0 {
                        out.push(' ');
                    }
                    write_expr(out, x, indent);
                }
                out.push(')');
                return;
            }
            // Leading atoms on the first line; every later item on its own line.
            let lead = v.iter().take_while(|x| matches!(x, SExpr::Atom(_))).count();
            for (k, x) in v.iter().enumerate() {
                if k < lead {
                    if k > 0 {
                        out.push(' ');
                    }
                    write_expr(out, x, indent);
                } else {
                    out.push('\n');
                    for _ in 0..=indent {
                        out.push('\t');
                    }
                    write_expr(out, x, indent + 1);
                }
            }
            out.push('\n');
            for _ in 0..indent {
                out.push('\t');
            }
            out.push(')');
        }
    }
}

/// KiCad-style text with a trailing newline.
pub fn write(e: &SExpr) -> String {
    let mut s = String::new();
    write_expr(&mut s, e, 0);
    s.push('\n');
    s
}

/// Compact single-line text (for digests and debugging).
pub fn write_compact(e: &SExpr) -> String {
    let mut s = String::new();
    fn go(s: &mut String, e: &SExpr) {
        match e {
            SExpr::Atom(a) => write_atom(s, a),
            SExpr::List(v) => {
                s.push('(');
                for (k, x) in v.iter().enumerate() {
                    if k > 0 {
                        s.push(' ');
                    }
                    go(s, x);
                }
                s.push(')');
            }
        }
    }
    go(&mut s, e);
    let _ = write!(s, "");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_tokens() {
        let src = r#"(kicad_pcb (version 20240108) (generator "pcbnew")
  (net 1 "GND") (segment (start 1.000 2.5) (end 3 4) (width 0.25) (layer "F.Cu") (net 1) (uuid "a-b"))
  (gr_text "say \"hi\"" (at 0 0) (unknown_future_token (deep 1 2 (deeper x)))))"#;
        let e = parse(src).unwrap();
        let out = write(&e);
        assert_eq!(parse(&out).unwrap(), e);
        // Number text is kept verbatim.
        assert!(out.contains("1.000"));
        assert_eq!(
            e.child("segment").unwrap().child("start").unwrap().arg(0),
            Some("1.000")
        );
        assert_eq!(fmt_num(0.1 + 0.2), "0.3");
        assert_eq!(fmt_num(-0.0), "0");
        assert!(parse("(a (b)").is_err());
        assert!(parse("(a))").is_err());
    }
}
