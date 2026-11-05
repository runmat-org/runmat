//! SPICE netlist text → cards, models, subcircuits and control lines.
//!
//! - The first line is the title. `*` starts a comment line, `;` and `$ ` start trailing comments,
//!   `+` continues the previous line. Everything except the title is case-insensitive.
//! - Tokens split on whitespace, commas and parentheses; `{…}` and `'…'` stay whole; `k = v`
//!   becomes one key/value token.
//! - `.include file` and `.lib file [section]` are read through the caller's resolver.
//! - `.control … .endc` blocks, `.save`, `.print`, `.probe`, `.plot`, `.backanno` and `.end` are
//!   accepted and ignored. Other unknown dot-cards are errors, so nothing is silently dropped.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    W(String),
    Kv(String, String),
}

impl Tok {
    pub fn word(&self) -> Option<&str> {
        match self {
            Tok::W(s) => Some(s),
            Tok::Kv(..) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    pub line: usize,
    /// Lowercased element name (`r1`, `xu1`).
    pub name: String,
    pub toks: Vec<Tok>,
    /// The card's text after its name, as written (behavioural expressions read it).
    pub raw: String,
}

impl Card {
    pub fn kind(&self) -> char {
        self.name.chars().next().unwrap_or(' ')
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub name: String,
    pub kind: String,
    pub params: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Body {
    pub cards: Vec<Card>,
    pub models: Vec<Model>,
    pub subckts: Vec<Subckt>,
    pub params: Vec<(String, String)>,
    pub funcs: Vec<super::behave::Func>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Subckt {
    pub name: String,
    pub ports: Vec<String>,
    pub defaults: Vec<(String, String)>,
    pub body: Body,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parsed {
    pub title: String,
    pub body: Body,
    pub options: Vec<(String, String)>,
    /// `.op`, `.dc`, `.ac`, `.tran`, `.meas` lines as cards (name is the dot-word).
    pub controls: Vec<Card>,
    pub ics: Vec<(String, String)>,
    pub nodesets: Vec<(String, String)>,
    pub globals: Vec<String>,
    pub temp: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

/// Reads an included file by name.
pub type Resolver<'a> = &'a dyn Fn(&str) -> Option<String>;

pub fn parse(text: &str, resolve: Resolver<'_>) -> Result<Parsed, ParseError> {
    let mut lines = logical_lines(text);
    let title = if lines.is_empty() {
        String::new()
    } else {
        lines.remove(0).1.trim_start_matches('*').trim().to_string()
    };
    let lines = expand_includes(lines, resolve, 0)?;
    let mut out = Parsed { title, ..Default::default() };
    let mut stack: Vec<Subckt> = vec![];
    let mut in_control = false;
    for (ln, line) in lines {
        let toks = tokenize(&line);
        let Some(Tok::W(head)) = toks.first().cloned() else {
            continue;
        };
        let rest = toks[1..].to_vec();
        let raw = {
            let t = line.trim_start();
            t[t.find(char::is_whitespace).unwrap_or(t.len())..].trim().to_string()
        };
        if in_control {
            if head == ".endc" {
                in_control = false;
            }
            continue;
        }
        let err = |m: String| ParseError { line: ln, message: m };
        let body = match stack.last_mut() {
            Some(s) => &mut s.body,
            None => &mut out.body,
        };
        if !head.starts_with('.') {
            body.cards.push(Card { line: ln, name: head, toks: rest, raw });
            continue;
        }
        match head.as_str() {
            ".model" => body.models.push(model(&rest).map_err(err)?),
            ".subckt" => {
                let mut words = vec![];
                let mut defaults = vec![];
                for t in rest {
                    match t {
                        Tok::W(w) if w == "params:" => {}
                        Tok::W(w) => words.push(w),
                        Tok::Kv(k, v) => defaults.push((k, v)),
                    }
                }
                if words.is_empty() {
                    return Err(err(".subckt needs a name".into()));
                }
                let name = words.remove(0);
                stack.push(Subckt { name, ports: words, defaults, body: Body::default() });
            }
            ".ends" => {
                let s = stack.pop().ok_or_else(|| err(".ends without .subckt".into()))?;
                match stack.last_mut() {
                    Some(p) => p.body.subckts.push(s),
                    None => out.body.subckts.push(s),
                }
            }
            ".param" | ".params" => {
                for t in rest {
                    if let Tok::Kv(k, v) = t {
                        body.params.push((k, v));
                    }
                }
            }
            ".options" | ".option" | ".opt" | ".opts" => {
                for t in rest {
                    match t {
                        Tok::Kv(k, v) => out.options.push((k, v)),
                        Tok::W(w) => out.options.push((w, "1".into())),
                    }
                }
            }
            ".ic" | ".nodeset" => {
                for t in rest {
                    // `v(out)=1` tokenizes as `v`, `out=1`.
                    if let Tok::Kv(k, v) = t {
                        let node = k;
                        if head == ".ic" {
                            out.ics.push((node, v));
                        } else {
                            out.nodesets.push((node, v));
                        }
                    }
                }
            }
            ".global" => out.globals.extend(rest.into_iter().filter_map(|t| t.word().map(String::from))),
            ".temp" => out.temp = rest.first().and_then(|t| t.word().map(String::from)),
            ".op" | ".dc" | ".ac" | ".tran" | ".meas" | ".measure" => {
                out.controls.push(Card { line: ln, name: head, toks: rest, raw })
            }
            ".control" => in_control = true,
            ".end" | ".save" | ".print" | ".probe" | ".plot" | ".backanno" | ".width" | ".endl" => {}
            ".func" => body.funcs.push(super::behave::read_func(&raw).map_err(err)?),
            other => return Err(err(format!("unknown control line `{other}`"))),
        }
    }
    if let Some(s) = stack.last() {
        return Err(ParseError { line: 0, message: format!(".subckt {} has no .ends", s.name) });
    }
    Ok(out)
}

fn model(rest: &[Tok]) -> Result<Model, String> {
    let name = rest.first().and_then(|t| t.word()).ok_or(".model needs a name")?;
    let kind = rest.get(1).and_then(|t| t.word()).ok_or(".model needs a type")?;
    let params = rest[2..]
        .iter()
        .filter_map(|t| match t {
            Tok::Kv(k, v) => Some((k.clone(), v.clone())),
            Tok::W(_) => None,
        })
        .collect();
    Ok(Model { name: name.into(), kind: kind.into(), params })
}

/// Joins continuation lines and strips comments; keeps 1-based source line numbers.
fn logical_lines(text: &str) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = vec![];
    for (i, raw) in text.lines().enumerate() {
        let mut l = raw.trim_end().to_string();
        if i == 0 {
            out.push((1, l));
            continue;
        }
        if let Some(k) = l.find(';') {
            l.truncate(k);
        }
        if let Some(k) = l.find("$ ") {
            l.truncate(k);
        }
        let t = l.trim_start();
        if t.is_empty() || t.starts_with('*') {
            continue;
        }
        if let Some(cont) = t.strip_prefix('+') {
            let has_card = out.len() > 1;
            if let Some(last) = out.last_mut().filter(|_| has_card) {
                last.1.push(' ');
                last.1.push_str(cont);
                continue;
            }
        }
        out.push((i + 1, t.to_string()));
    }
    out
}

fn expand_includes(
    lines: Vec<(usize, String)>,
    resolve: Resolver<'_>,
    depth: usize,
) -> Result<Vec<(usize, String)>, ParseError> {
    let mut out = vec![];
    let mut section_skip = false;
    for (ln, l) in lines {
        let lower = l.to_ascii_lowercase();
        let mut words = lower.split_whitespace();
        let head = words.next().unwrap_or("");
        if head == ".endl" {
            section_skip = false;
            continue;
        }
        if section_skip {
            continue;
        }
        let is_inc = head == ".include" || head == ".inc";
        let is_lib = head == ".lib";
        if !(is_inc || is_lib) {
            out.push((ln, l));
            continue;
        }
        let args: Vec<&str> = l.split_whitespace().skip(1).collect();
        if is_lib && args.len() == 1 && depth > 0 {
            // A section header inside a library file read whole: keep its contents.
            continue;
        }
        let file = args.first().map(|s| s.trim_matches(|c| c == '"' || c == '\'')).unwrap_or("");
        if depth > 16 {
            return Err(ParseError { line: ln, message: "includes nest too deeply".into() });
        }
        let text = resolve(file).ok_or_else(|| ParseError {
            line: ln,
            message: format!("cannot read included file `{file}`"),
        })?;
        let mut inner: Vec<(usize, String)> = logical_lines(&format!("*\n{text}"));
        inner.remove(0);
        if is_lib && args.len() >= 2 {
            let sec = args[1].to_ascii_lowercase();
            let mut keep = vec![];
            let mut on = false;
            for (n, s) in inner {
                let w: Vec<String> = s.to_ascii_lowercase().split_whitespace().map(String::from).collect();
                if w.first().map(String::as_str) == Some(".lib") && w.len() == 2 {
                    on = w[1] == sec;
                    continue;
                }
                if w.first().map(String::as_str) == Some(".endl") {
                    on = false;
                    continue;
                }
                if on {
                    keep.push((n, s));
                }
            }
            inner = keep;
        }
        out.extend(expand_includes(inner, resolve, depth + 1)?);
    }
    Ok(out)
}

/// Splits a line into tokens (lowercased).
pub fn tokenize(line: &str) -> Vec<Tok> {
    let mut raw: Vec<String> = vec![];
    let mut cur = String::new();
    let mut depth = 0;
    let mut quote = false;
    let flush = |cur: &mut String, raw: &mut Vec<String>| {
        if !cur.is_empty() {
            raw.push(std::mem::take(cur));
        }
    };
    for c in line.chars() {
        if quote {
            cur.push(c);
            if c == '\'' {
                quote = false;
            }
            continue;
        }
        if depth > 0 {
            cur.push(c);
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            continue;
        }
        match c {
            '{' => {
                depth = 1;
                cur.push(c);
            }
            '\'' => {
                quote = true;
                cur.push(c);
            }
            ' ' | '\t' | ',' | '(' | ')' => flush(&mut cur, &mut raw),
            '=' => {
                flush(&mut cur, &mut raw);
                raw.push("=".into());
            }
            _ => cur.push(c.to_ascii_lowercase()),
        }
    }
    flush(&mut cur, &mut raw);
    let mut out = vec![];
    let mut i = 0;
    while i < raw.len() {
        if i + 2 < raw.len() + 1 && raw.get(i + 1).map(String::as_str) == Some("=") && i + 2 < raw.len() {
            out.push(Tok::Kv(raw[i].clone(), raw[i + 2].clone()));
            i += 3;
        } else if raw[i] == "=" {
            i += 1;
        } else {
            out.push(Tok::W(raw[i].clone()));
            i += 1;
        }
    }
    out
}

/// Option values by key.
pub fn options_map(p: &Parsed) -> BTreeMap<String, String> {
    p.options.iter().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokens_and_cards() {
        let t = tokenize("Q1 c b e QN(IS = 1e-15, BF=100) {2*rval} 'a b'");
        assert_eq!(t[0], Tok::W("q1".into()));
        assert_eq!(t[5], Tok::Kv("is".into(), "1e-15".into()));
        assert_eq!(t[6], Tok::Kv("bf".into(), "100".into()));
        assert_eq!(t[7], Tok::W("{2*rval}".into()));
        let deck = "title\nR1 a 0 1k ; comment\n+ tc=0\n.subckt amp in out params: g=2\nE1 out 0 in 0 {g}\n.ends\n.model D1N D(IS=1n)\n.op\n.end\n";
        let p = parse(deck, &|_| None).unwrap();
        assert_eq!(p.title, "title");
        assert_eq!(p.body.cards[0].toks.len(), 4);
        assert_eq!(p.body.subckts[0].defaults, vec![("g".into(), "2".into())]);
        assert_eq!(p.body.models[0].kind, "d");
        assert_eq!(p.controls[0].name, ".op");
    }
}
