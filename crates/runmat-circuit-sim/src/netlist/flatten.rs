//! Subcircuit expansion and parameter evaluation.
//!
//! Each `X` instance is expanded recursively. Inside instance `xu1`:
//! - element `r1` becomes `xu1.r1`, internal node `n2` becomes `xu1.n2`, port nodes map to the
//!   caller's nodes, and ground (`0`, `gnd`) and `.global` nodes stay as they are;
//! - models defined in the subcircuit become `xu1.<model>`, evaluated with the instance's
//!   parameters; references to them, to controlling sources (`F`, `H`) and to coupled inductors
//!   (`K`) are renamed to match;
//! - parameters: the subcircuit's defaults, overridden by the instance's `name=value` pairs
//!   (evaluated in the caller's scope), then the body's `.param` lines.

use super::expr::{eval, Scope};
use super::parse::{Body, Card, Model, Parsed, Subckt, Tok};
use super::value::parse_number;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
pub struct FlatElement {
    pub line: usize,
    pub name: String,
    pub kind: char,
    pub nodes: Vec<String>,
    pub args: Vec<Tok>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlatModel {
    pub name: String,
    pub kind: String,
    pub params: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Flat {
    pub elements: Vec<FlatElement>,
    pub models: BTreeMap<String, FlatModel>,
    pub params: Scope,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlattenError {
    pub line: usize,
    pub message: String,
}

pub fn is_ground(n: &str) -> bool {
    n == "0" || n == "gnd"
}

struct Ctx<'a> {
    globals: BTreeSet<String>,
    out: Flat,
    /// Enclosing bodies, innermost last, for model and subcircuit lookup.
    chain: Vec<&'a Body>,
}

pub fn flatten(p: &Parsed) -> Result<Flat, FlattenError> {
    let mut scope = Scope::new();
    eval_params(&p.body.params, &mut scope, 0)?;
    let mut cx = Ctx {
        globals: p.globals.iter().cloned().collect(),
        out: Flat::default(),
        chain: vec![],
    };
    expand(&mut cx, &p.body, "", &BTreeMap::new(), &scope, 0)?;
    cx.out.params = scope;
    Ok(cx.out)
}

fn eval_params(params: &[(String, String)], scope: &mut Scope, line: usize) -> Result<(), FlattenError> {
    for (k, v) in params {
        let x = eval(v, scope).map_err(|m| FlattenError { line, message: format!("parameter `{k}`: {m}") })?;
        scope.insert(k.clone(), x);
    }
    Ok(())
}

fn eval_tok(t: &Tok, scope: &Scope, line: usize) -> Result<Tok, FlattenError> {
    let is_expr = |s: &str| s.starts_with('{') || s.starts_with('\'');
    let ev = |s: &str| {
        eval(s, scope)
            .map(|v| format!("{v:e}"))
            .map_err(|m| FlattenError { line, message: m })
    };
    Ok(match t {
        Tok::W(s) if is_expr(s) => Tok::W(ev(s)?),
        Tok::Kv(k, v) if is_expr(v) => Tok::Kv(k.clone(), ev(v)?),
        Tok::Kv(k, v) if parse_number(v).is_none() => match eval(v, scope) {
            Ok(x) => Tok::Kv(k.clone(), format!("{x:e}")),
            Err(_) => t.clone(),
        },
        _ => t.clone(),
    })
}

fn find_model<'a>(chain: &[&'a Body], name: &str) -> Option<(usize, &'a Model)> {
    chain
        .iter()
        .enumerate()
        .rev()
        .find_map(|(d, b)| b.models.iter().find(|m| m.name == name).map(|m| (d, m)))
}

fn find_subckt<'a>(chain: &[&'a Body], name: &str) -> Option<&'a Subckt> {
    chain.iter().rev().find_map(|b| b.subckts.iter().find(|s| s.name == name))
}

/// Indices of the node tokens of a card.
fn node_positions(c: &Card, is_model: &dyn Fn(&str) -> bool) -> Result<Vec<usize>, String> {
    let w = |i: usize| c.toks.get(i).and_then(|t| t.word());
    let words = c.toks.iter().take_while(|t| t.word().is_some()).count();
    let need = |n: usize| -> Result<Vec<usize>, String> {
        if words < n {
            Err(format!("`{}` needs {n} nodes", c.name))
        } else {
            Ok((0..n).collect())
        }
    };
    match c.kind() {
        'r' | 'c' | 'l' | 'v' | 'i' | 'd' | 'f' | 'h' | 'w' => need(2),
        'e' | 'g' => {
            if w(2) == Some("poly") {
                let n: usize = w(3).and_then(|s| s.parse().ok()).ok_or("POLY needs a dimension")?;
                let mut v: Vec<usize> = vec![0, 1];
                v.extend(4..4 + 2 * n);
                if words < 4 + 2 * n {
                    return Err(format!("`{}`: POLY({n}) needs {} controlling nodes", c.name, 2 * n));
                }
                Ok(v)
            } else if w(2).is_some_and(|s| s == "value" || s == "table" || s == "laplace") {
                Err(format!("`{}`: behavioural {} sources are not supported yet", c.name, w(2).unwrap()))
            } else {
                need(4)
            }
        }
        'q' => {
            // A fourth word is the substrate node unless it names a model.
            if !w(3).is_some_and(is_model) && w(4).is_some_and(is_model) {
                return Ok(vec![0, 1, 2, 3]);
            }
            need(3)
        }
        'm' | 's' => need(4),
        'x' => {
            let n = c.toks.iter().take_while(|t| matches!(t, Tok::W(s) if s != "params:")).count();
            if n < 1 {
                return Err(format!("`{}` names no subcircuit", c.name));
            }
            Ok((0..n - 1).collect())
        }
        'k' => Ok(vec![]),
        k => Err(format!("`{}`: element type `{k}` is not supported", c.name)),
    }
}

fn expand<'a>(
    cx: &mut Ctx<'a>,
    body: &'a Body,
    prefix: &str,
    ports: &BTreeMap<String, String>,
    scope: &Scope,
    depth: usize,
) -> Result<(), FlattenError> {
    if depth > 64 {
        return Err(FlattenError { line: 0, message: "subcircuits nest more than 64 deep".into() });
    }
    cx.chain.push(body);
    let scoped = |n: &str| if prefix.is_empty() { n.to_string() } else { format!("{prefix}.{n}") };
    let map_node = |n: &str, cx: &Ctx| -> String {
        if is_ground(n) {
            "0".into()
        } else if let Some(m) = ports.get(n) {
            m.clone()
        } else if cx.globals.contains(n) || prefix.is_empty() {
            n.to_string()
        } else {
            format!("{prefix}.{n}")
        }
    };
    // Local models, one copy per instance (their parameters may use the instance's).
    let depth_here = cx.chain.len() - 1;
    for m in &body.models {
        let mut params = BTreeMap::new();
        for (k, v) in &m.params {
            let x = eval(v, scope).map_err(|e| FlattenError { line: 0, message: format!("model {}: {k}: {e}", m.name) })?;
            params.insert(k.clone(), x);
        }
        let name = scoped(&m.name);
        cx.out.models.insert(name.clone(), FlatModel { name, kind: m.kind.clone(), params });
    }
    for c in &body.cards {
        let chain = cx.chain.clone();
        let is_model = |s: &str| find_model(&chain, s).is_some();
        let pos = node_positions(c, &is_model).map_err(|m| FlattenError { line: c.line, message: m })?;
        let mut nodes = vec![];
        let mut args = vec![];
        for (i, t) in c.toks.iter().enumerate() {
            if pos.contains(&i) {
                nodes.push(map_node(t.word().unwrap_or_default(), cx));
            } else {
                args.push(eval_tok(t, scope, c.line)?);
            }
        }
        // Rename references to models, controlling sources and inductors.
        let rename = |s: &str| -> String {
            match find_model(&chain, s) {
                Some((d, _)) if d == depth_here => scoped(s),
                Some((d, _)) if d > 0 => {
                    // A model of an enclosing subcircuit instance: its prefix is ours minus the
                    // inner levels.
                    let parts: Vec<&str> = prefix.split('.').collect();
                    let keep = parts.len() - (depth_here - d);
                    format!("{}.{s}", parts[..keep].join("."))
                }
                _ => s.to_string(),
            }
        };
        match c.kind() {
            'f' | 'h' => {
                let n = if args.first().and_then(|t| t.word()) == Some("poly") {
                    args.get(1).and_then(|t| t.word()).and_then(|s| s.parse().ok()).unwrap_or(1) + 2
                } else {
                    1
                };
                let start = if n == 1 { 0 } else { 2 };
                for t in args.iter_mut().take(n).skip(start) {
                    if let Tok::W(s) = t {
                        *s = scoped(s);
                    }
                }
            }
            'w' => {
                if let Some(Tok::W(s)) = args.first_mut() {
                    *s = scoped(s);
                }
                for t in args.iter_mut().skip(1) {
                    if let Tok::W(s) = t {
                        *s = rename(s);
                    }
                }
            }
            'k' => {
                for t in args.iter_mut().take(2) {
                    if let Tok::W(s) = t {
                        *s = scoped(s);
                    }
                }
            }
            'x' => {}
            _ => {
                for t in args.iter_mut() {
                    if let Tok::W(s) = t {
                        if parse_number(s).is_none() {
                            *s = rename(s);
                        }
                    }
                }
            }
        }
        if c.kind() == 'x' {
            let sub_name = c.toks[nodes.len()].word().unwrap_or_default().to_string();
            // An `X` naming a model (an op-amp macromodel) is a device, not a subcircuit.
            if find_subckt(&chain, &sub_name).is_none() && find_model(&chain, &sub_name).is_some() {
                args.retain(|t| t.word() != Some(sub_name.as_str()));
                args.insert(0, Tok::W(rename(&sub_name)));
                cx.out.elements.push(FlatElement { line: c.line, name: scoped(&c.name), kind: 'x', nodes, args });
                continue;
            }
            let sub = find_subckt(&chain, &sub_name)
                .ok_or_else(|| FlattenError { line: c.line, message: format!("unknown subcircuit `{sub_name}`") })?;
            if sub.ports.len() != nodes.len() {
                return Err(FlattenError {
                    line: c.line,
                    message: format!("`{}` connects {} nodes; subcircuit `{sub_name}` has {} ports", c.name, nodes.len(), sub.ports.len()),
                });
            }
            let mut inner = Scope::new();
            for (k, v) in cx.out.params.iter().chain(scope.iter()) {
                inner.insert(k.clone(), *v);
            }
            for (k, v) in &sub.defaults {
                let x = eval(v, &inner).map_err(|m| FlattenError { line: c.line, message: format!("{sub_name}: `{k}`: {m}") })?;
                inner.insert(k.clone(), x);
            }
            for t in &args {
                if let Tok::Kv(k, v) = t {
                    let x = eval(v, scope).map_err(|m| FlattenError { line: c.line, message: format!("`{k}`: {m}") })?;
                    inner.insert(k.clone(), x);
                }
            }
            eval_params(&sub.body.params, &mut inner, c.line)?;
            let port_map: BTreeMap<String, String> = sub.ports.iter().cloned().zip(nodes).collect();
            let inst = scoped(&c.name);
            // Subcircuits are resolved in their definition's chain: push the defining body.
            expand(cx, &sub.body, &inst, &port_map, &inner, depth + 1)?;
            continue;
        }
        cx.out.elements.push(FlatElement { line: c.line, name: scoped(&c.name), kind: c.kind(), nodes, args });
    }
    cx.chain.pop();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::parse::parse;

    #[test]
    fn nested_subcircuits_and_params() {
        let deck = "t\n.param rv=2k\n.subckt div a b params: r=1k\nR1 a m {r}\nR2 m b {r*2}\n.ends\n\
                    .subckt two in out\nX1 in mid div r={rv}\nX2 mid out div\n.ends\nXT 1 0 two\n";
        let p = parse(deck, &|_| None).unwrap();
        let f = flatten(&p).unwrap();
        let names: Vec<&str> = f.elements.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["xt.x1.r1", "xt.x1.r2", "xt.x2.r1", "xt.x2.r2"]);
        assert_eq!(f.elements[0].nodes, ["1", "xt.x1.m"]);
        assert_eq!(f.elements[1].nodes, ["xt.x1.m", "xt.mid"]);
        assert_eq!(f.elements[3].nodes, ["xt.x2.m", "0"]);
        assert_eq!(f.elements[1].args[0], Tok::W(format!("{:e}", 4000.0)));
        assert_eq!(f.elements[3].args[0], Tok::W(format!("{:e}", 2000.0)));
    }
}
