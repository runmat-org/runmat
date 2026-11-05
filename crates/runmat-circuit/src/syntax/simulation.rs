//! `simulation <name> { … }` sections: reading into [`SimulationDef`] and printing back.
//!
//! Statements: `sheet = <sheet>`, `source <name> = { … }`, `model <part> = { … }`,
//! `tolerance <name> = { match = "R*", value = 1 % }`, `analysis <name> = op | dc { … } |
//! ac { … } | tran { … } | monte_carlo { … }`, `probe = [net.A, …]`,
//! `measure <name> = { of = <analysis>, <kind> = net.X, … }` and `option <name> = <value>`.

use super::values::{diagnostic, val, Val, V};
use super::{name, quote, record, Read, Reader};
use crate::simulation::{MeasureKind, SimAnalysis, SimMeasure, SimModel, SimSource, SimTolerance, SimWave, SimulationDef, SourceKind};
use runmat_design_syntax::ast::{self, Expr, Headed, Statement};
use runmat_design_syntax::json::format_f64;
use runmat_rules::{parse_quantity, BuiltinUnits};
use runmat_types::Span;
use std::collections::BTreeMap;

pub const WORD: &str = super::words::SIMULATION;
pub const STATEMENTS: &[&str] = &["sheet", "source", "model", "tolerance", "analysis", "probe", "measure", "option"];

fn si(v: &Val) -> Read<f64> {
    match &v.v {
        V::Num(t) => t.parse().map_err(|_| diagnostic(v.span, "expected a number")),
        V::Qty(t, unit) if unit == "dB" => t.parse().map_err(|_| diagnostic(v.span, "expected a number of dB")),
        V::Qty(t, unit) => parse_quantity(&format!("{t} {unit}"), &BuiltinUnits)
            .map(|q| q.value)
            .map_err(|e| diagnostic(v.span, e)),
        _ => Err(diagnostic(v.span, "expected a number or quantity")),
    }
}

fn word(v: &Val) -> Read<String> {
    match &v.v {
        V::Word(s) | V::Str(s) => Ok(s.clone()),
        _ => Err(diagnostic(v.span, "expected a name")),
    }
}

/// `net.X` or `"net.+5V"` → the net's name.
fn net(v: &Val) -> Read<String> {
    let w = word(v)?;
    w.strip_prefix("net.").map(str::to_string).ok_or_else(|| diagnostic(v.span, format!("expected a net (`net.{w}`)")))
}

fn rec(v: &Val) -> Read<&[(String, Val)]> {
    match &v.v {
        V::Rec(r) => Ok(r),
        _ => Err(diagnostic(v.span, "expected a record `{ … }`")),
    }
}

/// An expression, allowing `word { record }` applications: `(word, argument)`.
fn applied(expr: Option<Expr>, span: Span) -> Read<(Option<String>, Option<Val>)> {
    match expr {
        Some(Expr::PrefixApplication(a)) => {
            let callee = a.callee().map(|c| val(&c)).transpose()?.map(|v| word(&v)).transpose()?;
            Ok((callee, a.argument().map(|x| val(&x)).transpose()?))
        }
        Some(e) => {
            let v = val(&e)?;
            match &v.v {
                V::Word(w) => Ok((Some(w.clone()), None)),
                _ => Ok((None, Some(v))),
            }
        }
        None => Err(diagnostic(span, "expected a value")),
    }
}

fn fields<'a>(v: &'a Val, allowed: &[&str], what: &str) -> Read<BTreeMap<&'a str, &'a Val>> {
    let mut out = BTreeMap::new();
    for (k, x) in rec(v)? {
        if !allowed.contains(&k.as_str()) {
            return Err(diagnostic(x.span, format!("{what} has no `{k}` (it takes {})", allowed.join(", "))));
        }
        out.insert(k.as_str(), x);
    }
    Ok(out)
}

fn req<'a>(f: &BTreeMap<&str, &'a Val>, key: &str, span: Span, what: &str) -> Read<&'a Val> {
    f.get(key).copied().ok_or_else(|| diagnostic(span, format!("{what} needs `{key}`")))
}

pub(super) fn read(reader: &mut Reader<'_>, block: &ast::BlockStatement) -> Option<SimulationDef> {
    let mut sim = SimulationDef {
        name: block.name().map(|n| n.text()).unwrap_or_default(),
        source: reader.key(block.span()),
        ..Default::default()
    };
    let body = block.block()?;
    for statement in body.statements() {
        let span = statement.span();
        let r: Read<()> = (|| {
            match &statement {
                Statement::Assignment(a) if a.target() == ["sheet"] => {
                    sim.sheet = Some(word(&val(&a.value().ok_or_else(|| diagnostic(span, "`sheet` needs a sheet name"))?)?)?);
                }
                Statement::Assignment(a) if a.target() == ["probe"] => {
                    let v = val(&a.value().ok_or_else(|| diagnostic(span, "`probe` needs a list of nets"))?)?;
                    sim.probes = match &v.v {
                        V::List(items) => items.iter().map(net).collect::<Read<_>>()?,
                        _ => vec![net(&v)?],
                    };
                }
                Statement::Value(value) => {
                    let key = value.keyword();
                    let nm = value.name().map(|n| n.text()).unwrap_or_default();
                    match key.as_str() {
                        "source" => {
                            let v = val(&value.value().ok_or_else(|| diagnostic(span, "a source needs `{ p = net.X, … }`"))?)?;
                            sim.sources.insert(nm, read_source(&v, span)?);
                        }
                        "model" => {
                            let v = val(&value.value().ok_or_else(|| diagnostic(span, "a model needs `{ opamp = { … } }` or `{ library, subckt }`"))?)?;
                            sim.models.insert(nm, read_model(&v, span)?);
                        }
                        "tolerance" => {
                            let v = val(&value.value().ok_or_else(|| diagnostic(span, "a tolerance needs `{ match, value }`"))?)?;
                            let f = fields(&v, &["match", "value"], "a tolerance")?;
                            let t = SimTolerance { pattern: word(req(&f, "match", span, "a tolerance")?)?, value: si(req(&f, "value", span, "a tolerance")?)? };
                            sim.tolerances.insert(nm, t);
                        }
                        "analysis" => {
                            let a = read_analysis(applied(value.value(), span)?, span)?;
                            sim.analyses.insert(nm, a);
                        }
                        "measure" => {
                            let v = val(&value.value().ok_or_else(|| diagnostic(span, "a measure needs a record"))?)?;
                            sim.measures.insert(nm, read_measure(&v, span)?);
                        }
                        "option" => {
                            let v = val(&value.value().ok_or_else(|| diagnostic(span, "an option needs a value"))?)?;
                            let text = match &v.v {
                                V::Num(t) | V::Word(t) | V::Str(t) => t.clone(),
                                _ => format_f64(si(&v)?),
                            };
                            sim.options.insert(nm, text);
                        }
                        other => return Err(diagnostic(span, format!("`{other}` does not start a simulation statement ({})", STATEMENTS.join(", ")))),
                    }
                }
                _ => return Err(diagnostic(span, format!("expected one of {}", STATEMENTS.join(", ")))),
            }
            Ok(())
        })();
        reader.note(r);
    }
    for (k, m) in &sim.measures {
        if !sim.analyses.contains_key(&m.of) {
            reader.diagnostics.push(diagnostic(block.span(), format!("measure `{k}` reads analysis `{}`, which this simulation does not declare", m.of)));
        }
    }
    Some(sim)
}

fn read_source(v: &Val, span: Span) -> Read<SimSource> {
    let f = fields(v, &["p", "n", "dc", "ac", "pulse", "sin", "pwl", "current"], "a source")?;
    let kind = match f.get("current") {
        Some(c) if word(c)? == "true" => SourceKind::Current,
        _ => SourceKind::Voltage,
    };
    let num = |k: &str| f.get(k).map(|x| si(x)).transpose();
    let mut wave = None;
    if let Some(p) = f.get("pulse") {
        let g = fields(p, &["from", "to", "delay", "rise", "fall", "width", "period"], "a pulse")?;
        let n = |k: &str, d: f64| g.get(k).map(|x| si(x)).transpose().map(|x| x.unwrap_or(d));
        let rise = n("rise", 0.0)?;
        wave = Some(SimWave::Pulse { from: n("from", 0.0)?, to: si(req(&g, "to", span, "a pulse")?)?, delay: n("delay", 0.0)?, rise, fall: n("fall", rise)?, width: n("width", f64::INFINITY)?, period: n("period", f64::INFINITY)? });
    } else if let Some(s) = f.get("sin") {
        let g = fields(s, &["offset", "amplitude", "freq", "delay"], "a sine")?;
        let n = |k: &str, d: f64| g.get(k).map(|x| si(x)).transpose().map(|x| x.unwrap_or(d));
        wave = Some(SimWave::Sin { offset: n("offset", 0.0)?, amplitude: si(req(&g, "amplitude", span, "a sine")?)?, freq: si(req(&g, "freq", span, "a sine")?)?, delay: n("delay", 0.0)? });
    } else if let Some(p) = f.get("pwl") {
        let V::List(items) = &p.v else { return Err(diagnostic(p.span, "pwl is a list of [time, value] pairs")) };
        let mut pts = vec![];
        for it in items {
            match &it.v {
                V::List(pair) if pair.len() == 2 => pts.push((si(&pair[0])?, si(&pair[1])?)),
                _ => return Err(diagnostic(it.span, "expected [time, value]")),
            }
        }
        wave = Some(SimWave::Pwl(pts));
    }
    Ok(SimSource {
        kind,
        p: net(req(&f, "p", span, "a source")?)?,
        n: f.get("n").map(|x| net(x)).transpose()?.unwrap_or_else(|| "GND".into()),
        dc: num("dc")?.unwrap_or(0.0),
        ac: num("ac")?.unwrap_or(0.0),
        wave,
    })
}

fn read_model(v: &Val, span: Span) -> Read<SimModel> {
    let f = fields(v, &["opamp", "library", "subckt", "pins"], "a model")?;
    let pins = match f.get("pins") {
        Some(p) => match &p.v {
            V::List(items) => items.iter().map(word).collect::<Read<Vec<_>>>()?,
            _ => return Err(diagnostic(p.span, "pins is a list of pin names")),
        },
        None => vec![],
    };
    if let Some(o) = f.get("opamp") {
        let params = rec(o)?.iter().map(|(k, x)| Ok((k.clone(), si(x)?))).collect::<Read<BTreeMap<_, _>>>()?;
        if pins.len() != 3 && pins.len() != 5 {
            return Err(diagnostic(span, "an op-amp model lists its pins as [in+, in−, out] or [in+, in−, out, v+, v−]"));
        }
        return Ok(SimModel::OpAmp { params, pins });
    }
    Ok(SimModel::Subckt { library: word(req(&f, "library", span, "a model")?)?, name: word(req(&f, "subckt", span, "a model")?)?, pins })
}

fn read_analysis((callee, arg): (Option<String>, Option<Val>), span: Span) -> Read<SimAnalysis> {
    let w = callee.ok_or_else(|| diagnostic(span, "an analysis is op, dc { … }, ac { … }, tran { … } or monte_carlo { … }"))?;
    let empty = Val { v: V::Rec(vec![]), span };
    let a = arg.unwrap_or(empty);
    let what = format!("a `{w}` analysis");
    Ok(match w.as_str() {
        "op" => SimAnalysis::Op,
        "dc" => {
            let f = fields(&a, &["source", "from", "to", "step"], &what)?;
            SimAnalysis::Dc { source: word(req(&f, "source", span, &what)?)?, from: si(req(&f, "from", span, &what)?)?, to: si(req(&f, "to", span, &what)?)?, step: si(req(&f, "step", span, &what)?)? }
        }
        "ac" => {
            let f = fields(&a, &["from", "to", "points"], &what)?;
            SimAnalysis::Ac { from: si(req(&f, "from", span, &what)?)?, to: si(req(&f, "to", span, &what)?)?, points: f.get("points").map(|x| si(x)).transpose()?.unwrap_or(50.0) as u32 }
        }
        "tran" => {
            let f = fields(&a, &["stop", "step", "max_step"], &what)?;
            let stop = si(req(&f, "stop", span, &what)?)?;
            SimAnalysis::Tran { stop, step: f.get("step").map(|x| si(x)).transpose()?.unwrap_or(stop / 1000.0), max_step: f.get("max_step").map(|x| si(x)).transpose()? }
        }
        "monte_carlo" => {
            let f = fields(&a, &["of", "runs", "seed", "distribution"], &what)?;
            let uniform = match f.get("distribution") {
                Some(d) => match word(d)?.as_str() {
                    "uniform" => true,
                    "normal" | "gauss" => false,
                    o => return Err(diagnostic(d.span, format!("distribution is normal or uniform, not `{o}`"))),
                },
                None => false,
            };
            SimAnalysis::MonteCarlo { of: word(req(&f, "of", span, &what)?)?, runs: si(req(&f, "runs", span, &what)?)? as u32, seed: f.get("seed").map(|x| si(x)).transpose()?.unwrap_or(1.0) as u64, uniform }
        }
        o => return Err(diagnostic(span, format!("`{o}` is not an analysis (op, dc, ac, tran, monte_carlo)"))),
    })
}

fn read_measure(v: &Val, span: Span) -> Read<SimMeasure> {
    let mut of = None;
    let mut kind = None;
    let mut args = BTreeMap::new();
    for (k, x) in rec(v)? {
        if k == "of" {
            of = Some(word(x)?);
        } else if let Some((_, mk)) = MeasureKind::ALL.iter().find(|(w, _)| w == k) {
            if kind.is_some() {
                return Err(diagnostic(x.span, "a measure computes one quantity"));
            }
            kind = Some((*mk, net(x)?));
        } else if ["at", "from", "to", "band", "drop"].contains(&k.as_str()) {
            args.insert(k.clone(), si(x)?);
        } else {
            return Err(diagnostic(x.span, format!("a measure has no `{k}`")));
        }
    }
    let (kind, net) = kind.ok_or_else(|| {
        let all: Vec<&str> = MeasureKind::ALL.iter().map(|(w, _)| *w).collect();
        diagnostic(span, format!("a measure names what it computes: {}", all.join(", ")))
    })?;
    Ok(SimMeasure { of: of.ok_or_else(|| diagnostic(span, "a measure needs `of = <analysis>`"))?, kind, net, args })
}

fn q(v: f64) -> String {
    format_f64(v)
}

fn net_ref(n: &str) -> String {
    let t = format!("net.{n}");
    if n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && n.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
        t
    } else {
        quote(&t)
    }
}

fn pins(p: &[String]) -> String {
    let items: Vec<String> = p.iter().map(|s| name(s)).collect();
    format!("[{}]", items.join(", "))
}

/// The section's text (one statement per line, SI values).
pub(super) fn print(sim: &SimulationDef) -> String {
    let mut lines = vec![];
    if let Some(s) = &sim.sheet {
        lines.push(format!("sheet = {}", name(s)));
    }
    for (k, s) in &sim.sources {
        let mut e: Vec<(String, String)> = vec![("p".into(), net_ref(&s.p))];
        if s.n != "GND" {
            e.push(("n".into(), net_ref(&s.n)));
        }
        if s.kind == SourceKind::Current {
            e.push(("current".into(), "true".into()));
        }
        if s.dc != 0.0 {
            e.push(("dc".into(), q(s.dc)));
        }
        if s.ac != 0.0 {
            e.push(("ac".into(), q(s.ac)));
        }
        match &s.wave {
            Some(SimWave::Pulse { from, to, delay, rise, fall, width, period }) => {
                let mut p = vec![("from".to_string(), q(*from)), ("to".into(), q(*to)), ("delay".into(), q(*delay)), ("rise".into(), q(*rise)), ("fall".into(), q(*fall))];
                if width.is_finite() {
                    p.push(("width".into(), q(*width)));
                }
                if period.is_finite() {
                    p.push(("period".into(), q(*period)));
                }
                e.push(("pulse".into(), record(&p)));
            }
            Some(SimWave::Sin { offset, amplitude, freq, delay }) => {
                e.push(("sin".into(), record(&[("offset".into(), q(*offset)), ("amplitude".into(), q(*amplitude)), ("freq".into(), q(*freq)), ("delay".into(), q(*delay))])));
            }
            Some(SimWave::Pwl(pts)) => {
                let items: Vec<String> = pts.iter().map(|(t, v)| format!("[{}, {}]", q(*t), q(*v))).collect();
                e.push(("pwl".into(), format!("[{}]", items.join(", "))));
            }
            None => {}
        }
        lines.push(format!("source {} = {}", name(k), record(&e)));
    }
    for (k, m) in &sim.models {
        let e = match m {
            SimModel::OpAmp { params, pins: p } => {
                let ps: Vec<(String, String)> = params.iter().map(|(a, b)| (a.clone(), q(*b))).collect();
                vec![("opamp".into(), record(&ps)), ("pins".into(), pins(p))]
            }
            SimModel::Subckt { library, name: n, pins: p } => vec![("library".into(), quote(library)), ("subckt".into(), name(n)), ("pins".into(), pins(p))],
        };
        lines.push(format!("model {} = {}", name(k), record(&e)));
    }
    for (k, t) in &sim.tolerances {
        lines.push(format!("tolerance {} = {}", name(k), record(&[("match".into(), quote(&t.pattern)), ("value".into(), q(t.value))])));
    }
    for (k, a) in &sim.analyses {
        let body = match a {
            SimAnalysis::Op => "op".to_string(),
            SimAnalysis::Dc { source, from, to, step } => format!("dc {}", record(&[("source".into(), name(source)), ("from".into(), q(*from)), ("to".into(), q(*to)), ("step".into(), q(*step))])),
            SimAnalysis::Ac { from, to, points } => format!("ac {}", record(&[("from".into(), q(*from)), ("to".into(), q(*to)), ("points".into(), points.to_string())])),
            SimAnalysis::Tran { stop, step, max_step } => {
                let mut e = vec![("stop".to_string(), q(*stop)), ("step".into(), q(*step))];
                if let Some(m) = max_step {
                    e.push(("max_step".into(), q(*m)));
                }
                format!("tran {}", record(&e))
            }
            SimAnalysis::MonteCarlo { of, runs, seed, uniform } => {
                let mut e = vec![("of".to_string(), name(of)), ("runs".into(), runs.to_string()), ("seed".into(), seed.to_string())];
                if *uniform {
                    e.push(("distribution".into(), "uniform".into()));
                }
                format!("monte_carlo {}", record(&e))
            }
        };
        lines.push(format!("analysis {} = {body}", name(k)));
    }
    if !sim.probes.is_empty() {
        let items: Vec<String> = sim.probes.iter().map(|p| net_ref(p)).collect();
        lines.push(format!("probe = [{}]", items.join(", ")));
    }
    for (k, m) in &sim.measures {
        let mut e = vec![("of".to_string(), name(&m.of)), (m.kind.word().to_string(), net_ref(&m.net))];
        e.extend(m.args.iter().map(|(a, b)| (a.clone(), q(*b))));
        lines.push(format!("measure {} = {}", name(k), record(&e)));
    }
    for (k, v) in &sim.options {
        let text = if v.parse::<f64>().is_ok() || v.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') { v.clone() } else { quote(v) };
        lines.push(format!("option {} = {text}", name(k)));
    }
    let mut out = format!("{WORD} {} {{\n", name(&sim.name));
    for l in lines {
        out.push_str(runmat_design_syntax::json::INDENT);
        out.push_str(&l);
        out.push('\n');
    }
    out.push('}');
    out
}
