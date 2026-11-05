//! Printing `simulation` sections: one statement per line, values with their units and an SI
//! prefix (`10 ns`, `3 MHz`, `1 %`), each reading back to exactly the value it came from.
//!
//! Each statement has its own printer, so a structured edit (add a probe, add an analysis, set a
//! tolerance) changes one line of the file and leaves the rest as written.

use super::{name, quote, record};
use crate::simulation::{SimAnalysis, SimMeasure, SimModel, SimProbe, SimSource, SimTolerance, SimWave, SimulationDef, SourceKind};
use runmat_design_syntax::json::format_f64;
use runmat_rules::{parse_quantity, BuiltinUnits};

const PREFIXES: &[(&str, f64)] = &[("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12), ("f", 1e-15)];

fn reads_as(text: &str, v: f64) -> bool {
    parse_quantity(text, &BuiltinUnits).is_ok_and(|q| q.value == v)
}

/// The mantissa `m` printed in as few decimals as read back to `m · scale` = `v` exactly.
fn mantissa(v: f64, scale: f64, suffix: &str) -> Option<String> {
    let m = v / scale;
    (0..=15).map(|d| format!("{m:.d$} {suffix}")).chain(std::iter::once(format!("{} {suffix}", format_f64(m)))).find(|t| reads_as(t, v))
}

/// `v` in `unit` (an SI base symbol, `%`, `dB` or none) for a `.circuit` file.
pub fn qty(v: f64, unit: &str) -> String {
    match unit {
        "" => format_f64(v),
        "dB" => format!("{} dB", format_f64(v)),
        "%" => mantissa(v, 0.01, "%").unwrap_or_else(|| format_f64(v)),
        _ if v == 0.0 || !v.is_finite() => format!("{} {unit}", format_f64(v)),
        _ => {
            let a = v.abs();
            PREFIXES
                .iter()
                .filter(|(_, s)| a >= *s * (1.0 - 1e-12) && a < *s * 1000.0)
                .find_map(|(p, s)| mantissa(v, *s, &format!("{p}{unit}")))
                .unwrap_or_else(|| format!("{} {unit}", format_f64(v)))
        }
    }
}

/// The value `v` means when written in `unit`: rounded to 12 significant digits under the prefix
/// [`qty`] would pick, then read as the file reads it. Edits store this, so a value typed as
/// 2e-5 s prints as `20 us` and reads back to itself.
pub fn as_written(v: f64, unit: &str) -> f64 {
    if !v.is_finite() || v == 0.0 || matches!(unit, "" | "dB") {
        return v;
    }
    let scale = if unit == "%" {
        0.01
    } else {
        PREFIXES.iter().find(|(_, s)| v.abs() >= *s * (1.0 - 1e-12) && v.abs() < *s * 1000.0).map(|(_, s)| *s).unwrap_or(1.0)
    };
    let m: f64 = format!("{:.11e}", v / scale).parse().unwrap_or(v / scale);
    let suffix = if unit == "%" { "%".to_string() } else { PREFIXES.iter().find(|(_, s)| *s == scale).map(|(p, _)| format!("{p}{unit}")).unwrap_or_else(|| unit.to_string()) };
    parse_quantity(&format!("{} {suffix}", format_f64(m)), &BuiltinUnits).map(|q| q.value).unwrap_or(v)
}

pub fn net_ref(n: &str) -> String {
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

/// The unit of a source's level: volts, or amperes for a current source.
fn level_unit(s: &SimSource) -> &'static str {
    if s.kind == SourceKind::Current {
        "A"
    } else {
        "V"
    }
}

pub fn source_line(k: &str, s: &SimSource) -> String {
    let u = level_unit(s);
    let mut e: Vec<(String, String)> = vec![("p".into(), net_ref(&s.p))];
    if s.n != "GND" {
        e.push(("n".into(), net_ref(&s.n)));
    }
    if s.kind == SourceKind::Current {
        e.push(("current".into(), "true".into()));
    }
    if s.dc != 0.0 {
        e.push(("dc".into(), qty(s.dc, u)));
    }
    if s.ac != 0.0 {
        e.push(("ac".into(), qty(s.ac, u)));
    }
    match &s.wave {
        Some(SimWave::Pulse { from, to, delay, rise, fall, width, period }) => {
            let mut p = vec![("from".to_string(), qty(*from, u)), ("to".into(), qty(*to, u)), ("delay".into(), qty(*delay, "s")), ("rise".into(), qty(*rise, "s")), ("fall".into(), qty(*fall, "s"))];
            if width.is_finite() {
                p.push(("width".into(), qty(*width, "s")));
            }
            if period.is_finite() {
                p.push(("period".into(), qty(*period, "s")));
            }
            e.push(("pulse".into(), record(&p)));
        }
        Some(SimWave::Sin { offset, amplitude, freq, delay }) => {
            e.push(("sin".into(), record(&[("offset".into(), qty(*offset, u)), ("amplitude".into(), qty(*amplitude, u)), ("freq".into(), qty(*freq, "Hz")), ("delay".into(), qty(*delay, "s"))])));
        }
        Some(SimWave::Pwl(pts)) => {
            let items: Vec<String> = pts.iter().map(|(t, v)| format!("[{}, {}]", qty(*t, "s"), qty(*v, u))).collect();
            e.push(("pwl".into(), format!("[{}]", items.join(", "))));
        }
        None => {}
    }
    format!("source {} = {}", name(k), record(&e))
}

/// The unit of an op-amp macromodel parameter.
fn opamp_unit(param: &str) -> &'static str {
    match param {
        "gbw" => "Hz",
        "rout" | "rin" => "ohm",
        "vdrop" | "vos" => "V",
        _ => "",
    }
}

pub fn model_line(k: &str, m: &SimModel) -> String {
    let e = match m {
        SimModel::OpAmp { params, pins: p } => {
            let ps: Vec<(String, String)> = params.iter().map(|(a, b)| (a.clone(), qty(*b, opamp_unit(a)))).collect();
            vec![("opamp".into(), record(&ps)), ("pins".into(), pins(p))]
        }
        SimModel::Subckt { library, name: n, pins: p } => vec![("library".into(), quote(library)), ("subckt".into(), name(n)), ("pins".into(), pins(p))],
    };
    format!("model {} = {}", name(k), record(&e))
}

pub fn tolerance_line(k: &str, t: &SimTolerance) -> String {
    format!("tolerance {} = {}", name(k), record(&[("match".into(), quote(&t.pattern)), ("value".into(), qty(t.value, "%"))]))
}

/// The unit a DC sweep steps in: its source's level unit when the section defines the source.
fn sweep_unit(sim: &SimulationDef, source: &str) -> &'static str {
    sim.sources.get(source).map(level_unit).unwrap_or("V")
}

pub fn analysis_line(sim: &SimulationDef, k: &str, a: &SimAnalysis) -> String {
    let body = match a {
        SimAnalysis::Op => "op".to_string(),
        SimAnalysis::Dc { source, from, to, step } => {
            let u = sweep_unit(sim, source);
            format!("dc {}", record(&[("source".into(), name(source)), ("from".into(), qty(*from, u)), ("to".into(), qty(*to, u)), ("step".into(), qty(*step, u))]))
        }
        SimAnalysis::Ac { from, to, points } => format!("ac {}", record(&[("from".into(), qty(*from, "Hz")), ("to".into(), qty(*to, "Hz")), ("points".into(), points.to_string())])),
        SimAnalysis::Tran { stop, step, max_step } => {
            let mut e = vec![("stop".to_string(), qty(*stop, "s")), ("step".into(), qty(*step, "s"))];
            if let Some(m) = max_step {
                e.push(("max_step".into(), qty(*m, "s")));
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
    format!("analysis {} = {body}", name(k))
}

pub fn probe_line(probes: &[SimProbe]) -> String {
    let items: Vec<String> = probes
        .iter()
        .map(|p| match p {
            SimProbe::Net(n) => net_ref(n),
            // Pins quote as the `connect` lists write them (`"RS.1"`, `"U.IN+"`).
            SimProbe::Pin { .. } => quote(&p.name()),
        })
        .collect();
    format!("probe = [{}]", items.join(", "))
}

/// The x unit of the analysis a measure reads (a Monte Carlo reads its nominal analysis).
fn x_unit(sim: &SimulationDef, of: &str) -> &'static str {
    match sim.analyses.get(of) {
        Some(SimAnalysis::MonteCarlo { of: inner, .. }) => match sim.analyses.get(inner) {
            Some(SimAnalysis::Ac { .. }) => "Hz",
            Some(SimAnalysis::Dc { source, .. }) => sweep_unit(sim, source),
            _ => "s",
        },
        Some(SimAnalysis::Ac { .. }) => "Hz",
        Some(SimAnalysis::Dc { source, .. }) => sweep_unit(sim, source),
        _ => "s",
    }
}

pub fn measure_line(sim: &SimulationDef, k: &str, m: &SimMeasure) -> String {
    let xu = x_unit(sim, &m.of);
    let mut e = vec![("of".to_string(), name(&m.of)), (m.kind.word().to_string(), net_ref(&m.net))];
    e.extend(m.args.iter().map(|(a, b)| {
        let unit = match a.as_str() {
            "band" => "%",
            "drop" => "dB",
            _ => xu,
        };
        (a.clone(), qty(*b, unit))
    }));
    let limit = |v: f64| match m.kind.unit() {
        "%" => format!("{} %", format_f64(v)),
        "°" => format!("{} deg", format_f64(v)),
        u => qty(v, u),
    };
    let mut spec = vec![];
    if let Some(v) = m.spec_min {
        spec.push(("min".to_string(), limit(v)));
    }
    if let Some(v) = m.spec_max {
        spec.push(("max".to_string(), limit(v)));
    }
    if !spec.is_empty() {
        e.push(("spec".into(), record(&spec)));
    }
    format!("measure {} = {}", name(k), record(&e))
}

pub fn option_line(k: &str, v: &str) -> String {
    let text = if v.parse::<f64>().is_ok() || v.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') { v.to_string() } else { quote(v) };
    format!("option {} = {text}", name(k))
}

/// The section's statements, in the order [`print`] writes them.
pub fn lines(sim: &SimulationDef) -> Vec<String> {
    let mut lines = vec![];
    if let Some(s) = &sim.sheet {
        lines.push(format!("sheet = {}", name(s)));
    }
    lines.extend(sim.sources.iter().map(|(k, s)| source_line(k, s)));
    lines.extend(sim.models.iter().map(|(k, m)| model_line(k, m)));
    lines.extend(sim.tolerances.iter().map(|(k, t)| tolerance_line(k, t)));
    lines.extend(sim.analyses.iter().map(|(k, a)| analysis_line(sim, k, a)));
    if !sim.probes.is_empty() {
        lines.push(probe_line(&sim.probes));
    }
    lines.extend(sim.measures.iter().map(|(k, m)| measure_line(sim, k, m)));
    lines.extend(sim.options.iter().map(|(k, v)| option_line(k, v)));
    lines
}

/// The section's text.
pub fn print(sim: &SimulationDef) -> String {
    let mut out = format!("{} {} {{\n", super::simulation::WORD, name(&sim.name));
    for l in lines(sim) {
        out.push_str(runmat_design_syntax::json::INDENT);
        out.push_str(&l);
        out.push('\n');
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::{as_written, qty};
    use runmat_rules::{parse_quantity, BuiltinUnits};

    #[test]
    fn quantities_print_with_a_prefix_and_read_back_exactly() {
        let cases = [(1e-8, "s", "10 ns"), (2e-5, "s", "20 us"), (3e6, "Hz", "3 MHz"), (100.0, "Hz", "100 Hz"), (0.25, "V", "250 mV"), (4.096, "V", "4.096 V"), (0.01, "%", "1 %"), (0.05, "%", "5 %"), (10.0, "ohm", "10 ohm"), (1.5e3, "ohm", "1.5 kohm"), (0.0, "s", "0 s"), (-1.75, "V", "-1.75 V"), (3.0103, "dB", "3.0103 dB")];
        // As stored: the value the text reads as (a typed 2e-5 is not bit-equal to 20 × 1e-6).
        for (v, unit, text) in cases {
            assert_eq!(qty(as_written(v, unit), unit), text);
        }
        for v in [1.0 / 3.0, 159_154.943_091_895_3, 1e-15 * 7.0, 6.02e23, 1e-20, 2.2e-9 * 1.05] {
            for unit in ["s", "V", "Hz", "F", "A", "%"] {
                let t = qty(v, unit);
                assert_eq!(parse_quantity(&t, &BuiltinUnits).unwrap().value, v, "{t}");
            }
        }
    }

    #[test]
    fn typed_values_store_as_they_are_written() {
        for (v, unit, text) in [(2e-5, "s", "20 us"), (2e-8, "s", "20 ns"), (0.001, "%", "0.1 %"), (1e8, "Hz", "100 MHz"), (0.3, "V", "300 mV"), (159_154.943_091_895_35, "Hz", "159.154943092 kHz")] {
            let w = as_written(v, unit);
            assert_eq!(qty(w, unit), text);
            assert_eq!(parse_quantity(text, &BuiltinUnits).unwrap().value, w);
        }
    }
}
