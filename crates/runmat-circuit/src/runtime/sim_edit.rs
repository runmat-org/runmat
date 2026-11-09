//! Structured edits to a design's `simulation` section, each the smallest change to the file:
//!
//! - `simulate.probe` (a picked net or pin), `simulate.probe-net` and `simulate.probe-pin` (the
//!   net or pin selected on the schematic, from its context menu): adds `net.X` (its voltage) or
//!   `R1.2` (the current into the pin) to `probe = [ … ]` (or writes that statement);
//! - `simulate.monte-carlo`, and `simulate.op-point`, `simulate.dc-sweep`,
//!   `simulate.transient`, `simulate.ac`, `simulate.noise`, `simulate.temperature` with their kind fixed: writes `analysis <name> = …`,
//!   replacing the analysis of that name when there is one;
//! - `simulate.set-tolerance`: writes `tolerance <name> = { match, value }`, replacing the one of
//!   that name or pattern.
//!
//! A new statement goes after the last one of its kind (or of the kinds printed before it), so
//! the section keeps the printer's order. A design without a section gets `simulation main { … }`
//! at the end of its `circuit` block. Values print with units (`10 ns`, `3 MHz`, `1 %`).

use super::edit::{EditError, EditOutput, EditRequest};
use super::{file_change, Project};
use crate::simulation::{SimAnalysis, SimProbe, SimTolerance, SimulationDef};
use crate::syntax::board_edit::SourceChange;
use crate::syntax::parse_circuit;
use crate::syntax::simulation_print::{analysis_line, as_written, probe_line, tolerance_line};
use crate::syntax::simulation_spans::{simulation_layout, SectionAt};
use runmat_design_syntax::edit::TextEdit;
use runmat_types::Span;
use serde_json::{Map, Value};

/// Statement kinds in print order: a new statement goes after the last of its kind or of the
/// kinds before it.
const ORDER: &[&str] = &[
    "sheet",
    "source",
    "model",
    "tolerance",
    "analysis",
    "probe",
    "measure",
    "option",
];

pub(super) fn run(project: &Project, request: &EditRequest) -> Result<EditOutput, EditError> {
    let v = &request.values;
    let text = &project.circuit_text;
    let layout = simulation_layout(text);
    let wanted = string(v, "simulation");
    let section =
        match wanted.as_deref() {
            Some(name) => Some(layout.sections.iter().find(|s| s.name == name).ok_or_else(
                || {
                    EditError::new(format!(
                        "{} has no simulation `{name}`",
                        project.circuit_path
                    ))
                },
            )?),
            None => match layout.sections.as_slice() {
                [] => None,
                [one] => Some(one),
                many => {
                    return Err(EditError::new(format!(
                        "{} has several simulations ({}); name one",
                        project.circuit_path,
                        many.iter()
                            .map(|s| s.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )))
                }
            },
        };
    let empty = SimulationDef {
        name: wanted.clone().unwrap_or_else(|| "main".into()),
        ..Default::default()
    };
    let def = section
        .and_then(|s| project.own.simulations.get(&s.name))
        .unwrap_or(&empty);
    let (keyword, name, line, summary) = match request.command.as_str() {
        "simulate.probe" | "simulate.probe-net" | "simulate.probe-pin" => {
            let at = string(v, "at")
                .or_else(|| string(v, "net"))
                .or_else(|| request.selection.first().cloned())
                .ok_or_else(|| EditError::new("name the net or pin to probe"))?;
            let probe = match at.strip_prefix("net:").or_else(|| at.strip_prefix("net.")) {
                Some(net) => SimProbe::Net(net.to_string()),
                None => {
                    let pin = at.strip_prefix("pin:").unwrap_or(&at);
                    match pin.split_once('.').or_else(|| pin.split_once(':')) {
                        Some((component, pin))
                            if string(v, "kind").as_deref() != Some("voltage") =>
                        {
                            SimProbe::Pin {
                                component: component.into(),
                                pin: pin.into(),
                            }
                        }
                        _ => SimProbe::Net(at.clone()),
                    }
                }
            };
            if def.probes.contains(&probe) {
                return Err(EditError::new(format!(
                    "{} is already probed",
                    probe.label()
                )));
            }
            let mut probes = def.probes.clone();
            probes.push(probe.clone());
            (
                "probe",
                String::new(),
                probe_line(&probes),
                format!("probe {}", probe.label()),
            )
        }
        "simulate.set-tolerance" => {
            let pattern = string(v, "match")
                .ok_or_else(|| EditError::new("name the parts it applies to (`R*`, `C?`, `R1`)"))?;
            let value = number(v, "value")
                .or_else(|| number(v, "percent").map(|p| p / 100.0))
                .map(|x| as_written(x, "%"))
                .ok_or_else(|| EditError::new("give the tolerance (`1 %`)"))?;
            if !(value > 0.0 && value < 1.0) {
                return Err(EditError::new("a tolerance is between 0 % and 100 %"));
            }
            let name = string(v, "name")
                .or_else(|| {
                    def.tolerances
                        .iter()
                        .find(|(_, t)| t.pattern == pattern)
                        .map(|(k, _)| k.clone())
                })
                .unwrap_or_else(|| tolerance_name(&pattern));
            let t = SimTolerance {
                pattern: pattern.clone(),
                value,
            };
            (
                "tolerance",
                name.clone(),
                tolerance_line(&name, &t),
                format!("tolerance {name} ({pattern} ±{} %)", value * 100.0),
            )
        }
        command => {
            let kind = match command {
                "simulate.op-point" => "op".to_string(),
                "simulate.dc-sweep" => "dc".into(),
                "simulate.transient" => "tran".into(),
                "simulate.ac" => "ac".into(),
                "simulate.noise" => "noise".into(),
                "simulate.temperature" => "temperature".into(),
                "simulate.monte-carlo" => string(v, "kind").ok_or_else(|| {
                    EditError::new("pick the analysis (op, dc, ac, tran, monte_carlo)")
                })?,
                other => {
                    return Err(EditError::new(format!(
                        "`{other}` is not a simulation edit"
                    )))
                }
            };
            let kind = kind.as_str();
            let a = analysis(kind, v, def)?;
            let name = string(v, "name").unwrap_or_else(|| default_name(kind, def));
            (
                "analysis",
                name.clone(),
                analysis_line(def, &name, &a),
                format!("analysis {name} ({kind})"),
            )
        }
    };
    let edit = match section {
        Some(s) => statement_edit(text, s, keyword, &name, &line),
        None => {
            let close = layout.circuit_close.ok_or_else(|| {
                EditError::new(format!("{} has no `circuit` block", project.circuit_path))
            })?;
            let at = line_start(text, close);
            let indent = indent_of(text, at) + "    ";
            TextEdit::replace(
                Span { start: at, end: at },
                format!(
                    "\n{indent}simulation {} {{\n{indent}    {line}\n{indent}}}\n",
                    def.name
                ),
            )
        }
    };
    let mut source = text.clone();
    source.replace_range(edit.span.start..edit.span.end, &edit.text);
    let parsed = parse_circuit(&project.circuit_path, &source);
    if !parsed.diagnostics.is_empty() {
        return Err(EditError {
            message: "the edit would not read back".into(),
            diagnostics: Vec::new(),
        });
    }
    let change = SourceChange {
        edits: vec![edit],
        source,
        reprinted: false,
    };
    Ok(EditOutput {
        summary: format!("{summary} in simulation {}", def.name),
        files: vec![file_change(&project.circuit_path, text, change)],
    })
}

/// Replaces the statement `keyword name` when the section has it; else inserts the line after
/// the last statement of that kind or of the kinds before it, or first in the section.
fn statement_edit(text: &str, s: &SectionAt, keyword: &str, name: &str, line: &str) -> TextEdit {
    if let Some(at) = s.find(keyword, name) {
        return TextEdit::replace(at.span, line);
    }
    let rank = ORDER
        .iter()
        .position(|k| *k == keyword)
        .unwrap_or(ORDER.len());
    let indent = s
        .statements
        .first()
        .map(|f| indent_of(text, line_start(text, f.span.start)))
        .unwrap_or_else(|| indent_of(text, line_start(text, s.span.start)) + "    ");
    let anchor = s.last_of(&ORDER[..=rank.min(ORDER.len() - 1)]);
    let at = match anchor {
        Some(a) => line_end(text, a.span.end),
        None => match s.statements.first() {
            Some(f) => line_start(text, f.span.start),
            None => line_start(text, s.close),
        },
    };
    let piece = if anchor.is_some() {
        format!("\n{indent}{line}")
    } else {
        format!("{indent}{line}\n")
    };
    TextEdit::replace(Span { start: at, end: at }, piece)
}

fn line_start(text: &str, at: usize) -> usize {
    text[..at].rfind('\n').map(|k| k + 1).unwrap_or(0)
}

fn line_end(text: &str, at: usize) -> usize {
    text[at..].find('\n').map(|k| at + k).unwrap_or(text.len())
}

fn indent_of(text: &str, line_start: usize) -> String {
    text[line_start..]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect()
}

fn string(v: &Map<String, Value>, id: &str) -> Option<String> {
    v.get(id)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

fn number(v: &Map<String, Value>, id: &str) -> Option<f64> {
    v.get(id).and_then(Value::as_f64).filter(|x| x.is_finite())
}

fn need(v: &Map<String, Value>, id: &str, what: &str) -> Result<f64, EditError> {
    number(v, id).ok_or_else(|| EditError::new(format!("give the {what} (`{id}`)")))
}

fn analysis(
    kind: &str,
    v: &Map<String, Value>,
    def: &SimulationDef,
) -> Result<SimAnalysis, EditError> {
    let a = analysis_raw(kind, v, def)?;
    let w = as_written;
    Ok(match a {
        SimAnalysis::Dc {
            source,
            from,
            to,
            step,
        } => {
            let u = if def
                .sources
                .get(&source)
                .is_some_and(|s| s.kind == crate::simulation::SourceKind::Current)
            {
                "A"
            } else {
                "V"
            };
            SimAnalysis::Dc {
                source,
                from: w(from, u),
                to: w(to, u),
                step: w(step, u),
            }
        }
        SimAnalysis::Ac { from, to, points } => SimAnalysis::Ac {
            from: w(from, "Hz"),
            to: w(to, "Hz"),
            points,
        },
        SimAnalysis::Noise {
            output,
            reference,
            input,
            from,
            to,
            points,
        } => SimAnalysis::Noise {
            output,
            reference,
            input,
            from: w(from, "Hz"),
            to: w(to, "Hz"),
            points,
        },
        SimAnalysis::Tran {
            stop,
            step,
            max_step,
        } => SimAnalysis::Tran {
            stop: w(stop, "s"),
            step: w(step, "s"),
            max_step: max_step.map(|m| w(m, "s")),
        },
        other => other,
    })
}

fn analysis_raw(
    kind: &str,
    v: &Map<String, Value>,
    def: &SimulationDef,
) -> Result<SimAnalysis, EditError> {
    Ok(match kind {
        "op" => SimAnalysis::Op,
        "dc" => {
            let source =
                string(v, "source").ok_or_else(|| EditError::new("name the source to sweep"))?;
            let (from, to, step) = (
                number(v, "start")
                    .or_else(|| number(v, "from"))
                    .unwrap_or(0.0),
                need(v, "stop", "sweep's end").or_else(|_| need(v, "to", "sweep's end"))?,
                need(v, "step", "sweep step")?,
            );
            if step.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || to <= from {
                return Err(EditError::new(
                    "a DC sweep goes up from `start` to `stop` in steps above zero",
                ));
            }
            SimAnalysis::Dc {
                source,
                from,
                to,
                step,
            }
        }
        "ac" => {
            let from = number(v, "start")
                .or_else(|| number(v, "from"))
                .ok_or_else(|| EditError::new("give the start frequency"))?;
            let to = number(v, "stop")
                .or_else(|| number(v, "to"))
                .ok_or_else(|| EditError::new("give the stop frequency"))?;
            let points = number(v, "points").unwrap_or(20.0);
            if !(from > 0.0 && to > from && points >= 1.0) {
                return Err(EditError::new("an AC sweep goes up from a start frequency above zero, with at least one point per decade"));
            }
            SimAnalysis::Ac {
                from,
                to,
                points: points as u32,
            }
        }
        "tran" => {
            let (stop, step) = (need(v, "stop", "stop time")?, need(v, "step", "time step")?);
            if !(stop > 0.0 && step > 0.0 && step < stop) {
                return Err(EditError::new(
                    "a transient needs a stop time above zero and a smaller step",
                ));
            }
            SimAnalysis::Tran {
                stop,
                step,
                max_step: number(v, "max_step"),
            }
        }
        "monte_carlo" => {
            let of = string(v, "of")
                .ok_or_else(|| EditError::new("name the analysis the Monte Carlo repeats"))?;
            if !def.analyses.contains_key(&of) {
                return Err(EditError::new(format!(
                    "the simulation has no analysis `{of}` to repeat"
                )));
            }
            let runs = number(v, "runs").unwrap_or(100.0);
            if !(1.0..=1e6).contains(&runs) {
                return Err(EditError::new("a Monte Carlo makes 1 to 1,000,000 runs"));
            }
            SimAnalysis::MonteCarlo {
                of,
                runs: runs as u32,
                seed: number(v, "seed").unwrap_or(1.0) as u64,
                uniform: string(v, "distribution").is_some_and(|d| d == "uniform"),
            }
        }
        "noise" => {
            let output =
                string(v, "output").ok_or_else(|| EditError::new("name the output net"))?;
            let output = output.strip_prefix("net.").unwrap_or(&output).to_string();
            let input = string(v, "input")
                .ok_or_else(|| EditError::new("name the input source the noise is referred to"))?;
            if !def.sources.contains_key(&input) {
                return Err(EditError::new(format!(
                    "the simulation has no source `{input}` to refer the noise to"
                )));
            }
            let from = number(v, "start")
                .or_else(|| number(v, "from"))
                .ok_or_else(|| EditError::new("give the start frequency"))?;
            let to = number(v, "stop")
                .or_else(|| number(v, "to"))
                .ok_or_else(|| EditError::new("give the stop frequency"))?;
            let points = number(v, "points").unwrap_or(20.0);
            if !(from > 0.0 && to > from && points >= 1.0) {
                return Err(EditError::new("a noise sweep goes up from a start frequency above zero, with at least one point per decade"));
            }
            SimAnalysis::Noise {
                output,
                reference: None,
                input,
                from,
                to,
                points: points as u32,
            }
        }
        "temperature" => {
            let of = string(v, "of")
                .ok_or_else(|| EditError::new("name the analysis the sweep repeats"))?;
            match def.analyses.get(&of) {
                None => {
                    return Err(EditError::new(format!(
                        "the simulation has no analysis `{of}` to repeat"
                    )))
                }
                Some(a) if a.reruns().is_some() => {
                    return Err(EditError::new(format!(
                        "`{of}` already reruns another analysis; name that one"
                    )))
                }
                _ => {}
            }
            let values: Vec<f64> = match v.get("values") {
                Some(Value::Array(items)) => items.iter().filter_map(Value::as_f64).collect(),
                Some(Value::String(t)) => t
                    .split([',', ' '])
                    .filter(|x| !x.is_empty())
                    .map(|x| x.trim_end_matches("degC").parse::<f64>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| {
                        EditError::new("temperatures are numbers in °C (`-40, 25, 85`)")
                    })?,
                _ => vec![],
            };
            if values.is_empty() || values.iter().any(|t| !t.is_finite() || *t < -273.15) {
                return Err(EditError::new(
                    "give the temperatures in °C (`-40, 25, 85`)",
                ));
            }
            SimAnalysis::Temperature { of, values }
        }
        other => {
            return Err(EditError::new(format!(
                "`{other}` is not an analysis (op, dc, ac, tran, monte_carlo, noise, temperature)"
            )))
        }
    })
}

/// `step`, `response`, `sweep`, `bias`, `spread`: the next free one with a number when taken.
fn default_name(kind: &str, def: &SimulationDef) -> String {
    let base = match kind {
        "tran" => "step",
        "ac" => "response",
        "dc" => "sweep",
        "op" => "bias",
        "noise" => "hiss",
        "temperature" => "corners",
        _ => "spread",
    };
    std::iter::once(base.to_string())
        .chain((2..).map(|k| format!("{base}{k}")))
        .find(|n| !def.analyses.contains_key(n))
        .expect("a free name")
}

/// `R*` → `resistors`, `C*` → `capacitors`, `L*` → `inductors`; else the pattern's letters.
fn tolerance_name(pattern: &str) -> String {
    match pattern {
        "R*" => "resistors".into(),
        "C*" => "capacitors".into(),
        "L*" => "inductors".into(),
        p => {
            let s: String = p
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase();
            if s.is_empty() {
                "parts".into()
            } else {
                s
            }
        }
    }
}
