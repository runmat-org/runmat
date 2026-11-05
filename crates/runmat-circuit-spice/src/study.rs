//! Runs a design's `simulation` section on the native engine.
//!
//! The design (or the section's `sheet`, as the whole circuit) is elaborated and emitted as a
//! deck with the section's sources as stimuli and its models for parts that carry none.
//! Components with no model and no primitive value are left out and listed in the warnings;
//! every node then gets a 1 TΩ shunt so the parts left behind cannot float the matrix. Each
//! analysis runs once; a `monte_carlo` analysis reruns the analysis it names with the section's
//! tolerances. Measurements read their analysis, or every Monte Carlo run of it.

use crate::netlist::{emit, Deck, EmitOptions, Stimulus, Waveform};
use runmat_circuit::elab::{elaborate, ElabOptions};
use runmat_circuit::ir::Design;
use runmat_circuit::library::{Pool, SpiceModel};
use runmat_circuit::simulation::{MeasureKind, SimAnalysis, SimMeasure, SimModel, SimWave, SimulationDef, SourceKind};
use runmat_circuit_sim as sim;
use sim::measure::{evaluate, Measure};
use sim::montecarlo::{Distribution, McOutcome, MonteCarlo, Stats, Target, Tolerance};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct MeasureValue {
    pub of: String,
    /// The value, for a measurement of a single analysis.
    pub value: Option<f64>,
    /// Per-run values and their statistics, for a measurement of a Monte Carlo analysis.
    pub runs: Vec<f64>,
    pub stats: Option<Stats>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Study {
    pub simulation: String,
    pub deck: Deck,
    /// Results by analysis name (Monte Carlo analyses keep their first run here).
    pub results: BTreeMap<String, sim::SimResult>,
    pub monte_carlo: BTreeMap<String, McOutcome>,
    pub measures: BTreeMap<String, MeasureValue>,
    /// Probed net → signal name in the results (`v(<node>)`).
    pub probes: Vec<(String, String)>,
    pub usage: sim::Usage,
    pub warnings: Vec<String>,
}

fn wave(s: &runmat_circuit::simulation::SimSource) -> Waveform {
    match &s.wave {
        Some(SimWave::Pulse { from, to, delay, rise, fall, width, period }) => {
            Waveform::Pulse { v1: *from, v2: *to, delay: *delay, rise: *rise, fall: *fall, width: *width, period: *period }
        }
        Some(SimWave::Sin { offset, amplitude, freq, .. }) => Waveform::Sin { offset: *offset, amplitude: *amplitude, freq: *freq },
        Some(SimWave::Pwl(p)) => Waveform::Pwl(p.clone()),
        None if s.ac != 0.0 => Waveform::Ac { dc: s.dc, mag: s.ac, phase: 0.0 },
        None => Waveform::Dc(s.dc),
    }
}

/// The deck text for one analysis run: the emitted deck with AC on every source that has one.
fn deck_text(deck: &Deck, sim_def: &SimulationDef) -> String {
    let mut lines = deck.lines();
    for line in lines.iter_mut() {
        for (k, s) in &sim_def.sources {
            let prefix = format!("{}{k} ", if s.kind == SourceKind::Voltage { 'V' } else { 'I' });
            if line.starts_with(&prefix) && s.ac != 0.0 && s.wave.is_some() {
                line.push_str(&format!(" AC {:e}", s.ac));
            }
        }
    }
    lines.join("\n") + "\n"
}

fn measure(m: &SimMeasure, node: &str) -> Measure {
    let v = format!("v({node})");
    let a = |k: &str| m.args.get(k).copied();
    match m.kind {
        MeasureKind::At => Measure::At { signal: v, x: a("at").unwrap_or(0.0) },
        MeasureKind::GainAt => Measure::At { signal: format!("vdb({node})"), x: a("at").unwrap_or(0.0) },
        MeasureKind::PhaseAt => Measure::At { signal: format!("vp({node})"), x: a("at").unwrap_or(0.0) },
        MeasureKind::Max => Measure::Max { signal: v, from: a("from"), to: a("to") },
        MeasureKind::Min => Measure::Min { signal: v, from: a("from"), to: a("to") },
        MeasureKind::PeakToPeak => Measure::PeakToPeak { signal: v, from: a("from"), to: a("to") },
        MeasureKind::Average => Measure::Average { signal: v, from: a("from"), to: a("to") },
        MeasureKind::Rms => Measure::Rms { signal: v, from: a("from"), to: a("to") },
        MeasureKind::Final => Measure::Final { signal: v },
        MeasureKind::RiseTime => Measure::RiseTime { signal: v, low: 0.1, high: 0.9 },
        MeasureKind::Settling => Measure::Settling { signal: v, band: a("band").unwrap_or(0.01) },
        MeasureKind::Overshoot => Measure::Overshoot { signal: v },
        MeasureKind::Bandwidth => Measure::Bandwidth { signal: v, drop_db: a("drop").unwrap_or(3.0103) },
        MeasureKind::PeakFrequency => Measure::PeakFrequency { signal: v },
        MeasureKind::Q => Measure::Q { signal: v },
        MeasureKind::UnityGain => Measure::UnityGain { signal: v },
        MeasureKind::PhaseMargin => Measure::PhaseMargin { signal: v },
    }
}

fn analysis(a: &SimAnalysis, sources: &SimulationDef) -> Result<sim::Analysis, String> {
    Ok(match a {
        SimAnalysis::Op => sim::Analysis::Op,
        SimAnalysis::Dc { source, from, to, step } => {
            let s = sources.sources.get(source).ok_or_else(|| format!("the DC sweep names `{source}`, which is not a source of this simulation"))?;
            let letter = if s.kind == SourceKind::Voltage { 'v' } else { 'i' };
            sim::Analysis::Dc { source: format!("{letter}{}", source.to_ascii_lowercase()), start: *from, stop: *to, step: *step }
        }
        SimAnalysis::Ac { from, to, points } => sim::Analysis::Ac { sweep: sim::AcSweep::Dec, points: *points as usize, fstart: *from, fstop: *to },
        SimAnalysis::Tran { stop, step, max_step } => sim::Analysis::Tran(sim::TranSpec { tstep: *step, tstop: *stop, tstart: 0.0, tmax: *max_step, uic: false }),
        SimAnalysis::MonteCarlo { .. } => return Err("a Monte Carlo analysis names another analysis".into()),
    })
}

/// Runs simulation `name` of `design`. `resolve` reads model libraries; `threads` caps Monte
/// Carlo parallelism (all cores when `None`).
pub fn run(design: &Design, pool: &Pool, name: &str, resolve: &dyn Fn(&str) -> Option<String>, threads: Option<usize>) -> Result<Study, String> {
    let def = design.simulations.get(name).ok_or_else(|| format!("no simulation named `{name}`"))?;
    let mut d = design.clone();
    if let Some(sheet) = &def.sheet {
        if !d.sheets.contains_key(sheet) {
            return Err(format!("simulation `{name}` names sheet `{sheet}`, which the design does not declare"));
        }
        d.root = sheet.clone();
    }
    let nl = elaborate(&d, pool, &ElabOptions::default(), None);
    let mut warnings: Vec<String> = nl.diagnostics.iter().map(|x| format!("{}: {}", x.code, x.message)).collect();
    let mut opts = EmitOptions { title: format!("{} / {name}", design.name), sense_pins: false, skip_unmodelled: true, ..Default::default() };
    for (k, s) in &def.sources {
        let w = wave(s);
        opts.stimuli.push(match s.kind {
            SourceKind::Voltage => Stimulus::Voltage { name: k.clone(), pos: s.p.clone(), neg: s.n.clone(), wave: w },
            SourceKind::Current => Stimulus::Current { name: k.clone(), pos: s.p.clone(), neg: s.n.clone(), wave: w },
        });
    }
    for (part, m) in &def.models {
        let (model, library, pins) = match m {
            SimModel::OpAmp { params, pins } => {
                let model = format!("rm_opamp_{}", crate::netlist::node_name(part));
                let ps: Vec<String> = params.iter().map(|(k, v)| format!("{k}={v:e}")).collect();
                opts.extra.push(format!(".model {model} OPAMP({})", ps.join(" ")));
                (model, None, pins.clone())
            }
            SimModel::Subckt { library, name, pins } => (name.clone(), Some(library.clone()), pins.clone()),
        };
        opts.part_models.insert(part.clone(), SpiceModel { model, library, pin_order: pins.into_iter().map(|p| (String::new(), p)).collect(), device: None });
    }
    for (k, v) in &def.options {
        opts.extra.push(format!(".options {k}={v}"));
    }
    let deck = emit(&nl, pool, &opts).map_err(|e| e.iter().map(|x| format!("{}: {}", x.designator, x.message)).collect::<Vec<_>>().join("; "))?;
    let modelled: std::collections::BTreeSet<&str> = deck.elements.iter().filter_map(|e| e.designator.as_deref()).collect();
    let skipped: Vec<&str> = nl.components.iter().map(|c| c.designator.as_str()).filter(|d| !modelled.contains(d)).collect();
    let mut text = deck_text(&deck, def);
    if !skipped.is_empty() {
        warnings.push(format!("left out (no model or value): {}", skipped.join(", ")));
        text = text.replace("\n.end\n", "\n.options rshunt=1e12\n.end\n");
    }
    let sim_design = sim::Design::from_netlist(&text, resolve).map_err(|e| e.to_string())?;
    let mut study = Study { simulation: name.into(), deck, results: BTreeMap::new(), monte_carlo: BTreeMap::new(), measures: BTreeMap::new(), probes: vec![], usage: sim::Usage::default(), warnings, };
    study.probes = def.probes.iter().map(|n| (n.clone(), format!("v({})", study.deck.node(n).to_ascii_lowercase()))).collect();
    let add = |u: &mut sim::Usage, o: &sim::Usage| -> Result<(), String> {
        *u = u.checked_add(o).map_err(|_| "operation count overflow".to_string())?;
        Ok(())
    };
    for (k, a) in &def.analyses {
        if matches!(a, SimAnalysis::MonteCarlo { .. }) {
            continue;
        }
        let o = sim::simulate(&sim_design, &analysis(a, def)?).map_err(|e| format!("analysis `{k}`: {e}"))?;
        add(&mut study.usage, &o.usage)?;
        for w in o.warnings {
            if !study.warnings.contains(&w) {
                study.warnings.push(w);
            }
        }
        study.results.insert(k.clone(), o.result);
    }
    for (k, a) in &def.analyses {
        let SimAnalysis::MonteCarlo { of, runs, seed, uniform } = a else { continue };
        let base = def.analyses.get(of).ok_or_else(|| format!("Monte Carlo `{k}` names analysis `{of}`, which is not declared"))?;
        let tolerances = def
            .tolerances
            .values()
            .map(|t| Tolerance {
                target: Target::Elements(t.pattern.clone()),
                dist: if *uniform { Distribution::Uniform { half_width: t.value } } else { Distribution::Gaussian { sigma: t.value / 3.0 } },
            })
            .collect();
        let measures = def
            .measures
            .iter()
            .filter(|(_, m)| m.of == *k)
            .map(|(n, m)| (n.clone(), measure(m, &study.deck.node(&m.net).to_ascii_lowercase())))
            .collect();
        let mc = MonteCarlo { runs: *runs as usize, seed: *seed, tolerances, analysis: analysis(base, def)?, measures, keep_results: 20 };
        let out = sim::montecarlo::run(&sim_design, &mc, threads).map_err(|e| format!("analysis `{k}`: {e}"))?;
        add(&mut study.usage, &out.usage)?;
        if let Some(first) = out.kept.first() {
            study.results.insert(k.clone(), first.clone());
        }
        study.monte_carlo.insert(k.clone(), out);
    }
    for (n, m) in &def.measures {
        let node = study.deck.node(&m.net).to_ascii_lowercase();
        let v = if let Some(mc) = study.monte_carlo.get(&m.of) {
            let runs = mc.values.get(n).cloned().unwrap_or_default();
            MeasureValue { of: m.of.clone(), value: None, stats: Some(Stats::of(&runs)), runs, error: None }
        } else {
            match study.results.get(&m.of).map(|r| evaluate(&measure(m, &node), r)) {
                Some(Ok(x)) => MeasureValue { of: m.of.clone(), value: Some(x), runs: vec![], stats: None, error: None },
                Some(Err(e)) => MeasureValue { of: m.of.clone(), value: None, runs: vec![], stats: None, error: Some(e) },
                None => MeasureValue { of: m.of.clone(), value: None, runs: vec![], stats: None, error: Some(format!("analysis `{}` did not run", m.of)) },
            }
        };
        study.measures.insert(n.clone(), v);
    }
    Ok(study)
}
