//! Runs a design's `simulation` section on the native engine.
//!
//! The design (or the section's `sheet`, as the whole circuit) is elaborated and emitted as a
//! deck with the section's sources as stimuli and its models for parts that carry none.
//! Components with no model and no primitive value are left out and listed in the warnings;
//! every node then gets a 1 TΩ shunt so the parts left behind cannot float the matrix. Each
//! analysis runs once; a `monte_carlo` analysis reruns the analysis it names with each
//! component's own tolerance band (`value = 1 kohm ± 1 %` on the component or its part), where a
//! section `tolerance` statement that matches the element replaces it. Measurements read their
//! analysis, or every Monte Carlo run of it.
//!
//! With an operation limit (the host's granted operations), each analysis gets what is left;
//! one that reaches it keeps the points it solved, and the analyses after it don't run.

use crate::netlist::{emit, Deck, EmitOptions, SensePins, Stimulus, Waveform};
use runmat_circuit::elab::{elaborate, ElabOptions};
use runmat_circuit::ir::Design;
use runmat_circuit::library::{Pool, SpiceModel};
use runmat_circuit::simulation::{MeasureKind, SimAnalysis, SimMeasure, SimModel, SimProbe, SimWave, SimulationDef, SourceKind};
use runmat_circuit_sim as sim;
use sim::measure::{evaluate, Measure};
use sim::montecarlo::{glob, Distribution, McOutcome, MonteCarlo, Stats, Target, Tolerance};
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
    /// Each probe and its signal in the results: `v(<node>)` for a net, `i(<sense source>)`
    /// (the current into the pin) for a pin.
    pub probes: Vec<(SimProbe, String)>,
    /// The tolerance applied to each varied element in Monte Carlo runs.
    pub tolerances: Vec<AppliedTolerance>,
    pub usage: sim::Usage,
    pub warnings: Vec<String>,
    /// Set when the operation limit stopped the study: what stopped, and what didn't run.
    pub stopped: Option<String>,
}

/// A tolerance as Monte Carlo applies it to one deck element (or a pattern of them).
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedTolerance {
    /// The element name or the section statement's pattern.
    pub elements: String,
    /// Band as fractions of nominal: `[min, max]` (0.99, 1.01 for ±1 %).
    pub band: (f64, f64),
    /// `component` (the part's own value) or `tolerance <name>` (the section's statement).
    pub from: String,
}

/// How a study runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunOptions {
    /// Monte Carlo threads; all cores when `None`.
    pub threads: Option<usize>,
    /// Operations the whole study may spend; `None`: no limit.
    pub limit: Option<u64>,
}

/// The tolerances a Monte Carlo of `def` applies on `deck`: the section's statements, then each
/// component band on an element no statement matches.
fn tolerances(def: &SimulationDef, deck: &Deck) -> Vec<AppliedTolerance> {
    let mut out: Vec<AppliedTolerance> =
        def.tolerances.iter().map(|(k, t)| AppliedTolerance { elements: t.pattern.clone(), band: (1.0 - t.value, 1.0 + t.value), from: format!("tolerance {k}") }).collect();
    for e in &deck.elements {
        let Some(band) = e.band else { continue };
        let name = e.name.to_ascii_lowercase();
        if def.tolerances.values().any(|t| glob(&t.pattern, &name)) {
            continue;
        }
        out.push(AppliedTolerance { elements: name, band, from: "component".into() });
    }
    out
}

fn sampled(t: &AppliedTolerance, uniform: bool) -> Tolerance {
    let half = 0.5 * (t.band.1 - t.band.0);
    let dist = if uniform { Distribution::Uniform { half_width: half } } else { Distribution::Gaussian { sigma: half / 3.0 } };
    Tolerance { target: Target::Elements(t.elements.clone()), dist, center: 0.5 * (t.band.0 + t.band.1) - 1.0 }
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

/// Runs simulation `name` of `design` with no operation limit. `resolve` reads model libraries;
/// `threads` caps Monte Carlo parallelism (all cores when `None`).
pub fn run(design: &Design, pool: &Pool, name: &str, resolve: &dyn Fn(&str) -> Option<String>, threads: Option<usize>) -> Result<Study, String> {
    run_with(design, pool, name, resolve, RunOptions { threads, limit: None })
}

/// Runs simulation `name` of `design` under `options`.
pub fn run_with(design: &Design, pool: &Pool, name: &str, resolve: &dyn Fn(&str) -> Option<String>, options: RunOptions) -> Result<Study, String> {
    let mut r = Runner::new(design, pool, name, resolve, options)?;
    while !r.step()? {}
    Ok(r.finish())
}

impl Runner {
    /// Elaborates the design, emits its deck and plans the units of work.
    pub fn new(design: &Design, pool: &Pool, name: &str, resolve: &dyn Fn(&str) -> Option<String>, options: RunOptions) -> Result<Runner, String> {
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
    let mut opts = EmitOptions { title: format!("{} / {name}", design.name), sense: SensePins::None, skip_unmodelled: true, ..Default::default() };
    // Only probed pins get a sense source, so an unprobed design solves the matrix it always did.
    let pins: Vec<(String, String)> = def.probes.iter().filter_map(|p| if let SimProbe::Pin { component, pin } = p { Some((component.clone(), pin.clone())) } else { None }).collect();
    if !pins.is_empty() {
        opts.sense = SensePins::Only(pins);
    }
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
    let applied = tolerances(def, &deck);
    let circuit = sim_design.circuit().map_err(|e| e.to_string())?;
    let mut study = Study {
        simulation: name.into(),
        deck,
        results: BTreeMap::new(),
        monte_carlo: BTreeMap::new(),
        measures: BTreeMap::new(),
        probes: vec![],
        tolerances: applied,
        usage: sim::Usage::default(),
        warnings,
        stopped: None,
    };
    let mut probes = vec![];
    for p in &def.probes {
        let signal = match p {
            SimProbe::Net(n) => format!("v({})", study.deck.node(n).to_ascii_lowercase()),
            SimProbe::Pin { component, pin } => match study.deck.senses.iter().find(|s| s.designator == *component && s.pin.eq_ignore_ascii_case(pin)) {
                Some(s) => format!("i({})", s.source.to_ascii_lowercase()),
                None => {
                    study.warnings.push(format!("probe {}: no such pin on a simulated part", p.label()));
                    continue;
                }
            },
        };
        probes.push((p.clone(), signal));
    }
    study.probes = probes;
    let mut units = vec![];
    for (k, a) in &def.analyses {
        if !matches!(a, SimAnalysis::MonteCarlo { .. }) {
            units.push(Unit::Analysis(k.clone()));
        }
    }
    for (k, a) in &def.analyses {
        if let SimAnalysis::MonteCarlo { runs, .. } = a {
            let total = *runs as usize;
            for first in (0..total).step_by(RUNS_PER_UNIT) {
                units.push(Unit::Batch { name: k.clone(), first, runs: RUNS_PER_UNIT.min(total - first), total });
            }
        }
    }
    Ok(Runner { def: def.clone(), sim_design, circuit, study, units, next: 0, options, not_run: vec![] })
    }

    /// Changes the operation limit for the units still to run (a lease's grant can grow).
    pub fn set_limit(&mut self, limit: Option<u64>) {
        self.options.limit = limit;
    }
}

/// One unit of a study's work: an analysis, or a batch of Monte Carlo runs.
#[derive(Clone, Debug)]
enum Unit {
    Analysis(String),
    Batch { name: String, first: usize, runs: usize, total: usize },
}

/// Monte Carlo runs per unit: the host reports usage and can cancel between units.
pub const RUNS_PER_UNIT: usize = 64;

/// Where a stepwise study is.
#[derive(Clone, Debug, PartialEq)]
pub struct Progress {
    pub done: usize,
    pub total: usize,
    /// What the next unit does (`analysis step`, `spread: runs 65–128 of 1000`); empty when done.
    pub next: String,
}

/// A study run one unit at a time, so a host can show progress, report usage and cancel
/// between units. [`run_with`] is the same steps in one call.
pub struct Runner {
    def: SimulationDef,
    sim_design: sim::Design,
    circuit: sim::Circuit,
    study: Study,
    units: Vec<Unit>,
    next: usize,
    options: RunOptions,
    not_run: Vec<String>,
}

impl Runner {
    pub fn progress(&self) -> Progress {
        let next = match self.units.get(self.next) {
            Some(Unit::Analysis(k)) => format!("analysis {k}"),
            Some(Unit::Batch { name, first, runs, total }) => format!("{name}: runs {}–{} of {total}", first + 1, first + runs),
            None => String::new(),
        };
        Progress { done: self.next, total: self.units.len(), next }
    }

    pub fn usage(&self) -> &sim::Usage {
        &self.study.usage
    }

    pub fn finished(&self) -> bool {
        self.next >= self.units.len()
    }

    fn left(&self) -> Option<u64> {
        let spent = self.study.usage.total().map(|t| t.operations()).unwrap_or(u64::MAX);
        self.options.limit.map(|l| l.saturating_sub(spent))
    }

    fn add(&mut self, o: &sim::Usage) -> Result<(), String> {
        self.study.usage = self.study.usage.checked_add(o).map_err(|_| "operation count overflow".to_string())?;
        Ok(())
    }

    /// Runs the next unit. Returns whether the study is finished.
    pub fn step(&mut self) -> Result<bool, String> {
        let Some(unit) = self.units.get(self.next).cloned() else { return Ok(true) };
        self.next += 1;
        if self.study.stopped.is_some() {
            let (Unit::Analysis(k) | Unit::Batch { name: k, .. }) = &unit;
            if !self.not_run.contains(k) && !self.study.monte_carlo.contains_key(k) {
                self.not_run.push(k.clone());
            }
            return Ok(self.finished());
        }
        match unit {
            Unit::Analysis(k) => {
                let a = analysis(&self.def.analyses[&k], &self.def)?;
                let (o, spent) = sim::simulate_spent(&self.circuit, &a, self.left());
                self.add(&spent)?;
                match o {
                    Err(sim::SimError::Stopped) => {
                        self.study.stopped = Some(format!("analysis `{k}` reached the granted operations before its operating point converged"));
                    }
                    r => {
                        let o = r.map_err(|e| format!("analysis `{k}`: {e}"))?;
                        if o.stopped {
                            self.study.stopped = Some(format!("analysis `{k}` stopped at the granted operations; it holds the points solved before then"));
                        }
                        for w in o.warnings {
                            if !self.study.warnings.contains(&w) {
                                self.study.warnings.push(w);
                            }
                        }
                        self.study.results.insert(k, o.result);
                    }
                }
            }
            Unit::Batch { name, first, runs, total } => {
                let SimAnalysis::MonteCarlo { of, seed, uniform, .. } = &self.def.analyses[&name] else { unreachable!() };
                let base = self.def.analyses.get(of).ok_or_else(|| format!("Monte Carlo `{name}` names analysis `{of}`, which is not declared"))?;
                let tolerances = self.study.tolerances.iter().map(|t| sampled(t, *uniform)).collect();
                let measures = self
                    .def
                    .measures
                    .iter()
                    .filter(|(_, m)| m.of == name)
                    .map(|(n, m)| (n.clone(), measure(m, &self.study.deck.node(&m.net).to_ascii_lowercase())))
                    .collect();
                let threads = if cfg!(target_arch = "wasm32") { None } else { self.options.threads };
                let mc = MonteCarlo { runs, first, seed: *seed, tolerances, analysis: analysis(base, &self.def)?, measures, keep_results: 20, limit: self.left() };
                let out = sim::montecarlo::run(&self.sim_design, &mc, threads).map_err(|e| format!("analysis `{name}`: {e}"))?;
                self.add(&out.usage)?;
                let merged = match self.study.monte_carlo.remove(&name) {
                    Some(mut m) => {
                        m.merge(out);
                        m
                    }
                    None => out,
                };
                if merged.stopped {
                    self.study.stopped = Some(format!("Monte Carlo `{name}` stopped at the granted operations after {} of {total} runs", merged.completed));
                }
                if let Some(first) = merged.kept.first() {
                    self.study.results.insert(name.clone(), first.clone());
                }
                self.study.monte_carlo.insert(name, merged);
            }
        }
        Ok(self.finished())
    }

    /// Stops here: the units not yet run are listed as cancelled.
    pub fn cancel(&mut self) {
        if self.finished() {
            return;
        }
        let p = self.progress();
        let mut left: Vec<String> = vec![];
        for u in &self.units[self.next..] {
            let (Unit::Analysis(k) | Unit::Batch { name: k, .. }) = u;
            if !left.contains(k) {
                left.push(k.clone());
            }
        }
        self.study.stopped = Some(format!("cancelled before {} ({} of {} steps done); not finished: {}", p.next, p.done, p.total, left.join(", ")));
        self.next = self.units.len();
    }

    /// The study: measurements evaluated on what ran.
    pub fn finish(mut self) -> Study {
        if let (Some(s), false) = (&mut self.study.stopped, self.not_run.is_empty()) {
            s.push_str(&format!("; not run: {}", self.not_run.join(", ")));
        }
        let mut study = self.study;
        for mc in study.monte_carlo.values_mut() {
            mc.stats = mc.values.iter().map(|(k, v)| (k.clone(), Stats::of(v))).collect();
        }
        for (n, m) in &self.def.measures {
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
        study
    }
}
