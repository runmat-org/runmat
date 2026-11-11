//! Monte Carlo over component tolerances.
//!
//! Run `k` draws its deviations from its own generator, seeded from `(seed, k)` (SplitMix64
//! into xoshiro256**), so a run's sample is the same on any number of threads. Element values
//! (`R`, `C`, `L`, and the DC value of `V` and `I`) and model parameters scale by `1 + δ`, with
//! `δ` normal (relative `sigma`) or uniform (relative `±half_width`). Runs execute in parallel on
//! a Rayon pool; measurements are kept in run order.

use crate::measure::{evaluate, Measure};
use crate::meter::Usage;
use crate::netlist::{parse_number, Flat, Tok};
use crate::{simulate_spent, Analysis, Circuit, Design, SimError, SimResult};
use rayon::prelude::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Distribution {
    /// Relative standard deviation.
    Gaussian { sigma: f64 },
    /// Relative half-width.
    Uniform { half_width: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    /// Elements whose flattened name matches a glob (`r*`, `xu1.r?`, `c5`, `?[sf]`).
    Elements(String),
    ModelParam {
        model: String,
        param: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tolerance {
    pub target: Target,
    pub dist: Distribution,
    /// Relative shift of the band's center from nominal (0 for a symmetric `± %`; an
    /// asymmetric band such as `+10 % / −5 %` centers at +2.5 %).
    pub center: f64,
}

impl Tolerance {
    /// A band centered on nominal.
    pub fn new(target: Target, dist: Distribution) -> Self {
        Tolerance {
            target,
            dist,
            center: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MonteCarlo {
    pub runs: usize,
    /// The first run's index: runs `first..first + runs` (each run's draw depends only on the
    /// seed and its index, so a Monte Carlo split into ranges gives the same runs).
    pub first: usize,
    pub seed: u64,
    pub tolerances: Vec<Tolerance>,
    pub analysis: Analysis,
    pub measures: Vec<(String, Measure)>,
    /// How many runs (the first ones) keep their full results, for overlay plots.
    pub keep_results: usize,
    /// The operations the whole Monte Carlo may spend; `None`: no limit. Runs go in batches of
    /// [`BATCH`], each run of a batch limited to an equal share of what is left, so which runs
    /// complete depends on the limit only, never on the thread count.
    pub limit: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl Stats {
    pub fn of(v: &[f64]) -> Stats {
        let ok: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        let n = ok.len();
        if n == 0 {
            return Stats::default();
        }
        let mean = ok.iter().sum::<f64>() / n as f64;
        let var = if n > 1 {
            ok.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        Stats {
            n,
            mean,
            std: var.sqrt(),
            min: ok.iter().copied().fold(f64::INFINITY, f64::min),
            max: ok.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct McOutcome {
    /// Per measurement, one value per run (NaN where the run or measurement failed).
    pub values: BTreeMap<String, Vec<f64>>,
    pub stats: BTreeMap<String, Stats>,
    pub failures: Vec<(usize, String)>,
    pub usage: Usage,
    pub kept: Vec<SimResult>,
    /// Runs that finished (the first `completed` of `runs`); fewer when the limit stopped it.
    pub completed: usize,
    /// The operation limit stopped the Monte Carlo before every run finished.
    pub stopped: bool,
}

/// Runs per batch when a Monte Carlo has an operation limit.
pub const BATCH: usize = 32;

impl McOutcome {
    /// Appends the outcome of the runs after this one's (a later range of the same Monte Carlo).
    pub fn merge(&mut self, o: McOutcome) {
        for (k, v) in o.values {
            self.values.entry(k).or_default().extend(v);
        }
        self.stats = self
            .values
            .iter()
            .map(|(k, v)| (k.clone(), Stats::of(v)))
            .collect();
        self.failures.extend(o.failures);
        self.usage = self.usage.checked_add(&o.usage).unwrap_or(self.usage);
        self.kept.extend(o.kept);
        self.completed += o.completed;
        self.stopped |= o.stopped;
    }
}

/// Equal-width histogram: bin edges (`bins + 1`) and counts.
pub fn histogram(v: &[f64], bins: usize) -> (Vec<f64>, Vec<usize>) {
    let s = Stats::of(v);
    let bins = bins.max(1);
    let span = if s.max > s.min { s.max - s.min } else { 1.0 };
    let edges: Vec<f64> = (0..=bins)
        .map(|k| s.min + span * k as f64 / bins as f64)
        .collect();
    let mut counts = vec![0; bins];
    for x in v.iter().filter(|x| x.is_finite()) {
        let k = (((x - s.min) / span) * bins as f64).floor() as usize;
        counts[k.min(bins - 1)] += 1;
    }
    (edges, counts)
}

struct Rng([u64; 4]);

fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

impl Rng {
    fn new(seed: u64, run: u64) -> Rng {
        let mut z = seed ^ run.wrapping_mul(0xD1B5_4A32_D192_ED03);
        Rng([
            splitmix(&mut z),
            splitmix(&mut z),
            splitmix(&mut z),
            splitmix(&mut z),
        ])
    }
    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }
    /// Uniform in (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn draw(&mut self, d: Distribution) -> f64 {
        match d {
            Distribution::Gaussian { sigma } => sigma * self.normal(),
            Distribution::Uniform { half_width } => half_width * (2.0 * self.unit() - 1.0),
        }
    }
}

/// Case-insensitive glob over a lowercase name: `*`, `?` and `[…]` sets (`[sf]`, `[1-4]`,
/// `[!r]`).
pub fn glob(pattern: &str, name: &str) -> bool {
    fn go(p: &[u8], n: &[u8]) -> bool {
        match (p.first(), n.first()) {
            (None, None) => true,
            (Some(b'*'), _) => go(&p[1..], n) || (!n.is_empty() && go(p, &n[1..])),
            (Some(b'?'), Some(_)) => go(&p[1..], &n[1..]),
            (Some(b'['), Some(&c)) => match p.iter().skip(1).position(|&x| x == b']') {
                // `[…]` matches one character of the set (`a-z` ranges; `!` or `^` first negates).
                Some(k) if k > 0 => {
                    let set = &p[1..1 + k];
                    let (neg, set) = match set.first() {
                        Some(b'!' | b'^') => (true, &set[1..]),
                        _ => (false, set),
                    };
                    let mut hit = false;
                    let mut i = 0;
                    while i < set.len() {
                        if i + 2 < set.len() && set[i + 1] == b'-' {
                            hit |= (set[i]..=set[i + 2]).contains(&c);
                            i += 3;
                        } else {
                            hit |= set[i] == c;
                            i += 1;
                        }
                    }
                    hit != neg && go(&p[2 + k..], &n[1..])
                }
                _ => c == b'[' && go(&p[1..], &n[1..]),
            },
            (Some(a), Some(b)) if a == b => go(&p[1..], &n[1..]),
            _ => false,
        }
    }
    go(pattern.to_ascii_lowercase().as_bytes(), name.as_bytes())
}

fn scale_first_value(args: &mut [Tok], factor: f64) {
    for t in args.iter_mut() {
        match t {
            Tok::W(s) => {
                if let Some(v) = parse_number(s) {
                    *s = format!("{:e}", v * factor);
                    return;
                }
                if s == "dc" {
                    continue;
                }
            }
            Tok::Kv(..) => {}
        }
    }
}

/// The flattened netlist of run `k`.
pub fn perturbed(flat: &Flat, tol: &[Tolerance], seed: u64, run: u64) -> Flat {
    let mut f = flat.clone();
    let mut rng = Rng::new(seed, run);
    for t in tol {
        match &t.target {
            Target::Elements(pat) => {
                for e in f
                    .elements
                    .iter_mut()
                    .filter(|e| matches!(e.kind, 'r' | 'c' | 'l' | 'v' | 'i') && glob(pat, &e.name))
                {
                    let factor = 1.0 + t.center + rng.draw(t.dist);
                    scale_first_value(&mut e.args, factor);
                }
            }
            Target::ModelParam { model, param } => {
                let factor = 1.0 + t.center + rng.draw(t.dist);
                if let Some(v) = f
                    .models
                    .get_mut(model)
                    .and_then(|m| m.params.get_mut(param))
                {
                    *v *= factor;
                }
            }
        }
    }
    f
}

/// Runs the Monte Carlo on `threads` threads (all cores when `None`).
pub fn run(
    design: &Design,
    mc: &MonteCarlo,
    threads: Option<usize>,
) -> Result<McOutcome, SimError> {
    if mc.runs == 0 {
        return Err(SimError::BadRequest(
            "Monte Carlo needs at least one run".into(),
        ));
    }
    for t in &mc.tolerances {
        if let Target::ModelParam { model, param } = &t.target {
            let ok = design
                .flat
                .models
                .get(model)
                .is_some_and(|m| m.params.contains_key(param));
            if !ok {
                return Err(SimError::BadRequest(format!(
                    "model `{model}` has no parameter `{param}` to vary"
                )));
            }
        }
    }
    // A run's usage counts even when it fails or stops: the work was done.
    let one = |k: usize, limit: Option<u64>| -> (Result<SimResult, String>, Usage, bool) {
        let flat = perturbed(&design.flat, &mc.tolerances, mc.seed, k as u64);
        let c = match Circuit::build(&flat, design.options.clone(), &design.ics, &design.nodesets) {
            Ok(c) => c,
            Err(e) => return (Err(e.to_string()), Usage::default(), false),
        };
        match simulate_spent(&c, &mc.analysis, limit) {
            (Ok(o), u) => {
                let stopped = o.stopped;
                (Ok(o.result), u, stopped)
            }
            (Err(SimError::Stopped), u) => (Err(SimError::Stopped.to_string()), u, true),
            (Err(e), u) => (Err(e.to_string()), u, false),
        }
    };
    let pool = match threads {
        Some(n) => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
                .map_err(|e| SimError::BadRequest(e.to_string()))?,
        ),
        None => None,
    };
    let batch = |range: std::ops::Range<usize>, limit: Option<u64>| {
        let body = || {
            range
                .clone()
                .into_par_iter()
                .map(|k| one(k, limit))
                .collect::<Vec<_>>()
        };
        match &pool {
            Some(p) => p.install(body),
            None => body(),
        }
    };
    let mut values: BTreeMap<String, Vec<f64>> = mc
        .measures
        .iter()
        .map(|(n, _)| (n.clone(), Vec::with_capacity(mc.runs)))
        .collect();
    let mut failures = vec![];
    let mut usage = Usage::default();
    let mut kept = vec![];
    let mut completed = 0usize;
    let mut stopped = false;
    let last = mc.first + mc.runs;
    let mut next = mc.first;
    while next < last && !stopped {
        let (range, per_run) = match mc.limit {
            None => (next..last, None),
            Some(l) => {
                let spent = usage
                    .total()
                    .map_err(|_| SimError::CountOverflow)?
                    .operations();
                let end = (next + BATCH).min(last);
                (
                    next..end,
                    Some(l.saturating_sub(spent) / (end - next) as u64),
                )
            }
        };
        next = range.end;
        let base = range.start;
        for (j, (r, u, run_stopped)) in batch(range, per_run).into_iter().enumerate() {
            let k = base + j;
            usage = usage.checked_add(&u).map_err(|_| SimError::CountOverflow)?;
            if run_stopped || stopped {
                // A run cut short is not a sample; the runs after it in the batch are dropped too,
                // so the completed runs are always the first ones.
                stopped = true;
                continue;
            }
            completed += 1;
            match r {
                Ok(res) => {
                    for (name, m) in &mc.measures {
                        let v = evaluate(m, &res).unwrap_or_else(|e| {
                            failures.push((k, format!("{name}: {e}")));
                            f64::NAN
                        });
                        values.get_mut(name).expect("measure").push(v);
                    }
                    if k < mc.keep_results {
                        kept.push(res);
                    }
                }
                Err(e) => {
                    failures.push((k, e));
                    for v in values.values_mut() {
                        v.push(f64::NAN);
                    }
                }
            }
        }
    }
    let stats = values
        .iter()
        .map(|(k, v)| (k.clone(), Stats::of(v)))
        .collect();
    Ok(McOutcome {
        values,
        stats,
        failures,
        usage,
        kept,
        completed,
        stopped,
    })
}
