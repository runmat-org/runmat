use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) const DEFAULT_RNG_SEED: u64 = 0x9e3779b97f4a7c15;
pub(crate) const DEFAULT_USER_SEED: u64 = 0;
const RNG_MULTIPLIER: u64 = 6364136223846793005;
const RNG_INCREMENT: u64 = 1;
const RNG_SHIFT: u32 = 11;
const RNG_SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
const MIN_UNIFORM: f64 = f64::MIN_POSITIVE;

fn random_error(label: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(label).build()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RngAlgorithm {
    RunMatLcg,
}

impl RngAlgorithm {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            RngAlgorithm::RunMatLcg => "twister",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RngSnapshot {
    pub state: u64,
    pub seed: Option<u64>,
    pub algorithm: RngAlgorithm,
}

impl RngSnapshot {
    pub(crate) fn new(state: u64, seed: Option<u64>, algorithm: RngAlgorithm) -> Self {
        Self {
            state,
            seed,
            algorithm,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RuntimeRng {
    state: u64,
    seed: Option<u64>,
    algorithm: RngAlgorithm,
}

impl RuntimeRng {
    pub(crate) fn new() -> Self {
        Self {
            state: DEFAULT_RNG_SEED,
            seed: Some(DEFAULT_USER_SEED),
            algorithm: RngAlgorithm::RunMatLcg,
        }
    }

    fn snapshot(&self) -> RngSnapshot {
        RngSnapshot {
            state: self.state,
            seed: self.seed,
            algorithm: self.algorithm,
        }
    }
}

impl From<RngSnapshot> for RuntimeRng {
    fn from(snapshot: RngSnapshot) -> Self {
        Self {
            state: snapshot.state,
            seed: snapshot.seed,
            algorithm: snapshot.algorithm,
        }
    }
}

static RNG_STATE: OnceLock<Mutex<RuntimeRng>> = OnceLock::new();
const LEGACY_TOKEN_BASE: u64 = (1_u64 << 52) + 0x5eed;
const LEGACY_TOKEN_CAPACITY: usize = 1024;

struct LegacyTokenState {
    next: u64,
    entries: HashMap<u64, RngSnapshot>,
    order: VecDeque<u64>,
}

impl LegacyTokenState {
    fn new() -> Self {
        Self {
            next: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn insert(&mut self, snapshot: RngSnapshot) -> u64 {
        let token = LEGACY_TOKEN_BASE + self.next;
        self.next = self.next.wrapping_add(1);
        if self.entries.insert(token, snapshot).is_none() {
            self.order.push_back(token);
        }
        while self.order.len() > LEGACY_TOKEN_CAPACITY {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        token
    }
}

static LEGACY_TOKENS: OnceLock<Mutex<LegacyTokenState>> = OnceLock::new();

fn fallback_rng_state() -> &'static Mutex<RuntimeRng> {
    RNG_STATE.get_or_init(|| Mutex::new(RuntimeRng::new()))
}

fn with_rng_state<T>(
    label: &str,
    operation: impl FnOnce(&mut RuntimeRng) -> T,
) -> BuiltinResult<T> {
    if let Some(context) = crate::context::legacy::active() {
        let mut state = context.state().random.try_borrow_mut().map_err(|_| {
            random_error(label, format!("{label}: random stream is already in use"))
        })?;
        return Ok(operation(&mut state));
    }
    let mut state = fallback_rng_state()
        .lock()
        .map_err(|_| random_error(label, format!("{label}: failed to acquire RNG lock")))?;
    Ok(operation(&mut state))
}

fn legacy_tokens() -> &'static Mutex<LegacyTokenState> {
    LEGACY_TOKENS.get_or_init(|| Mutex::new(LegacyTokenState::new()))
}

fn mix_seed(seed: u64) -> u64 {
    if seed == 0 {
        return DEFAULT_RNG_SEED;
    }
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    let mixed = z ^ (z >> 31);
    if mixed == 0 {
        DEFAULT_RNG_SEED
    } else {
        mixed
    }
}

pub(crate) fn snapshot() -> BuiltinResult<RngSnapshot> {
    with_rng_state("rng", |state| state.snapshot())
}

pub(crate) fn apply_snapshot(snapshot: RngSnapshot) -> BuiltinResult<RngSnapshot> {
    with_rng_state("rng", |state| {
        let previous = state.snapshot();
        state.state = snapshot.state;
        state.seed = snapshot.seed;
        state.algorithm = snapshot.algorithm;
        previous
    })
}

pub(crate) fn set_seed(seed: u64) -> BuiltinResult<RngSnapshot> {
    let state = mix_seed(seed);
    apply_snapshot(RngSnapshot::new(state, Some(seed), RngAlgorithm::RunMatLcg))
}

pub(crate) fn legacy_seed_value() -> BuiltinResult<u64> {
    let snapshot = snapshot()?;
    if let Some(seed) = snapshot.seed {
        if snapshot.state == mix_seed(seed) {
            return Ok(seed);
        }
    }
    let mut tokens = legacy_tokens()
        .lock()
        .map_err(|_| random_error("rand", "rand: failed to acquire legacy RNG token lock"))?;
    Ok(tokens.insert(snapshot))
}

pub(crate) fn set_legacy_seed(seed_or_token: u64) -> BuiltinResult<RngSnapshot> {
    if let Some(snapshot) = legacy_tokens()
        .lock()
        .map_err(|_| random_error("rand", "rand: failed to acquire legacy RNG token lock"))?
        .entries
        .get(&seed_or_token)
        .copied()
    {
        apply_snapshot(snapshot)
    } else {
        set_seed(seed_or_token)
    }
}

pub(crate) fn set_default() -> BuiltinResult<RngSnapshot> {
    apply_snapshot(default_snapshot())
}

pub(crate) fn default_snapshot() -> RngSnapshot {
    RngSnapshot::new(
        DEFAULT_RNG_SEED,
        Some(DEFAULT_USER_SEED),
        RngAlgorithm::RunMatLcg,
    )
}

/// Reserves one stable random stream per logical `parfor` iteration. Stream
/// identity depends on the coordinator's current session stream and logical
/// iteration ordinal, never on worker count, chunking, completion order, or
/// retry attempts.
pub fn reserve_parallel_random_streams(
    context: &crate::context::RuntimeContext,
    count: usize,
) -> BuiltinResult<Vec<runmat_execution::ParallelRandomStream>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let stream_count = u64::try_from(count).map_err(|_| {
        random_error(
            "parfor",
            "parfor: logical iteration count exceeds the supported random-stream range",
        )
    })?;
    let mut random = context.state().random.try_borrow_mut().map_err(|_| {
        random_error(
            "parfor",
            "parfor: coordinator random stream is already in use",
        )
    })?;
    let root = next_u64_state(&mut random.state);
    Ok((0..stream_count)
        .map(
            |ordinal| runmat_execution::ParallelRandomStream::RunMatLcgV1 {
                state: mix_seed(root ^ ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            },
        )
        .collect())
}

/// Temporarily installs an iteration-owned stream in one runtime context.
/// Restoring the coordinator/worker stream on drop keeps random effects scoped
/// to the exact logical iteration, including error and cancellation exits.
pub struct ParallelRandomStreamGuard {
    context: crate::context::RuntimeContext,
    previous: RngSnapshot,
}

impl Drop for ParallelRandomStreamGuard {
    fn drop(&mut self) {
        let mut state = self.context.state().random.borrow_mut();
        *state = RuntimeRng::from(self.previous);
    }
}

pub fn enter_parallel_random_stream(
    context: &crate::context::RuntimeContext,
    stream: runmat_execution::ParallelRandomStream,
) -> BuiltinResult<ParallelRandomStreamGuard> {
    let state = match stream {
        runmat_execution::ParallelRandomStream::RunMatLcgV1 { state } => state,
    };
    let mut random = context.state().random.try_borrow_mut().map_err(|_| {
        random_error(
            "parfor",
            "parfor: iteration random stream is already in use",
        )
    })?;
    let previous = random.snapshot();
    *random = RuntimeRng::from(RngSnapshot::new(state, None, RngAlgorithm::RunMatLcg));
    drop(random);
    Ok(ParallelRandomStreamGuard {
        context: context.clone(),
        previous,
    })
}

pub(crate) fn generate_uniform(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    with_rng_state(label, |state| {
        (0..len)
            .map(|_| next_uniform_state(&mut state.state))
            .collect()
    })
}

/// Generate unbiased offsets in `0..span` from the authoritative 64-bit RNG stream.
///
/// A span of `2^64` denotes the complete `u64` domain. Smaller spans use
/// rejection before the modulo operation, so no destination value receives an
/// extra source word when the span does not divide `2^64`.
pub(crate) fn generate_integer_offsets(
    span: u128,
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<u64>> {
    if span == 0 || span > (1_u128 << 64) {
        return Err(random_error(
            label,
            format!("{label}: integer sampling span must be in 1..=2^64"),
        ));
    }
    with_rng_state(label, |state| {
        generate_integer_offsets_from_state(&mut state.state, span, len)
    })
}

fn generate_integer_offsets_from_state(state: &mut u64, span: u128, len: usize) -> Vec<u64> {
    if span == 1 {
        return vec![0; len];
    }
    if span == (1_u128 << 64) {
        return (0..len).map(|_| next_u64_state(state)).collect();
    }

    let span = span as u64;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let word = next_u64_state(state);
        if let Some(offset) = unbiased_offset_from_word(word, span) {
            out.push(offset);
        }
    }
    out
}

fn unbiased_offset_from_word(word: u64, span: u64) -> Option<u64> {
    debug_assert!(span > 1);
    let rejection_floor = span.wrapping_neg() % span;
    (word >= rejection_floor).then(|| word % span)
}

pub(crate) fn generate_uniform_single(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    generate_uniform(len, label).map(|data| {
        data.into_iter()
            .map(|v| {
                let value = v as f32;
                value as f64
            })
            .collect()
    })
}

pub(crate) fn skip_uniform(len: usize, label: &str) -> BuiltinResult<()> {
    if len == 0 {
        return Ok(());
    }
    with_rng_state(label, |state| {
        state.state = advance_state(state.state, len as u64);
    })
}

fn advance_state(state: u64, mut delta: u64) -> u64 {
    if delta == 0 {
        return state;
    }
    let mut cur_mult = RNG_MULTIPLIER;
    let mut cur_plus = RNG_INCREMENT;
    let mut acc_mult = 1u64;
    let mut acc_plus = 0u64;
    while delta > 0 {
        if (delta & 1) != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = cur_plus.wrapping_mul(cur_mult.wrapping_add(1));
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        delta >>= 1;
    }
    acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
}

pub(crate) fn generate_complex(len: usize, label: &str) -> BuiltinResult<Vec<(f64, f64)>> {
    with_rng_state(label, |state| {
        (0..len)
            .map(|_| {
                let re = next_uniform_state(&mut state.state);
                let im = next_uniform_state(&mut state.state);
                (re, im)
            })
            .collect()
    })
}

pub(crate) fn next_uniform_state(state: &mut u64) -> f64 {
    let word = next_u64_state(state);
    let bits = word >> RNG_SHIFT;
    (bits as f64) * RNG_SCALE
}

fn next_u64_state(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(RNG_MULTIPLIER)
        .wrapping_add(RNG_INCREMENT);
    *state
}

fn next_normal_pair(state: &mut u64) -> (f64, f64) {
    let mut u1 = next_uniform_state(state);
    if u1 <= 0.0 {
        u1 = MIN_UNIFORM;
    }
    let u2 = next_uniform_state(state);
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = 2.0 * PI * u2;
    (radius * angle.cos(), radius * angle.sin())
}

pub(crate) fn generate_exponential(mu: f64, len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    with_rng_state(label, |state| {
        (0..len)
            .map(|_| {
                let u = next_uniform_state(&mut state.state).max(MIN_UNIFORM);
                -mu * u.ln()
            })
            .collect()
    })
}

/// Generate one exponential variate per output element, expanding a scalar mean
/// or pairing an array of means elementwise with the shared host stream.
pub(crate) fn generate_exponential_array(
    mu: &[f64],
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    debug_assert!(mu.len() == 1 || mu.len() == len);
    with_rng_state(label, |state| {
        (0..len)
            .map(|index| {
                let u = next_uniform_state(&mut state.state).max(MIN_UNIFORM);
                let mean = if mu.len() == 1 { mu[0] } else { mu[index] };
                -mean * u.ln()
            })
            .collect()
    })
}

pub(crate) fn generate_normal_scaled(
    mu: f64,
    sigma: f64,
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state(label, |state| {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let (z0, z1) = next_normal_pair(&mut state.state);
            out.push(mu + sigma * z0);
            if out.len() < len {
                out.push(mu + sigma * z1);
            }
        }
        out
    })
}

pub(crate) fn generate_student_t(nu: &[f64], len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    with_rng_state(label, |state| {
        let mut out = Vec::with_capacity(len);
        let mut spare_normal = None;
        for idx in 0..len {
            let df = if nu.len() == 1 { nu[0] } else { nu[idx] };
            let z = if let Some(value) = spare_normal.take() {
                value
            } else {
                let (z0, z1) = next_normal_pair(&mut state.state);
                spare_normal = Some(z1);
                z0
            };
            if df == f64::INFINITY {
                out.push(z);
            } else {
                let chi_square = sample_gamma_shape_scale(&mut state.state, df / 2.0, 2.0);
                out.push(z / (chi_square / df).sqrt());
            }
        }
        out
    })
}

pub(crate) fn generate_gamma_shape_scale(
    shape: &[f64],
    scale: &[f64],
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state(label, |state| {
        let mut out = Vec::with_capacity(len);
        for idx in 0..len {
            let a = if shape.len() == 1 {
                shape[0]
            } else {
                shape[idx]
            };
            let b = if scale.len() == 1 {
                scale[0]
            } else {
                scale[idx]
            };
            out.push(if a == 0.0 {
                0.0
            } else {
                sample_gamma_shape_scale(&mut state.state, a, b)
            });
        }
        out
    })
}

pub(crate) fn generate_binomial(
    trials: &[f64],
    probability: &[f64],
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state(label, |state| {
        let mut out = Vec::with_capacity(len);
        for idx in 0..len {
            let n = if trials.len() == 1 {
                trials[0]
            } else {
                trials[idx]
            };
            let p = if probability.len() == 1 {
                probability[0]
            } else {
                probability[idx]
            };
            out.push(sample_binomial(&mut state.state, n, p));
        }
        out
    })
}

pub(crate) fn generate_weibull(
    scale: &[f64],
    shape: &[f64],
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state(label, |state| {
        let mut out = Vec::with_capacity(len);
        for idx in 0..len {
            let a = if scale.len() == 1 {
                scale[0]
            } else {
                scale[idx]
            };
            let b = if shape.len() == 1 {
                shape[0]
            } else {
                shape[idx]
            };
            let u = next_uniform_state(&mut state.state).max(MIN_UNIFORM);
            out.push(a * (-u.ln()).powf(1.0 / b));
        }
        out
    })
}

fn sample_gamma_shape_scale(state: &mut u64, shape: f64, scale: f64) -> f64 {
    if shape < 1.0 {
        let u = next_uniform_state(state).max(MIN_UNIFORM);
        return sample_gamma_shape_scale(state, shape + 1.0, scale) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let (x, _) = next_normal_pair(state);
        let v_base = 1.0 + c * x;
        if v_base <= 0.0 {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = next_uniform_state(state);
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 || u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return scale * d * v;
        }
    }
}

fn sample_binomial(state: &mut u64, trials: f64, probability: f64) -> f64 {
    if trials == 0.0 || probability == 0.0 {
        return 0.0;
    }
    if probability == 1.0 {
        return trials;
    }

    let effective_p = probability.min(1.0 - probability);
    let mean = trials * effective_p;
    let sampled = if trials <= 10_000.0 {
        sample_binomial_direct(state, trials as u64, effective_p)
    } else if mean < 30.0 {
        sample_poisson_knuth(state, mean).min(trials)
    } else {
        sample_binomial_normal_approx(state, trials, effective_p)
    };

    if probability <= 0.5 {
        sampled
    } else {
        trials - sampled
    }
}

fn sample_binomial_direct(state: &mut u64, trials: u64, probability: f64) -> f64 {
    let mut successes = 0_u64;
    for _ in 0..trials {
        if next_uniform_state(state) < probability {
            successes += 1;
        }
    }
    successes as f64
}

fn sample_poisson_knuth(state: &mut u64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    let threshold = (-lambda).exp();
    let mut product = 1.0;
    let mut count = 0_u64;
    loop {
        count += 1;
        product *= next_uniform_state(state).max(MIN_UNIFORM);
        if product <= threshold {
            return (count - 1) as f64;
        }
    }
}

fn sample_binomial_normal_approx(state: &mut u64, trials: f64, probability: f64) -> f64 {
    let mean = trials * probability;
    let variance = mean * (1.0 - probability);
    if variance <= 0.0 {
        return mean.round().clamp(0.0, trials);
    }
    let (z, _) = next_normal_pair(state);
    (mean + variance.sqrt() * z).round().clamp(0.0, trials)
}

pub(crate) fn generate_uniform_scaled(
    a: f64,
    b: f64,
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    with_rng_state(label, |state| {
        (0..len)
            .map(|_| a + (b - a) * next_uniform_state(&mut state.state))
            .collect()
    })
}

pub(crate) fn generate_normal(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    with_rng_state(label, |state| {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let (z0, z1) = next_normal_pair(&mut state.state);
            out.push(z0);
            if out.len() < len {
                out.push(z1);
            }
        }
        out
    })
}

pub(crate) fn generate_normal_complex(len: usize, label: &str) -> BuiltinResult<Vec<(f64, f64)>> {
    with_rng_state(label, |state| {
        (0..len)
            .map(|_| next_normal_pair(&mut state.state))
            .collect()
    })
}

#[cfg(test)]
pub(crate) fn reset_rng() {
    if let Some(mutex) = RNG_STATE.get() {
        if let Ok(mut guard) = mutex.lock() {
            *guard = RuntimeRng::from(default_snapshot());
        }
    } else {
        let _ = RNG_STATE.set(Mutex::new(RuntimeRng::new()));
    }
    if let Some(mutex) = LEGACY_TOKENS.get() {
        if let Ok(mut guard) = mutex.lock() {
            *guard = LegacyTokenState::new();
        }
    }
}

#[cfg(test)]
pub(crate) fn expected_exponential_sequence(mu: f64, count: usize) -> Vec<f64> {
    let mut seed = DEFAULT_RNG_SEED;
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        let u = next_uniform_state(&mut seed).max(MIN_UNIFORM);
        seq.push(-mu * u.ln());
    }
    seq
}

#[cfg(test)]
pub(crate) fn expected_normal_scaled_sequence(mu: f64, sigma: f64, count: usize) -> Vec<f64> {
    let mut seed = DEFAULT_RNG_SEED;
    let mut seq = Vec::with_capacity(count);
    while seq.len() < count {
        let (z0, z1) = next_normal_pair(&mut seed);
        seq.push(mu + sigma * z0);
        if seq.len() < count {
            seq.push(mu + sigma * z1);
        }
    }
    seq
}

#[cfg(test)]
pub(crate) fn expected_uniform_scaled_sequence(a: f64, b: f64, count: usize) -> Vec<f64> {
    let mut seed = DEFAULT_RNG_SEED;
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        seq.push(a + (b - a) * next_uniform_state(&mut seed));
    }
    seq
}

#[cfg(test)]
pub(crate) fn expected_uniform_sequence(count: usize) -> Vec<f64> {
    let mut seed = DEFAULT_RNG_SEED;
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        seq.push(next_uniform_state(&mut seed));
    }
    seq
}

#[cfg(test)]
pub(crate) fn expected_integer_offset_sequence(span: u128, count: usize) -> Vec<u64> {
    let mut seed = DEFAULT_RNG_SEED;
    generate_integer_offsets_from_state(&mut seed, span, count)
}

#[cfg(test)]
pub(crate) fn expected_complex_sequence(count: usize) -> Vec<(f64, f64)> {
    let mut seed = DEFAULT_RNG_SEED;
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        let re = next_uniform_state(&mut seed);
        let im = next_uniform_state(&mut seed);
        seq.push((re, im));
    }
    seq
}

#[cfg(test)]
pub(crate) fn expected_normal_sequence(count: usize) -> Vec<f64> {
    let mut seed = DEFAULT_RNG_SEED;
    let mut seq = Vec::with_capacity(count);
    while seq.len() < count {
        let (z0, z1) = next_normal_pair(&mut seed);
        seq.push(z0);
        if seq.len() < count {
            seq.push(z1);
        }
    }
    seq
}

#[cfg(test)]
pub(crate) fn expected_complex_normal_sequence(count: usize) -> Vec<(f64, f64)> {
    let mut seed = DEFAULT_RNG_SEED;
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        seq.push(next_normal_pair(&mut seed));
    }
    seq
}

#[cfg(test)]
pub(crate) fn test_guard() -> super::test_support::GlobalStateTestGuard {
    super::test_support::global_state_test_guard()
}

#[cfg(test)]
mod integer_offset_tests {
    use super::*;

    #[test]
    fn rejection_floor_removes_the_incomplete_modulo_bucket() {
        let span = 10_u64;
        let rejection_floor = span.wrapping_neg() % span;
        assert_eq!(rejection_floor, 6);
        for word in 0..rejection_floor {
            assert_eq!(unbiased_offset_from_word(word, span), None);
        }
        assert_eq!(unbiased_offset_from_word(6, span), Some(6));
        assert_eq!(unbiased_offset_from_word(10, span), Some(0));
        assert_eq!(unbiased_offset_from_word(u64::MAX, span), Some(5));
        assert_eq!(
            (u128::from(u64::MAX) + 1 - u128::from(rejection_floor)) % 10,
            0
        );
    }

    #[test]
    fn full_width_offsets_preserve_every_rng_word() {
        let mut state = DEFAULT_RNG_SEED;
        let expected: Vec<u64> = (0..32).map(|_| next_u64_state(&mut state)).collect();
        let mut state = DEFAULT_RNG_SEED;
        assert_eq!(
            generate_integer_offsets_from_state(&mut state, 1_u128 << 64, 32),
            expected
        );
    }

    #[test]
    fn integer_offset_sequences_are_reproducible_for_non_power_of_two_spans() {
        let first = expected_integer_offset_sequence(10, 512);
        let second = expected_integer_offset_sequence(10, 512);
        assert_eq!(first, second);
        assert!(first.iter().all(|value| *value < 10));
        assert!(first.contains(&0));
        assert!(first.contains(&9));
    }
}

#[cfg(test)]
mod parallel_stream_tests {
    use std::rc::Rc;

    use super::*;
    use crate::execution::RuntimeExecutionService;

    fn context() -> crate::context::RuntimeContext {
        crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    fn iteration_values(
        context: &crate::context::RuntimeContext,
        streams: &[runmat_execution::ParallelRandomStream],
    ) -> Vec<Vec<f64>> {
        streams
            .iter()
            .map(|stream| {
                let _scope = context.enter();
                let _stream = enter_parallel_random_stream(context, *stream).unwrap();
                generate_uniform(3, "test").unwrap()
            })
            .collect()
    }

    #[test]
    fn session_streams_are_isolated_and_parallel_substreams_ignore_execution_order() {
        let first = context();
        let second = context();
        {
            let _scope = first.enter();
            set_seed(41).unwrap();
        }
        {
            let _scope = second.enter();
            set_seed(41).unwrap();
        }
        let first_streams = reserve_parallel_random_streams(&first, 4).unwrap();
        let second_streams = reserve_parallel_random_streams(&second, 4).unwrap();
        assert_eq!(first_streams, second_streams);

        let in_order = iteration_values(&first, &first_streams);
        let mut reverse_streams = second_streams.clone();
        reverse_streams.reverse();
        let mut reversed = iteration_values(&second, &reverse_streams);
        reversed.reverse();
        assert_eq!(in_order, reversed);

        let first_parent = {
            let _scope = first.enter();
            generate_uniform(2, "test").unwrap()
        };
        let second_parent = {
            let _scope = second.enter();
            generate_uniform(2, "test").unwrap()
        };
        assert_eq!(first_parent, second_parent);
    }
}
