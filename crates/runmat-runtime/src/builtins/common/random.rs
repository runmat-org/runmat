use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

pub(crate) const DEFAULT_RNG_SEED: u64 = 0x9e3779b97f4a7c15;
pub(crate) const DEFAULT_USER_SEED: u64 = 0;
const RNG_MULTIPLIER: u64 = 6364136223846793005;
const RNG_INCREMENT: u64 = 1;
const RNG_SHIFT: u32 = 11;
const RNG_SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
const MIN_UNIFORM: f64 = f64::MIN_POSITIVE;

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

#[derive(Clone, Copy)]
struct GlobalRng {
    state: u64,
    seed: Option<u64>,
    algorithm: RngAlgorithm,
}

impl GlobalRng {
    fn new() -> Self {
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

impl From<RngSnapshot> for GlobalRng {
    fn from(snapshot: RngSnapshot) -> Self {
        Self {
            state: snapshot.state,
            seed: snapshot.seed,
            algorithm: snapshot.algorithm,
        }
    }
}

static RNG_STATE: OnceLock<Mutex<GlobalRng>> = OnceLock::new();

fn rng_state() -> &'static Mutex<GlobalRng> {
    RNG_STATE.get_or_init(|| Mutex::new(GlobalRng::new()))
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

pub(crate) fn snapshot() -> Result<RngSnapshot, String> {
    rng_state()
        .lock()
        .map(|guard| guard.snapshot())
        .map_err(|_| "rng: failed to acquire RNG lock".to_string())
}

pub(crate) fn apply_snapshot(snapshot: RngSnapshot) -> Result<RngSnapshot, String> {
    let mut guard = rng_state()
        .lock()
        .map_err(|_| "rng: failed to acquire RNG lock".to_string())?;
    let previous = guard.snapshot();
    guard.state = snapshot.state;
    guard.seed = snapshot.seed;
    guard.algorithm = snapshot.algorithm;
    Ok(previous)
}

pub(crate) fn set_seed(seed: u64) -> Result<RngSnapshot, String> {
    let state = mix_seed(seed);
    apply_snapshot(RngSnapshot::new(state, Some(seed), RngAlgorithm::RunMatLcg))
}

pub(crate) fn set_default() -> Result<RngSnapshot, String> {
    apply_snapshot(default_snapshot())
}

pub(crate) fn default_snapshot() -> RngSnapshot {
    RngSnapshot::new(
        DEFAULT_RNG_SEED,
        Some(DEFAULT_USER_SEED),
        RngAlgorithm::RunMatLcg,
    )
}

pub(crate) fn generate_uniform(len: usize, label: &str) -> Result<Vec<f64>, String> {
    let mut guard = rng_state()
        .lock()
        .map_err(|_| format!("{label}: failed to acquire RNG lock"))?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(next_uniform_state(&mut guard.state));
    }
    Ok(out)
}

pub(crate) fn generate_complex(len: usize, label: &str) -> Result<Vec<(f64, f64)>, String> {
    let mut guard = rng_state()
        .lock()
        .map_err(|_| format!("{label}: failed to acquire RNG lock"))?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let re = next_uniform_state(&mut guard.state);
        let im = next_uniform_state(&mut guard.state);
        out.push((re, im));
    }
    Ok(out)
}

pub(crate) fn next_uniform_state(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(RNG_MULTIPLIER)
        .wrapping_add(RNG_INCREMENT);
    let bits = *state >> RNG_SHIFT;
    (bits as f64) * RNG_SCALE
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

pub(crate) fn generate_normal(len: usize, label: &str) -> Result<Vec<f64>, String> {
    let mut guard = rng_state()
        .lock()
        .map_err(|_| format!("{label}: failed to acquire RNG lock"))?;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let (z0, z1) = next_normal_pair(&mut guard.state);
        out.push(z0);
        if out.len() < len {
            out.push(z1);
        }
    }
    Ok(out)
}

pub(crate) fn generate_normal_complex(len: usize, label: &str) -> Result<Vec<(f64, f64)>, String> {
    let mut guard = rng_state()
        .lock()
        .map_err(|_| format!("{label}: failed to acquire RNG lock"))?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let (re, im) = next_normal_pair(&mut guard.state);
        out.push((re, im));
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) fn reset_rng() {
    if let Some(mutex) = RNG_STATE.get() {
        if let Ok(mut guard) = mutex.lock() {
            *guard = GlobalRng::from(default_snapshot());
        }
    } else {
        let _ = RNG_STATE.set(Mutex::new(GlobalRng::new()));
    }
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
static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

#[cfg(test)]
pub(crate) fn test_lock() -> &'static Mutex<()> {
    TEST_MUTEX.get_or_init(|| Mutex::new(()))
}
