use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod distributions;
pub(crate) mod gpu;
pub(crate) mod gpu_effect;
pub(crate) mod gpu_execution;
pub(crate) mod gpu_integer;
pub(crate) mod gpu_integer_batch;
pub(crate) mod gpu_like;
mod mt19937;
mod normal;
mod parallel;
mod reservation;
mod state;
mod turn;
mod uniform;
pub(crate) use distributions::{
    fill_exponential_array, generate_binomial, generate_exponential, generate_gamma_shape_scale,
    generate_normal_scaled, generate_student_t, generate_weibull,
};
pub(crate) use normal::{
    fill_normal, fill_normal_cooperative, generate_normal, generate_normal_cooperative,
};
use normal::{next_normal, next_normal_pair};
pub(crate) use parallel::ParallelRandomScope;
pub use parallel::{
    enter_parallel_random_stream, enter_parallel_random_stream_async,
    reserve_parallel_random_streams, reserve_parallel_random_streams_async,
    reserve_parallel_randomness_context_async, ParallelRandomStreamGuard,
};
pub(crate) use state::{RandomState, RngAlgorithm, RngSnapshot, RuntimeRng};
pub(crate) use turn::acquire as acquire_turn;
pub(crate) use turn::acquire_original as acquire_original_turn;
#[cfg(test)]
pub(crate) use turn::try_acquire as try_acquire_turn;
pub(crate) use turn::RandomTurn;
pub(crate) use turn::RandomTurnGate;
pub(crate) use uniform::{fill_uniform_scaled, generate_uniform, generate_uniform_cooperative};

pub(crate) const DEFAULT_RNG_SEED: u64 = runmat_execution::task::RUNMAT_MT_DEFAULT_SEED;
pub(crate) const DEFAULT_USER_SEED: u64 = 0;
const RNG_MULTIPLIER: u64 = 6364136223846793005;
const RNG_INCREMENT: u64 = 1;
const RNG_SHIFT: u32 = 11;
const RNG_SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
const MIN_UNIFORM: f64 = f64::MIN_POSITIVE;

fn random_error(label: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(label).build()
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

fn with_rng_state_in<T>(
    context: Option<&crate::context::RuntimeContext>,
    label: &str,
    operation: impl FnOnce(&mut RuntimeRng) -> T,
) -> BuiltinResult<T> {
    if let Some(context) = context {
        let scope = context.parallel_random_scope();
        let state = scope.map_or(&context.state().random, |scope| &scope.state);
        let mut state = state.try_borrow_mut().map_err(|_| {
            random_error(label, format!("{label}: random stream is already in use"))
        })?;
        return Ok(operation(&mut state));
    }
    let mut state = fallback_rng_state()
        .lock()
        .map_err(|_| random_error(label, format!("{label}: failed to acquire RNG lock")))?;
    Ok(operation(&mut state))
}

fn with_rng_state_mut<T>(
    turn: &mut RandomTurn,
    label: &str,
    may_draw: bool,
    operation: impl FnOnce(&mut RuntimeRng) -> T,
) -> BuiltinResult<T> {
    let context = turn.owner().cloned();
    turn.with_state(label, |state| {
        if may_draw {
            state.state.prepare_mutation(context.as_ref())?;
        }
        Ok(operation(state))
    })?
}

/// A cost-class fact only: inspect the explicitly owned source stream without
/// cloning or materializing its state, acquiring a draw turn, or advancing it.
/// A selected caller revalidates this fact after acquiring its ordered turn.
pub(crate) fn normal_source_algorithm(
    context: &crate::context::RuntimeContext,
) -> BuiltinResult<RngAlgorithm> {
    with_rng_state_in(Some(context), "randn", |rng| rng.state.algorithm())
}

pub(crate) fn normal_source_algorithm_in_turn(
    turn: &mut RandomTurn,
) -> BuiltinResult<RngAlgorithm> {
    turn.with_state("randn", |rng| rng.state.algorithm())
}

fn legacy_tokens() -> &'static Mutex<LegacyTokenState> {
    LEGACY_TOKENS.get_or_init(|| Mutex::new(LegacyTokenState::new()))
}

#[cfg(test)]
pub(crate) fn snapshot() -> BuiltinResult<RngSnapshot> {
    let mut turn = try_acquire_turn(None, "rng")?;
    snapshot_in_turn(&mut turn)
}

pub(crate) fn snapshot_in_turn(turn: &mut RandomTurn) -> BuiltinResult<RngSnapshot> {
    turn.with_state("rng", |state| state.snapshot())
}

pub(crate) fn apply_snapshot_in_turn(
    turn: &mut RandomTurn,
    mut snapshot: RngSnapshot,
) -> BuiltinResult<RngSnapshot> {
    let owner = turn.owner().cloned();
    turn.with_state("rng", |state| {
        // A parsed full-state value may have been retained before this
        // turn began. Rehome its exact bytes before publishing it.
        if matches!(snapshot.state, RandomState::Mt(_)) {
            snapshot.state.prepare_mutation(owner.as_ref())?;
        }
        let previous = state.snapshot();
        state.state = snapshot.state;
        state.seed = snapshot.seed;
        Ok(previous)
    })?
}

#[cfg(test)]
pub(crate) fn set_seed(seed: u64) -> BuiltinResult<RngSnapshot> {
    let mut turn = try_acquire_turn(None, "rng")?;
    set_seed_in_turn(&mut turn, seed)
}

pub(crate) fn set_seed_in_turn(turn: &mut RandomTurn, seed: u64) -> BuiltinResult<RngSnapshot> {
    apply_snapshot_in_turn(
        turn,
        RngSnapshot::new(RandomState::seeded(seed), Some(seed)),
    )
}

pub(crate) fn legacy_seed_value_in_turn(turn: &mut RandomTurn) -> BuiltinResult<u64> {
    let snapshot = snapshot_in_turn(turn)?;
    if let Some(seed) = snapshot.seed {
        if snapshot.state == RandomState::seeded(seed) {
            return Ok(seed);
        }
    }
    let mut tokens = legacy_tokens()
        .lock()
        .map_err(|_| random_error("rand", "rand: failed to acquire legacy RNG token lock"))?;
    Ok(tokens.insert(snapshot))
}

pub(crate) fn set_legacy_seed_in_turn(
    turn: &mut RandomTurn,
    seed_or_token: u64,
) -> BuiltinResult<RngSnapshot> {
    if let Some(snapshot) = legacy_tokens()
        .lock()
        .map_err(|_| random_error("rand", "rand: failed to acquire legacy RNG token lock"))?
        .entries
        .get(&seed_or_token)
        .cloned()
    {
        apply_snapshot_in_turn(turn, snapshot)
    } else {
        set_seed_in_turn(turn, seed_or_token)
    }
}

pub(crate) fn set_default_in_turn(turn: &mut RandomTurn) -> BuiltinResult<RngSnapshot> {
    apply_snapshot_in_turn(turn, default_snapshot())
}

pub(crate) fn default_snapshot() -> RngSnapshot {
    RngSnapshot::new(
        RandomState::seeded(DEFAULT_USER_SEED),
        Some(DEFAULT_USER_SEED),
    )
}

/// Generate unbiased offsets in `0..span` from the authoritative 64-bit RNG stream.
///
/// A span of `2^64` denotes the complete `u64` domain. Smaller spans use
/// rejection before the modulo operation, so no destination value receives an
/// extra source word when the span does not divide `2^64`.
pub(crate) fn generate_integer_offsets(
    turn: &mut RandomTurn,
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
    let mut offsets = Vec::new();
    offsets.try_reserve_exact(len).map_err(|source| {
        build_runtime_error(format!("{label}: cannot reserve integer offsets"))
            .with_builtin(label)
            .with_source(source)
            .build()
    })?;
    with_rng_state_mut(turn, label, len != 0, |state| {
        generate_integer_offsets_from_state(&mut state.state, span, len, &mut offsets);
    })?;
    Ok(offsets)
}

/// Fill a caller-owned, pre-admitted native U64 workspace using the same
/// rejection and source-word order as `generate_integer_offsets`.
pub(crate) fn fill_integer_offsets(
    turn: &mut RandomTurn,
    span: u128,
    offsets: &mut [u64],
    label: &str,
) -> BuiltinResult<()> {
    if span == 0 || span > (1_u128 << 64) {
        return Err(random_error(
            label,
            format!("{label}: integer sampling span must be in 1..=2^64"),
        ));
    }
    with_rng_state_mut(turn, label, !offsets.is_empty() && span != 1, |state| {
        for slot in offsets {
            *slot = if span == 1 {
                0
            } else if span == (1_u128 << 64) {
                next_u64_state(&mut state.state)
            } else {
                let span = span as u64;
                loop {
                    let word = next_u64_state(&mut state.state);
                    if let Some(offset) = unbiased_offset_from_word(word, span) {
                        break offset;
                    }
                }
            };
        }
    })?;
    Ok(())
}

/// Draw one unbiased offset per shrinking Fisher–Yates span under one stream
/// lock. Rejection consumes additional source words but never biases a span.
pub(crate) fn generate_permutation_offsets(
    turn: &mut RandomTurn,
    n: usize,
    k: usize,
) -> BuiltinResult<Vec<usize>> {
    let mut offsets = Vec::new();
    offsets.try_reserve_exact(k).map_err(|source| {
        build_runtime_error("randperm: cannot reserve permutation offsets")
            .with_builtin("randperm")
            .with_source(source)
            .build()
    })?;
    with_rng_state_mut(turn, "randperm", k != 0 && n > 1, |state| {
        for i in 0..k {
            let span = (n - i) as u64;
            if span == 1 {
                offsets.push(0);
                continue;
            }
            let offset = loop {
                let word = next_u64_state(&mut state.state);
                if let Some(offset) = unbiased_offset_from_word(word, span) {
                    break offset;
                }
            };
            offsets.push(offset as usize);
        }
    })?;
    Ok(offsets)
}

/// Fill an already admitted offset buffer without allocating after the first
/// stream word. The shrinking-span loop is the same Fisher–Yates owner as
/// `generate_permutation_offsets`.
pub(crate) fn fill_permutation_offsets(
    turn: &mut RandomTurn,
    n: usize,
    offsets: &mut [u64],
) -> BuiltinResult<()> {
    if offsets.len() > n {
        return Err(random_error("randperm", "randperm: K exceeds N"));
    }
    with_rng_state_mut(turn, "randperm", !offsets.is_empty() && n > 1, |state| {
        for (i, slot) in offsets.iter_mut().enumerate() {
            let span = (n - i) as u64;
            *slot = if span == 1 {
                0
            } else {
                loop {
                    let word = next_u64_state(&mut state.state);
                    if let Some(offset) = unbiased_offset_from_word(word, span) {
                        break offset;
                    }
                }
            };
        }
    })?;
    Ok(())
}

fn generate_integer_offsets_from_state(
    state: &mut RandomState,
    span: u128,
    len: usize,
    out: &mut Vec<u64>,
) {
    if span == 1 {
        out.resize(len, 0);
        return;
    }
    if span == (1_u128 << 64) {
        for _ in 0..len {
            out.push(next_u64_state(state));
        }
        return;
    }

    let span = span as u64;
    while out.len() < len {
        let word = next_u64_state(state);
        if let Some(offset) = unbiased_offset_from_word(word, span) {
            out.push(offset);
        }
    }
}

fn unbiased_offset_from_word(word: u64, span: u64) -> Option<u64> {
    debug_assert!(span > 1);
    let rejection_floor = span.wrapping_neg() % span;
    (word >= rejection_floor).then(|| word % span)
}

pub(crate) fn generate_uniform_single(
    turn: &mut RandomTurn,
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    generate_uniform(turn, len, label).map(|data| data.into_iter().map(single_open_unit).collect())
}

pub(crate) fn single_open_unit(value: f64) -> f64 {
    // Rounding the largest binary64 values to Single can produce exactly 1.0.
    (value as f32).min(f32::from_bits(0x3f7f_ffff)) as f64
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

pub(crate) fn generate_complex(
    turn: &mut RandomTurn,
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<(f64, f64)>> {
    with_rng_state_mut(turn, label, len != 0, |state| {
        (0..len)
            .map(|_| {
                let re = next_uniform_state(&mut state.state);
                let im = next_uniform_state(&mut state.state);
                (re, im)
            })
            .collect()
    })
}

pub(crate) fn next_uniform_state(state: &mut RandomState) -> f64 {
    match state {
        RandomState::Mt(_) => state
            .mt_mut()
            .expect("MT draw must have a unique admitted state")
            .next_f64(),
        RandomState::MtSeed(_) => panic!("MT draw requires admitted materialization"),
        RandomState::Lcg(value) => {
            let word = next_lcg_word(value);
            let bits = word >> RNG_SHIFT;
            (bits as f64) * RNG_SCALE
        }
    }
}

fn next_u64_state(state: &mut RandomState) -> u64 {
    match state {
        RandomState::Mt(_) => {
            let engine = state
                .mt_mut()
                .expect("MT draw must have a unique admitted state");
            (u64::from(engine.next_u32()) << 32) | u64::from(engine.next_u32())
        }
        RandomState::MtSeed(_) => panic!("MT draw requires admitted materialization"),
        RandomState::Lcg(value) => next_lcg_word(value),
    }
}

fn next_lcg_word(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(RNG_MULTIPLIER)
        .wrapping_add(RNG_INCREMENT);
    *state
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
    let mut seed = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        let u = next_uniform_state(&mut seed).max(MIN_UNIFORM);
        seq.push(-mu * u.ln());
    }
    seq
}

#[cfg(test)]
pub(crate) fn expected_normal_scaled_sequence(mu: f64, sigma: f64, count: usize) -> Vec<f64> {
    let mut seed = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
    let mut seq = Vec::with_capacity(count);
    while seq.len() < count {
        if matches!(&seed, RandomState::Mt(_)) {
            seq.push(mu + sigma * next_normal(&mut seed));
            continue;
        }
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
    let mut seed = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        seq.push(a + (b - a) * next_uniform_state(&mut seed));
    }
    seq
}

#[cfg(test)]
pub(crate) fn expected_uniform_sequence(count: usize) -> Vec<f64> {
    let mut seed = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        seq.push(next_uniform_state(&mut seed));
    }
    seq
}

#[cfg(test)]
pub(crate) fn expected_integer_offset_sequence(span: u128, count: usize) -> Vec<u64> {
    let mut seed = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
    let mut offsets = Vec::with_capacity(count);
    generate_integer_offsets_from_state(&mut seed, span, count, &mut offsets);
    offsets
}

#[cfg(test)]
pub(crate) fn expected_permutation_offsets(n: usize, k: usize) -> Vec<usize> {
    let mut seed = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
    (0..k)
        .map(|i| {
            let span = (n - i) as u64;
            if span == 1 {
                return 0;
            }
            loop {
                if let Some(offset) = unbiased_offset_from_word(next_u64_state(&mut seed), span) {
                    break offset as usize;
                }
            }
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn expected_complex_sequence(count: usize) -> Vec<(f64, f64)> {
    let mut seed = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
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
    let mut seed = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
    let mut seq = Vec::with_capacity(count);
    while seq.len() < count {
        if matches!(&seed, RandomState::Mt(_)) {
            seq.push(next_normal(&mut seed));
            continue;
        }
        let (z0, z1) = next_normal_pair(&mut seed);
        seq.push(z0);
        if seq.len() < count {
            seq.push(z1);
        }
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
    fn single_rounding_keeps_half_open_uniform_range() {
        let near_one = f64::from_bits(1.0_f64.to_bits() - 1);
        assert_eq!(
            single_open_unit(near_one),
            f64::from(f32::from_bits(0x3f7f_ffff))
        );
        assert_eq!(single_open_unit(0.0), 0.0);
    }

    #[test]
    fn full_width_offsets_preserve_every_rng_word() {
        let mut state = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
        let expected: Vec<u64> = (0..32).map(|_| next_u64_state(&mut state)).collect();
        let mut state = RandomState::seeded_unbounded(DEFAULT_USER_SEED);
        let mut offsets = Vec::with_capacity(32);
        generate_integer_offsets_from_state(&mut state, 1_u128 << 64, 32, &mut offsets);
        assert_eq!(offsets, expected);
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

    fn generate_uniform(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
        let mut turn = try_acquire_turn(None, label)?;
        super::generate_uniform(&mut turn, len, label)
    }

    fn context() -> crate::context::RuntimeContext {
        crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn caller_owned_offsets_keep_integer_and_permutation_word_order() {
        for span in [1, 10, 1_u128 << 64] {
            let expected_context = context();
            let actual_context = context();
            let mut expected_turn = try_acquire_turn(Some(&expected_context), "randi").unwrap();
            let expected = generate_integer_offsets(&mut expected_turn, span, 31, "randi").unwrap();
            let mut actual_turn = try_acquire_turn(Some(&actual_context), "randi").unwrap();
            let mut actual = [0_u64; 31];
            fill_integer_offsets(&mut actual_turn, span, &mut actual, "randi").unwrap();
            assert_eq!(actual.as_slice(), expected.as_slice());
        }
        let expected_context = context();
        let actual_context = context();
        let mut expected_turn = try_acquire_turn(Some(&expected_context), "randperm").unwrap();
        let expected = generate_permutation_offsets(&mut expected_turn, 37, 19).unwrap();
        let mut actual_turn = try_acquire_turn(Some(&actual_context), "randperm").unwrap();
        let mut actual = [0_u64; 19];
        fill_permutation_offsets(&mut actual_turn, 37, &mut actual).unwrap();
        let expected: Vec<_> = expected.into_iter().map(|offset| offset as u64).collect();
        assert_eq!(actual.as_slice(), expected.as_slice());
    }

    fn iteration_values(
        context: &crate::context::RuntimeContext,
        streams: &[runmat_execution::ParallelRandomStream],
    ) -> Vec<Vec<f64>> {
        streams
            .iter()
            .map(|stream| {
                let _scope = context.enter();
                let _stream = enter_parallel_random_stream(context, stream.clone()).unwrap();
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
        assert!(first_streams.iter().all(|stream| matches!(
            stream,
            runmat_execution::ParallelRandomStream::Mt19937InitialSeedV1 { .. }
        )));

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

    #[test]
    fn versioned_lcg_parallel_replay_preserves_old_word_conversion() {
        let context = context();
        let parent = with_rng_state_in(Some(&context), "test", |rng| rng.snapshot()).unwrap();
        let stream = runmat_execution::ParallelRandomStream::RunMatLcgV1 { state: 41 };
        {
            let _context = context.enter();
            let _scope = enter_parallel_random_stream(&context, stream).unwrap();
            let mut word_state = 41_u64;
            let expected = (next_lcg_word(&mut word_state) >> RNG_SHIFT) as f64 * RNG_SCALE;
            assert_eq!(generate_uniform(1, "test").unwrap(), vec![expected]);
        }
        assert_eq!(
            with_rng_state_in(Some(&context), "test", |rng| rng.snapshot()).unwrap(),
            parent
        );
    }

    #[test]
    fn new_lcg_parallel_range_replays_the_existing_iteration_states() {
        let compact_context = context();
        let vector_context = context();
        let compact_guard = enter_parallel_random_stream(
            &compact_context,
            runmat_execution::ParallelRandomStream::RunMatLcgV1 { state: 41 },
        )
        .unwrap();
        let vector_guard = enter_parallel_random_stream(
            &vector_context,
            runmat_execution::ParallelRandomStream::RunMatLcgV1 { state: 41 },
        )
        .unwrap();
        let compact = futures::executor::block_on(reserve_parallel_randomness_context_async(
            &compact_context,
            5,
        ))
        .unwrap();
        let vector = reserve_parallel_random_streams(&vector_context, 5).unwrap();
        assert!(matches!(
            &compact,
            runmat_execution::ParallelRandomnessContext::RunMatLcgStateRangeV1 { .. }
        ));
        for (index, expected) in vector.into_iter().enumerate() {
            assert_eq!(compact.deterministic_stream(index), Some(expected));
        }
        assert_eq!(
            with_rng_state_in(Some(&compact_context), "test", |rng| rng.snapshot()).unwrap(),
            with_rng_state_in(Some(&vector_context), "test", |rng| rng.snapshot()).unwrap()
        );
        drop((compact_guard, vector_guard));
    }

    #[test]
    fn scoped_parallel_draws_never_replace_coordinator_state() {
        let coordinator = context();
        let streams = reserve_parallel_random_streams(&coordinator, 1).unwrap();
        let parent_after_reservation =
            with_rng_state_in(Some(&coordinator), "test", |rng| rng.snapshot()).unwrap();
        let task = coordinator.clone().with_parallel_random_scope().unwrap();
        {
            let _scope = task.enter();
            let _stream = enter_parallel_random_stream(&task, streams[0].clone()).unwrap();
            let first = generate_uniform(3, "test").unwrap();
            assert_eq!(first.len(), 3);
            assert_eq!(
                with_rng_state_in(Some(&coordinator), "test", |rng| rng.snapshot()).unwrap(),
                parent_after_reservation
            );
        }
        assert_eq!(
            with_rng_state_in(Some(&coordinator), "test", |rng| rng.snapshot()).unwrap(),
            parent_after_reservation
        );
    }
}
