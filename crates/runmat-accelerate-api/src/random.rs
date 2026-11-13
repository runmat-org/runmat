//! Counter-based random primitives shared by the session stream and provider
//! conformance tests. This is not a stream issuer or a provider fallback.

use serde::{Deserialize, Serialize};

pub const PROVIDER_GPU_RANDOM_OPERATION: &str = "random.gpu_child";
pub const PROVIDER_GPU_INTEGER_OPERATION: &str = "random.gpu_child.integer";
pub const PROVIDER_GPU_INTEGER_BATCH_OPERATION: &str = "random.gpu_child.integer.batch";

/// The exact generator selected by a session GPU stream. Additional public
/// algorithms require their own real state/round function, not an alias of
/// Threefry or a provider's process-global generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuRandomAlgorithm {
    Threefry4x64_20,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuNormalTransform {
    BoxMuller,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuRandomDistribution {
    Uniform,
    StandardNormal,
}

/// Exact integer domain for one GPU-child request. A provider may execute it
/// natively or the runtime may choose host computation before any effect and
/// restore the native class to the selected device. The range is inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuBoundedIntegerDomain {
    pub lower: i128,
    pub span: u128,
}

/// One native bounded-integer generation candidate. Its output class is
/// native, not inferred from the provider's preferred floating precision.
#[derive(Clone, Copy, Debug)]
pub struct GpuIntegerRequest<'a> {
    pub shape: &'a [usize],
    pub class: crate::NumericElementType,
    pub domain: GpuBoundedIntegerDomain,
    pub reservation: GpuRandomReservation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuIntegerWorkload {
    pub domain: GpuBoundedIntegerDomain,
    pub reservation: GpuRandomReservation,
}

/// An unpublished device result and its per-item completion mask. The runtime
/// retains this state across bounded provider batches; only `Complete` may be
/// validated and published as a language value. Clones share authenticated
/// allocation leases and do not create new views or copy payload bytes.
#[derive(Clone)]
pub struct GpuIntegerPending {
    pub output: crate::GpuTensorHandle,
    pub accepted: crate::GpuTensorHandle,
    pub remaining: u64,
}

/// A finite batch of independent Threefry attempt lanes for each unresolved
/// item. A batch never hides an unbounded shader loop. The provider must
/// physically settle its submissions and count readback before returning.
#[derive(Clone, Copy)]
pub struct GpuIntegerBatchRequest<'a> {
    pub integer: GpuIntegerRequest<'a>,
    pub first_attempt: u64,
    pub attempts_per_item: u32,
    pub pending: Option<&'a GpuIntegerPending>,
}

pub enum GpuIntegerBatchProgress {
    Pending(GpuIntegerPending),
    Complete(GpuIntegerPending),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuIntegerBatchWorkload {
    pub first_attempt: u64,
    pub attempts_per_item: u32,
    /// Existing authenticated output/mask remain live during a continuation.
    pub pending: bool,
}

/// Peak physical inventory for one acknowledged batch, not a fabricated
/// bound on the unbounded number of rejection batches. The coordinator must
/// re-admit each later batch against its then-live output and mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuIntegerBatchPhysicalProposal {
    pub output_buffer_bytes: u64,
    pub accepted_buffer_bytes: u64,
    pub count_buffer_bytes: u64,
    pub uniform_buffer_bytes: u64,
    pub staging_buffer_bytes: u64,
    /// Shared WGPU map-completion control allocation retained through its
    /// physical callback, including caller cancellation.
    pub readback_control_host_bytes: u64,
    /// Transient logical-view snapshots while two pending buffers are
    /// authenticated. The first batch has no pending buffers and quotes zero.
    pub pending_view_workspace_bytes: u64,
    pub uniform_allocations: u32,
    pub queue_submissions: u32,
}

impl GpuIntegerBatchPhysicalProposal {
    pub fn peak_bytes(self) -> Option<u64> {
        self.output_buffer_bytes
            .checked_add(self.accepted_buffer_bytes)?
            .checked_add(self.count_buffer_bytes)?
            .checked_add(self.uniform_buffer_bytes)?
            .checked_add(self.staging_buffer_bytes)
    }

    /// Bytes newly retained by this batch. A continuation returns the same
    /// authenticated output and acceptance allocations it received, so their
    /// ownership transition has no incremental retained allocation.
    pub fn newly_retained_output_bytes(self, batch: GpuIntegerBatchWorkload) -> Option<u64> {
        if batch.pending {
            Some(0)
        } else {
            self.output_buffer_bytes
                .checked_add(self.accepted_buffer_bytes)
        }
    }
}

impl GpuIntegerBatchRequest<'_> {
    pub fn validate(self) -> Option<usize> {
        let count = self.integer.element_count()?;
        if self.attempts_per_item == 0
            || self.attempts_per_item > 8
            || self
                .first_attempt
                .checked_add(u64::from(self.attempts_per_item))
                .is_none()
            || self
                .pending
                .is_some_and(|pending| pending.remaining == 0 || pending.remaining > count as u64)
        {
            return None;
        }
        Some(count)
    }
}

impl GpuIntegerRequest<'_> {
    pub fn element_count(self) -> Option<usize> {
        let count = if self.shape.contains(&0) {
            0
        } else {
            self.shape
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent))?
        };
        (u64::try_from(count).ok()? == self.reservation.count).then_some(count)?;
        self.reservation
            .first
            .checked_add(u128::from(self.reservation.count))?;
        GpuBoundedIntegerDomain::new(self.domain.lower, self.domain.span)?;
        Some(count)
    }

    /// The first native shader candidate has no rejection path: modulo by a
    /// power-of-two span is exact, including the full 2^32 domain. General
    /// spans require bounded, acknowledged retry batches before admission.
    pub fn fixed_work_32(self) -> Option<(u32, u32)> {
        self.element_count()?;
        if !self.domain.span.is_power_of_two() || self.domain.span > (1_u128 << 32) {
            return None;
        }
        let upper = self
            .domain
            .lower
            .checked_add(self.domain.span as i128 - 1)?;
        let supported = match self.class {
            crate::NumericElementType::F32
            | crate::NumericElementType::F64
            | crate::NumericElementType::I32 => {
                self.domain.lower >= i32::MIN as i128 && upper <= i32::MAX as i128
            }
            crate::NumericElementType::U32 => self.domain.lower >= 0 && upper <= u32::MAX as i128,
            _ => false,
        };
        if !supported || self.reservation.algorithm != GpuRandomAlgorithm::Threefry4x64_20 {
            return None;
        }
        let mask = (self.domain.span - 1) as u32;
        Some((self.domain.lower as i32 as u32, mask))
    }

    /// All non-power-of-two domains representable by the currently selected
    /// native 32-bit output classes. The acceptance limit is
    /// `floor(2^64 / span) * span`, exactly the shared scalar attempt owner:
    /// words at or above it are rejected, and the remainder is unbiased.
    pub fn rejection_domain_32(self) -> Option<(u32, u64, u64)> {
        self.element_count()?;
        if self.domain.span <= 1
            || self.domain.span.is_power_of_two()
            || self.domain.span > u64::MAX as u128
            || self.reservation.algorithm != GpuRandomAlgorithm::Threefry4x64_20
        {
            return None;
        }
        let upper = self
            .domain
            .lower
            .checked_add(self.domain.span as i128 - 1)?;
        let lower = match self.class {
            crate::NumericElementType::F32
            | crate::NumericElementType::F64
            | crate::NumericElementType::I32
                if self.domain.lower >= i32::MIN as i128 && upper <= i32::MAX as i128 =>
            {
                self.domain.lower as i32 as u32
            }
            crate::NumericElementType::U32
                if self.domain.lower >= 0 && upper <= u32::MAX as i128 =>
            {
                self.domain.lower as u32
            }
            _ => return None,
        };
        let span = self.domain.span as u64;
        let accept_limit = ((1_u128 << 64) / self.domain.span * self.domain.span) as u64;
        Some((lower, span, accept_limit))
    }
}

/// A nonempty permutation uses exactly `n` disjoint child-stream item lanes:
/// reservoir step `t` owns lane `t`, and the final shuffle step `i` owns lane
/// `i`. Rejections stay within that item's attempt lane. This permits an
/// O(k)-storage implementation without a provider-global RNG or biased float
/// index conversion.
#[derive(Clone, Copy, Debug)]
pub struct GpuPermutationRequest {
    pub n: u64,
    pub k: u64,
    pub reservation: GpuRandomReservation,
}

impl GpuPermutationRequest {
    pub fn validate(self) -> Option<()> {
        if self.k > self.n
            || self.n > (1_u64 << 53)
            || self.reservation.count != if self.k == 0 { 0 } else { self.n }
            || self
                .reservation
                .first
                .checked_add(u128::from(self.reservation.count))
                .is_none()
        {
            return None;
        }
        Some(())
    }

    /// The complete result has MATLAB's Double row-vector class and shape.
    pub fn output_requested_bytes(self) -> Option<u64> {
        self.validate()?;
        self.k.checked_mul(8)
    }

    pub fn reservoir_attempt(self, total: u64, attempt: u64) -> Option<Option<u64>> {
        self.validate()?;
        if total <= self.k || total > self.n {
            return None;
        }
        self.reservation
            .bounded_word_attempt(total - 1, u128::from(total), attempt)
    }

    pub fn shuffle_attempt(self, index: u64, attempt: u64) -> Option<Option<u64>> {
        self.validate()?;
        if index >= self.k {
            return None;
        }
        self.reservation
            .bounded_word_attempt(index, u128::from(self.k - index), attempt)
    }
}

impl GpuBoundedIntegerDomain {
    pub fn new(lower: i128, span: u128) -> Option<Self> {
        if span == 0 || span > (1_u128 << 64) {
            return None;
        }
        lower.checked_add((span - 1) as i128)?;
        Some(Self { lower, span })
    }

    /// A rejected word consumes an attempt lane, never an output item or a
    /// later call's counter range. Callers checkpoint between attempts.
    pub fn sample_attempt(
        self,
        reservation: GpuRandomReservation,
        item: u64,
        attempt: u64,
    ) -> Option<Option<i128>> {
        reservation
            .bounded_word_attempt(item, self.span, attempt)
            .map(|accepted| accepted.map(|offset| self.lower + i128::from(offset)))
    }
}

/// A request-local child-stream range, never a provider-global mutable seed.
/// The selected provider must use exactly the supplied logical item counters.
#[derive(Clone, Copy, Debug)]
pub struct GpuRandomRequest<'a> {
    pub shape: &'a [usize],
    pub precision: crate::ProviderPrecision,
    pub storage: crate::GpuTensorStorage,
    pub distribution: GpuRandomDistribution,
    pub reservation: GpuRandomReservation,
}

/// The exact request facts projected into canonical whole-work placement.
/// The output representation owns the shape/class; this carries the stream
/// range and transform without duplicating dimensions or issuing authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuRandomWorkload {
    pub distribution: GpuRandomDistribution,
    pub reservation: GpuRandomReservation,
}

impl GpuRandomRequest<'_> {
    /// A complex standard-normal value has two independent components whose
    /// individual variance is one half. Keep the real request's Box--Muller
    /// mapping and range unchanged; only its complex component is scaled.
    pub fn normal_component_f64(self, item: u64) -> Option<f64> {
        let normal = self.reservation.normal_f64(item)?;
        Some(match self.storage {
            crate::GpuTensorStorage::Real => normal,
            crate::GpuTensorStorage::ComplexInterleaved => normal * std::f64::consts::FRAC_1_SQRT_2,
        })
    }

    pub fn normal_component_f32(self, item: u64) -> Option<f32> {
        let normal = self.reservation.normal_f32(item)?;
        Some(match self.storage {
            crate::GpuTensorStorage::Real => normal,
            crate::GpuTensorStorage::ComplexInterleaved => normal * std::f32::consts::FRAC_1_SQRT_2,
        })
    }

    pub fn element_count(self) -> Option<usize> {
        let count = if self.shape.contains(&0) {
            0
        } else {
            self.shape
                .iter()
                .try_fold(1usize, |count, dim| count.checked_mul(*dim))?
        };
        let components = count.checked_mul(match self.storage {
            crate::GpuTensorStorage::Real => 1,
            crate::GpuTensorStorage::ComplexInterleaved => 2,
        })?;
        if u64::try_from(components).ok()? != self.reservation.count {
            return None;
        }
        self.reservation
            .first
            .checked_add(u128::from(self.reservation.count))?;
        Some(components)
    }
}

/// Provider-local physical quote for the exact selected range. It includes
/// fresh output storage and every simultaneously retained uniform buffer; the
/// caller separately admits provider host registration metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuRandomPhysicalProposal {
    pub output_buffer_bytes: u64,
    pub uniform_buffer_bytes: u64,
    pub uniform_allocations: u32,
    pub dispatches: u32,
    pub queue_submissions: u32,
}

impl GpuRandomPhysicalProposal {
    pub fn requested_incremental_peak_bytes(self) -> Option<u64> {
        self.output_buffer_bytes
            .checked_add(self.uniform_buffer_bytes)
    }
}

/// One committed, disjoint child-stream range. The session owns issuance;
/// providers may use these exact counters only under their selected resource
/// and output-adoption contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuRandomReservation {
    pub algorithm: GpuRandomAlgorithm,
    pub seed: u64,
    pub first: u128,
    pub count: u64,
    pub normal_transform: GpuNormalTransform,
}

impl GpuRandomReservation {
    pub fn word(self, item: u64, attempt: u64) -> Option<u64> {
        if item >= self.count {
            return None;
        }
        let position = self.first.checked_add(u128::from(item))?;
        let counter = [position as u64, (position >> 64) as u64, attempt, 0];
        match self.algorithm {
            GpuRandomAlgorithm::Threefry4x64_20 => {
                Some(threefry4x64_20(counter, [self.seed, 0, 0, 0])[0])
            }
        }
    }

    /// Binary64 uniform in [0, 1), independent of provider chunk width.
    pub fn uniform_f64(self, item: u64) -> Option<f64> {
        const UNIT: f64 = 1.0 / 9_007_199_254_740_992.0;
        Some(((self.word(item, 0)? >> 11) as f64) * UNIT)
    }

    /// Direct 24-bit binary32 generation avoids rounding a value up to one.
    pub fn uniform_f32(self, item: u64) -> Option<f32> {
        const UNIT: f32 = 1.0 / 16_777_216.0;
        Some(((self.word(item, 0)? >> 40) as u32 as f32) * UNIT)
    }

    /// Independent per-item Box--Muller transform. This does not claim the
    /// undocumented MATLAB seed/lane-to-output mapping is bitwise identical.
    pub fn normal_f64(self, item: u64) -> Option<f64> {
        if item >= self.count || self.normal_transform != GpuNormalTransform::BoxMuller {
            return None;
        }
        let position = self.first.checked_add(u128::from(item))?;
        let counter = [position as u64, (position >> 64) as u64, 0, 0];
        let pair = match self.algorithm {
            GpuRandomAlgorithm::Threefry4x64_20 => threefry4x64_20(counter, [self.seed, 0, 0, 0]),
        };
        const UNIT: f64 = 1.0 / 9_007_199_254_740_992.0;
        let u1 = 1.0 - ((pair[0] >> 11) as f64) * UNIT;
        let u2 = ((pair[1] >> 11) as f64) * UNIT;
        Some((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos())
    }

    /// Single-precision Box--Muller uses the same top-24-bit uniform inputs
    /// and binary32 operations as the selected Single shader. Transcendental
    /// implementations may differ in their final low bits by provider.
    pub fn normal_f32(self, item: u64) -> Option<f32> {
        if item >= self.count || self.normal_transform != GpuNormalTransform::BoxMuller {
            return None;
        }
        let position = self.first.checked_add(u128::from(item))?;
        let counter = [position as u64, (position >> 64) as u64, 0, 0];
        let pair = match self.algorithm {
            GpuRandomAlgorithm::Threefry4x64_20 => threefry4x64_20(counter, [self.seed, 0, 0, 0]),
        };
        const UNIT: f32 = 1.0 / 16_777_216.0;
        let u1 = 1.0 - ((pair[0] >> 40) as u32 as f32) * UNIT;
        let u2 = ((pair[1] >> 40) as u32 as f32) * UNIT;
        Some((-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos())
    }

    /// One rejection-sampling attempt for an unbiased integer in [0, span).
    /// `None` means invalid bounds/item; `Some(None)` means retry after the
    /// caller's normal cancellation/work checkpoint. The attempt lane is
    /// isolated from every other item's committed stream range.
    pub fn bounded_word_attempt(self, item: u64, span: u128, attempt: u64) -> Option<Option<u64>> {
        const DOMAIN: u128 = 1_u128 << 64;
        if span == 0 || span > DOMAIN {
            return None;
        }
        if span == DOMAIN {
            return self.word(item, attempt).map(Some);
        }
        let limit = (DOMAIN / span) * span;
        let word = self.word(item, attempt)?;
        Some((u128::from(word) < limit).then_some((u128::from(word) % span) as u64))
    }
}

/// Random123 Threefry4x64 with twenty rounds. Counter and key words use the
/// published Random123 lane order; stream seed-to-key policy belongs to the
/// runtime session owner, not to this primitive.
pub fn threefry4x64_20(counter: [u64; 4], key: [u64; 4]) -> [u64; 4] {
    const ROTATIONS: [(u32, u32); 8] = [
        (14, 16),
        (52, 57),
        (23, 40),
        (5, 37),
        (25, 33),
        (46, 12),
        (58, 22),
        (32, 32),
    ];
    const PARITY: u64 = 0x1BD1_1BDA_A9FC_1A22;
    let keys = [
        key[0],
        key[1],
        key[2],
        key[3],
        PARITY ^ key[0] ^ key[1] ^ key[2] ^ key[3],
    ];
    let mut words = [
        counter[0].wrapping_add(keys[0]),
        counter[1].wrapping_add(keys[1]),
        counter[2].wrapping_add(keys[2]),
        counter[3].wrapping_add(keys[3]),
    ];
    for round in 0..20 {
        let (left, right) = ROTATIONS[round % ROTATIONS.len()];
        if round % 2 == 0 {
            mix(&mut words, 0, 1, left);
            mix(&mut words, 2, 3, right);
        } else {
            mix(&mut words, 0, 3, left);
            mix(&mut words, 2, 1, right);
        }
        if round % 4 == 3 {
            let injection = round / 4 + 1;
            for lane in 0..4 {
                words[lane] = words[lane].wrapping_add(keys[(injection + lane) % 5]);
            }
            words[3] = words[3].wrapping_add(injection as u64);
        }
    }
    words
}

fn mix(words: &mut [u64; 4], first: usize, second: usize, rotation: u32) {
    words[first] = words[first].wrapping_add(words[second]);
    words[second] = words[second].rotate_left(rotation) ^ words[first];
}

#[cfg(test)]
mod tests {
    use super::{
        threefry4x64_20, GpuBoundedIntegerDomain, GpuIntegerBatchPhysicalProposal,
        GpuIntegerBatchWorkload, GpuIntegerRequest, GpuNormalTransform, GpuPermutationRequest,
        GpuRandomAlgorithm, GpuRandomDistribution, GpuRandomRequest, GpuRandomReservation,
    };

    #[test]
    fn integer_batch_continuation_reuses_its_retained_outputs() {
        let physical = GpuIntegerBatchPhysicalProposal {
            output_buffer_bytes: 16,
            accepted_buffer_bytes: 16,
            count_buffer_bytes: 4,
            uniform_buffer_bytes: 8,
            staging_buffer_bytes: 4,
            readback_control_host_bytes: 0,
            pending_view_workspace_bytes: 0,
            uniform_allocations: 1,
            queue_submissions: 2,
        };
        let initial = GpuIntegerBatchWorkload {
            first_attempt: 0,
            attempts_per_item: 8,
            pending: false,
        };
        let continuation = GpuIntegerBatchWorkload {
            first_attempt: 8,
            pending: true,
            ..initial
        };
        assert_eq!(physical.peak_bytes(), Some(48));
        assert_eq!(physical.newly_retained_output_bytes(initial), Some(32));
        assert_eq!(physical.newly_retained_output_bytes(continuation), Some(0));
    }

    #[test]
    fn fixed_native_integer_work_is_only_exact_power_of_two_domain() {
        let reservation = GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 3,
            first: 0,
            count: 4,
            normal_transform: GpuNormalTransform::BoxMuller,
        };
        let shape = [2, 2];
        let request = GpuIntegerRequest {
            shape: &shape,
            class: crate::NumericElementType::I32,
            domain: GpuBoundedIntegerDomain::new(-4, 8).unwrap(),
            reservation,
        };
        assert_eq!(request.fixed_work_32(), Some(((-4_i32) as u32, 7)));
        assert_eq!(
            GpuIntegerRequest {
                domain: GpuBoundedIntegerDomain::new(-4, 7).unwrap(),
                ..request
            }
            .fixed_work_32(),
            None
        );
        assert_eq!(
            GpuIntegerRequest {
                class: crate::NumericElementType::U64,
                ..request
            }
            .fixed_work_32(),
            None
        );
    }

    #[test]
    fn batched_native_acceptance_limit_matches_scalar_attempts() {
        let reservation = GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 31,
            first: 9,
            count: 3,
            normal_transform: GpuNormalTransform::BoxMuller,
        };
        for span in [3_u128, 7, 257, u32::MAX as u128] {
            let domain = GpuBoundedIntegerDomain::new(0, span).unwrap();
            let request = GpuIntegerRequest {
                shape: &[1, 3],
                class: crate::NumericElementType::U32,
                domain,
                reservation,
            };
            let (_, actual_span, accept_limit) = request.rejection_domain_32().unwrap();
            assert_eq!(actual_span as u128, span);
            for item in 0..3 {
                for attempt in 0..32 {
                    let word = reservation.word(item, attempt).unwrap();
                    let from_kernel_rule = (word < accept_limit).then_some(word % actual_span);
                    assert_eq!(
                        reservation.bounded_word_attempt(item, span, attempt),
                        Some(from_kernel_rule),
                    );
                }
            }
        }
    }
    use crate::ProviderPrecision;

    #[test]
    fn bounded_integer_domain_preserves_full_width_and_disjoint_attempt_lanes() {
        let reservation = GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 7,
            first: 19,
            count: 2,
            normal_transform: GpuNormalTransform::BoxMuller,
        };
        let full = GpuBoundedIntegerDomain::new(i64::MIN as i128, 1_u128 << 64)
            .expect("full signed 64-bit domain");
        assert_eq!(
            full.sample_attempt(reservation, 0, 0),
            Some(Some(
                i64::MIN as i128 + reservation.word(0, 0).unwrap() as i128
            ))
        );
        let narrow = GpuBoundedIntegerDomain::new(4, 3).expect("narrow domain");
        for item in 0..2 {
            for attempt in 0..32 {
                if let Some(value) = narrow.sample_attempt(reservation, item, attempt).unwrap() {
                    assert!((4..=6).contains(&value));
                }
            }
        }
        assert_eq!(narrow.sample_attempt(reservation, 2, 0), None);
        assert!(GpuBoundedIntegerDomain::new(0, 0).is_none());
        assert!(GpuBoundedIntegerDomain::new(0, (1_u128 << 64) + 1).is_none());
    }

    #[test]
    fn permutation_reservoir_and_shuffle_use_disjoint_unbiased_item_lanes() {
        let request = GpuPermutationRequest {
            n: 7,
            k: 3,
            reservation: GpuRandomReservation {
                algorithm: GpuRandomAlgorithm::Threefry4x64_20,
                seed: 11,
                first: 37,
                count: 7,
                normal_transform: GpuNormalTransform::BoxMuller,
            },
        };
        assert_eq!(request.output_requested_bytes(), Some(24));
        for total in 4..=7 {
            let offset = (0..100)
                .find_map(|attempt| request.reservoir_attempt(total, attempt).flatten())
                .expect("rejection eventually accepts");
            assert!(offset < total);
        }
        for index in 0..3 {
            let offset = (0..100)
                .find_map(|attempt| request.shuffle_attempt(index, attempt).flatten())
                .expect("rejection eventually accepts");
            assert!(offset < 3 - index);
        }
        assert_eq!(request.reservoir_attempt(3, 0), None);
        assert_eq!(request.shuffle_attempt(3, 0), None);
    }

    #[test]
    fn random123_known_answer_vectors() {
        // D. E. Shaw Research Random123 tests/kat_vectors, Threefry4x64 R20.
        assert_eq!(
            threefry4x64_20([0; 4], [0; 4]),
            [
                0x0921_8ebd_e6c8_5537,
                0x5594_1f52_66d8_6105,
                0x4bd2_5e16_2824_34dc,
                0xee29_ec84_6bd2_e40b,
            ]
        );
        assert_eq!(
            threefry4x64_20([u64::MAX; 4], [u64::MAX; 4]),
            [
                0x29c2_4097_942b_ba1b,
                0x0371_bbfb_0f6f_4e11,
                0x3c23_1ffa_33f8_3a1c,
                0xcd29_113f_de32_d168,
            ]
        );
    }

    #[test]
    fn reserved_items_are_chunk_independent_and_bounded() {
        let range = GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 7,
            first: u128::from(u64::MAX) - 1,
            count: 4,
            normal_transform: GpuNormalTransform::BoxMuller,
        };
        let before_boundary = range.word(1, 0).expect("counter before word boundary");
        let after_boundary = range.word(2, 0).expect("counter after word boundary");
        assert_ne!(before_boundary, after_boundary);
        assert_eq!(range.word(2, 0), Some(after_boundary));
        assert!(range.word(4, 0).is_none());
        for item in 0..range.count {
            let double = range.uniform_f64(item).expect("binary64 uniform");
            let single = range.uniform_f32(item).expect("binary32 uniform");
            let normal = range.normal_f64(item).expect("Box-Muller normal");
            let single_normal = range.normal_f32(item).expect("Single Box-Muller normal");
            assert!((0.0..1.0).contains(&double));
            assert!((0.0..1.0).contains(&single));
            assert!(normal.is_finite());
            assert!(single_normal.is_finite());
            assert!(normal.is_finite());
            assert_eq!(range.bounded_word_attempt(item, 1, 0), Some(Some(0)));
            assert_eq!(
                range.bounded_word_attempt(item, 1_u128 << 64, 0),
                range.word(item, 0).map(Some)
            );
        }
    }

    #[test]
    fn physical_request_requires_the_exact_reserved_geometry() {
        let reservation = GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 0,
            first: 11,
            count: 6,
            normal_transform: GpuNormalTransform::BoxMuller,
        };
        let request = GpuRandomRequest {
            shape: &[2, 3],
            precision: ProviderPrecision::F64,
            storage: crate::GpuTensorStorage::Real,
            distribution: GpuRandomDistribution::Uniform,
            reservation,
        };
        let complex_normal = GpuRandomRequest {
            shape: &[3],
            storage: crate::GpuTensorStorage::ComplexInterleaved,
            distribution: GpuRandomDistribution::StandardNormal,
            ..request
        };
        for item in 0..reservation.count {
            assert_eq!(
                request.normal_component_f64(item),
                reservation.normal_f64(item)
            );
            assert_eq!(
                request.normal_component_f32(item),
                reservation.normal_f32(item)
            );
            assert_eq!(
                complex_normal.normal_component_f64(item),
                reservation
                    .normal_f64(item)
                    .map(|value| value * std::f64::consts::FRAC_1_SQRT_2)
            );
            assert_eq!(
                complex_normal.normal_component_f32(item),
                reservation
                    .normal_f32(item)
                    .map(|value| value * std::f32::consts::FRAC_1_SQRT_2)
            );
        }
        assert_eq!(request.element_count(), Some(6));
        assert_eq!(
            GpuRandomRequest {
                storage: crate::GpuTensorStorage::ComplexInterleaved,
                reservation: GpuRandomReservation {
                    count: 12,
                    ..reservation
                },
                ..request
            }
            .element_count(),
            Some(12)
        );
        assert_eq!(
            GpuRandomRequest {
                shape: &[2, 2],
                ..request
            }
            .element_count(),
            None
        );
        assert_eq!(
            GpuRandomRequest {
                reservation: GpuRandomReservation {
                    first: u128::MAX,
                    ..reservation
                },
                ..request
            }
            .element_count(),
            None
        );
        assert_eq!(
            GpuRandomRequest {
                shape: &[usize::MAX, 2, 0],
                reservation: GpuRandomReservation {
                    count: 0,
                    ..reservation
                },
                ..request
            }
            .element_count(),
            Some(0)
        );
    }
}
