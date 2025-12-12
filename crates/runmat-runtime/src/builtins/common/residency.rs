//! Shared heuristics for deciding when newly constructed arrays should prefer
//! GPU residency, even when none of their inputs already live on the device.

use runmat_accelerate_api::{provider, sequence_threshold_hint};

const DEFAULT_SEQUENCE_MIN_LEN: usize = 4_096;
const MIN_THRESHOLD: usize = 1_024;

/// Kinds of sequence-producing builtins that can consult the residency helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceIntent {
    Linspace,
    Logspace,
    Colon,
    MeshAxis,
    Generic,
}

impl SequenceIntent {
    fn scale(self) -> f64 {
        match self {
            SequenceIntent::MeshAxis => 0.5,
            SequenceIntent::Colon => 1.0,
            SequenceIntent::Linspace => 1.0,
            SequenceIntent::Logspace => 1.0,
            SequenceIntent::Generic => 1.0,
        }
    }
}

/// Describes why we recommended GPU or CPU residency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyReason {
    ExplicitGpuInput,
    DisabledByEnv,
    ProviderUnavailable,
    ZeroLength,
    BelowThreshold,
    ThresholdHint,
}

/// Final decision returned by the residency helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyDecision {
    pub prefer_gpu: bool,
    pub reason: ResidencyReason,
}

impl ResidencyDecision {
    fn gpu(reason: ResidencyReason) -> Self {
        Self {
            prefer_gpu: true,
            reason,
        }
    }

    fn cpu(reason: ResidencyReason) -> Self {
        Self {
            prefer_gpu: false,
            reason,
        }
    }
}

/// Decide whether a sequence of `len` elements for the supplied intent should
/// prefer GPU residency.
///
/// `explicit_gpu_input` should be `true` when any of the arguments already
/// reside on the GPU (for example, `gpuArray.linspace(...)`).
pub fn sequence_gpu_preference(
    len: usize,
    intent: SequenceIntent,
    explicit_gpu_input: bool,
) -> ResidencyDecision {
    if explicit_gpu_input {
        return ResidencyDecision::gpu(ResidencyReason::ExplicitGpuInput);
    }

    if len == 0 {
        return ResidencyDecision::cpu(ResidencyReason::ZeroLength);
    }

    if sequence_heuristics_disabled() {
        return ResidencyDecision::cpu(ResidencyReason::DisabledByEnv);
    }

    if provider().is_none() {
        return ResidencyDecision::cpu(ResidencyReason::ProviderUnavailable);
    }

    let threshold = threshold_for_intent(intent);
    if len >= threshold {
        return ResidencyDecision::gpu(ResidencyReason::ThresholdHint);
    }

    ResidencyDecision::cpu(ResidencyReason::BelowThreshold)
}

fn threshold_for_intent(intent: SequenceIntent) -> usize {
    let env_override = std::env::var("RUNMAT_SEQUENCE_GPU_MIN")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok());

    let base = env_override
        .or_else(sequence_threshold_hint)
        .unwrap_or(DEFAULT_SEQUENCE_MIN_LEN);

    let scaled = (base as f64 * intent.scale()).round() as isize;
    scaled.max(MIN_THRESHOLD as isize) as usize
}

fn sequence_heuristics_disabled() -> bool {
    matches!(
        std::env::var("RUNMAT_SEQUENCE_GPU_DISABLE"),
        Ok(flag) if flag.trim().eq_ignore_ascii_case("1")
            || flag.trim().eq_ignore_ascii_case("true")
            || flag.trim().eq_ignore_ascii_case("yes")
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_accelerate::simple_provider;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn reset_env() {
        std::env::remove_var("RUNMAT_SEQUENCE_GPU_DISABLE");
        std::env::remove_var("RUNMAT_SEQUENCE_GPU_MIN");
    }

    #[test]
    fn explicit_gpu_short_circuits() {
        let _guard = ENV_LOCK.lock().unwrap();
        reset_env();
        let decision = sequence_gpu_preference(4, SequenceIntent::Linspace, true);
        assert!(decision.prefer_gpu);
        assert_eq!(decision.reason, ResidencyReason::ExplicitGpuInput);
    }

    #[test]
    fn env_disable_blocks_gpu() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("RUNMAT_SEQUENCE_GPU_DISABLE", "1");
        let decision = sequence_gpu_preference(10_000, SequenceIntent::Linspace, false);
        assert!(!decision.prefer_gpu);
        assert_eq!(decision.reason, ResidencyReason::DisabledByEnv);
        reset_env();
    }

    #[test]
    fn env_min_len_controls_threshold() {
        let _guard = ENV_LOCK.lock().unwrap();
        reset_env();
        simple_provider::register_inprocess_provider();
        std::env::set_var("RUNMAT_SEQUENCE_GPU_MIN", "8192");
        let decision = sequence_gpu_preference(10_000, SequenceIntent::Linspace, false);
        assert!(decision.prefer_gpu);
        assert_eq!(decision.reason, ResidencyReason::ThresholdHint);
        reset_env();
    }
}
