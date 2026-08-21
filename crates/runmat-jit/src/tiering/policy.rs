use runmat_execution::Digest;

use super::{CompilationMode, TieringConfig};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TierAvailability {
    pub generic_ready: bool,
    pub specialized_profiles: Vec<Digest>,
    pub pending_compilations: usize,
    pub retained_versions: usize,
    pub retained_code_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TierDecision {
    Interpret,
    ExecuteGeneric,
    ExecuteSpecialized {
        profile: Digest,
    },
    CompileGeneric {
        mode: CompilationMode,
    },
    CompileSpecialized {
        profile: Digest,
        mode: CompilationMode,
    },
}

pub(super) fn decide(
    config: TieringConfig,
    invocations: u64,
    backedges: u64,
    dominant_profile: Option<(Digest, u64, u64)>,
    availability: &TierAvailability,
) -> TierDecision {
    let specialized_ready = dominant_profile
        .as_ref()
        .and_then(|(profile, _, failures)| {
            if *failures >= 2 {
                return None;
            }
            availability
                .specialized_profiles
                .iter()
                .find(|candidate| *candidate == profile)
                .copied()
        });
    if let Some(profile) = specialized_ready {
        return TierDecision::ExecuteSpecialized { profile };
    }
    let within_pending_budget = availability.pending_compilations < config.max_pending_compilations;
    let within_compile_budget = within_pending_budget
        && availability.retained_versions < config.max_versions_per_entry
        && availability.retained_code_bytes < config.max_code_bytes;
    if availability.generic_ready {
        let specialization_hot = dominant_profile.is_some_and(|(_, samples, failures)| {
            samples >= config.specialized_hot_threshold && failures < 2
        });
        if specialization_hot && within_pending_budget {
            let profile = dominant_profile
                .expect("hot specialization has a dominant profile")
                .0;
            return TierDecision::CompileSpecialized {
                profile,
                mode: config.compilation_mode(),
            };
        }
        return TierDecision::ExecuteGeneric;
    }
    if (invocations >= config.generic_hot_threshold || backedges >= config.loop_hot_threshold)
        && within_compile_budget
    {
        TierDecision::CompileGeneric {
            mode: config.compilation_mode(),
        }
    } else {
        TierDecision::Interpret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> Digest {
        Digest::sha256([seed])
    }

    #[test]
    fn failing_specialization_is_quarantined_even_when_code_is_retained() {
        let profile = digest(1);
        let availability = TierAvailability {
            generic_ready: true,
            specialized_profiles: vec![profile],
            ..TierAvailability::default()
        };
        assert_eq!(
            decide(
                TieringConfig::default(),
                100,
                0,
                Some((profile, 100, 2)),
                &availability
            ),
            TierDecision::ExecuteGeneric
        );
    }

    #[test]
    fn every_compile_budget_is_a_hard_boundary() {
        let config = TieringConfig {
            generic_hot_threshold: 1,
            ..TieringConfig::default()
        };
        for availability in [
            TierAvailability {
                pending_compilations: config.max_pending_compilations,
                ..TierAvailability::default()
            },
            TierAvailability {
                retained_versions: config.max_versions_per_entry,
                ..TierAvailability::default()
            },
            TierAvailability {
                retained_code_bytes: config.max_code_bytes,
                ..TierAvailability::default()
            },
        ] {
            assert_eq!(
                decide(config, 1, 0, None, &availability),
                TierDecision::Interpret
            );
        }
    }

    #[test]
    fn hot_new_profile_can_replace_an_old_specialization_at_the_version_cap() {
        let profile = digest(2);
        let config = TieringConfig {
            specialized_hot_threshold: 2,
            max_versions_per_entry: 2,
            ..TieringConfig::default()
        };
        let availability = TierAvailability {
            generic_ready: true,
            retained_versions: 2,
            retained_code_bytes: config.max_code_bytes,
            ..TierAvailability::default()
        };
        assert_eq!(
            decide(config, 10, 0, Some((profile, 2, 0)), &availability),
            TierDecision::CompileSpecialized {
                profile,
                mode: CompilationMode::Background,
            }
        );
    }
}
