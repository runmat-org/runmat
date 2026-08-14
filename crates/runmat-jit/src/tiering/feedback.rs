use std::cell::RefCell;
use std::collections::BTreeMap;

use runmat_execution::Digest;
use runmat_types::{ProgramFunctionId, ProgramPointId, ValueFact};
use serde::{Deserialize, Serialize};

use super::policy::decide;
use super::{TierAvailability, TierDecision, TieringConfig};

const FEEDBACK_SCHEMA_VERSION: u16 = 1;
const MAX_OBSERVATIONS: u64 = (1 << 53) - 1;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierSiteId {
    pub entry: String,
    pub function: ProgramFunctionId,
    pub loop_header: Option<ProgramPointId>,
}

impl TierSiteId {
    fn validate(&self) -> Result<(), &'static str> {
        if self.entry.is_empty()
            || self.entry.len() > 512
            || self.entry.chars().any(char::is_control)
            || self
                .loop_header
                .is_some_and(|point| point.function != self.function)
        {
            return Err("tier site identity is invalid");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepresentationProfile {
    pub digest: Digest,
    pub facts: Vec<ValueFact>,
}

impl RepresentationProfile {
    pub fn from_facts(facts: Vec<ValueFact>, max_bytes: usize) -> Result<Self, &'static str> {
        if facts.len() > 64 {
            return Err("tier representation profile has too many values");
        }
        let encoded = serde_json::to_vec(&facts)
            .map_err(|_| "tier representation profile could not be encoded")?;
        if encoded.len() > max_bytes {
            return Err("tier representation profile exceeds its byte bound");
        }
        Ok(Self {
            digest: Digest::sha256(encoded),
            facts,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierFeedbackSnapshot {
    pub schema_version: u16,
    pub sites: Vec<TierSiteSnapshot>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierSiteSnapshot {
    pub site: TierSiteId,
    #[serde(with = "decimal_u64")]
    pub invocations: u64,
    #[serde(with = "decimal_u64")]
    pub backedges: u64,
    #[serde(with = "decimal_u64")]
    pub total_elapsed_ns: u64,
    #[serde(with = "decimal_u64")]
    pub latest_tick: u64,
    pub profiles: Vec<TierProfileSnapshot>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierProfileSnapshot {
    pub digest: Digest,
    #[serde(with = "decimal_u64")]
    pub samples: u64,
    #[serde(with = "decimal_u64")]
    pub failures: u64,
    #[serde(with = "decimal_u64")]
    pub latest_tick: u64,
}

#[derive(Default)]
struct TieringState {
    tick: u64,
    sites: BTreeMap<TierSiteId, SiteFeedback>,
}

#[derive(Default)]
struct SiteFeedback {
    invocations: u64,
    backedges: u64,
    total_elapsed_ns: u64,
    latest_tick: u64,
    profiles: BTreeMap<Digest, ProfileFeedback>,
}

#[derive(Default)]
struct ProfileFeedback {
    samples: u64,
    failures: u64,
    latest_tick: u64,
}

pub struct TieringSession {
    config: TieringConfig,
    state: RefCell<TieringState>,
}

impl Default for TieringSession {
    fn default() -> Self {
        Self::new(TieringConfig::default()).expect("default tiering configuration must be valid")
    }
}

impl TieringSession {
    pub fn new(config: TieringConfig) -> Result<Self, &'static str> {
        Ok(Self {
            config: config.validate()?,
            state: RefCell::new(TieringState::default()),
        })
    }

    pub fn observe_invocation(
        &self,
        site: TierSiteId,
        profile: &RepresentationProfile,
        elapsed_ns: u64,
        succeeded: bool,
    ) -> Result<(), &'static str> {
        site.validate()?;
        let encoded = serde_json::to_vec(&profile.facts)
            .map_err(|_| "tier representation profile could not be encoded")?;
        if encoded.len() > self.config.max_profile_bytes {
            return Err("tier representation profile exceeds its byte bound");
        }
        if Digest::sha256(encoded) != profile.digest {
            return Err("tier representation profile digest does not match its facts");
        }
        let mut state = self.state.borrow_mut();
        let tick = next_tick(&mut state)?;
        let feedback = state.sites.entry(site.clone()).or_default();
        feedback.invocations = bounded_count_add(feedback.invocations, 1);
        feedback.total_elapsed_ns = feedback.total_elapsed_ns.saturating_add(elapsed_ns);
        feedback.latest_tick = tick;
        let profile_feedback = feedback.profiles.entry(profile.digest).or_default();
        if succeeded {
            profile_feedback.samples = bounded_count_add(profile_feedback.samples, 1);
        } else {
            profile_feedback.failures = bounded_count_add(profile_feedback.failures, 1);
        }
        profile_feedback.latest_tick = tick;
        evict_profiles(feedback, self.config.max_profiles_per_site);
        evict_sites(&mut state, self.config.max_sites, Some(&site));
        Ok(())
    }

    pub fn observe_backedge(&self, site: TierSiteId, count: u64) -> Result<(), &'static str> {
        site.validate()?;
        let mut state = self.state.borrow_mut();
        let tick = next_tick(&mut state)?;
        let feedback = state.sites.entry(site.clone()).or_default();
        feedback.backedges = bounded_count_add(feedback.backedges, count);
        feedback.latest_tick = tick;
        evict_sites(&mut state, self.config.max_sites, Some(&site));
        Ok(())
    }

    pub fn decide(&self, site: &TierSiteId, availability: &TierAvailability) -> TierDecision {
        let state = self.state.borrow();
        let Some(feedback) = state.sites.get(site) else {
            return TierDecision::Interpret;
        };
        decide(
            self.config,
            feedback.invocations,
            feedback.backedges,
            dominant_profile(feedback),
            availability,
        )
    }

    pub fn snapshot(&self) -> TierFeedbackSnapshot {
        let sites = self
            .state
            .borrow()
            .sites
            .iter()
            .map(|(site, feedback)| TierSiteSnapshot {
                site: site.clone(),
                invocations: feedback.invocations,
                backedges: feedback.backedges,
                total_elapsed_ns: feedback.total_elapsed_ns,
                latest_tick: feedback.latest_tick,
                profiles: feedback
                    .profiles
                    .iter()
                    .map(|(digest, profile)| TierProfileSnapshot {
                        digest: *digest,
                        samples: profile.samples,
                        failures: profile.failures,
                        latest_tick: profile.latest_tick,
                    })
                    .collect(),
            })
            .collect();
        TierFeedbackSnapshot {
            schema_version: FEEDBACK_SCHEMA_VERSION,
            sites,
        }
    }
}

fn dominant_profile(feedback: &SiteFeedback) -> Option<(Digest, u64, u64)> {
    feedback
        .profiles
        .iter()
        .max_by(|left, right| {
            left.1
                .samples
                .cmp(&right.1.samples)
                .then_with(|| right.0.cmp(left.0))
        })
        .map(|(digest, profile)| (*digest, profile.samples, profile.failures))
}

fn next_tick(state: &mut TieringState) -> Result<u64, &'static str> {
    state.tick = state
        .tick
        .checked_add(1)
        .ok_or("tier feedback tick exhausted")?;
    Ok(state.tick)
}

fn bounded_count_add(value: u64, add: u64) -> u64 {
    value.saturating_add(add).min(MAX_OBSERVATIONS)
}

fn evict_profiles(feedback: &mut SiteFeedback, limit: usize) {
    while feedback.profiles.len() > limit {
        let key = feedback
            .profiles
            .iter()
            .min_by_key(|(digest, profile)| (profile.latest_tick, *digest))
            .map(|(digest, _)| *digest)
            .expect("non-empty profile map exceeds its bound");
        feedback.profiles.remove(&key);
    }
}

fn evict_sites(state: &mut TieringState, limit: usize, protected: Option<&TierSiteId>) {
    while state.sites.len() > limit {
        let key = state
            .sites
            .iter()
            .filter(|(site, _)| protected != Some(*site))
            .min_by_key(|(site, feedback)| (feedback.latest_tick, *site))
            .map(|(site, _)| site.clone());
        let Some(key) = key else {
            break;
        };
        state.sites.remove(&key);
    }
}

mod decimal_u64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use runmat_types::{DynamicReason, ProgramFunctionId, ProgramPointId, ShapeFact, ValueFact};

    use super::*;
    use crate::tiering::{CompilationMode, TierAvailability, TierDecision};

    fn site(name: &str, loop_header: bool) -> TierSiteId {
        let function = ProgramFunctionId(3);
        TierSiteId {
            entry: name.into(),
            function,
            loop_header: loop_header.then_some(ProgramPointId {
                function,
                block: 2,
                position: 0,
            }),
        }
    }

    fn profile(seed: u8) -> RepresentationProfile {
        let mut fact = ValueFact::unknown(DynamicReason::Unspecified);
        fact.shape = ShapeFact::Scalar;
        RepresentationProfile::from_facts(vec![fact; usize::from(seed.max(1))], 64 * 1024).unwrap()
    }

    #[test]
    fn cold_generic_specialized_progression_is_monotonic_and_deterministic() {
        let session = TieringSession::new(TieringConfig {
            generic_hot_threshold: 2,
            specialized_hot_threshold: 3,
            loop_hot_threshold: 4,
            deterministic: true,
            ..TieringConfig::default()
        })
        .unwrap();
        let site = site("entry", false);
        let profile = profile(1);
        let mut availability = TierAvailability::default();
        assert_eq!(
            session.decide(&site, &availability),
            TierDecision::Interpret
        );
        session
            .observe_invocation(site.clone(), &profile, 100, true)
            .unwrap();
        session
            .observe_invocation(site.clone(), &profile, 90, true)
            .unwrap();
        assert_eq!(
            session.decide(&site, &availability),
            TierDecision::CompileGeneric {
                mode: CompilationMode::DeterministicSynchronous
            }
        );
        availability.generic_ready = true;
        assert_eq!(
            session.decide(&site, &availability),
            TierDecision::ExecuteGeneric
        );
        session
            .observe_invocation(site.clone(), &profile, 80, true)
            .unwrap();
        assert_eq!(
            session.decide(&site, &availability),
            TierDecision::CompileSpecialized {
                profile: profile.digest,
                mode: CompilationMode::DeterministicSynchronous
            }
        );
        availability.specialized_profiles.push(profile.digest);
        assert_eq!(
            session.decide(&site, &availability),
            TierDecision::ExecuteSpecialized {
                profile: profile.digest
            }
        );
    }

    #[test]
    fn loop_heat_can_request_generic_compilation_without_function_replay() {
        let session = TieringSession::new(TieringConfig {
            generic_hot_threshold: 100,
            specialized_hot_threshold: 100,
            loop_hot_threshold: 3,
            ..TieringConfig::default()
        })
        .unwrap();
        let site = site("loop", true);
        session.observe_backedge(site.clone(), 3).unwrap();
        assert_eq!(
            session.decide(&site, &TierAvailability::default()),
            TierDecision::CompileGeneric {
                mode: CompilationMode::Background
            }
        );
    }

    #[test]
    fn feedback_is_bounded_failure_aware_and_javascript_safe() {
        let session = TieringSession::new(TieringConfig {
            generic_hot_threshold: 1,
            specialized_hot_threshold: 2,
            max_sites: 2,
            max_profiles_per_site: 2,
            ..TieringConfig::default()
        })
        .unwrap();
        let target = site("target", false);
        for profile in [profile(1), profile(2), profile(3)] {
            session
                .observe_invocation(target.clone(), &profile, u64::MAX, true)
                .unwrap();
        }
        let failing = profile(3);
        session
            .observe_invocation(target.clone(), &failing, 1, false)
            .unwrap();
        session
            .observe_invocation(target.clone(), &failing, 1, false)
            .unwrap();
        session
            .observe_invocation(site("other", false), &profile(1), 1, true)
            .unwrap();
        session
            .observe_invocation(site("newest", false), &profile(1), 1, true)
            .unwrap();

        let snapshot = session.snapshot();
        assert_eq!(snapshot.sites.len(), 2);
        assert!(snapshot.sites.iter().all(|site| site.profiles.len() <= 2));
        let encoded = serde_json::to_string(&snapshot).unwrap();
        assert!(encoded.contains("\"total_elapsed_ns\":\""));
    }

    #[test]
    fn invalid_profile_digest_is_rejected_before_feedback_changes() {
        let session = TieringSession::default();
        let mut invalid = profile(1);
        invalid.digest = profile(2).digest;
        assert_eq!(
            session.observe_invocation(site("entry", false), &invalid, 1, true),
            Err("tier representation profile digest does not match its facts")
        );
        assert!(session.snapshot().sites.is_empty());
    }
}
