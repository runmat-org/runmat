use super::*;
use crate::execution::{SequenceIdentity, SolverResultIdentity, StepIdentity};
use crate::results::field::RecoveredFieldSample;
use crate::results::observation::{
    ObservationEventAttribution, ObservationResultIdentity, SolverEventAttribution,
    SolverEventSource,
};
use runmat_analysis_core::SolverObservationEvent;

fn entries() -> Vec<ObservationHistoryEntry> {
    (0..3)
        .map(|ordinal| {
            let coordinate = RecoveredFieldSample::TimeSeconds(f64::from(ordinal));
            let result = ObservationResultIdentity(Digest::sha256([ordinal as u8]));
            ObservationHistoryEntry {
                sample: super::super::contract::derive_sample_identity(coordinate, None, result)
                    .unwrap(),
                coordinate,
                event_attribution: None,
                result,
            }
        })
        .collect()
}

fn stream(evicted: bool) -> ObservationHistoryStream {
    let mut stream = ObservationHistoryStream {
        schema_version: 1,
        identity: ObservationHistoryStreamIdentity(Digest::sha256([])),
        plan_identity: ResolvedObservationPlanIdentity(Digest::sha256("plan")),
        topology_digest: Digest::sha256("topology"),
        independent_domain: IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 2.0,
            interval_count: 2,
        },
        observation_id: ObservationId("temperature".into()),
        quantity: ObservationQuantity::Temperature,
        location: FieldLocation::Node,
        value_shape: ObservationValueShape::Scalar,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::TimeSeconds(vec![0.0, 1.0, 2.0]),
        retention: if evicted {
            RetentionPolicy::LatestSamples { count: 2 }
        } else {
            RetentionPolicy::Canonical
        },
        partial_access: PartialAccessPolicy::ManifestAndChunks,
        maximum_produced_samples: 3,
        maximum_retained_samples: if evicted { 2 } else { 3 },
    };
    stream.identity = derive_stream_identity(&stream).unwrap();
    stream.validate().unwrap();
    stream
}

fn fixture(evicted: bool) -> ObservationHistoryCheckpointManifest {
    let stream = stream(evicted);
    let samples = entries();
    let initial = initial_observed_prefix_digest(stream.identity).unwrap();
    let mut prefix = initial;
    let mut retained_prefix = initial;
    let mut predecessor = None;
    let mut terminal_predecessor = None;
    for (ordinal, entry) in samples.iter().enumerate() {
        prefix = extend_observed_prefix_digest(prefix, entry.sample).unwrap();
        terminal_predecessor = predecessor;
        predecessor = Some(
            ObservationHistoryCheckpointIdentity::from_prefixes(
                stream.identity,
                predecessor,
                ordinal as u64 + 1,
                ordinal as u64 + 1,
                initial,
                prefix,
            )
            .unwrap(),
        );
        if evicted && ordinal == 0 {
            retained_prefix = prefix;
        }
    }
    let mut manifest = ObservationHistoryCheckpointManifest {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        identity: ObservationHistoryCheckpointIdentity(Digest::sha256([])),
        stream,
        predecessor: terminal_predecessor,
        observed_sample_count: 3,
        retained_prefix_digest: retained_prefix,
        observed_prefix_digest: prefix,
        samples: samples.into_iter().skip(usize::from(evicted)).collect(),
    };
    manifest.identity = derive_checkpoint_identity(&manifest).unwrap();
    manifest.validate().unwrap();
    manifest
}

fn rehash_entry(entry: &mut ObservationHistoryEntry) {
    entry.sample = super::super::contract::derive_sample_identity(
        entry.coordinate,
        entry.event_attribution,
        entry.result,
    )
    .unwrap();
}

fn rejects_with_fresh_header(mut manifest: ObservationHistoryCheckpointManifest) {
    manifest.identity = derive_checkpoint_identity(&manifest).unwrap();
    assert!(manifest.validate().is_err());
}

#[test]
fn checkpoint_prefix_roundtrip_preserves_retained_and_evicted_commitments() {
    for evicted in [false, true] {
        let manifest = fixture(evicted);
        let bytes = manifest.canonical_bytes().unwrap();
        assert_eq!(bytes, manifest.canonical_bytes().unwrap());
        assert_eq!(
            ObservationHistoryCheckpointManifest::from_canonical_bytes(&bytes).unwrap(),
            manifest
        );
        let mut legacy = manifest.clone();
        legacy.schema_version = 1;
        assert!(legacy.validate().is_err());
        let old_envelope = encode_contract(
            OBSERVATION_PREFIX,
            "observation-history-checkpoint-manifest",
            &manifest,
            OBSERVATION_LIMITS,
        )
        .unwrap();
        assert!(ObservationHistoryCheckpointManifest::from_canonical_bytes(&old_envelope).is_err());
    }
}

#[test]
fn checkpoint_prefix_commitment_rejects_rehashed_entry_substitution_and_order() {
    for evicted in [false, true] {
        let manifest = fixture(evicted);
        let mut changed = manifest.clone();
        changed.samples[0].result = ObservationResultIdentity(Digest::sha256("other result"));
        rehash_entry(&mut changed.samples[0]);
        rejects_with_fresh_header(changed);

        let mut changed = manifest.clone();
        changed.samples.swap(0, 1);
        rejects_with_fresh_header(changed);

        let mut changed = manifest;
        changed.samples[0].coordinate = RecoveredFieldSample::TimeSeconds(0.5);
        rehash_entry(&mut changed.samples[0]);
        rejects_with_fresh_header(changed);
    }
}

#[test]
fn checkpoint_prefix_commitment_binds_event_attribution() {
    let mut manifest = fixture(false);
    let sequence = SequenceIdentity::new(Digest::sha256("sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("step")).unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("solver")).unwrap();
    manifest.samples[0].event_attribution = Some(ObservationEventAttribution::Solver(
        SolverEventAttribution {
            event: SolverObservationEvent::FirstYield,
            sample: manifest.samples[0].coordinate,
            occurrence_index: 0,
            source: SolverEventSource::CompletedResult { result },
        },
    ));
    rehash_entry(&mut manifest.samples[0]);
    rejects_with_fresh_header(manifest);
}

#[test]
fn checkpoint_prefix_rejects_changed_retained_evicted_counts_and_prefixes() {
    for evicted in [false, true] {
        let manifest = fixture(evicted);
        let mut changed = manifest.clone();
        changed.samples.remove(0);
        rejects_with_fresh_header(changed);
        let mut changed = manifest.clone();
        changed.observed_sample_count += 1;
        rejects_with_fresh_header(changed);
        let mut changed = manifest.clone();
        changed.retained_prefix_digest = Digest::sha256("other retained prefix");
        rejects_with_fresh_header(changed);
        let mut changed = manifest.clone();
        changed.observed_prefix_digest = Digest::sha256("other observed prefix");
        rejects_with_fresh_header(changed);
        let mut changed = manifest;
        changed.predecessor = Some(changed.identity);
        assert!(changed.validate().is_err());
    }
}

#[test]
fn checkpoint_prefix_identities_use_only_constant_size_headers() {
    let stream = stream(false);
    let initial = initial_observed_prefix_digest(stream.identity).unwrap();
    let mut prefix = initial;
    let mut predecessor = None;
    let samples = entries();
    for (ordinal, sample) in samples.iter().enumerate() {
        prefix = extend_observed_prefix_digest(prefix, sample.sample).unwrap();
        let count = ordinal as u64 + 1;
        let identity = ObservationHistoryCheckpointIdentity::from_prefixes(
            stream.identity,
            predecessor,
            count,
            count,
            initial,
            prefix,
        )
        .unwrap();
        let manifest = ObservationHistoryCheckpointManifest {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            identity,
            stream: stream.clone(),
            predecessor,
            observed_sample_count: count,
            retained_prefix_digest: initial,
            observed_prefix_digest: prefix,
            samples: samples[..=ordinal].to_vec(),
        };
        manifest.validate().unwrap();
        assert_eq!(identity, derive_checkpoint_identity(&manifest).unwrap());
        predecessor = Some(identity);
    }
    assert!(ObservationHistoryCheckpointIdentity::from_prefixes(
        stream.identity,
        None,
        0,
        0,
        initial,
        initial,
    )
    .is_err());
}

#[test]
fn checkpoint_prefix_iterator_checks_terminal_link_without_prefix_copies() {
    let manifest = fixture(false);
    let mut last = None;
    for (ordinal, item) in manifest.canonical_prefix_identities().unwrap().enumerate() {
        let (predecessor, sample_ordinal, identity) = item.unwrap();
        assert_eq!(sample_ordinal, ordinal as u64);
        assert_eq!(predecessor, last);
        last = Some(identity);
    }
    assert_eq!(last, Some(manifest.identity));
    let mut substituted = manifest;
    substituted.predecessor = Some(ObservationHistoryCheckpointIdentity(Digest::sha256(
        "foreign",
    )));
    substituted.identity = derive_checkpoint_identity(&substituted).unwrap();
    // General checkpoints allow a supplied predecessor; canonical one-sample lineage does not.
    substituted.validate().unwrap();
    assert!(substituted.canonical_prefix_identities().is_err());
    assert!(fixture(true).canonical_prefix_identities().is_err());
}
