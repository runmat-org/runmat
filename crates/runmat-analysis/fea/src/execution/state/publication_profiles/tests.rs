use super::*;
use runmat_analysis_core::{ProcedureStep, StateId};

fn request(id: &str) -> SemanticStatePublicationRequest {
    let step = ProcedureStep::structural_static("profiles");
    let sequence = SequenceIdentity::new(Digest::sha256(b"sequence")).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &step).unwrap();
    let mut declaration = step.state_outputs[0].clone();
    declaration.state_id = StateId(id.into());
    SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::StructuralPrimaryFields,
        producer,
        declaration,
        Digest::sha256(b"topology"),
        SolverResultIdentity::from_step_result(producer, Digest::sha256(b"result")).unwrap(),
    )
    .unwrap()
}

fn entry(id: &str) -> StatePublicationProfileEntry {
    StatePublicationProfileEntry::new(
        request(id),
        SerializedCanonicalProfile {
            encoded_bytes_upper_bound: 1024,
            materialization_resident_bytes_estimate: 100_000,
        },
    )
    .unwrap()
}

fn catalog(
    entries: Vec<StatePublicationProfileEntry>,
) -> AnalysisExecutionResult<StatePublicationProfiles> {
    let request = request("source");
    StatePublicationProfiles::new(
        request.producer,
        request.source_result,
        request.topology_digest,
        Digest::sha256(b"payload"),
        entries,
    )
}

#[test]
fn roundtrip_empty_and_sorted_catalogs_and_exact_lookup() {
    for entries in [vec![], vec![entry("z"), entry("a")]] {
        let value = catalog(entries).unwrap();
        let encoded = value.canonical_bytes().unwrap();
        assert_eq!(
            value,
            StatePublicationProfiles::from_canonical_bytes(&encoded).unwrap()
        );
        assert_eq!(
            value,
            serde_json::from_value::<StatePublicationProfiles>(
                serde_json::to_value(&value).unwrap()
            )
            .unwrap()
        );
        value
            .validate_source(
                value.producer,
                value.source_result,
                value.topology_digest,
                value.source_payload_digest,
            )
            .unwrap();
        assert!(value
            .validate_source(
                value.producer,
                value.source_result,
                value.topology_digest,
                Digest::sha256(b"other")
            )
            .is_err());
    }
    let a = catalog(vec![entry("z"), entry("a")]).unwrap();
    let b = catalog(vec![entry("a"), entry("z")]).unwrap();
    assert_eq!(a.identity(), b.identity());
    assert_eq!(a.entries()[0].request.declaration.state_id.0, "a");
    assert_eq!(
        a.entry(&request("a")).unwrap().profile(),
        entry("a").profile()
    );
    assert!(a.entry(&request("absent")).is_err());
    let mut changed = request("a");
    changed.topology_digest = Digest::sha256(b"other");
    assert!(a.entry(&changed).is_err());
    changed = request("a");
    changed.declaration.persistence = StatePersistence::Canonical;
    assert_ne!(
        changed.declaration.persistence,
        request("a").declaration.persistence
    );
    assert!(a.entry(&changed).is_err());
}

#[test]
fn all_source_declaration_and_profile_substitutions_change_identity() {
    let original = catalog(vec![entry("a")]).unwrap();
    for field in [
        "producer",
        "source_result",
        "topology_digest",
        "source_payload_digest",
        "identity",
    ] {
        let mut wire = serde_json::to_value(&original).unwrap();
        wire[field] = serde_json::to_value(Digest::sha256(b"substituted")).unwrap();
        assert!(
            serde_json::from_value::<StatePublicationProfiles>(wire).is_err(),
            "{field}"
        );
    }
    for field in [
        "encoded_bytes_upper_bound",
        "materialization_resident_bytes_estimate",
    ] {
        let mut wire = serde_json::to_value(&original).unwrap();
        wire["entries"][0][field] = serde_json::json!(123456);
        assert!(serde_json::from_value::<StatePublicationProfiles>(wire).is_err());
    }
    let mut wire = serde_json::to_value(&original).unwrap();
    wire["entries"][0]["request"]["declaration"]["state_id"] = serde_json::json!("different");
    assert!(serde_json::from_value::<StatePublicationProfiles>(wire).is_err());
    let mut changed = entry("a");
    changed.encoded_bytes_upper_bound += 1;
    assert_ne!(
        original.identity(),
        catalog(vec![changed]).unwrap().identity()
    );
    assert_ne!(
        original.identity(),
        catalog(vec![entry("b")]).unwrap().identity()
    );
    let changed_payload = StatePublicationProfiles::new(
        original.producer,
        original.source_result,
        original.topology_digest,
        Digest::sha256(b"other"),
        original.entries.clone(),
    )
    .unwrap();
    assert_ne!(original.identity(), changed_payload.identity());
}

#[test]
fn malformed_catalogs_and_wire_fail_closed() {
    assert!(catalog(vec![entry("a"), entry("a")]).is_err());
    assert!(catalog(vec![entry("a"); STATE_PUBLICATION_PROFILES_MAX_ENTRIES + 1]).is_err());
    for mutate in [
        |value: &mut StatePublicationProfileEntry| value.encoded_bytes_upper_bound = 0,
        |value: &mut StatePublicationProfileEntry| {
            value.encoded_bytes_upper_bound = STATE_LIMITS.maximum_encoded_bytes as u64 + 1
        },
        |value: &mut StatePublicationProfileEntry| {
            value.materialization_resident_bytes_estimate = 0
        },
        |value: &mut StatePublicationProfileEntry| {
            value.materialization_resident_bytes_estimate = u64::MAX
        },
        |value: &mut StatePublicationProfileEntry| value.request.schema_version = 2,
        |value: &mut StatePublicationProfileEntry| {
            value.request.topology_digest = Digest::sha256(b"other")
        },
        |value: &mut StatePublicationProfileEntry| {
            value.request.declaration.state_id.0 = "x".repeat(STATE_LIMITS.maximum_string_bytes + 1)
        },
    ] {
        let mut value = entry("a");
        mutate(&mut value);
        assert!(catalog(vec![value]).is_err());
    }
    let value = catalog(vec![entry("a"), entry("b")]).unwrap();
    for mutate in [
        |wire: &mut serde_json::Value| wire["schema_version"] = serde_json::json!(2),
        |wire: &mut serde_json::Value| wire["unknown"] = serde_json::json!(true),
        |wire: &mut serde_json::Value| wire["entries"][1] = wire["entries"][0].clone(),
        |wire: &mut serde_json::Value| wire["entries"].as_array_mut().unwrap().reverse(),
        |wire: &mut serde_json::Value| {
            wire["entries"][0]["encoded_bytes_upper_bound"] = serde_json::json!(0)
        },
    ] {
        let mut wire = serde_json::to_value(&value).unwrap();
        mutate(&mut wire);
        assert!(serde_json::from_value::<StatePublicationProfiles>(wire.clone()).is_err());
        let encoded = encode_contract(PREFIX, DOMAIN, &wire, LIMITS).unwrap();
        assert!(StatePublicationProfiles::from_canonical_bytes(&encoded).is_err());
    }
}

#[test]
fn presolve_projection_matches_schema_and_bounds_actual_encoding() {
    for count in [0, 1, STATE_PUBLICATION_PROFILES_MAX_ENTRIES] {
        for length in [1, 24, 256] {
            let entries: Vec<_> = (0..count)
                .map(|index| entry(&format!("{}-{index}", "x".repeat(length))))
                .collect();
            let requests: Vec<_> = entries.iter().map(|entry| entry.request.clone()).collect();
            let projection =
                StatePublicationProfiles::encoding_profile_for_requests(&requests).unwrap();
            let value = catalog(entries).unwrap();
            assert_eq!(
                projection,
                profile_serialized_shape(&value, LIMITS).unwrap()
            );
            assert!(
                projection.encoded_bytes_upper_bound
                    >= value.canonical_bytes().unwrap().len() as u64
            );
        }
    }
    assert!(
        StatePublicationProfiles::encoding_profile_for_requests(&[request("a"), request("a")])
            .is_err()
    );
    assert!(
        StatePublicationProfiles::encoding_profile_for_requests(&vec![
            request("a");
            STATE_PUBLICATION_PROFILES_MAX_ENTRIES
                + 1
        ])
        .is_err()
    );
    assert!(
        StatePublicationProfiles::encoding_profile_for_requests(&[request(
            &"x".repeat(STATE_LIMITS.maximum_string_bytes + 1)
        )])
        .is_err()
    );
}
