use super::*;
use runmat_execution::value::{artifact_object_value_id, ValueRef, ValueRefKind};

fn source() -> ValueRef {
    let digest = Digest::sha256(b"original");
    let context = Digest::sha256(b"source-phase-test");
    ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: artifact_object_value_id(digest, context, "source-phase-test"),
        logical_digest: digest,
        encoded_length: 8,
        media_type: "application/vnd.runmat.test".into(),
        value_schema: "source-phase-test/v1".into(),
        encryption_context: context,
        kind: ValueRefKind::ResultObject,
        authorization_scope: "source-phase-test".into(),
        resident_fence: None,
    }
}

#[test]
fn profiled_source_inventory_is_bound_and_profile_reads_hold_the_captured_allowance() {
    let maximum = 32 << 20;
    let mut builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"profiled-source"), maximum).unwrap();
    builder.admit_preparation(&vec![1.25_f64; 512]).unwrap();
    let preparation = builder.preparation_resident_bytes;
    let mut original = source();
    original.encoded_length = 1 << 20;
    assert!(builder
        .admit_source_objects(std::slice::from_ref(&original))
        .is_err());
    let available = builder.source_workspace_limit().unwrap();
    {
        let phase = builder.begin_source_profile(available).unwrap();
        assert_eq!(builder.active_source_workspace().unwrap(), available);
        assert!(builder.admit(1).is_err());
        assert!(builder
            .admit_profiled_source_objects(std::slice::from_ref(&original), 4 << 20)
            .is_err());
        drop(phase);
    }
    assert_eq!(builder.preparation_resident_bytes, preparation);
    assert_eq!(builder.active_source_workspace().unwrap(), 0);
    assert!(builder
        .admit_profiled_source_objects(std::slice::from_ref(&original), original.encoded_length - 1)
        .is_err());
    builder
        .admit_profiled_source_objects(std::slice::from_ref(&original), 4 << 20)
        .unwrap();
    let phase = builder.begin_source_phase().unwrap();
    assert_eq!(
        builder
            .source_replay_allowance(std::slice::from_ref(&original))
            .unwrap(),
        4 << 20
    );
    for changed in 0..4 {
        let mut substituted = original.clone();
        match changed {
            0 => substituted.logical_digest = Digest::sha256(b"foreign-source"),
            1 => substituted.encoded_length += 1,
            2 => substituted.authorization_scope.push_str("foreign"),
            _ => substituted.encoded_length = 8,
        }
        assert!(builder.require_source_objects(&[substituted]).is_err());
    }
    drop(phase);
    assert_eq!(builder.preparation_resident_bytes, preparation);
}

#[test]
fn replay_peak_is_ephemeral_while_input_and_overlapping_output_remain_charged() {
    let definition = Digest::sha256(b"source-phase-lifetime");
    let mut builder = AnalysisPublicationBuilder::new(definition, 2 << 20).unwrap();
    builder.admit_preparation(&vec![1.25_f64; 512]).unwrap();
    let input = builder.preparation_resident_bytes;
    let source = source();
    builder
        .admit_source_objects(std::slice::from_ref(&source))
        .unwrap();
    let peak = builder.source_workspace_bytes;
    assert_eq!(builder.preparation_resident_bytes, input);
    builder.maximum_bytes = input + peak;
    builder.admit(peak).unwrap();
    assert!(builder
        .require_source_objects(std::slice::from_ref(&source))
        .is_err());
    let phase = builder.begin_source_phase().unwrap();
    builder
        .require_source_objects(std::slice::from_ref(&source))
        .unwrap();
    assert!(matches!(
        builder.admit(1),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert!(builder.begin_source_phase().is_err());
    assert!(builder
        .admit_source_objects(std::slice::from_ref(&source))
        .is_err());
    assert_eq!(builder.active_source_workspace().unwrap(), peak);
    drop(phase);
    assert_eq!(builder.active_source_workspace().unwrap(), 0);
    builder.admit(peak).unwrap();
    assert_eq!(builder.preparation_resident_bytes, input);
    assert!(matches!(
        builder.admit(peak + 1),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
}

#[test]
fn failed_admission_and_unwind_release_the_phase_but_active_work_blocks_finalization() {
    let definition = Digest::sha256(b"source-phase-failure");
    let mut builder = AnalysisPublicationBuilder::new(definition, 2 << 20).unwrap();
    builder
        .admit_preparation(&"retained original input")
        .unwrap();
    builder.admit_source_objects(&[source()]).unwrap();
    let required = builder.preparation_resident_bytes + builder.source_workspace_bytes;
    builder.maximum_bytes = required - 1;
    assert!(matches!(
        builder.begin_source_phase(),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(builder.active_source_workspace().unwrap(), 0);
    builder.maximum_bytes = required;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _phase = builder.begin_source_phase().unwrap();
        panic!("source operation stopped");
    }));
    assert!(unwind.is_err());
    assert_eq!(builder.active_source_workspace().unwrap(), 0);
    let phase = builder.begin_source_phase().unwrap();
    let result = builder.finish(serde_json::json!({"study_definition_digest": definition}));
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("source phase is still active"));
    drop(phase);
}

#[test]
fn sequential_catalogs_reuse_one_peak_and_late_sources_recheck_retained_input() {
    let mut builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"sequential-sources"), 2 << 20).unwrap();
    let small = source();
    let mut larger = small.clone();
    larger.encoded_length = 512;
    builder.admit_source_objects(&[small]).unwrap();
    builder
        .admit_source_objects(std::slice::from_ref(&larger))
        .unwrap();
    let peak = builder.source_workspace_bytes;
    assert_eq!(peak, resources::source_workspace(&[larger]).unwrap());
    builder.admit_source_objects(&[source()]).unwrap();
    assert_eq!(builder.source_workspace_bytes, peak);
    let phase = builder.begin_source_phase().unwrap();
    drop(phase);
    builder.admit_preparation(&vec![2.0_f64; 1024]).unwrap();
    builder.maximum_bytes = builder.preparation_resident_bytes + peak - 1;
    assert!(matches!(
        builder.begin_source_phase(),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(builder.active_source_workspace().unwrap(), 0);
}

#[test]
fn transport_transition_keeps_input_charged_and_cannot_authorize_replay() {
    let mut builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"transport-phase"), 2 << 20).unwrap();
    builder.admit_preparation(&vec![1.0_f64; 128]).unwrap();
    builder.admit_source_objects(&[source()]).unwrap();
    let input = builder.preparation_resident_bytes;
    let phase = builder.begin_source_phase().unwrap();
    let transport = builder.source_workspace_bytes + 64;
    builder.maximum_bytes = input + transport;
    phase
        .with_transport(&builder, transport, || {
            assert_eq!(builder.active_source_workspace().unwrap(), transport);
            assert!(builder.require_source_objects(&[source()]).is_err());
            assert!(builder.source_phase_active().unwrap());
            assert!(matches!(
                builder.admit(1),
                Err(AnalysisPublicationError::ResourceLimit { .. })
            ));
            Ok(())
        })
        .unwrap();
    assert!(phase.with_transport(&builder, 0, || Ok(())).is_err());
    drop(phase);
    assert_eq!(builder.active_source_workspace().unwrap(), 0);
    assert_eq!(builder.preparation_resident_bytes, input);
}

#[test]
fn rejected_transport_restores_replay_and_foreign_guards_cannot_mutate_it() {
    let mut builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"transport-rejection"), 2 << 20).unwrap();
    builder.admit_source_objects(&[source()]).unwrap();
    builder.maximum_bytes = builder.source_workspace_bytes;
    let phase = builder.begin_source_phase().unwrap();
    let mut installed = false;
    assert!(phase
        .with_transport(&builder, builder.maximum_bytes + 1, || {
            installed = true;
            Ok(())
        })
        .is_err());
    assert!(!installed);
    builder.require_source_objects(&[source()]).unwrap();
    let foreign =
        AnalysisPublicationBuilder::new(Digest::sha256(b"foreign-transport"), 2 << 20).unwrap();
    assert!(phase.with_transport(&foreign, 0, || Ok(())).is_err());
    assert_eq!(foreign.active_source_workspace().unwrap(), 0);
    assert_eq!(
        builder.active_source_workspace().unwrap(),
        builder.maximum_bytes
    );
    drop(phase);
}

#[test]
fn frozen_inventory_rejects_entry_and_transition_without_losing_the_live_charge() {
    let mut builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"frozen-source"), 2 << 20).unwrap();
    builder.admit_source_objects(&[source()]).unwrap();
    let phase = builder.begin_source_phase().unwrap();
    *builder.admission.lock().unwrap() = true;
    let before = builder.active_source_workspace().unwrap();
    assert!(phase.with_transport(&builder, 0, || Ok(())).is_err());
    assert_eq!(builder.active_source_workspace().unwrap(), before);
    drop(phase);
    assert!(builder.begin_source_phase().is_err());
    assert_eq!(builder.active_source_workspace().unwrap(), 0);
}
