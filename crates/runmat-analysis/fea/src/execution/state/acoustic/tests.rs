use super::*;
use crate::topology::test_meshes::{quadratic_slab_mesh, slab_mesh};
use crate::{
    execution::{FieldRecoveryPlan, FieldRecoveryRequest, FieldTopologyIdentity, SequenceIdentity},
    physics::structural::topology::CanonicalStructuralTopologyIdentity,
    results::field::{RecoveredField, RecoveredFieldArtifact},
};
use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, DeviceFieldRef, IndependentDomain, StateCompleteness,
    StateKind,
};

#[test]
fn acoustic_result_publishes_one_cold_topology_bound_complex_state() {
    for quadratic in [false, true] {
        let mesh = if quadratic {
            quadratic_slab_mesh(2)
        } else {
            slab_mesh(2)
        };
        let workload = crate::execution::acoustic::tests::workload();
        let result = workload.execute(&mesh).unwrap();
        let state =
            CanonicalSemanticState::from_acoustic_result(&workload, &mesh, &result).unwrap();
        assert_eq!(state.kind, StateKind::AcousticComplex);
        assert_eq!(
            state.source_result,
            result.source_result_identity().unwrap()
        );
        let payload = state.payload.acoustic().unwrap();
        assert_eq!(payload.frequency, workload.frequency);
        assert_eq!(payload.ordered_node_ids, result.ordered_node_ids());
        assert_eq!(payload.ordered_element_ids, result.ordered_element_ids());
        assert_eq!(payload.fields.fields().len(), 4);
        assert_eq!(
            CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap())
                .unwrap(),
            state
        );
    }
}

#[test]
fn acoustic_detached_publication_is_source_bound_and_borrowed_profile_bounded() {
    for quadratic in [false, true] {
        let mesh = if quadratic {
            quadratic_slab_mesh(2)
        } else {
            slab_mesh(2)
        };
        let workload = crate::execution::acoustic::tests::workload();
        let result = workload.execute(&mesh).unwrap();
        let result_bytes = result.canonical_bytes().unwrap();
        let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
        let bound = workload
            .state_publication_profiles_encoding_profile(topology)
            .unwrap();
        let result = AcousticHarmonicSolverResult::from_canonical_bytes(&result_bytes).unwrap();
        let requests = result.state_publication_requests().unwrap();
        assert_eq!(
            requests,
            workload
                .state_publication_requests(topology, result.source_result_identity().unwrap())
                .unwrap()
        );
        let catalog = result
            .state_publication_profiles(Digest::sha256(&result_bytes))
            .unwrap();
        assert!(catalog.canonical_bytes().unwrap().len() as u64 <= bound.encoded_bytes_upper_bound);
        assert!(result
            .state_publication_profiles(Digest::sha256(b"foreign-payload"))
            .is_err());
        let request = &requests[0];
        let projected = request
            .encoding_profile(
                crate::execution::SemanticStatePublicationProfileSource::AcousticNativeFields(
                    &result,
                ),
            )
            .unwrap();
        // No workload/mesh is passed into the detached publication path. Result arrays are copied
        // into the shared artifact exactly once; the profiler never materializes that artifact.
        let state = CanonicalSemanticState::from_acoustic_publication(request, &result).unwrap();
        assert!(
            state.canonical_bytes().unwrap().len() as u64 <= projected.encoded_bytes_upper_bound
        );
        assert_eq!(catalog.entry(request).unwrap().profile(), projected);
        assert_eq!(
            state,
            CanonicalSemanticState::from_acoustic_result(&workload, &mesh, &result).unwrap()
        );
        let mut foreign = request.clone();
        foreign.declaration.state_id.0.push_str("-foreign");
        assert!(CanonicalSemanticState::from_acoustic_publication(&foreign, &result).is_err());
        foreign = request.clone();
        foreign.producer = StepIdentity::new(
            SequenceIdentity::new(Digest::sha256(b"foreign-sequence")).unwrap(),
            Digest::sha256(b"foreign-step"),
        )
        .unwrap();
        assert!(foreign
            .encoding_profile(
                crate::execution::SemanticStatePublicationProfileSource::AcousticNativeFields(
                    &result
                )
            )
            .is_err());
        foreign = request.clone();
        foreign.topology_digest = Digest::sha256(b"foreign-mesh");
        assert!(CanonicalSemanticState::from_acoustic_publication(&foreign, &result).is_err());
        foreign = request.clone();
        foreign.source_result = SolverResultIdentity::from_step_result(
            request.producer,
            Digest::sha256(b"foreign-result"),
        )
        .unwrap();
        assert!(CanonicalSemanticState::from_acoustic_publication(&foreign, &result).is_err());
    }
}

#[test]
fn acoustic_sweep_entries_keep_distinct_original_complex_state_authority() {
    let mesh = slab_mesh(2);
    let mut template = crate::execution::acoustic::tests::workload();
    template.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.2, 0.4]);
    let sweep = crate::execution::AcousticFrequencyWorkload::new(template.clone()).unwrap();
    let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let mut states = Vec::new();
    for entry in sweep.plan().entries().unwrap() {
        let workload = sweep.entry(entry).unwrap();
        assert_eq!(workload.model, template.model);
        let result = workload.execute(&mesh).unwrap();
        let requests = workload
            .state_publication_requests(topology, result.source_result_identity().unwrap())
            .unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests, result.state_publication_requests().unwrap());
        let profiles = result
            .state_publication_profiles(Digest::sha256(result.canonical_bytes().unwrap()))
            .unwrap();
        assert_eq!(profiles.entries().len(), 1);
        let state =
            CanonicalSemanticState::from_acoustic_result(&workload, &mesh, &result).unwrap();
        assert_eq!(
            state
                .payload
                .acoustic()
                .unwrap()
                .frequency
                .canonical_value(),
            entry.frequency_hz()
        );
        assert_eq!(
            state.source_result,
            result.source_result_identity().unwrap()
        );
        assert_eq!(
            state,
            CanonicalSemanticState::from_acoustic_publication(&requests[0], &result).unwrap()
        );
        states.push(state);
    }
    assert_eq!(states[0].state_id, states[1].state_id);
    assert_ne!(states[0].source_result, states[1].source_result);
    assert_ne!(states[0].identity, states[1].identity);
}

#[test]
fn acoustic_state_rejects_substituted_result_mesh_and_payload() {
    let mesh = slab_mesh(2);
    let workload = crate::execution::acoustic::tests::workload();
    let result = workload.execute(&mesh).unwrap();
    let state = CanonicalSemanticState::from_acoustic_result(&workload, &mesh, &result).unwrap();

    let mut other_workload = workload.clone();
    other_workload.solver.relative_tolerance *= 0.5;
    assert!(CanonicalSemanticState::from_acoustic_result(&other_workload, &mesh, &result).is_err());
    let other_mesh = slab_mesh(3);
    assert!(CanonicalSemanticState::from_acoustic_result(&workload, &other_mesh, &result).is_err());

    let mut sweep = workload.clone();
    sweep.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.2, 0.4]);
    assert!(CanonicalSemanticState::from_acoustic_result(&sweep, &mesh, &result).is_err());

    let mut declaration = workload.model.sequence.steps[0].state_outputs[0].clone();
    declaration.completeness = StateCompleteness::Partial;
    assert!(CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer: state.producer,
        declaration,
        topology_digest: state.topology_digest,
        source_result: state.source_result,
        payload: state.payload.clone(),
    })
    .is_err());

    let payload = state.payload.acoustic().unwrap();
    let mut fields = payload.fields.fields().to_vec();
    let first = fields[0].clone();
    let element_count = first.storage().shape.iter().product();
    fields[0] = RecoveredField::new(
        first.definition().clone(),
        first.sample(),
        AnalysisField {
            field_id: first.storage().field_id.clone(),
            layout: first.definition().layout(),
            shape: first.storage().shape.clone(),
            values: AnalysisFieldValues::DeviceRef(DeviceFieldRef {
                backend: "test-device".into(),
                token: "resident".into(),
                element_count,
            }),
        },
    )
    .unwrap();
    if let Ok(device_fields) = RecoveredFieldArtifact::new(
        payload.fields.recovery_request,
        payload.fields.source,
        payload.fields.topology_digest,
        payload.fields.entity_counts,
        fields,
    ) {
        let mut device_payload = payload.clone();
        device_payload.fields = device_fields;
        assert!(device_payload
            .validate_for(state.topology_digest, state.source_result, state.producer)
            .is_err());
    }

    let mut nonfinite_storage = first.storage().clone();
    let AnalysisFieldValues::HostF64(nonfinite_values) = &mut nonfinite_storage.values else {
        unreachable!("native acoustic fields are host-backed")
    };
    nonfinite_values[0] = f64::NAN;
    if let Ok(nonfinite) = RecoveredField::new(
        first.definition().clone(),
        first.sample(),
        nonfinite_storage,
    ) {
        let mut fields = payload.fields.fields().to_vec();
        fields[0] = nonfinite;
        if let Ok(nonfinite_fields) = RecoveredFieldArtifact::new(
            payload.fields.recovery_request,
            payload.fields.source,
            payload.fields.topology_digest,
            payload.fields.entity_counts,
            fields,
        ) {
            let mut nonfinite_payload = payload.clone();
            nonfinite_payload.fields = nonfinite_fields;
            assert!(nonfinite_payload
                .validate_for(state.topology_digest, state.source_result, state.producer)
                .is_err());
        }
    }

    let structural_request = FieldRecoveryRequest::new(
        state.producer,
        payload.recovery_request.source,
        FieldTopologyIdentity::Structural {
            topology: CanonicalStructuralTopologyIdentity::Solid {
                mesh_digest: mesh.canonical_digest,
            },
        },
        payload.recovery_request.plan.clone(),
    )
    .unwrap();
    let structural_fields = RecoveredFieldArtifact::new(
        structural_request.identity(),
        structural_request.source,
        structural_request.topology.discretization_digest(),
        payload.fields.entity_counts,
        payload.fields.fields().to_vec(),
    )
    .unwrap();
    let mut structural_payload = payload.clone();
    structural_payload.recovery_request = structural_request;
    structural_payload.fields = structural_fields;
    assert!(structural_payload
        .validate_for(state.topology_digest, state.source_result, state.producer)
        .is_err());

    let empty_request = FieldRecoveryRequest::new(
        state.producer,
        payload.recovery_request.source,
        payload.recovery_request.topology,
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let empty_plan_fields = RecoveredFieldArtifact::new(
        empty_request.identity(),
        empty_request.source,
        empty_request.topology.discretization_digest(),
        payload.fields.entity_counts,
        payload.fields.fields().to_vec(),
    )
    .unwrap();
    let mut empty_plan_payload = payload.clone();
    empty_plan_payload.recovery_request = empty_request;
    empty_plan_payload.fields = empty_plan_fields;
    assert!(empty_plan_payload
        .validate_for(state.topology_digest, state.source_result, state.producer)
        .is_err());

    let mut substituted_plan = payload.recovery_request.plan.clone();
    substituted_plan.requirements[0]
        .cost
        .arithmetic_operation_count += 1;
    let substituted_request = FieldRecoveryRequest::new(
        state.producer,
        payload.recovery_request.source,
        payload.recovery_request.topology,
        substituted_plan,
    )
    .unwrap();
    let substituted_fields = RecoveredFieldArtifact::new(
        substituted_request.identity(),
        substituted_request.source,
        substituted_request.topology.discretization_digest(),
        payload.fields.entity_counts,
        payload.fields.fields().to_vec(),
    )
    .unwrap();
    let mut substituted_payload = payload.clone();
    substituted_payload.recovery_request = substituted_request;
    substituted_payload.fields = substituted_fields;
    assert!(substituted_payload
        .validate_for(state.topology_digest, state.source_result, state.producer)
        .is_err());

    let mut malformed = serde_json::to_value(&state).unwrap();
    malformed["payload"]["state"]["frequency"]["hertz"] = serde_json::json!(0.3);
    let malformed: CanonicalSemanticState = serde_json::from_value(malformed).unwrap();
    assert!(malformed.validate().is_err());
    assert!(malformed.canonical_bytes().is_err());
}
