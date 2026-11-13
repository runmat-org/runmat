use super::*;
use crate::execution::{
    AnalysisNumericalObjectKind, SemanticStatePublicationSource, SequenceIdentity,
    SolverResultIdentity, StatePublicationProfileEntry, StepIdentity,
};
use runmat_analysis_core::{ProcedureStep, StateKind};
use runmat_canonical_codec::SerializedCanonicalProfile;

fn fixture(
    source: SemanticStatePublicationSource,
) -> (
    SemanticStatePublicationRequest,
    StatePublicationProfiles,
    PreparedPhaseResourceFacts,
) {
    let step = ProcedureStep::structural_static("publication");
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::new(Digest::sha256(b"sequence")).unwrap(),
        &step,
    )
    .unwrap();
    let mut declaration = step.state_outputs[0].clone();
    declaration.kind = match source {
        SemanticStatePublicationSource::NonlinearSolidOriginalState => StateKind::MaterialHistory,
        SemanticStatePublicationSource::FluidPrimaryFields => StateKind::Fluid,
        SemanticStatePublicationSource::MaxwellNativeFields => StateKind::ElectromagneticComplex,
        SemanticStatePublicationSource::StructuralPrimaryFields => StateKind::StructuralPrimary,
        SemanticStatePublicationSource::ThermalPrimaryField => StateKind::Thermal,
        SemanticStatePublicationSource::AcousticNativeFields => StateKind::AcousticComplex,
        SemanticStatePublicationSource::AcousticResponseFields => StateKind::AcousticPressure,
        SemanticStatePublicationSource::StaticElectricalNativeFields
        | SemanticStatePublicationSource::MagnetostaticNativeFields => {
            StateKind::ElectromagneticStatic
        }
        SemanticStatePublicationSource::StructuralLinearization => StateKind::GeometricStiffness,
        SemanticStatePublicationSource::ModalBasis => StateKind::ModalBasis,
        SemanticStatePublicationSource::StructuralResponseEnvelope => {
            StateKind::StructuralResponseEnvelope
        }
        SemanticStatePublicationSource::StructuralRandomStatistics => {
            StateKind::StructuralRandomStatistics
        }
    };
    let request = SemanticStatePublicationRequest::new(
        source,
        producer,
        declaration,
        Digest::sha256(b"topology"),
        SolverResultIdentity::from_step_result(producer, Digest::sha256(b"result")).unwrap(),
    )
    .unwrap();
    let entry = StatePublicationProfileEntry::new(
        request.clone(),
        SerializedCanonicalProfile {
            encoded_bytes_upper_bound: 1024,
            materialization_resident_bytes_estimate: 100_000,
        },
    )
    .unwrap();
    let catalog = StatePublicationProfiles::new(
        producer,
        request.source_result,
        request.topology_digest,
        Digest::sha256(b"payload"),
        vec![entry],
    )
    .unwrap();
    let facts = PreparedPhaseResourceFacts {
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(16_384),
            transport_closure_bytes: ByteCount(18_000),
            decoded_resident_bytes_admission_budget: ByteCount(200_000),
        },
        output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(1024),
            transport_closure_bytes_upper_bound: ByteCount(2048),
            construction_resident_bytes_admission_budget: ByteCount(102_048),
        },
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(1024),
        },
    };
    (request, catalog, facts)
}

#[test]
fn all_sources_bind_canonical_request_and_account_owned_state_and_verification() {
    for source in [
        SemanticStatePublicationSource::NonlinearSolidOriginalState,
        SemanticStatePublicationSource::StructuralPrimaryFields,
        SemanticStatePublicationSource::ThermalPrimaryField,
        SemanticStatePublicationSource::AcousticNativeFields,
        SemanticStatePublicationSource::StaticElectricalNativeFields,
        SemanticStatePublicationSource::MagnetostaticNativeFields,
        SemanticStatePublicationSource::StructuralLinearization,
        SemanticStatePublicationSource::ModalBasis,
        SemanticStatePublicationSource::StructuralResponseEnvelope,
        SemanticStatePublicationSource::StructuralRandomStatistics,
    ] {
        let (request, catalog, facts) = fixture(source);
        let plan = build_state_publication_resource_plan(
            &request,
            &catalog,
            Digest::sha256(b"payload"),
            facts.clone(),
        )
        .unwrap();
        assert_eq!(
            plan.input.phase,
            FeaResourcePhase::SemanticStatePublication(Digest::sha256(
                request.canonical_bytes().unwrap()
            ))
        );
        assert_eq!(
            plan.input
                .demand
                .output_facts(AnalysisNumericalObjectKind::SemanticState),
            Some(&facts.output)
        );
        assert_eq!(
            plan,
            FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap()
        );
        let FeaResourceDemand::SemanticStatePublication(demand) = &plan.input.demand else {
            panic!("wrong demand")
        };
        assert_eq!(demand.source, source);
        assert_eq!(demand.profile_catalog, catalog.identity());
        assert_eq!(
            demand.source_validation_workspace_bytes,
            facts.inputs.decoded_resident_bytes_admission_budget
        );
        let mut larger = facts;
        larger.inputs.decoded_resident_bytes_admission_budget.0 += 1000;
        let larger_plan = build_state_publication_resource_plan(
            &request,
            &catalog,
            Digest::sha256(b"payload"),
            larger,
        )
        .unwrap();
        // One retained decoded source and one independent source-validation workspace.
        assert_eq!(
            larger_plan.request.memory_bytes - plan.request.memory_bytes,
            2000
        );
        assert_ne!(larger_plan.identity, plan.identity);
    }
}

#[test]
fn original_publication_ceiling_governs_wall_and_exact_memory_admission() {
    let (request, catalog, facts) =
        fixture(SemanticStatePublicationSource::NonlinearSolidOriginalState);
    let mut ceiling = super::super::post_process::post_process_ceiling();
    ceiling.maximum_wall_time_milliseconds = 17;
    let first = build_state_publication_resource_plan_with_ceiling(
        &request,
        &catalog,
        Digest::sha256(b"payload"),
        facts.clone(),
        ceiling,
    )
    .unwrap();
    assert_eq!(first.input.worker.maximum_wall_milliseconds, 17);
    assert_eq!(first.request.max_wall_millis, 17);
    ceiling.maximum_resident_bytes = Some(first.request.memory_bytes);
    let exact = build_state_publication_resource_plan_with_ceiling(
        &request,
        &catalog,
        Digest::sha256(b"payload"),
        facts.clone(),
        ceiling,
    )
    .unwrap();
    assert_eq!(exact.request.memory_bytes, ceiling.resident_byte_budget());
    ceiling.maximum_resident_bytes = Some(ceiling.resident_byte_budget() - 1);
    assert!(build_state_publication_resource_plan_with_ceiling(
        &request,
        &catalog,
        Digest::sha256(b"payload"),
        facts,
        ceiling
    )
    .is_err());
}

#[test]
fn source_substitution_and_missing_exact_declaration_are_rejected() {
    let (mut request, catalog, facts) =
        fixture(SemanticStatePublicationSource::StructuralPrimaryFields);
    assert!(build_state_publication_resource_plan(
        &request,
        &catalog,
        Digest::sha256(b"wrong"),
        facts.clone()
    )
    .is_err());
    request.declaration.state_id.0.push_str("-other");
    assert!(build_state_publication_resource_plan(
        &request,
        &catalog,
        Digest::sha256(b"payload"),
        facts
    )
    .is_err());
}

#[test]
fn insufficient_output_and_source_reservations_are_rejected() {
    let (request, catalog, facts) = fixture(SemanticStatePublicationSource::ModalBasis);
    for change in 0..5 {
        let mut bad = facts.clone();
        match change {
            0 => bad.output.encoded_payload_bytes_upper_bound.0 -= 1,
            1 => bad.output.construction_resident_bytes_admission_budget.0 -= 1,
            2 => bad.inputs.decoded_resident_bytes_admission_budget = ByteCount(1),
            3 => bad.inputs.transport_closure_bytes = ByteCount(0),
            4 => bad.inputs.encoded_payload_bytes = ByteCount(1),
            _ => unreachable!(),
        }
        assert!(build_state_publication_resource_plan(
            &request,
            &catalog,
            Digest::sha256(b"payload"),
            bad
        )
        .is_err());
    }
    let plan = build_state_publication_resource_plan(
        &request,
        &catalog,
        Digest::sha256(b"payload"),
        facts,
    )
    .unwrap();
    let mut input = plan.input.clone();
    input.phase = FeaResourcePhase::SemanticStatePublication(Digest::sha256(b"another request"));
    assert!(build_resource_plan(input).is_err());
    let mut input = plan.input;
    let FeaResourceDemand::SemanticStatePublication(demand) = &mut input.demand else {
        panic!("wrong demand")
    };
    demand.source_validation_workspace_bytes.0 -= 1;
    assert!(build_resource_plan(input).is_err());
}
