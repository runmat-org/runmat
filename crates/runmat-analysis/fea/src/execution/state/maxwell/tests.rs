use super::*;
use crate::execution::{MaxwellHarmonicWorkload, SemanticStatePublicationProfileSource};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures, ElementOrder};

fn workload() -> MaxwellHarmonicWorkload {
    let frequency = 1e8;
    let mut model = crate::physics::magnetic::test_support::authored_model(1.0);
    model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(frequency),
    });
    let mut step = ProcedureStep::electromagnetic_harmonic("potential");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![frequency]);
    step.state_outputs = ["first", "second"]
        .into_iter()
        .map(|id| StateOutput {
            state_id: StateId(id.into()),
            kind: StateKind::ElectromagneticComplex,
            completeness: StateCompleteness::Complete,
            persistence: StatePersistence::Canonical,
        })
        .collect();
    model.sequence = AnalysisSequence::from_steps("maxwell", vec![step]);
    model.constraints = ["slab_left", "slab_right", "slab_insulated"]
        .into_iter()
        .map(|id| {
            AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: id.into(),
                kind: ElectricalConstraintKind::ElectricPerfectConductor {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(id)),
                    ),
                },
            })
        })
        .collect();
    let PhysicalSignal::ElectricCurrentDensity(signal) = &mut model.signals[0] else {
        panic!()
    };
    let vector = |v: [f64; 3]| SignalValue::Vector3(v.map(ElectricCurrentDensity::new));
    signal.precision = SignalPrecision::Complex64;
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::FrequencyHertz,
        knots: vec![SignalKnot::continuous(
            frequency,
            SignalOrdinate::Complex {
                real: vector([0.0, 0.6, 0.0]),
                imaginary: vector([0.0, 0.8, 0.0]),
            },
        )],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::HoldEndpoint,
    });
    MaxwellHarmonicWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
        frequency: runmat_analysis_core::Frequency::new(1e8),
    }
}

#[test]
fn both_orders_cold_complex_state_retains_original_electric_fields_and_exact_profiles() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = match order {
            ElementOrder::Tet4 => fixtures::slab_mesh(2),
            ElementOrder::Tet10 => fixtures::quadratic_slab_mesh(2),
        };
        let workload = workload();
        let result = workload.execute(&mesh).unwrap();
        let original = result.canonical_bytes().unwrap();
        let cold = MaxwellHarmonicSolverResult::from_canonical_bytes(&original).unwrap();
        drop(result);
        let requests = cold.state_publication_requests().unwrap();
        assert_eq!(requests.len(), 2);
        for request in requests {
            let state = CanonicalSemanticState::from_maxwell_publication(&request, &cold).unwrap();
            let imported =
                CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap())
                    .unwrap();
            assert_eq!(imported, state);
            let CanonicalSemanticStatePayload::MaxwellHarmonic { state: payload } =
                &imported.payload
            else {
                panic!()
            };
            assert_eq!(payload.result.input(), cold.input());
            assert_eq!(payload.result.response(), cold.response());
            assert_eq!(payload.result.canonical_bytes().unwrap(), original);
            let projected = request
                .encoding_profile(SemanticStatePublicationProfileSource::MaxwellNativeFields(
                    &cold,
                ))
                .unwrap();
            let actual = runmat_canonical_codec::profile_serialized(&state, STATE_LIMITS).unwrap();
            assert_eq!(projected, actual);
        }
        let profiles = cold
            .state_publication_profiles(Digest::sha256(&original))
            .unwrap();
        let encoded = workload
            .state_publication_profiles_encoding_profile(Digest::from_bytes(
                *mesh.canonical_digest.bytes(),
            ))
            .unwrap();
        let actual = runmat_canonical_codec::profile_serialized_shape(
            &profiles,
            CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
        )
        .unwrap();
        assert_eq!(encoded, actual);
        assert!(cold
            .state_publication_profiles(Digest::sha256(b"foreign source bytes"))
            .is_err());
    }
}

#[test]
fn publication_rejects_static_or_foreign_declaration_source_and_topology() {
    let result = workload().execute(&fixtures::slab_mesh(2)).unwrap();
    let request = result.state_publication_requests().unwrap().remove(0);
    let mut foreign = request.clone();
    foreign.declaration.kind = StateKind::ElectromagneticStatic;
    assert!(CanonicalSemanticState::from_maxwell_publication(&foreign, &result).is_err());
    foreign = request.clone();
    foreign.declaration.state_id = StateId("invented".into());
    assert!(result.validate_publication_request(&foreign).is_err());
    foreign = request.clone();
    foreign.source = SemanticStatePublicationSource::MagnetostaticNativeFields;
    assert!(result.validate_publication_request(&foreign).is_err());
    foreign = request.clone();
    foreign.topology_digest = Digest::sha256(b"foreign sample topology");
    assert!(result.validate_publication_request(&foreign).is_err());
    foreign = request.clone();
    foreign.source_result =
        SolverResultIdentity::from_step_result(request.producer, Digest::sha256(b"foreign result"))
            .unwrap();
    assert!(result.validate_publication_request(&foreign).is_err());
    let payload = CanonicalMaxwellStatePayload {
        result: Box::new(result),
    };
    assert!(payload
        .validate_for(
            foreign.topology_digest,
            foreign.source_result,
            foreign.producer
        )
        .is_err());
}
