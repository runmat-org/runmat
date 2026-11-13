use super::*;
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures, ElementOrder};

#[test]
fn actual_wire_decode_budget_is_checked_without_demanding_maximum_width_profile() {
    let w = workload();
    let m = fixtures::slab_mesh(2);
    let mut f = facts(&w, &m);
    let exact = runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(
        &w.canonical_bytes().unwrap(),
    )
    .unwrap()
        + runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(
            &m.canonical_encode().unwrap(),
        )
        .unwrap();
    let upper = profile_serialized(&w, LIMITS)
        .unwrap()
        .materialization_resident_bytes_estimate
        + profile_serialized(&m, LIMITS)
            .unwrap()
            .materialization_resident_bytes_estimate;
    assert!(upper > exact);
    f.inputs.decoded_resident_bytes_admission_budget = ByteCount(exact);
    let plan = build_maxwell_harmonic_resource_plan(&w, &m, f.clone()).unwrap();
    let FeaResourceDemand::MaxwellHarmonic(d) = &plan.input.demand else {
        panic!();
    };
    assert!(d.numerical_reservation_bytes.0 > upper);
    f.inputs.decoded_resident_bytes_admission_budget.0 -= 1;
    assert!(build_maxwell_harmonic_resource_plan(&w, &m, f).is_err());
}

fn workload() -> MaxwellHarmonicWorkload {
    let mut model = crate::physics::magnetic::test_support::authored_model(1.0);
    model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(1e8),
    });
    let mut step = ProcedureStep::electromagnetic_harmonic("potential");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![1e8]);
    step.state_outputs = ["one", "two"]
        .into_iter()
        .map(|id| StateOutput {
            state_id: StateId(id.repeat(256)),
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
    MaxwellHarmonicWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
        frequency: runmat_analysis_core::Frequency::new(1e8),
    }
}
fn facts(
    w: &MaxwellHarmonicWorkload,
    m: &SolverMeshArtifact,
) -> PreparedMaxwellHarmonicResourceFacts {
    let input = (w.canonical_bytes().unwrap().len() + m.canonical_encode().unwrap().len()) as u64;
    PreparedMaxwellHarmonicResourceFacts {
        local_execution: crate::planning::resource::LocalExecutionPlan::serial(),
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(input),
            transport_closure_bytes: ByteCount(input + 2048),
            decoded_resident_bytes_admission_budget: ByteCount(input * 32),
        },
        result_output: projected_canonical_output_resource_facts(
            maxwell_harmonic_result_payload_bytes_upper_bound(w, m).unwrap(),
        )
        .unwrap(),
        receipt_output: projected_canonical_output_resource_facts(
            crate::execution::MAXWELL_RECEIPT_MAX_ENCODED_BYTES,
        )
        .unwrap(),
        state_profiles_output: projected_canonical_output_resource_facts(
            w.state_publication_profiles_encoding_profile(Digest::from_bytes(
                *m.canonical_digest.bytes(),
            ))
            .unwrap()
            .encoded_bytes_upper_bound,
        )
        .unwrap(),
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(2048),
        },
    }
}
#[test]
fn exact_complex_support_codec_and_retained_cold_outputs_fit_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let m = match order {
            ElementOrder::Tet4 => fixtures::slab_mesh(2),
            _ => fixtures::quadratic_slab_mesh(2),
        };
        let w = workload();
        let f = facts(&w, &m);
        let plan = build_maxwell_harmonic_resource_plan(&w, &m, f.clone()).unwrap();
        assert_eq!(
            FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
            plan
        );
        let FeaResourceDemand::MaxwellHarmonic(d) = &plan.input.demand else {
            panic!("Maxwell demand");
        };
        let grounded = crate::physics::maxwell::grounded_face_identities(&w.model, &m).unwrap();
        let system = MaxwellSystemPlan::derive(&m, &grounded).unwrap();
        assert_eq!(d.system_plan_identity, system.identity().unwrap());
        assert_eq!(d.complex_dimension.0, d.free_edge_count.0);
        assert_eq!(d.nonzeros.0, system.pattern().column_indices().len() as u64);
        assert!(d.factorization.dimension.0 > 0);
        assert_eq!(
            d.quadrature_sample_count.0,
            (m.topology.volume_elements.len() * 15) as u64
        );
        let result = w.execute_with_resource_plan(&m, &plan).unwrap();
        assert_eq!(result, w.execute(&m).unwrap());
        let bytes = result.canonical_bytes().unwrap();
        assert!(bytes.len() as u64 <= f.result_output.encoded_payload_bytes_upper_bound.0);
        let expected = Digest::sha256(&bytes);
        assert_eq!(
            crate::execution::MaxwellHarmonicSolverResult::from_canonical_bytes_for(
                &bytes, expected, &w, &m
            )
            .unwrap(),
            result
        );
    }
}
#[test]
fn prepared_understatement_and_count_only_tiny_policy_fail_closed() {
    let w = workload();
    let m = fixtures::slab_mesh(2);
    let f = facts(&w, &m);
    let mut bad = f.clone();
    bad.result_output.encoded_payload_bytes_upper_bound.0 -= 1;
    assert!(build_maxwell_harmonic_resource_plan(&w, &m, bad).is_err());
    let mut bad = f.clone();
    bad.inputs.encoded_payload_bytes.0 -= 1;
    assert!(build_maxwell_harmonic_resource_plan(&w, &m, bad).is_err());
    let mut bad = f.clone();
    bad.inputs.decoded_resident_bytes_admission_budget = ByteCount(1);
    assert!(build_maxwell_harmonic_resource_plan(&w, &m, bad).is_err());
    let mut bad = f.clone();
    bad.state_profiles_output
        .encoded_payload_bytes_upper_bound
        .0 -= 1;
    assert!(build_maxwell_harmonic_resource_plan(&w, &m, bad).is_err());
    let mut tiny = w.clone();
    tiny.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let mut malformed = m.clone();
    malformed.topology.volume_elements[0]
        .node_ids
        .extend(100..200);
    let error = build_maxwell_harmonic_resource_plan(&tiny, &malformed, f).unwrap_err();
    assert!(error
        .to_string()
        .contains("Maxwell local resource preflight"));
    assert!(maxwell_harmonic_result_payload_bytes_upper_bound(&tiny, &malformed).is_err());
}
#[test]
fn direct_factor_envelope_and_forged_exact_facts_are_rejected() {
    let w = workload();
    let m = fixtures::slab_mesh(2);
    let plan = build_maxwell_harmonic_resource_plan(&w, &m, facts(&w, &m)).unwrap();
    for which in 0..4 {
        let mut input = plan.input.clone();
        let FeaResourceDemand::MaxwellHarmonic(d) = &mut input.demand else {
            panic!();
        };
        match which {
            0 => d.complex_dimension.0 += 1,
            1 => d.factorization.resident_bytes.0 -= 1,
            2 => d.nonzeros.0 = d.element_count.0 * 36 + 1,
            _ => d.edge_count_upper_bound.0 -= 1,
        }
        assert!(build_resource_plan(input).is_err());
    }
    assert_eq!(
        estimate::maxwell_harmonic::factor_envelope(0, 0).unwrap(),
        MaxwellDirectFactorEnvelope::default()
    );
    assert!(estimate::maxwell_harmonic::factor_envelope(u64::MAX, 0).is_err());
}
