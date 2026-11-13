use super::*;
use crate::execution::MaxwellHarmonicWorkload;
use crate::results::field::canonical_field_partitions;
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
fn complex_recovery_profiles_bound_actual_catalog_and_partitions_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = match order {
            ElementOrder::Tet4 => fixtures::slab_mesh(2),
            ElementOrder::Tet10 => fixtures::quadratic_slab_mesh(2),
        };
        let workload = workload();
        let result = workload.execute(&mesh).unwrap();
        let request = result.native_field_request(&workload, &mesh).unwrap();
        let fields = result
            .recover_native_fields(&workload, &mesh, &request)
            .unwrap();
        assert_eq!(
            fields.fields().len(),
            crate::execution::MAXWELL_NATIVE_FIELD_COUNT
        );
        let (bounds, workspace) = maxwell_field_recovery_shape(&request).unwrap();
        let mut actual = vec![fields.manifest().unwrap().canonical_bytes().unwrap().len() as u64];
        for field in fields.fields() {
            actual.extend(
                canonical_field_partitions(field)
                    .unwrap()
                    .iter()
                    .map(|p| p.canonical_bytes().unwrap().len() as u64),
            );
        }
        assert_eq!(actual.len(), bounds.len());
        assert_eq!(
            maxwell_field_recovery_partition_count(&request).unwrap(),
            bounds.len()
        );
        assert!(actual.iter().zip(&bounds).all(|(a, b)| a <= b));
        assert!(workspace.0 > actual.iter().sum::<u64>());
        let bytes = bounds.iter().sum::<u64>();
        let output = PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(bytes),
            transport_closure_bytes_upper_bound: ByteCount(bytes + 1024),
            construction_resident_bytes_admission_budget: ByteCount(2 * bytes + 2048),
        };
        let facts = PreparedPhaseResourceFacts {
            inputs: PreparedInputResourceFacts {
                encoded_payload_bytes: ByteCount(8),
                transport_closure_bytes: ByteCount(16),
                decoded_resident_bytes_admission_budget: ByteCount(1024),
            },
            output: output.clone(),
            output_set: PreparedOutputSetResourceFacts {
                encoded_bytes_upper_bound: ByteCount(1024),
            },
        };
        let validation = maxwell_recovery_source_validation_workspace(&workload, &mesh).unwrap();
        let plan = build_maxwell_field_recovery_resource_plan(
            &request,
            validation,
            facts.clone(),
            crate::planning::resource::LocalExecutionPlan::serial(),
        )
        .unwrap();
        assert_eq!(
            plan.input
                .demand
                .output_facts(crate::execution::AnalysisNumericalObjectKind::Field),
            Some(&output)
        );
        assert!(plan
            .input
            .demand
            .output_facts(crate::execution::AnalysisNumericalObjectKind::SolverResult)
            .is_none());
        let mut understated = facts;
        understated.output.encoded_payload_bytes_upper_bound.0 -= 1;
        assert!(build_maxwell_field_recovery_resource_plan(
            &request,
            validation,
            understated,
            crate::planning::resource::LocalExecutionPlan::serial(),
        )
        .is_err());
        assert_eq!(
            FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
            plan
        );
    }
}

#[test]
fn incomplete_or_incoherent_complex_native_inventory_is_rejected() {
    let mesh = fixtures::slab_mesh(2);
    let workload = workload();
    let result = workload.execute(&mesh).unwrap();
    let original = result.native_field_request(&workload, &mesh).unwrap();
    let mut plan = original.plan.clone();
    plan.requirements.pop();
    let request =
        FieldRecoveryRequest::new(original.producer, original.source, original.topology, plan)
            .unwrap();
    assert!(maxwell_field_recovery_shape(&request).is_err());
    let mut plan = original.plan.clone();
    plan.requirements[1].cost.output_entity_count += 1;
    let request =
        FieldRecoveryRequest::new(original.producer, original.source, original.topology, plan)
            .unwrap();
    assert!(maxwell_field_recovery_partition_count(&request).is_err());
}
