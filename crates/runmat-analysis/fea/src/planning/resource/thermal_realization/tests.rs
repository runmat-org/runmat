use super::*;
use crate::{
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::ThermalPreassemblyResourceFacts,
};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("thermal", vec![ProcedureStep::thermal_transient("heat")]);
    // The authored ceiling is part of the encoded model, so resident requirements are
    // measured under a non-binding authored ceiling (present, like the ceilings tests author).
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(runmat_types::wire_integer::MAX_WIRE_INTEGER);
    ThermalWorkload {
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: crate::ComputeBackend::Cpu,
        procedure: crate::ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

fn element_field(rows: usize) -> ThermalWorkload {
    let mut workload = workload();
    let mut action = workload.model.excitations[0].excitation().clone();
    action.target = ExcitationTarget::WholeModel;
    action.direction = None;
    action.distribution = ExcitationDistribution::TopologyField;
    action.projection.preserve_resultant = true;
    action.projection.preserve_moment = false;
    let signal = Signal {
        signal_id: action.signal_id.clone(),
        shape: SignalShape::TopologyField {
            topology_id: runmat_execution::Digest::from_bytes([8; 32]).to_string(),
            location: FieldLocation::Element,
            component_count: 1,
        },
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Tensor {
            shape: vec![rows, 1],
            values: vec![HeatGenerationDensity::new(2.0); rows],
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    };
    workload.model.signals = vec![PhysicalSignal::HeatGenerationDensity(signal)];
    workload.model.excitations = vec![ExcitationAction::HeatGeneration(action)];
    workload
}

fn field_signal(workload: &mut ThermalWorkload) -> &mut Signal<HeatGenerationDensity> {
    let PhysicalSignal::HeatGenerationDensity(signal) = &mut workload.model.signals[0] else {
        panic!("field fixture")
    };
    signal
}

fn field_facts(workload: &ThermalWorkload) -> ThermalRealizationResourceFacts {
    ThermalRealizationResourceFacts::derive(&workload.model, &workload.model.sequence.steps[0])
        .unwrap()
}

#[test]
fn element_scalar_sampling_payload_is_explicit_grows_and_preserves_scalar_reservation() {
    let small = field_facts(&element_field(1));
    let large = field_facts(&element_field(17));
    let shape_bytes = 2 * std::mem::size_of::<usize>() as u64;
    assert_eq!(small.field_payload_bytes.0, 32 + shape_bytes);
    assert_eq!(large.field_payload_bytes.0, 32 * 17 + shape_bytes);
    assert_eq!(
        large.field_payload_bytes.0 - small.field_payload_bytes.0,
        32 * 16
    );
    assert_eq!(large.vector_bytes, small.vector_bytes);
    assert_eq!(
        large.metadata_estimated_bytes,
        small.metadata_estimated_bytes
    );
    assert_eq!(
        large.reservation_bytes().unwrap().0 - small.reservation_bytes().unwrap().0,
        32 * 16
    );
    let scalar = field_facts(&workload());
    assert_eq!(scalar.field_payload_bytes, ByteCount(0));
    assert_eq!(
        scalar.reservation_bytes().unwrap().0,
        scalar.vector_bytes.0 + scalar.metadata_estimated_bytes.0 + scalar.signal_lookup_bytes.0
    );
    assert!(scalar_field_payload(u64::MAX, 1).is_err());
    assert!(scalar_field_payload(1, u64::MAX).is_err());
    let mut overflow = small;
    overflow.field_payload_bytes = ByteCount(u64::MAX);
    assert!(overflow.reservation_bytes().is_err());
}

#[test]
fn time_tensor_interpolation_reserves_endpoints_output_shapes_and_retained_field() {
    let mut workload = element_field(7);
    let signal = field_signal(&mut workload);
    let SignalRepresentation::Constant(value) = signal.representation.clone() else {
        panic!("constant fixture")
    };
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![
            SignalKnot::continuous(0.0, SignalOrdinate::Real(value.clone())),
            SignalKnot::continuous(1.0, SignalOrdinate::Real(value)),
        ],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    let facts = field_facts(&workload);
    assert_eq!(
        facts.field_payload_bytes.0,
        64 * 7 + 6 * std::mem::size_of::<usize>() as u64
    );
    let signal = field_signal(&mut workload);
    let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
        unreachable!()
    };
    let SignalKnotValue::Continuous(SignalOrdinate::Real(real)) = table.knots[0].value.clone()
    else {
        unreachable!()
    };
    table.knots[0].value = SignalKnotValue::Continuous(SignalOrdinate::Complex {
        real: real.clone(),
        imaginary: real,
    });
    assert_eq!(
        field_facts(&workload).field_payload_bytes,
        facts.field_payload_bytes
    );
}

#[test]
fn element_scalar_field_is_included_in_both_order_exact_preassembly_ceiling() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut workload = element_field(mesh.topology.volume_elements.len());
        let SignalShape::TopologyField { topology_id, .. } = &mut field_signal(&mut workload).shape
        else {
            unreachable!()
        };
        *topology_id =
            runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string();
        let facts = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &workload.solve_schedule().unwrap(),
            &crate::admission::admit_nonstructural(
                PhysicsFamily::Thermal,
                workload.procedure.procedure(),
                Some(&mesh),
                SourceImplementation::CanonicalVolumeProjection,
                BoundaryTransferImplementation::CanonicalBoundaryProjection,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(facts.realization.field_payload_bytes.0 > 0);
        let required = facts.reservation_bytes().unwrap().0;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required - 1);
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling { required: actual,.. })) if actual == required)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required);
        workload.execute(&mesh).unwrap();
    }
}

#[test]
fn field_envelope_rejects_malformed_shapes_locations_values_and_unmaterialized_sources() {
    for failure in 0..7 {
        let mut workload = element_field(3);
        let signal = field_signal(&mut workload);
        match failure {
            0 => {
                let SignalShape::TopologyField { location, .. } = &mut signal.shape else {
                    unreachable!()
                };
                *location = FieldLocation::Node;
            }
            1 => {
                let SignalShape::TopologyField {
                    component_count, ..
                } = &mut signal.shape
                else {
                    unreachable!()
                };
                *component_count = 2;
            }
            2 => {
                let SignalShape::TopologyField { topology_id, .. } = &mut signal.shape else {
                    unreachable!()
                };
                *topology_id = "foreign".into();
            }
            3 => {
                let SignalRepresentation::Constant(SignalValue::Tensor { shape, .. }) =
                    &mut signal.representation
                else {
                    unreachable!()
                };
                shape[0] = 4;
            }
            4 => {
                let SignalRepresentation::Constant(SignalValue::Tensor { values, .. }) =
                    &mut signal.representation
                else {
                    unreachable!()
                };
                values[0] = HeatGenerationDensity::new(f64::NAN);
            }
            5 => {
                signal.representation = SignalRepresentation::Constant(SignalValue::Scalar(
                    HeatGenerationDensity::new(2.0),
                ));
            }
            _ => {
                signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
                    domain: TabulatedSignalDomain::FrequencyHertz,
                    knots: Vec::new(),
                    interpolation: SignalInterpolation::Linear,
                    extrapolation: SignalExtrapolation::Reject,
                });
            }
        }
        assert!(ThermalRealizationResourceFacts::derive(
            &workload.model,
            &workload.model.sequence.steps[0]
        )
        .is_err());
    }
    let mut malformed = element_field(3);
    let signal = field_signal(&mut malformed);
    let SignalRepresentation::Constant(value) = signal.representation.clone() else {
        unreachable!()
    };
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![SignalKnot::continuous(
            0.0,
            SignalOrdinate::Complex {
                real: value,
                imaginary: SignalValue::Tensor {
                    shape: vec![2, 1],
                    values: vec![HeatGenerationDensity::new(1.0); 2],
                },
            },
        )],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    assert!(ThermalRealizationResourceFacts::derive(
        &malformed.model,
        &malformed.model.sequence.steps[0]
    )
    .is_err());
}

#[test]
fn inventory_counts_all_actions_and_each_case_membership_without_copying_selection() {
    let mut workload = workload();
    let mut duplicate = workload.model.excitation_cases[0].clone();
    duplicate.case_id.0 = "second_case".repeat(512);
    workload.model.excitation_cases.push(duplicate);
    let step = &workload.model.sequence.steps[0];
    let facts = ThermalRealizationResourceFacts::derive(&workload.model, step).unwrap();
    let realized =
        realize_step_excitations(&workload.model, step, ExcitationSamplePoint::time(0.0)).unwrap();
    assert_eq!(facts.active_actions.0, realized.len() as u64);
    assert_eq!(
        facts.case_memberships.0,
        realized
            .iter()
            .map(|entry| entry.case_ids.len() as u64)
            .sum::<u64>()
    );
    assert!(facts.case_memberships.0 > facts.active_actions.0);
    let selected = active_actions(&workload.model, Some(step))
        .unwrap()
        .next()
        .unwrap();
    assert!(std::ptr::eq(
        selected.case_ids().next().unwrap(),
        &workload.model.excitation_cases[0].case_id
    ));
    let source = super::super::ThermalSourceEvidenceResourceFacts::derive(
        &workload.model,
        step,
        &workload.solve_schedule().unwrap(),
    )
    .unwrap();
    assert_eq!(source.records_per_sample.0, 0); // Structural load metadata still realizes.
    assert!(facts.active_actions.0 > 0);
    assert_eq!(
        facts.vector_bytes.0,
        facts.active_actions.0 * std::mem::size_of::<RealizedExcitation>() as u64
            + facts.case_memberships.0 * std::mem::size_of::<ExcitationCaseId>() as u64
    );
    workload.model.excitation_cases[1].activation =
        ExcitationCaseActivation::Steps(vec!["other".into()]);
    let reduced =
        ThermalRealizationResourceFacts::derive(&workload.model, &workload.model.sequence.steps[0])
            .unwrap();
    assert_eq!(reduced.active_actions, facts.active_actions);
    assert!(reduced.metadata_estimated_bytes.0 < facts.metadata_estimated_bytes.0);
    assert!(reduced.case_memberships.0 < facts.case_memberships.0);
    // Resolved target paths are cloned by realization and must affect metadata admission.
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut entity = mesh.topology.boundary_faces[0].provenance[0].clone();
    entity
        .assembly_path
        .extend((0..100).map(|index| format!("component_{index}_{}", "x".repeat(180))));
    let mut target =
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("resolved_face"));
    target.bind_resolved_entities(vec![entity]).unwrap();
    let ExcitationAction::Force(action) = &mut workload.model.excitations[0] else {
        panic!("force fixture")
    };
    action.target = ExcitationTarget::Region(target);
    let nested =
        ThermalRealizationResourceFacts::derive(&workload.model, &workload.model.sequence.steps[0])
            .unwrap();
    assert_eq!(nested.vector_bytes, reduced.vector_bytes);
    assert!(nested.metadata_estimated_bytes.0 > reduced.metadata_estimated_bytes.0);
}

#[test]
fn realization_budget_is_composed_and_enforced_for_both_element_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let facts = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &workload.solve_schedule().unwrap(),
            &crate::admission::admit_nonstructural(
                runmat_analysis_core::PhysicsFamily::Thermal,
                workload.procedure.procedure(),
                Some(&mesh),
                runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
                runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
            )
            .unwrap(),
        )
        .unwrap();
        let required = facts.reservation_bytes().unwrap().0;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes =
            Some(required - facts.realization.reservation_bytes().unwrap().0);
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling { required: actual, .. })) if actual == required)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(required);
        workload.execute(&mesh).unwrap();
    }
}

#[test]
fn lookup_includes_unused_signals_and_realization_rejects_duplicate_ids() {
    let mut workload = workload();
    let base =
        ThermalRealizationResourceFacts::derive(&workload.model, &workload.model.sequence.steps[0])
            .unwrap();
    let mut unused = workload.model.signals[0].clone();
    let PhysicalSignal::Force(signal) = &mut unused else {
        panic!("force fixture")
    };
    signal.signal_id.0 = "unused_signal".into();
    workload.model.signals.push(unused);
    let step = &workload.model.sequence.steps[0];
    let grown = ThermalRealizationResourceFacts::derive(&workload.model, step).unwrap();
    assert_eq!(
        grown.signal_lookup_bytes.0,
        base.signal_lookup_bytes.0 + std::mem::size_of::<&PhysicalSignal>() as u64
    );
    assert_eq!(
        grown.metadata_estimated_bytes,
        base.metadata_estimated_bytes
    );
    realize_step_excitations(&workload.model, step, ExcitationSamplePoint::time(0.0)).unwrap();
    workload
        .model
        .signals
        .push(workload.model.signals[0].clone());
    assert!(realize_step_excitations(
        &workload.model,
        &workload.model.sequence.steps[0],
        ExcitationSamplePoint::time(0.0)
    )
    .is_err());
    let mut overflow = grown;
    overflow.vector_bytes = ByteCount(u64::MAX);
    assert!(overflow.reservation_bytes().is_err());
    assert!(scale(u64::MAX, 2).is_err());
}
