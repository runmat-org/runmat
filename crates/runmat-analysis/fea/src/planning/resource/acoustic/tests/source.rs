use super::*;

#[test]
fn long_named_complex_normal_velocity_evidence_fits_actual_payload() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let mut workload = workload();
        workload.model.constraints.clear();
        // Metadata scales with actual UTF-8 authored identifiers, not a fixed string ceiling.
        let action_id = "speaker_".repeat(1000);
        let signal_id = "drive_".repeat(1000);
        workload
            .model
            .signals
            .push(PhysicalSignal::Velocity(Signal {
                signal_id: SignalId(signal_id.clone()),
                shape: SignalShape::Scalar,
                frame: None,
                representation: SignalRepresentation::Tabulated(TabulatedSignal {
                    domain: TabulatedSignalDomain::FrequencyHertz,
                    knots: [0.01, 1.0]
                        .into_iter()
                        .map(|frequency| {
                            SignalKnot::continuous(
                                frequency,
                                SignalOrdinate::Complex {
                                    real: SignalValue::Scalar(Velocity::new(-1.0 / 6.0)),
                                    imaginary: SignalValue::Scalar(Velocity::new(-0.4 / 6.0)),
                                },
                            )
                        })
                        .collect(),
                    interpolation: SignalInterpolation::Linear,
                    extrapolation: SignalExtrapolation::Reject,
                }),
                precision: SignalPrecision::Complex64,
                evidence: SignalApproximationEvidence {
                    max_absolute_error: Some(0.0),
                    max_relative_error: Some(0.0),
                    derivative_max_absolute_error: None,
                    sample_count: 2,
                },
                provenance: SignalProvenance::Authored,
            }));
        workload
            .model
            .excitations
            .push(ExcitationAction::NormalVelocity(AnalysisExcitation {
                excitation_id: ExcitationId(action_id.clone()),
                target: ExcitationTarget::Region(RegionTarget::boundary("slab_left")),
                frame: CoordinateFrame::BoundaryNormalTangent,
                direction: None,
                distribution: ExcitationDistribution::UniformKinematic,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId(signal_id),
                active_step_ids: Vec::new(),
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1e-10,
                },
            }));
        workload.model.excitation_cases.push(ExcitationCase {
            case_id: ExcitationCaseId("speaker".into()),
            action_ids: vec![ExcitationId(action_id)],
            activation: ExcitationCaseActivation::Always,
        });
        let result = workload.execute(&mesh).unwrap();
        let prepared = facts(&workload, &mesh);
        assert!(
            result.canonical_bytes().unwrap().len() as u64
                <= prepared.result_output.encoded_payload_bytes_upper_bound.0
        );
        build_acoustic_resource_plan(&workload, &mesh, prepared).unwrap();
    }
}
