use super::*;

#[test]
fn long_named_complex_volume_flow_evidence_and_cold_result_fit_admitted_payload() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let mut workload = workload();
        workload.model.constraints.clear();
        let signal_id = SignalId("volume_flow_drive_".repeat(1000));
        let action_id = ExcitationId("volume_velocity_source_".repeat(1000));
        workload
            .model
            .signals
            .push(PhysicalSignal::VolumeFlowRate(Signal {
                signal_id: signal_id.clone(),
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
                                    real: SignalValue::Scalar(VolumeFlowRate::new(-1.0 / 6.0)),
                                    imaginary: SignalValue::Scalar(VolumeFlowRate::new(-0.4 / 6.0)),
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
            .push(ExcitationAction::VolumeVelocity(AnalysisExcitation {
                excitation_id: action_id.clone(),
                target: ExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("slab_left"),
                )),
                frame: CoordinateFrame::BoundaryNormalTangent,
                direction: None,
                distribution: ExcitationDistribution::PreserveResultant,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: signal_id.clone(),
                active_step_ids: Vec::new(),
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1e-10,
                },
            }));
        workload.model.excitation_cases.push(ExcitationCase {
            case_id: ExcitationCaseId("volume_flow_case_".repeat(1000)),
            action_ids: vec![action_id.clone()],
            activation: ExcitationCaseActivation::Always,
        });
        let cold_workload =
            AcousticHarmonicWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(cold_workload, workload);
        let prepared = facts(&cold_workload, &mesh);
        build_acoustic_resource_plan(&cold_workload, &mesh, prepared.clone()).unwrap();

        // This is the sole numerical solve. The retained result is subsequently
        // decoded and authenticated without reconstructing its numerical fields.
        let result = cold_workload.execute(&mesh).unwrap();
        assert_eq!(result.source_evidence().len(), 1);
        let evidence = &result.source_evidence()[0];
        assert_eq!(evidence.quantity, PhysicalQuantityKind::VolumeFlowRate);
        assert_eq!(evidence.excitation_id, action_id.0);
        assert_eq!(evidence.signal_id, signal_id.0);
        assert!(evidence.accepted);
        evidence.validate().unwrap();
        let bytes = result.canonical_bytes().unwrap();
        assert!(bytes.len() as u64 <= prepared.result_output.encoded_payload_bytes_upper_bound.0);
        let imported =
            crate::execution::AcousticHarmonicSolverResult::from_canonical_bytes(&bytes).unwrap();
        imported.validate_for(&workload, &mesh).unwrap();
        assert_eq!(imported, result);
        assert_eq!(imported.canonical_bytes().unwrap(), bytes);
        assert_eq!(imported.source_workload(), workload.identity().unwrap());
        assert_eq!(
            imported.source_result_identity().unwrap(),
            result.source_result_identity().unwrap()
        );

        // A valid but differently authored complex flow is foreign lineage,
        // even though it uses the same quantity, boundary and canonical mesh.
        let mut foreign_workload = workload.clone();
        let PhysicalSignal::VolumeFlowRate(signal) =
            foreign_workload.model.signals.last_mut().unwrap()
        else {
            panic!("expected the authored volume-flow signal");
        };
        let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
            panic!("expected the authored complex frequency table");
        };
        table.knots[0] = SignalKnot::continuous(
            0.01,
            SignalOrdinate::Complex {
                real: SignalValue::Scalar(VolumeFlowRate::new(-2.0 / 6.0)),
                imaginary: SignalValue::Scalar(VolumeFlowRate::new(-0.4 / 6.0)),
            },
        );
        assert!(imported.validate_for(&foreign_workload, &mesh).is_err());
        let foreign_mesh = slab_mesh(3);
        assert!(imported.validate_for(&workload, &foreign_mesh).is_err());
    }
}

#[test]
fn incident_projection_metadata_and_cold_result_fit_admitted_payload() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let mut workload = workload();
        workload.model.constraints.clear();
        let signal_id = SignalId("incident_drive_".repeat(1000));
        let action_id = ExcitationId("incident_source_".repeat(1000));
        workload
            .model
            .signals
            .push(PhysicalSignal::Pressure(Signal {
                signal_id: signal_id.clone(),
                shape: SignalShape::Scalar,
                frame: None,
                representation: SignalRepresentation::Tabulated(TabulatedSignal {
                    domain: TabulatedSignalDomain::FrequencyHertz,
                    knots: [0.01, 1.0]
                        .into_iter()
                        .map(|f| {
                            SignalKnot::continuous(
                                f,
                                SignalOrdinate::Complex {
                                    real: SignalValue::Scalar(Pressure::new(1.0)),
                                    imaginary: SignalValue::Scalar(Pressure::new(0.4)),
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
            .push(ExcitationAction::AcousticIncidentPressure(
                AnalysisExcitation {
                    excitation_id: action_id.clone(),
                    target: ExcitationTarget::Region(RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_face("slab_left"),
                    )),
                    frame: CoordinateFrame::BoundaryNormalTangent,
                    direction: None,
                    distribution: ExcitationDistribution::UniformKinematic,
                    kinematics: ExcitationKinematics::FixedFrame,
                    signal_id,
                    active_step_ids: vec![],
                    combination_group: None,
                    projection: ProjectionRequirement {
                        preserve_resultant: true,
                        preserve_moment: false,
                        relative_tolerance: 1e-10,
                    },
                },
            ));
        workload.model.excitation_cases.push(ExcitationCase {
            case_id: ExcitationCaseId("incident".into()),
            action_ids: vec![action_id],
            activation: ExcitationCaseActivation::Always,
        });
        let cold =
            AcousticHarmonicWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(cold, workload);
        let prepared = facts(&cold, &mesh);
        build_acoustic_resource_plan(&cold, &mesh, prepared.clone()).unwrap();
        let result = cold.execute(&mesh).unwrap();
        assert_eq!(result.source_evidence().len(), 1);
        let bytes = result.canonical_bytes().unwrap();
        assert!(bytes.len() as u64 <= prepared.result_output.encoded_payload_bytes_upper_bound.0);
        let imported =
            crate::execution::AcousticHarmonicSolverResult::from_canonical_bytes(&bytes).unwrap();
        imported.validate_for(&workload, &mesh).unwrap();
        assert_eq!(imported, result);
        assert_eq!(
            imported.source_result_identity().unwrap(),
            result.source_result_identity().unwrap()
        );
    }
}

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
                target: ExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("slab_left"),
                )),
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
