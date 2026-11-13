use super::*;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};

const LIMITS: CanonicalLimits = CanonicalLimits::new(1_000_000, 10_000, 100_000, 32);

fn evidence(sample: ProjectionSample) -> SpatialProjectionEvidence {
    SpatialProjectionEvidence {
        excitation_id: "heat.α".repeat(100),
        signal_id: "signal.β".repeat(100),
        quantity: PhysicalQuantityKind::HeatFlux,
        sample,
        analysis_projection_digest: Digest::sha256(b"actual projection"),
        integration_rule: SpatialIntegrationRule::BoundaryDunavantSeven,
        resolved_entity_count: usize::MAX,
        resolved_node_count: usize::MAX,
        reference_point_m: [f64::MAX, -f64::MAX, 0.0],
        balance: SpatialProjectionBalance::ScalarIntegral {
            requested_integral: ProjectionScalar {
                real: 1.0,
                imaginary: 0.0,
            },
            projected_integral: ProjectionScalar {
                real: 1.0,
                imaginary: 0.0,
            },
            normalization_scale: 1.0,
            relative_error: 0.0,
        },
        requirements: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: false,
            relative_tolerance: 1e-8,
        },
        accepted: true,
    }
}

#[test]
fn scalar_projection_profile_matches_actual_records_across_samples_and_rules() {
    for sample in [
        ProjectionSample::Static,
        ProjectionSample::TimeSeconds {
            value: f64::MAX,
            approach: SignalSampleApproach::At,
        },
        ProjectionSample::TimeSeconds {
            value: 0.0,
            approach: SignalSampleApproach::LeftLimit,
        },
        ProjectionSample::TimeSeconds {
            value: 0.0,
            approach: SignalSampleApproach::RightLimit,
        },
    ] {
        for (quantity, integration_rule) in [
            (
                PhysicalQuantityKind::HeatFlux,
                SpatialIntegrationRule::BoundaryDunavantSeven,
            ),
            (
                PhysicalQuantityKind::HeatGenerationDensity,
                SpatialIntegrationRule::VolumeKeastFifteen,
            ),
        ] {
            let mut actual = evidence(sample.clone());
            actual.quantity = quantity;
            actual.integration_rule = integration_rule;
            actual.validate().unwrap();
            let projection = SpatialProjectionEvidence::scalar_projection_serialization(
                &actual.excitation_id,
                &actual.signal_id,
                quantity,
                &sample,
                integration_rule,
                &actual.requirements,
            );
            // The projection has the evidence's exact shape; values do not matter.
            let projected =
                runmat_canonical_codec::profile_serialized_shape(&projection, LIMITS).unwrap();
            assert_eq!(
                projected,
                runmat_canonical_codec::profile_serialized_shape(&actual, LIMITS).unwrap()
            );
            let bytes = encode_contract(b"test\0", "source", &actual, LIMITS).unwrap();
            assert!(bytes.len() as u64 <= projected.encoded_bytes_upper_bound);
            assert!(
                runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(
                    &bytes
                )
                .unwrap()
                    <= projected.materialization_resident_bytes_estimate
            );
        }
    }
}

#[test]
fn shared_borrowed_wire_preserves_all_original_evidence_fields() {
    for sample in [
        ProjectionSample::Static,
        ProjectionSample::FrequencyHertz {
            value: 120.0,
            approach: SignalSampleApproach::LeftLimit,
        },
        ProjectionSample::StudyParameter {
            parameter_id: "parameter.α".to_owned(),
            canonical_value: -2.5,
            approach: SignalSampleApproach::RightLimit,
        },
    ] {
        let mut actual = evidence(sample);
        assert_original_wire(&actual);
        let resultant = ProjectionVector3 {
            real: [1.0, -2.0, 3.0],
            imaginary: [0.5, -0.25, 0.125],
        };
        let moment = ProjectionVector3 {
            real: [-3.0, 2.0, -1.0],
            imaginary: [-0.5, 0.25, -0.125],
        };
        actual.balance = SpatialProjectionBalance::VectorResultant {
            requested_resultant: resultant,
            projected_resultant: resultant,
            requested_moment: moment,
            projected_moment: moment,
            resultant_scale: 4.0,
            moment_scale: 4.0,
            resultant_relative_error: 0.0,
            moment_relative_error: 0.0,
        };
        actual.requirements.preserve_moment = true;
        actual.validate().unwrap();
        assert_original_wire(&actual);
    }
}

fn assert_original_wire(actual: &SpatialProjectionEvidence) {
    let original = serde_json::json!({
        "excitation_id": actual.excitation_id,
        "signal_id": actual.signal_id,
        "quantity": actual.quantity,
        "sample": actual.sample,
        "analysis_projection_digest": actual.analysis_projection_digest,
        "integration_rule": actual.integration_rule,
        "resolved_entity_count": actual.resolved_entity_count,
        "resolved_node_count": actual.resolved_node_count,
        "reference_point_m": actual.reference_point_m,
        "balance": actual.balance,
        "requirements": actual.requirements,
        "accepted": actual.accepted,
    });
    assert_eq!(serde_json::to_value(actual).unwrap(), original);
    assert_eq!(
        encode_contract(b"test\0", "source", actual, LIMITS).unwrap(),
        encode_contract(b"test\0", "source", &original, LIMITS).unwrap()
    );
    assert_eq!(
        serde_json::from_value::<SpatialProjectionEvidence>(original).unwrap(),
        *actual
    );
}
