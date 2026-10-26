use super::*;
use crate::excitation::{field::NodalVectorField, ExcitationSamplePoint, RealizedExcitationKind};
use runmat_analysis_core::{
    BodyForceDensity, CoordinateFrame, FieldLocation, Signal, SignalApproximationEvidence,
    SignalId, SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape, SignalValue,
};

#[test]
fn count_only_encoding_projection_bounds_actual_frozen_fields_and_all_sample_metadata() {
    let digest = Digest::sha256("mesh");
    for rows in [4, 10, 128, 1024] {
        let signal = Signal {
            signal_id: SignalId("density".into()),
            shape: SignalShape::TopologyField {
                topology_id: digest.to_string(),
                location: FieldLocation::Node,
                component_count: 3,
            },
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Constant(SignalValue::Tensor {
                shape: vec![rows, 3],
                values: vec![BodyForceDensity::new(2.0); rows * 3],
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
        let kind = RealizedExcitationKind::BodyForceField(
            NodalVectorField::sample(&signal, ExcitationSamplePoint::Static, |value| {
                value.canonical_value()
            })
            .unwrap(),
        );
        let projected = nodal_sample_encoding_workspace(digest, rows).unwrap();
        for sample in [
            ProjectionSample::Static,
            ProjectionSample::TimeSeconds {
                value: 0.25,
                approach: SignalSampleApproach::RightLimit,
            },
            ProjectionSample::FrequencyHertz {
                value: 20.0,
                approach: SignalSampleApproach::LeftLimit,
            },
            ProjectionSample::StudyParameter {
                parameter_id: "p".repeat(1024),
                canonical_value: 2.0,
                approach: SignalSampleApproach::LeftLimit,
            },
        ] {
            let value = (PhysicalQuantityKind::BodyForceDensity, sample, &kind);
            let actual_profile = profile_serialized(&value, LIMITS).unwrap();
            assert!(projected >= actual_profile.materialization_resident_bytes_estimate);
            let encoded = encode_contract(PREFIX, DOMAIN, &value, LIMITS).unwrap();
            assert!(projected >= runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(&encoded).unwrap());
        }
    }
}

#[test]
fn encoding_projection_rejects_fixed_collection_limit_without_allocating_rows() {
    assert!(nodal_sample_encoding_workspace(Digest::sha256("mesh"), 1025).is_err());
    assert!(nodal_sample_encoding_workspace(Digest::sha256("mesh"), usize::MAX).is_err());
}
