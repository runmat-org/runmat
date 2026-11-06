use super::*;
use crate::planning::resource::{
    PreparedInputResourceFacts, PreparedOutputResourceFacts, PreparedOutputSetResourceFacts,
};
mod governed;

fn prepared() -> PreparedPhaseResourceFacts {
    PreparedPhaseResourceFacts {
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(100),
            transport_closure_bytes: ByteCount(140),
            decoded_resident_bytes_admission_budget: ByteCount(300),
        },
        output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(200),
            transport_closure_bytes_upper_bound: ByteCount(240),
            construction_resident_bytes_admission_budget: ByteCount(500),
        },
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(30),
        },
    }
}

#[test]
fn original_wire_requires_exact_bytes_and_actual_decode_not_maximum_width() {
    let facts = prepared();
    admission::validate_prepared(100, 300, 200, 500, &facts).unwrap();
    let mut short = facts.clone();
    short.inputs.decoded_resident_bytes_admission_budget.0 -= 1;
    assert!(admission::validate_prepared(100, 300, 200, 500, &short).is_err());
    let mut substituted = facts.clone();
    substituted.inputs.encoded_payload_bytes.0 += 1;
    assert!(admission::validate_prepared(100, 300, 200, 500, &substituted).is_err());
    let mut substituted = facts;
    substituted.output.encoded_payload_bytes_upper_bound.0 += 1;
    assert!(admission::validate_prepared(100, 300, 200, 500, &substituted).is_err());
}

#[test]
fn incomplete_transport_or_construction_never_grants_admission() {
    let mut facts = prepared();
    facts.inputs.transport_closure_bytes = ByteCount(99);
    assert!(admission::validate_prepared(100, 300, 200, 500, &facts).is_err());
    let mut facts = prepared();
    facts.output.transport_closure_bytes_upper_bound = ByteCount(199);
    assert!(admission::validate_prepared(100, 300, 200, 500, &facts).is_err());
    let mut facts = prepared();
    facts.output.construction_resident_bytes_admission_budget = ByteCount(499);
    assert!(admission::validate_prepared(100, 300, 200, 500, &facts).is_err());
    let mut facts = prepared();
    facts.output_set.encoded_bytes_upper_bound = ByteCount(0);
    assert!(admission::validate_prepared(100, 300, 200, 500, &facts).is_err());
}

#[test]
fn sequential_lifetimes_exact_ceiling_and_overflow_are_checked() {
    // Retained 30 + decoded 50 live throughout; then the largest of scratch 40, the solve
    // 100, and encoding (verification 20 + output 2 x 5 + 7 = 37).
    let bytes = admission::phase_resident_bytes(100, 20, 30, 40, 50, 5, 7).unwrap();
    assert_eq!(bytes, 180);
    // Scratch above the solve is the peak; then encoding above both.
    assert_eq!(
        admission::phase_resident_bytes(10, 5, 30, 40, 50, 0, 0).unwrap(),
        120
    );
    assert_eq!(
        admission::phase_resident_bytes(10, 5, 30, 4, 50, 10, 20).unwrap(),
        125
    );
    admission::require_ceiling(bytes, bytes).unwrap();
    assert!(matches!(
        admission::require_ceiling(bytes, bytes - 1),
        Err(ResourcePlanningError::AccuracyCeiling { .. })
    ));
    assert!(matches!(
        admission::phase_resident_bytes(0, 0, u64::MAX, 0, 1, 0, 0),
        Err(ResourcePlanningError::Overflow(_))
    ));
    assert!(admission::require_ceiling(u64::MAX, u64::MAX).is_err());
}
