use super::super::*;
use serde::Serialize;

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Shape profiles reserve complete Serde integer value ranges across CBOR-width
//   transitions while exact-value encoding profiles remain compact. Strings, collection counts,
//   variants and optional branches remain the domain caller's envelope responsibility. This codec
//   policy witness does not establish whole-task memory or numerical qualification.
#[test]
fn shape_bounds_integer_transitions_without_changing_exact_encoding_profiles() {
    #[derive(Serialize)]
    struct Shape {
        budget: u64,
        indices: [usize; 3],
        signed: i64,
    }
    let limits = CanonicalLimits::new(1 << 20, 1 << 20, 1024, 64);
    let small = Shape {
        budget: 1,
        indices: [0; 3],
        signed: 0,
    };
    let bound = profile_serialized_shape(&small, limits).unwrap();
    for value in [23, 24, 255, 256, 65535, 65536, u32::MAX as u64, u64::MAX] {
        let actual = Shape {
            budget: value,
            indices: [value as usize; 3],
            signed: i64::MIN,
        };
        assert_eq!(profile_serialized_shape(&actual, limits).unwrap(), bound);
        let exact = profile_serialized(&actual, limits).unwrap();
        assert!(exact.encoded_bytes_upper_bound <= bound.encoded_bytes_upper_bound);
        assert!(
            exact.materialization_resident_bytes_estimate
                <= bound.materialization_resident_bytes_estimate
        );
    }
    let compact = profile_canonical_encoding(&small, limits).unwrap();
    assert!(compact.encoded_bytes_upper_bound < bound.encoded_bytes_upper_bound);
}

#[test]
fn shape_map_keys_cover_signed_and_unsigned_decimal_widths() {
    let limits = CanonicalLimits::new(4096, 4096, 1024, 32);
    let small = std::collections::BTreeMap::from([(0_i64, 0_u8)]);
    let large = std::collections::BTreeMap::from([(i64::MIN, u8::MAX)]);
    let bound = profile_serialized_shape(&small, limits).unwrap();
    assert_eq!(profile_serialized_shape(&large, limits).unwrap(), bound);
    assert!(
        profile_serialized(&large, limits)
            .unwrap()
            .encoded_bytes_upper_bound
            <= bound.encoded_bytes_upper_bound
    );
    let signed = std::collections::BTreeMap::from([(i128::MIN, 0_u8)]);
    let unsigned = std::collections::BTreeMap::from([(u128::MAX, 0_u8)]);
    for profile in [
        profile_serialized_shape(&signed, limits).unwrap(),
        profile_serialized_shape(&unsigned, limits).unwrap(),
    ] {
        assert!(
            profile.encoded_bytes_upper_bound
                >= profile_serialized(&signed, limits)
                    .unwrap()
                    .encoded_bytes_upper_bound
        );
        assert!(
            profile.encoded_bytes_upper_bound
                >= profile_serialized(&unsigned, limits)
                    .unwrap()
                    .encoded_bytes_upper_bound
        );
    }
}
