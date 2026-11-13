use super::*;
use crate::{encode_contract, materialization_resident_bytes_estimate, CanonicalLimits};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn every_chunk_boundary_preserves_independent_structural_inventory() {
    let value = serde_json::json!({"array": [null, true, -4, 0.125, u64::MAX], "text": "é𝄞"});
    let bytes = encode_contract(
        b"test-prefix\0",
        "unicode-domain",
        &value,
        CanonicalLimits::new(4096, 100, 100, 8),
    )
    .unwrap();
    let expected = MaterializationProfile {
        value_nodes: 8,
        map_entries: 2,
        string_bytes: 15,
        heap_containers: 2,
        maps: 1,
        value_string_bytes: 6,
    };
    for split in 0..=bytes.len() {
        let (profile, length) =
            chunked_profile([&bytes[..split], &[], &bytes[split..]], &NeverCancelled).unwrap();
        assert_eq!(profile, expected);
        assert_eq!(length, bytes.len() as u64);
    }
    let (profile, length) = chunked_profile(bytes.chunks(1), &NeverCancelled).unwrap();
    assert_eq!(profile, expected);
    assert_eq!(
        materialization_resident_bytes_estimate(profile, length).unwrap(),
        crate::canonical_payload_materialization_resident_bytes_estimate(&bytes).unwrap()
    );
}

#[test]
fn dense_numeric_inventory_uses_original_nodes_and_bytes() {
    let values = vec![0.125_f64; 100_000];
    let bytes = encode_contract(
        b"dense\0",
        "values",
        &values,
        CanonicalLimits::new(2_000_000, 200_000, 128, 8),
    )
    .unwrap();
    let (profile, length) = chunked_profile(bytes.chunks(257), &NeverCancelled).unwrap();
    assert_eq!(
        (
            profile.value_nodes,
            profile.map_entries,
            profile.string_bytes
        ),
        (100_001, 0, 0)
    );
    assert_eq!(length, bytes.len() as u64);
}

#[test]
fn malformed_chunked_values_reject_without_gathering() {
    for payload in [
        vec![0x9f, 0xff],                                           // indefinite array
        vec![0xbf, 0xff],                                           // indefinite map
        vec![0x7f, 0xff],                                           // indefinite string
        vec![0x42, 1, 2],                                           // unsupported bytes
        vec![0xc0, 0],                                              // unsupported tag
        vec![0x61, 0xff],                                           // malformed UTF-8
        vec![0x62, 0xc0, 0x80],                                     // overlong UTF-8
        vec![0x63, 0xed, 0xa0, 0x80],                               // surrogate
        vec![0x61, 0xc2], // incomplete scalar within declared string
        vec![0x78, 255],  // truncated long string
        vec![0x3b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff], // negative beyond i64
        vec![0, 0],       // trailing value
    ] {
        let bytes = [b"test\0\x82\x61x".as_slice(), &payload].concat();
        assert!(
            chunked_profile(bytes.chunks(1), &NeverCancelled).is_err(),
            "{payload:?}"
        );
    }
    assert!(chunked_profile([b"missing-terminator".as_slice()], &NeverCancelled).is_err());
    let bytes = [b"test\0\x82\x61x".as_slice(), &[0x81; 129], &[0]].concat();
    assert!(chunked_profile(bytes.chunks(1), &NeverCancelled).is_err());
}

#[test]
fn cancellation_stops_the_chunk_iterator_before_complete_input() {
    struct Cancel {
        calls: AtomicUsize,
    }
    impl CanonicalCodecControl for Cancel {
        fn is_cancelled(&self) -> bool {
            self.calls.fetch_add(1, Ordering::Relaxed) >= 3
        }
    }
    let cancelled = Cancel {
        calls: AtomicUsize::new(0),
    };
    let mut read = 0;
    let chunks = std::iter::repeat_n(b"x".as_slice(), 1000).inspect(|_| read += 1);
    assert!(chunked_profile(chunks, &cancelled).is_err());
    assert!(read < 1000);
}

#[test]
fn cancellation_interrupts_one_large_string_chunk() {
    struct Cancel(AtomicUsize);
    impl CanonicalCodecControl for Cancel {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) >= 5
        }
    }
    let bytes = vec![b'x'; 100_000];
    let control = Cancel(AtomicUsize::new(0));
    let mut reader = Reader {
        chunks: std::iter::empty(),
        current: &bytes,
        position: 0,
        control: &control,
    };
    let error = reader.utf8(bytes.len() as u64).unwrap_err();
    assert!(error.reason.contains("cancel"));
    assert!(reader.position > 0 && reader.position <= 6 * 1024);
    assert!(!reader.current.is_empty());
}

#[test]
fn admission_value_and_envelope_share_the_same_independent_structural_inventory() {
    let original = serde_json::json!({"array": [null, true, -4, 0.125, u64::MAX], "text": "é𝄞"});
    let limits = CanonicalLimits::new(4096, 100, 100, 8);
    let encoded = encode_contract(b"test-prefix\0", "unicode-domain", &original, limits).unwrap();
    let mut admitted = false;
    let decoded: serde_json::Value = crate::decode_contract_with_admission(
        b"test-prefix\0",
        "unicode-domain",
        &encoded,
        limits,
        &NeverCancelled,
        |value| {
            let (profile, bytes) = value_profile(value, &NeverCancelled)?;
            assert_eq!(
                (
                    profile.value_nodes,
                    profile.map_entries,
                    profile.string_bytes
                ),
                (8, 2, 15)
            );
            assert_eq!(bytes, value.len() as u64);
            let value_estimate = crate::canonical_value_materialization_resident_bytes_estimate(
                value,
                &NeverCancelled,
            )?;
            let envelope_estimate =
                crate::canonical_payload_materialization_resident_bytes_estimate(&encoded)?;
            assert_eq!(
                envelope_estimate - value_estimate,
                2 * (encoded.len() - value.len()) as u64
            );
            admitted = true;
            Ok(())
        },
    )
    .unwrap();
    assert!(admitted);
    assert_eq!(decoded, original);
    assert!(
        crate::canonical_value_materialization_resident_bytes_estimate(&[0x18], &NeverCancelled)
            .is_err()
    );
    assert!(
        crate::canonical_value_materialization_resident_bytes_estimate(
            &[0x01, 0x01],
            &NeverCancelled
        )
        .is_err()
    );
}
