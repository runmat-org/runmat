use super::*;
use runmat_canonical_codec::encode_contract;

#[test]
fn controlled_codec_retains_original_bytes_and_cold_values() {
    let original = vec!["α".repeat(512), "original".into()];
    let bytes = encode("codec-witness", &original).unwrap();
    assert_eq!(
        bytes,
        encode_contract(PLAN_PREFIX, "codec-witness", &original, PLAN_LIMITS).unwrap()
    );
    assert_eq!(
        decode::<Vec<String>>("codec-witness", &bytes).unwrap(),
        original
    );
    assert!(matches!(
        decode::<Vec<String>>("wrong-domain", &bytes),
        Err(ConstraintPlanError::Encoding(_))
    ));
}

#[test]
fn cancelled_identity_and_cold_decode_return_typed_cancellation() {
    let bytes = encode("codec-witness", &vec![1u32, 2, 3]).unwrap();
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        encode("codec-witness", &vec![1u32, 2, 3]).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        decode::<Vec<u32>>("codec-witness", &bytes).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        crate::constraints::plan::canonical_digest("codec-witness", &vec![1u32, 2, 3]).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}

#[test]
fn cancellation_raised_by_serialization_prevents_completed_identity() {
    struct CancelAfterValue(crate::progress::FeaCancellationToken);
    impl Serialize for CancelAfterValue {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let result = serializer.serialize_u32(42);
            self.0.cancel();
            result
        }
    }
    for profile_only in [true, false] {
        let token = crate::progress::FeaCancellationToken::new();
        let _guard =
            crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
                cancellation: token.clone(),
                ..Default::default()
            });
        let value = CancelAfterValue(token);
        if profile_only {
            assert_eq!(
                crate::constraints::plan::canonical_value_workspace_bytes(&value).unwrap_err(),
                ConstraintPlanError::Cancelled
            );
        } else {
            assert_eq!(
                crate::constraints::plan::canonical_digest("codec-witness", &value).unwrap_err(),
                ConstraintPlanError::Cancelled
            );
        }
    }
}
