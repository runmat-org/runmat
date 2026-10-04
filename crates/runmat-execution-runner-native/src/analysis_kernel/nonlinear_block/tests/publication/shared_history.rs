//! Equal independently decoded histories are not shared until full-content comparison succeeds.
use runmat_analysis_fea::execution::{
    semantic_state_retained_storage_bytes, CanonicalSemanticState, CanonicalSemanticStatePayload,
};
use runmat_canonical_codec::CanonicalLimits;
use std::sync::Arc;

fn payload(
    state: &CanonicalSemanticState,
) -> &Arc<runmat_analysis_fea::execution::CanonicalNonlinearSolidStatePayload> {
    let CanonicalSemanticStatePayload::NonlinearSolid { state } = &state.payload else {
        panic!("original nonlinear history required");
    };
    state
}

pub(super) fn verify(original: &CanonicalSemanticState) {
    let wire = original.canonical_bytes().unwrap();
    let mut cold = CanonicalSemanticState::from_canonical_bytes(&wire).unwrap();
    assert!(!Arc::ptr_eq(payload(original), payload(&cold)));
    let limits = CanonicalLimits::new(128 * 1024 * 1024, 100_000_000, 65_536, 64);
    let separate =
        semantic_state_retained_storage_bytes([original, &cold].into_iter(), limits).unwrap();
    cold.share_original_nonlinear_payload(original);
    assert!(Arc::ptr_eq(payload(original), payload(&cold)));
    assert_eq!(cold.canonical_bytes().unwrap(), wire);
    cold.validate().unwrap();
    let shared =
        semantic_state_retained_storage_bytes([original, &cold].into_iter(), limits).unwrap();
    assert!(shared < separate);
    assert!(
        shared > semantic_state_retained_storage_bytes([original].into_iter(), limits).unwrap()
    );

    let mut foreign = CanonicalSemanticState::from_canonical_bytes(&wire).unwrap();
    let source = foreign.source_result;
    foreign.source_result = runmat_analysis_fea::execution::SolverResultIdentity::from_step_result(
        original.producer,
        runmat_execution::Digest::sha256(b"another-original-source"),
    )
    .unwrap();
    foreign.share_original_nonlinear_payload(original);
    assert!(!Arc::ptr_eq(payload(original), payload(&foreign)));
    assert!(foreign.validate().is_err());
    foreign.source_result = source;
    let CanonicalSemanticStatePayload::NonlinearSolid { state } = &mut foreign.payload else {
        unreachable!();
    };
    Arc::make_mut(state).block.request.block_ordinal += 1;
    foreign.share_original_nonlinear_payload(original);
    assert!(!Arc::ptr_eq(payload(original), payload(&foreign)));
    assert!(foreign.validate().is_err());
}
