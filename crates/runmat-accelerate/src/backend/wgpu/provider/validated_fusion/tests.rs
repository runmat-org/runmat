use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn validated_publication_is_one_winner_and_cancelled_candidate_is_absent() {
    let cache = std::sync::Mutex::new(std::collections::HashMap::new());
    let cancelled = Arc::new(11_u8);
    drop(cancelled);
    assert!(cache.lock().unwrap().is_empty());
    let first = Arc::new(21_u8);
    let (winner, cold) = select_validated_winner(&cache, 7, first.clone()).unwrap();
    assert!(cold);
    assert!(Arc::ptr_eq(&winner, &first));
    let other = Arc::new(22_u8);
    let (winner, cold) = select_validated_winner(&cache, 7, other.clone()).unwrap();
    assert!(!cold);
    assert!(Arc::ptr_eq(&winner, &first));
    assert!(!Arc::ptr_eq(&winner, &other));
    assert_eq!(cache.lock().unwrap().len(), 1);
}
