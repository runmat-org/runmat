use super::is_scalar_shape;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn empty_and_overflowed_shapes_never_enter_scalar_dispatch() {
    assert!(is_scalar_shape(&[1, 1]));
    assert!(!is_scalar_shape(&[0, 1]));
    assert!(!is_scalar_shape(&[usize::MAX, 2]));
    assert!(!is_scalar_shape(&[usize::MAX, 2, 0]));
}
