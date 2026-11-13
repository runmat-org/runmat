use super::{build_two_pass_shaders, two_pass_tile_bytes};
use crate::backend::wgpu::types::NumericPrecision;

#[test]
fn two_pass_accumulator_tile_keeps_selected_precision_through_both_passes() {
    for (precision, scalar, element_bytes) in [
        (NumericPrecision::F32, "f32", 4u64),
        (NumericPrecision::F64, "f64", 8u64),
    ] {
        let (first, second) = build_two_pass_shaders(precision, 256);
        for shader in [&first, &second] {
            assert!(shader.contains(&format!("var<workgroup> tile: array<{scalar},256>")));
            assert!(shader.contains(&format!("var acc: {scalar}")));
            assert!(!shader.contains("@IEEE_NAN_PREDICATE@"));
            assert!(!shader.contains("{cast}"));
        }
        assert!(first.contains("partials.data[(slice * params.chunks) + chunk] = tile[0u]"));
        assert!(second.contains("output.data[slice] = (tile[0u]) * params.scale"));
        assert_eq!(
            two_pass_tile_bytes(precision, 256),
            Some(256 * element_bytes)
        );
    }
    let one_kib_limit = 1024u64;
    assert!(two_pass_tile_bytes(NumericPrecision::F32, 256).unwrap() <= one_kib_limit);
    assert!(two_pass_tile_bytes(NumericPrecision::F64, 256).unwrap() > one_kib_limit);
}
