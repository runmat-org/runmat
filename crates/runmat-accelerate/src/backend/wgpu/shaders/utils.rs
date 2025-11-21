pub fn is_nan_stub(scalar_ty: &str) -> &'static str {
    if scalar_ty == "f64" {
        "fn isNan(x: f64) -> bool { return x != x; }"
    } else {
        "fn isNan(x: f32) -> bool { return x != x; }"
    }
}

pub fn workgroup_tile_decl(_scalar_ty: &str, wg: u32) -> String {
    format!("var<workgroup> tile: array<f32,{}>;", wg)
}


