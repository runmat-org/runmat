//! Canonical distribution notice bytes for Runtime-derived native products.

pub const FILE_NAME: &str = "THIRD_PARTY_NOTICES.txt";

pub fn bytes() -> &'static [u8] {
    include_bytes!("builtins/math/linalg/ops/linear_operator/LAPACK-LICENSE.txt")
}
