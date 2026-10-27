//! Canonical distribution notice bytes for Runtime-derived native products.

pub const FILE_NAME: &str = "THIRD_PARTY_NOTICES.txt";

pub fn bytes() -> &'static [u8] {
    concat!(
        include_str!("builtins/math/linalg/ops/linear_operator/LAPACK-LICENSE.txt"),
        "\n",
        include_str!("builtins/math/linalg/ops/sparse_factor/managed_lu/FAER-LICENSE.txt"),
        "\n",
        include_str!("../licenses/UNICODE-LICENSE.txt"),
        "\n",
        include_str!("../licenses/CLARABEL-LICENSE.txt"),
        "\n",
        include_str!("../licenses/CLARABEL-DEPENDENCIES-NOTICES.txt"),
        "\n",
        include_str!("../licenses/SOBOL-NOTICES.txt"),
    )
    .as_bytes()
}
