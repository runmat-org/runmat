use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn compute_backend_identities_conform() {
    assert_family(&[
        (
            &cpu::FEA_COMPUTE_BACKEND_CPU_CATALOG_ENTRY,
            &cpu::POLICY_CALL,
        ),
        (
            &gpu::FEA_COMPUTE_BACKEND_GPU_CATALOG_ENTRY,
            &gpu::POLICY_CALL,
        ),
    ]);
}
