#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) struct BinaryArithmeticInferencePolicy
{
    pub real_result_domain: RealResultDomain,
    pub sparse_storage: SparseStorageRule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) enum SparseStorageRule {
    PreserveMatchingSparse,
    ValueDependent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) enum RealResultDomain {
    Real,
    RuntimeDependent,
}

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const REAL_RESULT:
    BinaryArithmeticInferencePolicy = BinaryArithmeticInferencePolicy {
    real_result_domain: RealResultDomain::Real,
    sparse_storage: SparseStorageRule::ValueDependent,
};

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const MATCHING_SPARSE_REAL_RESULT:
    BinaryArithmeticInferencePolicy = BinaryArithmeticInferencePolicy {
    real_result_domain: RealResultDomain::Real,
    sparse_storage: SparseStorageRule::PreserveMatchingSparse,
};

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const RUNTIME_DEPENDENT_RESULT: BinaryArithmeticInferencePolicy =
    BinaryArithmeticInferencePolicy {
        real_result_domain: RealResultDomain::RuntimeDependent,
        sparse_storage: SparseStorageRule::ValueDependent,
    };
