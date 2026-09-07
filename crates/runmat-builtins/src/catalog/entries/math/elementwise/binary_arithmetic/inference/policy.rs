#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) struct BinaryArithmeticInferencePolicy
{
    pub real_result_domain: RealResultDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) enum RealResultDomain {
    Real,
    RuntimeDependent,
}

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const REAL_RESULT:
    BinaryArithmeticInferencePolicy = BinaryArithmeticInferencePolicy {
    real_result_domain: RealResultDomain::Real,
};

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const RUNTIME_DEPENDENT_RESULT: BinaryArithmeticInferencePolicy =
    BinaryArithmeticInferencePolicy {
        real_result_domain: RealResultDomain::RuntimeDependent,
    };
