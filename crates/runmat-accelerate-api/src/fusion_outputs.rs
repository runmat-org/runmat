//! Class intent at the raw fused-kernel publication boundary. Final adapted
//! output types are deliberately not substituted for these physical facts.
use crate::{unsupported_provider_operation, ProviderElementType, ProviderPrecision};

/// Descriptive buffer/index ABI, not source arithmetic or allocation authority.
/// Ordinary typed shaders currently use one uniform real floating class and
/// the canonical column-major broadcast shape/stride uniform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ProviderFusedElementwiseAbi {
    BroadcastReal { precision: ProviderPrecision },
}
impl ProviderFusedElementwiseAbi {
    pub fn precision(self) -> ProviderPrecision {
        match self {
            Self::BroadcastReal { precision } => precision,
        }
    }
}

/// Request-derived physical preparation description. This does not authenticate
/// host views, issue allocations, or authorize source numerical conversions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderFusedElementwiseWorkload {
    pub abi: ProviderFusedElementwiseAbi,
    pub host_materialization_elements: u64,
    /// Peak one-at-a-time host materialization; visits above sum all copies.
    pub host_workspace_bytes: u64,
}
impl ProviderFusedElementwiseWorkload {
    pub fn checked_host_workspace_bytes(self) -> Option<u64> {
        let maximum =
            self.host_materialization_elements
                .checked_mul(match self.abi.precision() {
                    ProviderPrecision::F32 => 4,
                    ProviderPrecision::F64 => 8,
                })?;
        (self.host_workspace_bytes <= maximum).then_some(self.host_workspace_bytes)
    }
}

pub struct ProviderFusedElementwiseRequest<'a> {
    pub shader: &'a str,
    pub requirements: crate::ProviderShaderRequirements,
    pub inputs: &'a [crate::GpuTensorHandle],
    pub output_shape: &'a [usize],
    pub len: usize,
    pub output_classes: &'a [ProviderElementType],
    pub abi: ProviderFusedElementwiseAbi,
}

/// Validate before allocating output buffers or dispatching. Backend binding
/// limits are checked by that backend; no separate count can contradict this
/// list. A numeric raw output must match the executing shader precision.
pub fn validate_fused_output_classes(
    precision: ProviderPrecision,
    classes: &[ProviderElementType],
) -> anyhow::Result<()> {
    if classes.is_empty() {
        anyhow::bail!("fused output class list must not be empty");
    }
    let numeric = ProviderElementType::from(precision);
    if classes
        .iter()
        .any(|&class| class != numeric && class != ProviderElementType::Logical)
    {
        return Err(unsupported_provider_operation(
            "fused output class is unsupported at the raw kernel precision",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_hook_class_is_not_post_adaptation_result_class() {
        assert!(validate_fused_output_classes(
            ProviderPrecision::F64,
            &[ProviderElementType::F64, ProviderElementType::Logical]
        )
        .is_ok());
        let error =
            validate_fused_output_classes(ProviderPrecision::F64, &[ProviderElementType::F32])
                .unwrap_err();
        assert!(crate::is_unsupported_provider_operation(&error));
        assert!(validate_fused_output_classes(
            ProviderPrecision::F32,
            &[ProviderElementType::F32, ProviderElementType::Logical]
        )
        .is_ok());
        assert!(
            validate_fused_output_classes(ProviderPrecision::F32, &[ProviderElementType::F64])
                .is_err()
        );
    }
    #[test]
    fn empty_or_non_native_output_declarations_reject() {
        assert!(validate_fused_output_classes(ProviderPrecision::F64, &[]).is_err());
        for class in [
            ProviderElementType::ComplexF64,
            ProviderElementType::ComplexF32,
            ProviderElementType::U8,
            ProviderElementType::I64,
        ] {
            assert!(validate_fused_output_classes(ProviderPrecision::F64, &[class]).is_err());
        }
    }
}
