//! Class intent at the raw fused-kernel publication boundary. Final adapted
//! output types are deliberately not substituted for these physical facts.
use crate::{unsupported_provider_operation, ProviderElementType, ProviderPrecision};

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
