use super::InProcessNumericData;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::{GpuTensorHandle, NumericElementType};

pub(super) fn floating_payload(
    handle: &GpuTensorHandle,
    payload: &InProcessNumericData,
) -> Result<Vec<f64>> {
    match (handle.physical().element_type(), payload) {
        (NumericElementType::F64, InProcessNumericData::F64(values)) => {
            let mut output = Vec::new();
            output
                .try_reserve_exact(values.len())
                .map_err(|source| anyhow!(source).context("iir_filter: input allocation failed"))?;
            output.extend_from_slice(values);
            Ok(output)
        }
        (NumericElementType::F32, InProcessNumericData::F32(values)) => {
            let mut output = Vec::new();
            output
                .try_reserve_exact(values.len())
                .map_err(|source| anyhow!(source).context("iir_filter: input allocation failed"))?;
            output.extend(values.iter().copied().map(f64::from));
            Ok(output)
        }
        (declared, _) => Err(anyhow!(
            "iir_filter: physical element type {:?} does not match a floating payload",
            declared
        )),
    }
}

pub(super) fn output_payload(
    values: Vec<f64>,
    physical: NumericElementType,
) -> Result<InProcessNumericData> {
    match physical {
        NumericElementType::F64 => Ok(InProcessNumericData::F64(values)),
        NumericElementType::F32 => {
            let mut output = Vec::new();
            output.try_reserve_exact(values.len()).map_err(|source| {
                anyhow!(source).context("iir_filter: output allocation failed")
            })?;
            output.extend(values.into_iter().map(|value| value as f32));
            Ok(InProcessNumericData::F32(output))
        }
        other => Err(anyhow!(
            "iir_filter: unsupported signal element type {:?}",
            other
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simple_provider::InProcessProvider;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelProvider, HostTensorView, ProviderIirCoefficientSource, ProviderIirFilterOptions,
        ProviderPreparedIirCoefficients, ThreadProviderGuard,
    };

    #[test]
    fn publication_preserves_signal_precision_without_f64_mirror() {
        assert_eq!(
            output_payload(vec![1.25, -2.5], NumericElementType::F32).unwrap(),
            InProcessNumericData::F32(vec![1.25, -2.5])
        );
        assert_eq!(
            output_payload(vec![1.25], NumericElementType::F64).unwrap(),
            InProcessNumericData::F64(vec![1.25])
        );
    }

    #[test]
    fn one_prepared_coefficient_owner_is_reused_for_two_filter_passes() {
        let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
        let _scope = ThreadProviderGuard::set(Some(provider));
        let prepared = ProviderPreparedIirCoefficients::upload(
            provider,
            &[0.5, 0.5],
            &[1.0, 0.0],
            provider.precision(),
            &[],
        )
        .expect("prepare normalized FIR coefficients");
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 3.0, 5.0],
                shape: &[1, 3],
            })
            .unwrap();
        let first = block_on(provider.iir_filter(
            ProviderIirCoefficientSource::Prepared(&prepared),
            &input,
            ProviderIirFilterOptions { dim: 1, zi: None },
        ))
        .expect("first prepared pass");
        let second = block_on(provider.iir_filter(
            ProviderIirCoefficientSource::Prepared(&prepared),
            &first.output,
            ProviderIirFilterOptions { dim: 1, zi: None },
        ))
        .expect("second prepared pass");
        assert_eq!(
            block_on(provider.download(&first.output)).unwrap().data,
            vec![0.5, 2.0, 4.0]
        );
        assert_eq!(
            block_on(provider.download(&second.output)).unwrap().data,
            vec![0.25, 1.25, 3.0]
        );
        assert!(prepared
            .numerator()
            .allocation_lease()
            .validate_live()
            .is_ok());
        assert!(prepared
            .denominator()
            .allocation_lease()
            .validate_live()
            .is_ok());
    }
}
