use super::*;

#[derive(Clone, Copy)]
pub(super) enum Logarithm {
    Binary,
    Common,
    OnePlus,
}

impl InProcessProvider {
    pub(super) fn logarithm(
        &self,
        input: &GpuTensorHandle,
        operation: Logarithm,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            input.device_id() == self.device_id(),
            "logarithm input belongs to another provider"
        );
        ensure!(
            input.descriptor().element_type == Some(NumericElementType::F64)
                && input.descriptor().storage == Some(GpuTensorStorage::Real)
                && !runmat_accelerate_api::handle_is_logical(input),
            "in-process logarithm requires real native double input"
        );
        let guard = registry(self.device_id).lock().unwrap();
        let values = guard
            .get(input)
            .ok_or_else(|| anyhow!("logarithm input buffer is missing"))?;
        ensure_storage_len("logarithm", values, input.shape(), &GpuTensorStorage::Real)?;
        let output = values
            .iter()
            .map(|value| match operation {
                Logarithm::Binary => value.log2(),
                Logarithm::Common => value.log10(),
                Logarithm::OnePlus => value.ln_1p(),
            })
            .collect();
        drop(guard);
        self.allocate_tensor(output, input.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn public_logarithm_hooks_preserve_descriptors_and_numerical_boundaries() {
        let provider = InProcessProvider::new();
        for operation in [Logarithm::Binary, Logarithm::Common, Logarithm::OnePlus] {
            let values = [-1.0, -0.0, 1e-300, 1.0 + 1e-14, f64::INFINITY, f64::NAN];
            let input = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &[2, 3],
                })
                .unwrap();
            let output = block_on(match operation {
                Logarithm::Binary => provider.unary_log2(&input),
                Logarithm::Common => provider.unary_log10(&input),
                Logarithm::OnePlus => provider.unary_log1p(&input),
            })
            .unwrap();
            assert_ne!(input.buffer_id(), output.buffer_id());
            assert_eq!(output.device_id(), provider.device_id());
            assert_eq!(output.shape(), input.shape());
            assert_eq!(
                output.descriptor().element_type,
                Some(NumericElementType::F64)
            );
            assert_eq!(output.descriptor().storage, Some(GpuTensorStorage::Real));
            for (actual, original) in block_on(provider.download(&output))
                .unwrap()
                .data
                .iter()
                .zip(values)
            {
                let expected = match operation {
                    Logarithm::Binary => original.log2(),
                    Logarithm::Common => original.log10(),
                    Logarithm::OnePlus => original.ln_1p(),
                };
                assert!(
                    actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan()
                );
            }
            let other = InProcessProvider::new();
            let foreign = other
                .upload(&HostTensorView {
                    data: &values,
                    shape: input.shape(),
                })
                .unwrap();
            assert!(provider.logarithm(&foreign, operation).is_err());
            let mut wire = input.wire_descriptor();
            wire.view = wire.view.with_logical(true);
            assert!(provider.acquire_live(&wire).is_err());
            output.release().unwrap();
            input.release().unwrap();
        }
    }
}
