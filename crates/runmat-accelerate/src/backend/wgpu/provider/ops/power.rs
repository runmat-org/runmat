//! Resident paired power classification; only its scalar domain flag downloads.
use super::backend_types::WgpuProvider;
use crate::backend::wgpu::shaders::power::{classification_shader, shader, PowerShader};
use crate::backend::wgpu::types::NumericPrecision;
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, PowerResultDomain};

impl WgpuProvider {
    fn admitted_power_shape(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        input_precision: NumericPrecision,
    ) -> Result<(Vec<usize>, usize)> {
        for handle in [base, exponent] {
            let entry = self.get_entry_raw(handle)?;
            ensure!(
                handle.descriptor().element_type == Some(entry.element_type)
                    && runmat_accelerate_api::handle_class_identity(handle)
                        .is_none_or(|class| class == entry.element_type.class_identity()),
                "real power class metadata mismatch"
            );
            if entry.integer_type().is_some() || entry.precision != input_precision {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "real power requires matching floating operand precision",
                ));
            }
            ensure!(
                !runmat_accelerate_api::handle_is_logical(handle),
                "real power requires floating operands"
            );
            ensure!(
                entry.storage == GpuTensorStorage::Real
                    && handle.descriptor().storage == Some(GpuTensorStorage::Real),
                "real power requires real storage"
            );
            ensure!(
                runmat_accelerate_api::handle_precision(handle)
                    == Some(match input_precision {
                        crate::backend::wgpu::types::NumericPrecision::F32 =>
                            runmat_accelerate_api::ProviderPrecision::F32,
                        crate::backend::wgpu::types::NumericPrecision::F64 =>
                            runmat_accelerate_api::ProviderPrecision::F64,
                    }),
                "real power precision mismatch"
            );
            ensure!(
                entry.shape == handle.shape(),
                "real power shape metadata mismatch"
            );
            let len = handle
                .shape()
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(|| anyhow!("power input size overflow"))?;
            ensure!(len == entry.len, "real power storage length mismatch");
        }
        let shape = runmat_runtime::builtins::common::broadcast::broadcast_shapes(
            "power",
            base.shape(),
            exponent.shape(),
        )
        .map_err(|e| anyhow!(e))?;
        ensure!(
            shape.len() <= crate::backend::wgpu::params::BCAST_MAX_RANK,
            "power rank exceeds provider broadcast limit"
        );
        let len = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| anyhow!("power output size overflow"))?;
        ensure!(
            len <= u32::MAX as usize && shape.iter().all(|d| *d <= u32::MAX as usize),
            "power output exceeds GPU index limits"
        );
        for handle in [base, exponent] {
            super::backend_shared::fusion_operand_layout(
                handle,
                &self.get_entry_raw(handle)?,
                &shape,
            )?;
        }
        Ok((shape, len))
    }

    pub(crate) async fn classify_real_power_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<PowerResultDomain> {
        let input_precision = self.get_entry_raw(base)?.precision;
        if input_precision != self.precision
            && !(input_precision == NumericPrecision::F32
                && self.precision == NumericPrecision::F64)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "power classification requires exact floating input conversion",
            ));
        }
        let (shape, len) = self.admitted_power_shape(base, exponent, input_precision)?;
        if len == 0 {
            return Ok(PowerResultDomain::Real);
        }
        // Input buffers remain borrowed, including F32 on an F64 provider.
        // Exact widening occurs in registers. Flags and the existing scalar
        // reductions retain native precision, so probe allocations/readback
        // geometry is unchanged: no input staging or hidden layout dispatch.
        let flags = self.fused_elementwise_with_representation_telemetry_exec(
            &classification_shader(input_precision, self.precision),
            &[base.clone(), exponent.clone()],
            &shape,
            len,
            super::elementwise::FusedElementwiseRepresentation {
                input_precision,
                output_storage: GpuTensorStorage::Real,
                logical_output: false,
            },
        )?;
        // This buffer is owned here, so reshaping it cannot change an input's
        // metadata. Existing two-dimensional full reduction handles all ranks.
        let flat = self.reshape_exec(&flags, &[len, 1])?;
        let reduced = self.reduce_any_exec(&flat, true)?;
        let result = self.download_exec(&reduced).await?;
        ensure!(
            result.data.len() == 1 && (result.data[0] == 0.0 || result.data[0] == 1.0),
            "power domain reduction returned an invalid flag"
        );
        Ok(if result.data[0] == 0.0 {
            PowerResultDomain::Real
        } else {
            PowerResultDomain::Complex
        })
    }

    pub(crate) fn real_power_in_domain_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: PowerResultDomain,
    ) -> Result<GpuTensorHandle> {
        // Execution still owns a native-precision output contract. Supporting
        // a typed classification probe does not authorize a widened result.
        self.get_entry(base)?;
        self.get_entry(exponent)?;
        let (shape, len) = self.admitted_power_shape(base, exponent, self.precision)?;
        let (width, storage) = match domain {
            PowerResultDomain::Real => (1, GpuTensorStorage::Real),
            PowerResultDomain::Complex => (2, GpuTensorStorage::ComplexInterleaved),
        };
        let words = len
            .checked_mul(width)
            .ok_or_else(|| anyhow!("power output storage overflow"))?;
        ensure!(
            words <= u32::MAX as usize,
            "power output exceeds GPU index limits"
        );
        if words == 0 {
            let buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "power-empty")
            })??;
            return self.register_existing_buffer_with_storage(buffer, shape, 0, storage);
        }
        self.fused_elementwise_with_telemetry_and_storage_exec(
            &shader(self.precision, PowerShader::Evaluate(domain)),
            &[base.clone(), exponent.clone()],
            &shape,
            words,
            storage,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

    #[tokio::test]
    async fn actual_wgpu_power_classification_and_execution_preserve_semantics() {
        let provider = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("power execution qualification requires a compatible WGPU adapter");
        let tolerance = if provider.precision() == ProviderPrecision::F32 {
            2e-4
        } else {
            1e-10
        };
        for (shape, domain) in [
            (vec![1, 2], PowerResultDomain::Real),
            (vec![2, 1], PowerResultDomain::Complex),
        ] {
            let base = provider
                .upload(&HostTensorView {
                    data: &[-4.0, 4.0],
                    shape: &[1, 2],
                })
                .unwrap();
            let exponent = provider
                .upload(&HostTensorView {
                    data: &[2.0, 0.5],
                    shape: &shape,
                })
                .unwrap();
            assert_eq!(
                provider
                    .classify_real_power(&base, &exponent)
                    .await
                    .unwrap(),
                domain
            );
            let output = provider
                .real_power_in_domain(&base, &exponent, domain)
                .await
                .unwrap();
            let actual = provider.download(&output).await.unwrap();
            if domain == PowerResultDomain::Real {
                assert!((actual.data[0] - 16.0).abs() < tolerance);
                assert!((actual.data[1] - 2.0).abs() < tolerance);
            } else {
                assert_eq!(
                    output.descriptor().storage,
                    Some(GpuTensorStorage::ComplexInterleaved)
                );
                assert!((actual.data[3] - 2.0).abs() < tolerance);
            }
            for handle in [base, exponent, output] {
                handle.release().unwrap();
            }
        }
        let base = provider
            .upload(&HostTensorView {
                data: &[-1e-30, 1.0, f64::NAN, -0.0],
                shape: &[1, 4],
            })
            .unwrap();
        let exponent = provider
            .upload(&HostTensorView {
                data: &[0.5, f64::NAN, 0.0, -3.0],
                shape: &[1, 4],
            })
            .unwrap();
        assert_eq!(
            provider
                .classify_real_power(&base, &exponent)
                .await
                .unwrap(),
            PowerResultDomain::Complex
        );
        let output = provider
            .real_power_in_domain(&base, &exponent, PowerResultDomain::Complex)
            .await
            .unwrap();
        let actual = provider.download(&output).await.unwrap().data;
        assert!((actual[1] / 1e-15 - 1.0).abs() < tolerance);
        assert!(actual[2].is_nan() && actual[4].is_nan());
        assert_eq!(actual[6], f64::NEG_INFINITY);
        for handle in [base, exponent, output] {
            handle.release().unwrap();
        }
    }
}
