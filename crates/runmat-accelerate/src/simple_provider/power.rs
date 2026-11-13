//! Exact paired-domain classification for the in-process floating provider.
use super::*;
use runmat_accelerate_api::PowerResultDomain;
use runmat_runtime::builtins::common::broadcast::BroadcastPlan;

fn admitted_plan(base: &GpuTensorHandle, exponent: &GpuTensorHandle) -> Result<BroadcastPlan> {
    require_non_integer_elementwise_inputs("real power", base, exponent)?;
    ensure!(
        base.descriptor().storage == Some(GpuTensorStorage::Real)
            && exponent.descriptor().storage == Some(GpuTensorStorage::Real),
        "real power requires real storage"
    );
    ensure!(
        runmat_accelerate_api::handle_precision(base) == Some(ProviderPrecision::F64)
            && runmat_accelerate_api::handle_precision(exponent) == Some(ProviderPrecision::F64),
        "in-process real power requires native double inputs"
    );
    ensure!(
        !runmat_accelerate_api::handle_is_logical(base)
            && !runmat_accelerate_api::handle_is_logical(exponent),
        "real power requires floating, not logical inputs"
    );
    let shape = runtime_broadcast_shapes("real power", base.shape(), exponent.shape())
        .map_err(|e| anyhow!(e))?;
    shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| anyhow!("power broadcast size overflow"))?;
    BroadcastPlan::new(base.shape(), exponent.shape()).map_err(|e| anyhow!(e))
}

impl InProcessProvider {
    pub(super) fn classify_real_power_impl(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<PowerResultDomain> {
        ensure!(
            base.device_id() == self.device_id() && exponent.device_id() == self.device_id(),
            "power operands belong to another provider"
        );
        let plan = admitted_plan(base, exponent)?;
        let registry = registry(self.device_id).lock().unwrap();
        let base_values = registry
            .get(base)
            .ok_or_else(|| anyhow!("power base buffer missing"))?;
        let exponent_values = registry
            .get(exponent)
            .ok_or_else(|| anyhow!("power exponent buffer missing"))?;
        ensure_storage_len(
            "real power",
            base_values,
            base.shape(),
            &GpuTensorStorage::Real,
        )?;
        ensure_storage_len(
            "real power",
            exponent_values,
            exponent.shape(),
            &GpuTensorStorage::Real,
        )?;
        Ok(plan
            .iter()
            .fold(PowerResultDomain::Real, |domain, (_, a, b)| {
                domain.join(PowerResultDomain::for_real_pair(
                    base_values[a],
                    exponent_values[b],
                ))
            }))
    }

    pub(super) fn real_power_in_domain_impl(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: PowerResultDomain,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            base.device_id() == self.device_id() && exponent.device_id() == self.device_id(),
            "power operands belong to another provider"
        );
        let plan = admitted_plan(base, exponent)?;
        let width = if domain == PowerResultDomain::Complex {
            2
        } else {
            1
        };
        let capacity = plan
            .len()
            .checked_mul(width)
            .ok_or_else(|| anyhow!("power output size overflow"))?;
        let mut output = Vec::with_capacity(capacity);
        {
            let registry = registry(self.device_id).lock().unwrap();
            let base_values = registry
                .get(base)
                .ok_or_else(|| anyhow!("power base buffer missing"))?;
            let exponent_values = registry
                .get(exponent)
                .ok_or_else(|| anyhow!("power exponent buffer missing"))?;
            ensure_storage_len(
                "real power",
                base_values,
                base.shape(),
                &GpuTensorStorage::Real,
            )?;
            ensure_storage_len(
                "real power",
                exponent_values,
                exponent.shape(),
                &GpuTensorStorage::Real,
            )?;
            for (_, a, b) in plan.iter() {
                let (base, exponent) = (base_values[a], exponent_values[b]);
                let pair_domain = PowerResultDomain::for_real_pair(base, exponent);
                ensure!(
                    domain == PowerResultDomain::Complex || pair_domain == PowerResultDomain::Real,
                    "real power domain disagrees with operand values"
                );
                let (real, imag) = if base.is_nan() || exponent.is_nan() {
                    (f64::NAN, 0.0)
                } else if pair_domain == PowerResultDomain::Complex {
                    let magnitude = (exponent * base.abs().ln()).exp();
                    let phase = exponent * std::f64::consts::PI;
                    (magnitude * phase.cos(), magnitude * phase.sin())
                } else {
                    (base.powf(exponent), 0.0)
                };
                output.push(real);
                if domain == PowerResultDomain::Complex {
                    output.push(imag);
                }
            }
        }
        self.allocate_tensor_with_storage(
            output,
            plan.output_shape().to_vec(),
            if domain == PowerResultDomain::Complex {
                GpuTensorStorage::ComplexInterleaved
            } else {
                GpuTensorStorage::Real
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn actual_provider_classifies_paired_values_and_preserves_result_domain() {
        let provider = InProcessProvider::new();
        for (exponent_shape, expected) in [
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
                    shape: &exponent_shape,
                })
                .unwrap();
            let domain = block_on(provider.classify_real_power(&base, &exponent)).unwrap();
            assert_eq!(domain, expected);
            let output = block_on(provider.real_power_in_domain(&base, &exponent, domain)).unwrap();
            let values = block_on(provider.download(&output)).unwrap();
            if domain == PowerResultDomain::Real {
                assert_eq!(values.data, vec![16.0, 2.0]);
                assert_eq!(output.descriptor().storage, Some(GpuTensorStorage::Real));
            } else {
                assert_eq!(output.shape(), vec![2, 2]);
                assert_eq!(
                    output.descriptor().storage,
                    Some(GpuTensorStorage::ComplexInterleaved)
                );
                assert!((values.data[3] - 2.0).abs() < 1e-12);
                assert!(block_on(provider.real_power_in_domain(
                    &base,
                    &exponent,
                    PowerResultDomain::Real
                ))
                .is_err());
            }
            for handle in [base, exponent, output] {
                handle.release().unwrap();
            }
        }
    }

    #[test]
    fn tiny_complex_and_documented_nan_overrides_survive_provider_execution() {
        let provider = InProcessProvider::new();
        let base = provider
            .upload(&HostTensorView {
                data: &[-1e-300, 1.0, f64::NAN],
                shape: &[1, 3],
            })
            .unwrap();
        let exponent = provider
            .upload(&HostTensorView {
                data: &[0.5, f64::NAN, 0.0],
                shape: &[1, 3],
            })
            .unwrap();
        let domain = block_on(provider.classify_real_power(&base, &exponent)).unwrap();
        assert_eq!(domain, PowerResultDomain::Complex);
        let output = block_on(provider.real_power_in_domain(&base, &exponent, domain)).unwrap();
        let values = block_on(provider.download(&output)).unwrap().data;
        assert!((values[1] / 1e-150 - 1.0).abs() < 1e-12);
        assert!(values[2].is_nan() && values[4].is_nan());
        assert_eq!(values[3], 0.0);
        assert_eq!(values[5], 0.0);
        for handle in [base, exponent, output] {
            handle.release().unwrap();
        }
    }
}
