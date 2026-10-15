//! Resident paired power classification; only its scalar domain flag downloads.
use super::backend_types::WgpuProvider;
use crate::backend::wgpu::shaders::power::{shader, PowerShader};
use crate::backend::wgpu::types::NumericPrecision;
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, PowerResultDomain};

#[path = "power/flag_fusion.rs"]
mod flag_fusion;
#[path = "power/evaluation.rs"]
mod evaluation;
#[path = "power/projection.rs"]
mod projection;
#[path = "power/probe_emission.rs"]
mod probe_emission;
#[path = "power/probe.rs"]
mod probe;
pub(crate) use probe_emission::PowerProbeResourcePhase;
#[cfg(test)]
pub(crate) use probe_emission::ProbeSubmissionRecord;

#[cfg(test)]
type ProbeRecordResult = Option<ProbeSubmissionRecord>;
#[cfg(not(test))]
type ProbeRecordResult = ();

#[cfg(test)]
fn empty_probe_record() -> ProbeRecordResult {
    None
}

#[cfg(not(test))]
fn empty_probe_record() -> ProbeRecordResult {}

impl WgpuProvider {
    pub(crate) fn issue_power_probe_resource_ticket(
        &self,
        requested_bytes: u64,
    ) -> Result<Option<runmat_accelerate_api::observation::ProviderResourceTicket>> {
        runmat_accelerate_api::observation::current_provider_resource_scope()
            .map(|scope| scope.issue_resource_ticket(requested_bytes))
            .transpose()
            .map_err(Into::into)
    }

    /// Revalidate the service plan against this provider, then project the
    /// same FusionOut or typed-cast allocator used by its public result.
    pub(crate) fn plan_power_output_buffer_bytes_exec(
        &self,
        planned: &runmat_accelerate_api::power_service::PowerSymbolicPlan,
        domain: PowerResultDomain,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<u64>> {
        let current = planned.replan(self).map_err(
            runmat_accelerate_api::power_service::PowerServiceError::into_execution_error,
        )?;
        if current.as_ref() != Some(planned)
            || planned.feasibility_query(domain).is_some_and(|query| {
                !runmat_accelerate_api::AccelProvider::query_feasibility(self, &query)
                    .is_supported()
            })
        {
            return Ok(None);
        }
        if let Some(proposal) = planned.exact_physical_proposal() {
            // Replan above revalidated both source extent and the exact recipe
            // against this owner. Do not substitute a parallel size formula.
            return Ok((domain == PowerResultDomain::Real).then_some(proposal.output_buffer_bytes));
        }
        if planned.physical_recipe().route
            != runmat_accelerate_api::power_service::PowerPhysicalRoute::ExactSingleBase
            && !matches!(
                planned.mode(),
                runmat_accelerate_api::RealPowerArithmeticMode::NativeSingle
            )
            && !self.ieee_words_available(self.precision)
        {
            return Ok(None);
        }
        let Some(work) = planned.physical_work(domain) else {
            return Ok(None);
        };
        let public = work
            .single_publication
            .as_ref()
            .unwrap_or(&work.native_result);
        let Some(elements) = public.checked_element_count() else {
            return Ok(None);
        };
        let width = if domain == PowerResultDomain::Complex {
            2u64
        } else {
            1u64
        };
        let Some(words) = elements.checked_mul(width) else {
            return Ok(None);
        };
        let Some(bytes) = (if work.single_publication.is_some() {
            crate::backend::wgpu_storage::raw_word_requested_bytes(words)
        } else {
            let scalar_size = match public.element_type {
                runmat_accelerate_api::ProviderElementType::F32
                | runmat_accelerate_api::ProviderElementType::ComplexF32 => 4,
                runmat_accelerate_api::ProviderElementType::F64
                | runmat_accelerate_api::ProviderElementType::ComplexF64 => 8,
                _ => return Ok(None),
            };
            usize::try_from(words).ok().and_then(|len| {
                crate::backend::wgpu::residency::BufferResidency::requested_bytes(len, scalar_size)
            })
        }) else {
            return Ok(None);
        };
        if bytes > self.adapter_limits.max_buffer_size
            || bytes > u64::from(self.adapter_limits.max_storage_buffer_binding_size)
        {
            return Ok(None);
        }
        Ok(Some(bytes))
    }

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
        Ok(self
            .classify_real_power_with_probe_exec(base, exponent)
            .await?
            .0)
    }

    #[cfg(test)]
    pub(crate) async fn classify_real_power_observed_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<(PowerResultDomain, ProbeRecordResult)> {
        self.classify_real_power_with_probe_exec(base, exponent)
            .await
    }

    async fn classify_real_power_with_probe_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<(PowerResultDomain, ProbeRecordResult)> {
        let Some(prepared) = self.prepare_power_domain_probe_exec(base, exponent)? else {
            return Ok((PowerResultDomain::Real, empty_probe_record()));
        };
        self.execute_power_domain_probe_prepared_exec(base, exponent, prepared)
            .await
    }
    pub(crate) fn real_power_in_domain_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: PowerResultDomain,
    ) -> Result<GpuTensorHandle> {
        let input_precision = self.real_floating_entry(base)?.precision;
        self.real_floating_entry(exponent)?;
        if input_precision != self.precision
            && !(input_precision == NumericPrecision::F32
                && self.precision == NumericPrecision::F64)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "power execution requires an admitted floating physical class",
            ));
        }
        let (shape, len) = self.admitted_power_shape(base, exponent, input_precision)?;
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
        if input_precision == NumericPrecision::F32 && self.precision == NumericPrecision::F64 {
            self.require_ieee_words(input_precision)?;
            return self.fused_elementwise_with_representation_telemetry_exec(
                &shader(input_precision, PowerShader::Evaluate(domain)),
                &[base.clone(), exponent.clone()],
                &shape,
                words,
                super::elementwise::FusedElementwiseRepresentation {
                    input_precision,
                    output_precision: input_precision,
                    output_storage: storage,
                    logical_output: false,
                },
            );
        }
        if words == 0 {
            let buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "power-empty")
            })??;
            return self.register_existing_buffer_with_storage(buffer, shape, 0, storage);
        }
        self.require_ieee_words(input_precision)?;
        self.fused_elementwise_with_telemetry_and_storage_exec(
            &shader(input_precision, PowerShader::Evaluate(domain)),
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

    #[test]
    #[ignore = "requires a WGPU adapter and RUNMAT_WGPU_FORCE_PRECISION=f32"]
    fn f32_symbolic_output_projection_retains_typed_domains_and_rejects_stale_limits() {
        use crate::backend::wgpu::provider::WgpuProviderOptions;
        use runmat_accelerate_api::power_service::plan_resident_scalar_power;
        use runmat_accelerate_api::{
            PowerScalarControl, ProviderElementType, ProviderLayout, ProviderOperationIdentity,
            ProviderRepresentation, ProviderResidency, ProviderStorage, RealPowerOperand,
        };

        let mut provider = WgpuProvider::new(WgpuProviderOptions::default())
            .expect("qualification requires a real WGPU adapter");
        assert_eq!(provider.precision(), ProviderPrecision::F32);
        let plan = plan_resident_scalar_power(
            &provider,
            ProviderRepresentation {
                element_type: ProviderElementType::F32,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: vec![2, 4, 5],
                residency: ProviderResidency::Device,
            },
            PowerScalarControl::SingleBits(2.0f32.to_bits()),
            RealPowerOperand::Base,
            &[1, 1],
            ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name),
        )
        .unwrap()
        .expect("selected F32 POWER route");
        assert_eq!(
            plan.physical_recipe().route,
            runmat_accelerate_api::power_service::PowerPhysicalRoute::TypedDomain
        );
        assert_eq!(
            provider
                .plan_power_output_buffer_bytes(&plan, PowerResultDomain::Real)
                .unwrap(),
            Some(160)
        );
        assert_eq!(
            provider
                .plan_power_output_buffer_bytes(&plan, PowerResultDomain::Complex)
                .unwrap(),
            Some(320),
            "the typed route retains the supported complex output branch"
        );
        provider.adapter_limits.max_buffer_size = 159;
        assert_eq!(
            provider
                .plan_power_output_buffer_bytes(&plan, PowerResultDomain::Real)
                .unwrap(),
            None,
            "stale provider limits cannot reuse a previous output proposal"
        );
        assert_eq!(
            provider
                .plan_power_output_buffer_bytes(&plan, PowerResultDomain::Complex)
                .unwrap(),
            None,
            "stale provider limits cannot reuse the complex output proposal"
        );
    }

    #[tokio::test]
    #[ignore = "requires an actual shader-f64 WGPU adapter; run in the F64 qualification gate"]
    async fn f64_owner_executes_native_single_power_on_exact_nd_views() {
        use runmat_accelerate_api::{
            handle_precision, upload_numeric_protecting, HostNumericDataView, HostNumericTensorView,
        };

        let provider = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("F64 qualification requires an actual WGPU adapter");
        assert_eq!(provider.precision(), ProviderPrecision::F64);
        let values = [1.25_f32; 40];
        let exponent = [1.8_f32];
        let base = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: HostNumericDataView::F32(&values),
                shape: &[2, 4, 5],
                storage: GpuTensorStorage::Real,
            },
            &[],
        )
        .expect("Single N-D input on Double-preferred owner");
        let control = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: HostNumericDataView::F32(&exponent),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            },
            &[&base],
        )
        .expect("exact Single scalar control");
        let wrong_class = provider
            .upload(&HostTensorView {
                data: &[1.8],
                shape: &[1, 1],
            })
            .expect("Double control witness");
        assert!(provider
            .real_power_in_domain(&base, &wrong_class, PowerResultDomain::Real)
            .await
            .is_err());
        wrong_class.release().unwrap();
        let foreign = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 4, 5],
                runmat_accelerate_api::NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .expect("foreign Single witness");
        assert!(provider
            .real_power_in_domain(&foreign.handle, &control, PowerResultDomain::Real)
            .await
            .is_err());
        let wrong_shape = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: HostNumericDataView::F32(&[1.8; 3]),
                shape: &[3, 1],
                storage: GpuTensorStorage::Real,
            },
            &[&base],
        )
        .expect("Single incompatible-shape witness");
        assert!(provider
            .real_power_in_domain(&base, &wrong_shape, PowerResultDomain::Real)
            .await
            .is_err());
        wrong_shape.release().unwrap();
        assert_eq!(
            provider.classify_real_power(&base, &control).await.unwrap(),
            PowerResultDomain::Real
        );
        let output = provider
            .real_power_in_domain(&base, &control, PowerResultDomain::Real)
            .await
            .expect("native Single power on F64 owner");
        assert_eq!(output.shape(), [2, 4, 5]);
        assert_eq!(handle_precision(&output), Some(ProviderPrecision::F32));
        assert!(!output.shares_allocation_with(&base));
        assert!(!output.shares_allocation_with(&control));
        let actual = provider.download(&output).await.unwrap();
        assert_eq!(actual.data.len(), 40);
        let expected = 1.25_f32.powf(1.8_f32);
        assert!(actual
            .data
            .iter()
            .all(|value| (*value - f64::from(expected)).abs() < 2e-5));
        base.release().unwrap();
        control.release().unwrap();
        assert!(output.allocation_lease().validate_live().is_ok());
        output.release().unwrap();
    }

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
            let before = provider.telemetry_snapshot();
            let compound = runmat_accelerate_api::power_service::PowerPairCompound::plan(
                provider, &base, &exponent,
            )
            .unwrap()
            .expect("actual native paired POWER compound");
            let representations = [&base, &exponent].map(|source| {
                runmat_accelerate_api::ProviderFeasibilityQuery::exact_numeric_download(source)
                    .unwrap()
                    .inputs
                    .into_iter()
                    .next()
                    .unwrap()
            });
            let descriptions = [
                runmat_accelerate_api::PowerInputDescription::Live {
                    physical: *base.physical(),
                    view: base.view().clone(),
                },
                runmat_accelerate_api::PowerInputDescription::PlannedDense {
                    requested_bytes: provider
                        .plan_dense_real_output_buffer_bytes(&representations[1])
                        .unwrap()
                        .expect("future dense output extent"),
                },
            ];
            let future = runmat_accelerate_api::power_service::PowerPairProjection::plan(
                provider,
                representations,
                descriptions,
            )
            .unwrap()
            .expect("future dense operand has both native POWER alternatives");
            assert_eq!(future.physical().probe, *compound.probe().physical());
            assert_eq!(future.physical().real, *compound.real().physical());
            assert_eq!(future.physical().complex, *compound.complex().physical());
            let bound = future
                .bind(&base, &exponent)
                .expect("actual produced lease binds the projected alternatives");
            assert_eq!(bound.identity().unwrap(), compound.identity().unwrap());
            let foreign = runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    exponent.shape().to_vec(),
                    exponent.physical().element_type(),
                    GpuTensorStorage::Real,
                )
                .unwrap();
            assert!(future.bind(&base, &foreign.handle).is_err());
            foreign.handle.release().unwrap();
            assert_eq!(
                before,
                provider.telemetry_snapshot(),
                "physical planning cannot execute a probe"
            );
            let executed = bound.execute(None).await.unwrap();
            let (selected_output, evidence) = executed.into_parts();
            assert_eq!(evidence.domain(), domain);
            assert!(!selected_output.shares_allocation_with(&base));
            assert!(!selected_output.shares_allocation_with(&exponent));
            assert_eq!(
                selected_output.physical().element_type(),
                base.physical().element_type()
            );
            let selected = provider.download(&selected_output).await.unwrap();
            assert_eq!(selected.shape, actual.shape);
            assert_eq!(selected.storage, actual.storage);
            assert_eq!(selected.data.len(), actual.data.len());
            for (selected, ordinary) in selected.data.iter().zip(&actual.data) {
                assert!((selected - ordinary).abs() < tolerance);
            }
            selected_output.release().unwrap();
            let cancelled = runmat_execution::cancellation::CancellationSource::new();
            cancelled.cancel(runmat_execution::CancellationReason::User);
            let token = cancelled.token();
            let before = provider.telemetry_snapshot();
            assert!(compound.execute(Some(&token)).await.is_err());
            assert_eq!(
                before,
                provider.telemetry_snapshot(),
                "pre-effect cancellation must not execute any child"
            );
            base.allocation_lease().validate_live().unwrap();
            exponent.allocation_lease().validate_live().unwrap();
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

    #[tokio::test]
    async fn actual_scalar_power_projection_binds_upload_and_consumes_paired_probe() {
        use runmat_accelerate_api::power_service::{
            execute_resident_scalar_power_projected, prepare_resident_scalar_power,
            PowerTypedScalarProjection,
        };
        use runmat_accelerate_api::{
            PowerScalarControl, ProviderOperationIdentity, RealPowerOperand,
        };
        let provider = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("scalar POWER qualification requires a compatible WGPU adapter");
        let base = provider
            .upload(&HostTensorView {
                data: &[-4.0, 4.0],
                shape: &[1, 2],
            })
            .unwrap();
        let control = match provider.precision() {
            ProviderPrecision::F32 => PowerScalarControl::SingleBits(0.5_f32.to_bits()),
            ProviderPrecision::F64 => PowerScalarControl::DoubleBits(0.5_f64.to_bits()),
        };
        let prepared = prepare_resident_scalar_power(
            provider,
            &base,
            control,
            RealPowerOperand::Base,
            &[1, 1],
            ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name),
        )
        .unwrap()
        .expect("real scalar typed POWER route");
        let before = provider.telemetry_snapshot();
        let projection = PowerTypedScalarProjection::plan(provider, prepared.symbolic_plan())
            .unwrap()
            .expect("native upload and paired alternatives have exact physical stages");
        let resources = projection
            .resource_bounds()
            .expect("all native stages are bounded");
        assert!(resources.retained_bytes > 0 && resources.queue_slots > 0);
        assert!(matches!(
            resources.peak_bytes,
            runmat_execution::ResourceByteBound::Known(bytes)
                if bytes >= resources.retained_bytes
        ));
        assert_eq!(before, provider.telemetry_snapshot());
        let executed = execute_resident_scalar_power_projected(prepared, &projection, None)
            .await
            .unwrap();
        assert_eq!(executed.domain, PowerResultDomain::Complex);
        assert!(executed.scalar_evidence.is_some());
        assert!(!executed.handle.shares_allocation_with(&base));
        let actual = provider.download(&executed.handle).await.unwrap();
        let tolerance = if provider.precision() == ProviderPrecision::F32 {
            2e-4
        } else {
            1e-10
        };
        assert!((actual.data[1] - 2.0).abs() < tolerance);
        assert!((actual.data[2] - 2.0).abs() < tolerance);
        base.allocation_lease().validate_live().unwrap();
        executed.handle.release().unwrap();
        base.release().unwrap();
    }
}
