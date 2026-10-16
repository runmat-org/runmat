use crate::{
    upload_numeric_protecting_admitted, GpuTensorHandle, GpuTensorStorage, HostNumericDataView,
    HostNumericTensorView, PowerResultDomain, PowerScalarControl, ProviderExecutionCompletionError,
    ProviderExecutionError, ProviderPrecision, RealPowerArithmeticMode, RealPowerOperand,
    ResidencyRestoreGuard,
};

use super::prepare::{PowerPhysicalRoute, PreparedResidentScalarPower};
use super::validate::{publish_provenance, reject, storage, valid_output, PowerServiceError};
mod exact;
mod projected;
pub use projected::{
    execute_resident_scalar_power_projected, execute_resident_scalar_power_projected_admitted,
};

pub struct ExecutedResidentScalarPower {
    pub handle: GpuTensorHandle,
    pub domain: PowerResultDomain,
    pub scalar_evidence: Option<super::scalar_projection::PowerScalarExecutionEvidence>,
}

/// Execute the prepared physical route. No returned failure authorizes a host
/// replay: fill/upload/widen/classification may already have submitted work.
pub async fn execute_resident_scalar_power(
    prepared: PreparedResidentScalarPower,
) -> Result<ExecutedResidentScalarPower, PowerServiceError> {
    execute_resident_scalar_power_admitted(prepared, None).await
}

pub async fn execute_resident_scalar_power_admitted(
    prepared: PreparedResidentScalarPower,
    admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> Result<ExecutedResidentScalarPower, PowerServiceError> {
    prepared.validate_live()?;
    // Retain and restore every surrounding borrowed view, including on an
    // error or when the in-flight future is cancelled.
    let _protected_metadata = prepared
        .protected_with(&[])
        .into_iter()
        .map(ResidencyRestoreGuard::new)
        .collect::<Vec<_>>();
    if prepared.route == PowerPhysicalRoute::LegacyProvenReal {
        return execute_legacy_real(&prepared).await;
    }
    if prepared.route == PowerPhysicalRoute::ExactSingleBase {
        return exact::execute(&prepared, admission).await;
    }
    if admission.is_some() {
        return Err(PowerServiceError::pre(
            "finite selected POWER requires its exact prepared recipe",
        ));
    }
    let provider = prepared.provider;
    let resident = &prepared.resident;
    let mixed = matches!(
        prepared.mode,
        RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
    );
    let widened = mixed && crate::handle_precision(resident) == Some(ProviderPrecision::F32);
    let native = if widened {
        let _guard = ResidencyRestoreGuard::new(resident);
        let output = provider
            .unary_double(resident)
            .await
            .map_err(PowerServiceError::after)?;
        if !valid_output(
            &output,
            provider,
            resident.shape(),
            ProviderPrecision::F64,
            GpuTensorStorage::Real,
            &prepared.protected_with(&[resident]),
        ) {
            return Err(reject(output, "invalid POWER resident widening"));
        }
        output
    } else {
        resident.clone()
    };
    if widened {
        revalidate_after_effect(&prepared)?;
    }
    let scalar = prepare_scalar(&prepared, &native, widened, None)?;
    let (base, exponent) = match prepared.resident_operand {
        RealPowerOperand::Base => (&native, &scalar),
        RealPowerOperand::Exponent => (&scalar, &native),
    };
    let physical_shape =
        runmat_types::geometry::broadcast_concrete_shape(base.shape(), exponent.shape()).map_err(
            |error| {
                PowerServiceError::after(ProviderExecutionError::from(anyhow::Error::from(error)))
            },
        )?;
    revalidate_after_effect(&prepared)?;
    let domain = {
        let _base = ResidencyRestoreGuard::new(base);
        let _exponent = ResidencyRestoreGuard::new(exponent);
        provider
            .classify_real_power(base, exponent)
            .await
            .map_err(|error| PowerServiceError::after(ProviderExecutionError::from(error)))?
    };
    if !prepared
        .feasibility_query(domain)
        .is_some_and(|query| provider.query_feasibility(&query).is_supported())
    {
        return Err(PowerServiceError::after(ProviderExecutionError::from(
            anyhow::anyhow!("selected POWER result domain is no longer admitted"),
        )));
    }
    revalidate_after_effect(&prepared)?;
    let native_output = {
        let _base = ResidencyRestoreGuard::new(base);
        let _exponent = ResidencyRestoreGuard::new(exponent);
        let output = provider
            .real_power_in_domain(base, exponent, domain)
            .await
            .map_err(|error| PowerServiceError::after(ProviderExecutionError::from(error)))?;
        let precision = if mixed {
            ProviderPrecision::F64
        } else {
            crate::handle_precision(resident).ok_or_else(|| {
                PowerServiceError::after(anyhow::anyhow!("resident precision disappeared"))
            })?
        };
        if !valid_output(
            &output,
            provider,
            &physical_shape,
            precision,
            storage(domain),
            &prepared.protected_with(&[base, exponent]),
        ) {
            return Err(reject(output, "invalid POWER native result"));
        }
        output
    };
    let published = if mixed {
        revalidate_after_effect(&prepared)?;
        let _guard = ResidencyRestoreGuard::new(&native_output);
        let output = provider
            .unary_single(&native_output)
            .await
            .map_err(PowerServiceError::after)?;
        if !valid_output(
            &output,
            provider,
            native_output.shape(),
            ProviderPrecision::F32,
            storage(domain),
            &prepared.protected_with(&[&native_output, resident, &scalar]),
        ) {
            return Err(reject(output, "invalid POWER Single publication"));
        }
        output
    } else {
        native_output
    };
    let published = if published.shape() == prepared.output_shape.as_slice() {
        published
    } else {
        revalidate_after_effect(&prepared)?;
        let output = provider
            .reshape(&published, &prepared.output_shape)
            .map_err(|error| PowerServiceError::after(ProviderExecutionError::from(error)))?;
        if !output.shares_allocation_with(&published)
            || output.shape() != prepared.output_shape.as_slice()
            || output.physical() != published.physical()
            || output.allocation_lease().validate_live().is_err()
            || !crate::provider_for_handle(&output)
                .is_some_and(|owner| crate::same_provider_instance(owner, provider))
        {
            return Err(reject(output, "invalid POWER output reshape"));
        }
        output
    };
    if prepared
        .protected_with(&[])
        .iter()
        .any(|protected| published.shares_allocation_with(protected))
    {
        return Err(reject(published, "POWER reshape aliases protected lease"));
    }
    if let Err(error) = prepared.validate_live() {
        return Err(reject_with_cause(published, error));
    }
    let handle = publish_provenance(published, resident)?;
    Ok(ExecutedResidentScalarPower {
        handle,
        domain,
        scalar_evidence: None,
    })
}

async fn execute_legacy_real(
    prepared: &PreparedResidentScalarPower,
) -> Result<ExecutedResidentScalarPower, PowerServiceError> {
    let provider = prepared.provider;
    let resident = &prepared.resident;
    let precision = crate::handle_precision(resident)
        .ok_or_else(|| PowerServiceError::pre("legacy POWER class disappeared"))?;
    let scalar = {
        let _guard = ResidencyRestoreGuard::new(resident);
        let output = provider
            .fill_like(resident, prepared.control.value())
            .map_err(PowerServiceError::unproven)?
            .ok_or_else(|| PowerServiceError::pre("legacy POWER scalar constructor unavailable"))?;
        if !valid_output(
            &output,
            provider,
            resident.shape(),
            precision,
            GpuTensorStorage::Real,
            &prepared.protected_with(&[resident]),
        ) {
            return Err(reject(output, "invalid legacy POWER scalar fill"));
        }
        output
    };
    let (base, exponent) = match prepared.resident_operand {
        RealPowerOperand::Base => (resident, &scalar),
        RealPowerOperand::Exponent => (&scalar, resident),
    };
    revalidate_after_effect(prepared)?;
    let output = {
        let _base = ResidencyRestoreGuard::new(base);
        let _exponent = ResidencyRestoreGuard::new(exponent);
        provider
            .elem_pow(base, exponent)
            .await
            .map_err(PowerServiceError::after)?
    };
    if !valid_output(
        &output,
        provider,
        resident.shape(),
        precision,
        GpuTensorStorage::Real,
        &prepared.protected_with(&[resident, &scalar]),
    ) {
        return Err(reject(output, "invalid legacy POWER result"));
    }
    if let Err(error) = prepared.validate_live() {
        return Err(reject_with_cause(output, error));
    }
    let handle = publish_provenance(output, resident)?;
    Ok(ExecutedResidentScalarPower {
        handle,
        domain: PowerResultDomain::Real,
        scalar_evidence: None,
    })
}

fn prepare_scalar(
    prepared: &PreparedResidentScalarPower,
    native: &GpuTensorHandle,
    prior_effect: bool,
    admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> Result<GpuTensorHandle, PowerServiceError> {
    let provider = prepared.provider;
    let control = prepared.control;
    let mixed = matches!(
        prepared.mode,
        RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
    );
    // Mixed arithmetic deliberately constructs the Double scalar from the
    // original class/bits. Native Single on a Double provider uses a typed
    // F32 upload so fill_like cannot silently promote it.
    if prepared.exact_scalar_upload() {
        let shape = prepared.host_shape.as_slice();
        let single = [match control {
            PowerScalarControl::SingleBits(bits) => f32::from_bits(bits),
            PowerScalarControl::DoubleBits(_) => 0.0,
        }];
        let double = [if mixed {
            control.value()
        } else {
            match control {
                PowerScalarControl::DoubleBits(bits) => f64::from_bits(bits),
                _ => 0.0,
            }
        }];
        let data = if mixed || control.precision() == ProviderPrecision::F64 {
            HostNumericDataView::F64(&double)
        } else {
            HostNumericDataView::F32(&single)
        };
        let view = HostNumericTensorView {
            data,
            shape,
            storage: GpuTensorStorage::Real,
        };
        let output = upload_numeric_protecting_admitted(
            provider,
            &view,
            &prepared.protected_with(&[&prepared.resident, native]),
            admission,
        )
        .map_err(|error| {
            let carrier = error.into_execution_error();
            if prior_effect {
                PowerServiceError::after(carrier)
            } else {
                PowerServiceError(carrier)
            }
        })?;
        return Ok(output);
    }
    let _guard = ResidencyRestoreGuard::new(native);
    let output = provider
        .fill_like(native, control.value())
        .map_err(|error| {
            if prior_effect {
                PowerServiceError::after(error)
            } else {
                PowerServiceError::unproven(error)
            }
        })?
        .ok_or_else(|| {
            let error = anyhow::anyhow!("POWER scalar constructor unavailable");
            if prior_effect {
                PowerServiceError::after(error)
            } else {
                PowerServiceError::unproven(error)
            }
        })?;
    let precision = if mixed {
        ProviderPrecision::F64
    } else {
        crate::handle_precision(native).ok_or_else(|| {
            PowerServiceError::after(anyhow::anyhow!("resident precision disappeared"))
        })?
    };
    if !valid_output(
        &output,
        provider,
        native.shape(),
        precision,
        GpuTensorStorage::Real,
        &prepared.protected_with(&[native]),
    ) {
        return Err(reject(output, "invalid POWER scalar preparation"));
    }
    Ok(output)
}

fn reject_with_cause(output: GpuTensorHandle, error: PowerServiceError) -> PowerServiceError {
    let primary = error.into_execution_error().after_effect();
    match output.release().err() {
        Some(cleanup) => PowerServiceError(
            ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                Some(primary),
                vec![cleanup],
            ))
            .after_effect(),
        ),
        None => PowerServiceError(primary),
    }
}

fn revalidate_after_effect(
    prepared: &PreparedResidentScalarPower,
) -> Result<(), PowerServiceError> {
    prepared
        .validate_live()
        .map_err(|error| PowerServiceError::after(error.into_execution_error()))
}
