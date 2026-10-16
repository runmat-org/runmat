//! Execution of the pre-effect native scalar projection. The only domain
//! witness is the paired recipe's actually consumed probe.
use super::*;
use crate::power_service::{PowerPhysicalRoute, PowerTypedScalarProjection};

pub async fn execute_resident_scalar_power_projected(
    prepared: PreparedResidentScalarPower,
    projection: &PowerTypedScalarProjection,
    cancellation: Option<&runmat_execution::cancellation::CancellationToken>,
) -> Result<ExecutedResidentScalarPower, PowerServiceError> {
    execute_resident_scalar_power_projected_admitted(prepared, projection, cancellation, None).await
}

pub async fn execute_resident_scalar_power_projected_admitted(
    prepared: PreparedResidentScalarPower,
    projection: &PowerTypedScalarProjection,
    cancellation: Option<&runmat_execution::cancellation::CancellationToken>,
    admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> Result<ExecutedResidentScalarPower, PowerServiceError> {
    crate::placement::compound_execution::check_cancelled(cancellation, false)
        .map_err(PowerServiceError)?;
    prepared.validate_live()?;
    if prepared.route != PowerPhysicalRoute::TypedDomain
        || !projection.matches_bound(prepared.provider, &prepared.symbolic, &prepared.resident)
    {
        return Err(PowerServiceError::pre(
            "typed POWER scalar projection changed before execution",
        ));
    }
    if admission.is_some() && !prepared.exact_scalar_upload() {
        return Err(PowerServiceError::pre(
            "finite typed POWER requires an admitted scalar upload",
        ));
    }
    if admission.is_some() {
        let expected = prepared
            .output_shape
            .iter()
            .copied()
            .map(u64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| PowerServiceError::pre("finite typed POWER output shape overflows"))?;
        if projection.pair.physical().real.output.shape.as_slice() != expected.as_slice()
            || projection.pair.physical().complex.output.shape.as_slice() != expected.as_slice()
        {
            return Err(PowerServiceError::pre(
                "finite typed POWER requires shape-preserving selected outputs",
            ));
        }
    }
    if !projection.is_current_for(prepared.provider) {
        return Err(PowerServiceError::pre(
            "typed POWER scalar projection changed before execution",
        ));
    }
    let projection_identity = projection.identity().map_err(PowerServiceError::from)?;
    let _protected_metadata = prepared
        .protected_with(&[])
        .into_iter()
        .map(ResidencyRestoreGuard::new)
        .collect::<Vec<_>>();
    let provider = prepared.provider;
    let resident = &prepared.resident;
    let widened = if let Some(stage) = projection.widened {
        let _guard = ResidencyRestoreGuard::new(resident);
        let output = provider
            .unary_double_admitted(resident, admission)
            .await
            .map_err(PowerServiceError::after)?;
        if !valid_output(
            &output,
            provider,
            resident.shape(),
            ProviderPrecision::F64,
            GpuTensorStorage::Real,
            &prepared.protected_with(&[resident]),
        ) || output.physical().requested_bytes() != stage.output_requested_bytes
        {
            return Err(reject(
                output,
                "typed POWER widening differs from projection",
            ));
        }
        Some(output)
    } else {
        None
    };
    if widened.is_some() {
        revalidate_after_effect(&prepared)?;
    }
    crate::placement::compound_execution::check_cancelled(cancellation, widened.is_some())
        .map_err(PowerServiceError)?;
    if !projection.is_current_for(provider) {
        return Err(if widened.is_some() {
            PowerServiceError::after(anyhow::anyhow!(
                "typed POWER scalar stages changed before upload"
            ))
        } else {
            PowerServiceError::pre("typed POWER scalar stages changed before upload")
        });
    }
    let native = widened.as_ref().unwrap_or(resident);
    let scalar = prepare_scalar(&prepared, native, widened.is_some(), admission)?;
    if scalar.physical().requested_bytes() != projection.scalar_upload.output_requested_bytes {
        return Err(reject(
            scalar,
            "typed POWER scalar upload differs from projection",
        ));
    }
    crate::placement::compound_execution::check_cancelled(cancellation, true)
        .map_err(PowerServiceError)?;
    if !projection.is_current_for(provider) {
        return Err(PowerServiceError::after(anyhow::anyhow!(
            "typed POWER scalar stages changed after upload"
        )));
    }
    let (base, exponent) = match prepared.resident_operand {
        RealPowerOperand::Base => (native, &scalar),
        RealPowerOperand::Exponent => (&scalar, native),
    };
    revalidate_after_effect(&prepared)?;
    let pair = projection
        .pair
        .bind(base, exponent)
        .map_err(PowerServiceError::after)?;
    let (native_output, evidence) = pair
        .execute_admitted(cancellation, admission)
        .await
        .map_err(PowerServiceError::after)?
        .into_parts();
    let domain = evidence.domain();
    if let Err(error) = revalidate_after_effect(&prepared) {
        return Err(reject_with_cause(native_output, error));
    }
    if let Err(error) = crate::placement::compound_execution::check_cancelled(cancellation, true) {
        return Err(reject_with_cause(native_output, PowerServiceError(error)));
    }
    if !projection.is_current_for(provider) {
        return Err(reject_with_cause(
            native_output,
            PowerServiceError::after(anyhow::anyhow!(
                "typed POWER scalar stages changed before publication"
            )),
        ));
    }
    let published = if let Some(stage) = match domain {
        PowerResultDomain::Real => projection.narrow_real,
        PowerResultDomain::Complex => projection.narrow_complex,
    } {
        let result = {
            let _guard = ResidencyRestoreGuard::new(&native_output);
            provider
                .unary_single_admitted(&native_output, admission)
                .await
        };
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                return Err(reject_with_cause(
                    native_output,
                    PowerServiceError::after(error),
                ));
            }
        };
        if !valid_output(
            &output,
            provider,
            native_output.shape(),
            ProviderPrecision::F32,
            storage(domain),
            &prepared.protected_with(&[&native_output, resident, &scalar]),
        ) || output.physical().requested_bytes() != stage.output_requested_bytes
        {
            return Err(reject(
                output,
                "typed POWER Single publication differs from projection",
            ));
        }
        output
    } else {
        native_output
    };
    let published = if published.shape() == prepared.output_shape.as_slice() {
        published
    } else {
        if let Err(error) = revalidate_after_effect(&prepared) {
            return Err(reject_with_cause(published, error));
        }
        let output = match provider.reshape(&published, &prepared.output_shape) {
            Ok(output) => output,
            Err(error) => {
                return Err(reject_with_cause(
                    published,
                    PowerServiceError::after(error),
                ));
            }
        };
        if !output.shares_allocation_with(&published)
            || output.shape() != prepared.output_shape.as_slice()
            || output.physical() != published.physical()
            || output.allocation_lease().validate_live().is_err()
            || !crate::provider_for_handle(&output)
                .is_some_and(|owner| crate::same_provider_instance(owner, provider))
        {
            return Err(reject(
                output,
                "typed POWER reshape differs from projection",
            ));
        }
        output
    };
    if prepared
        .protected_with(&[])
        .iter()
        .any(|protected| published.shares_allocation_with(protected))
    {
        return Err(reject(
            published,
            "typed POWER result aliases protected lease",
        ));
    }
    if let Err(error) = prepared.validate_live() {
        return Err(reject_with_cause(published, error));
    }
    if let Err(error) = crate::placement::compound_execution::check_cancelled(cancellation, true) {
        return Err(reject_with_cause(published, PowerServiceError(error)));
    }
    let handle = publish_provenance(published, resident)?;
    let scalar_evidence =
        match crate::power_service::PowerScalarExecutionEvidence::from_consumed_pair(
            projection_identity,
            resident.view_identity(),
            &pair,
            &evidence,
        ) {
            Some(evidence) => evidence,
            None => {
                return Err(reject_with_cause(
                    handle,
                    PowerServiceError::after(anyhow::anyhow!(
                        "typed POWER consumed pair differs from frozen projection"
                    )),
                ));
            }
        };
    Ok(ExecutedResidentScalarPower {
        handle,
        domain,
        scalar_evidence: Some(scalar_evidence),
    })
}
