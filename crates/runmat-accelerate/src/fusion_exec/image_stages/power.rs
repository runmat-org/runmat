//! Selected POWER projection and live execution through the shared API owner.
//! Symbolic proposals do not allocate or grant the held image route placement.

use super::*;
use runmat_accelerate_api::{
    power_service::{
        execute_resident_scalar_power, plan_scalar_power_for_dense_output,
        prepare_resident_scalar_power, prepare_resident_scalar_power_against_plan,
        PowerSymbolicPlan,
    },
    PowerResultDomain, PowerScalarControl, ProviderElementType, ProviderImagePowerCall,
    ProviderLayout, ProviderOperationIdentity, ProviderRepresentation, ProviderResidency,
    ProviderStorage, RealPowerOperand,
};
use runmat_types::NumericClass;

pub(super) struct ImagePowerPlans {
    pub square: PowerSymbolicPlan,
    pub final_power: PowerSymbolicPlan,
}

fn future_representation(class: NumericClass, shape: [u64; 3]) -> Option<ProviderRepresentation> {
    Some(ProviderRepresentation {
        element_type: match class {
            NumericClass::Single => ProviderElementType::F32,
            NumericClass::Double => ProviderElementType::F64,
            _ => return None,
        },
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.to_vec(),
        residency: ProviderResidency::Device,
    })
}

pub(super) fn plan_selected(
    provider: &'static dyn AccelProvider,
    program: &ProviderImageStageWorkload,
) -> ProviderExecutionResult<Option<ImagePowerPlans>> {
    let Some(powers) = program.selected_powers.as_ref() else {
        return Ok(None);
    };
    if !powers.validate() || powers.square.base_class != NumericClass::Single {
        return Ok(None);
    }
    let plan = |call: &ProviderImagePowerCall| {
        plan_scalar_power_for_dense_output(
            provider,
            future_representation(call.base_class, program.input_shape).ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("selected POWER base class is unavailable"))
            })?,
            call.control,
            call.resident_operand,
            &call
                .exponent_shape
                .iter()
                .map(|&extent| usize::try_from(extent))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| ProviderExecutionError::from(anyhow::Error::from(error)))?,
            call.operation.clone(),
        )
        .map_err(|error| error.into_execution_error())
    };
    let (Some(square), Some(final_power)) = (plan(&powers.square)?, plan(&powers.final_power)?)
    else {
        return Ok(None);
    };
    let expected_shape = program
        .input_shape
        .iter()
        .map(|&extent| usize::try_from(extent))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ProviderExecutionError::from(anyhow::Error::from(error)))?;
    if square.mode() != powers.square.mode
        || final_power.mode() != powers.final_power.mode
        || square.output_shape() != expected_shape.as_slice()
        || final_power.output_shape() != expected_shape.as_slice()
    {
        return Ok(None);
    }
    Ok(Some(ImagePowerPlans {
        square,
        final_power,
    }))
}

impl ImageNormalizeStageSession {
    pub(super) async fn execute_stage_power(
        &self,
        base: &GpuTensorHandle,
        selected: Option<(&ProviderImagePowerCall, &PowerSymbolicPlan)>,
        legacy_single_bits: u32,
        protected: &[&GpuTensorHandle],
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        if self.program.selected_powers.is_some() && selected.is_none() {
            return Err(
                ProviderExecutionError::from(anyhow!("selected POWER plan is missing"))
                    .after_effect(),
            );
        }
        let prepared = if let Some((call, plan)) = selected {
            prepare_resident_scalar_power_against_plan(
                self.provider,
                base,
                plan,
                call.control,
                call.resident_operand,
                &call
                    .exponent_shape
                    .iter()
                    .map(|&extent| usize::try_from(extent))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| ProviderExecutionError::from(anyhow::Error::from(error)))?,
                call.operation.clone(),
            )
            .map_err(|error| error.into_execution_error().after_effect())?
        } else {
            // This path is only the explicit held legacy fixture. Its F32 bits
            // never stand in for an original selected Double control.
            prepare_resident_scalar_power(
                self.provider,
                base,
                PowerScalarControl::SingleBits(legacy_single_bits),
                RealPowerOperand::Base,
                &[1, 1],
                ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name),
            )
            .map_err(|error| error.into_execution_error().after_effect())?
            .ok_or_else(|| {
                ProviderExecutionError::from(anyhow!("held legacy POWER route is unavailable"))
                    .after_effect()
            })?
        };
        let prepared = prepared
            .protect_borrowed(protected)
            .map_err(|error| error.into_execution_error().after_effect())?;
        let executed = execute_resident_scalar_power(prepared)
            .await
            .map_err(|error| error.into_execution_error().after_effect())?;
        if executed.domain != PowerResultDomain::Real {
            let cleanup = executed.handle.release().err();
            let primary = ProviderExecutionError::from(anyhow!(
                "selected image POWER returned a complex result"
            ))
            .after_effect();
            return Err(match cleanup {
                Some(cleanup) => ProviderExecutionError::from_source(
                    ProviderExecutionCompletionError::new(Some(primary), vec![cleanup]),
                )
                .after_effect(),
                None => primary,
            });
        }
        Ok(executed.handle)
    }
}
