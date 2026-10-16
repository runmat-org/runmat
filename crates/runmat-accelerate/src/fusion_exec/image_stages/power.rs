//! Selected POWER projection and live execution through the shared API owner.
//! Symbolic proposals do not allocate or grant the held image route placement.

use super::*;
use runmat_accelerate_api::{
    power_service::{
        execute_resident_scalar_power_admitted, execute_resident_scalar_power_projected_admitted,
        plan_scalar_power_for_dense_output, prepare_resident_scalar_power,
        prepare_resident_scalar_power_against_plan, PowerPhysicalRoute, PowerSymbolicPlan,
        PowerTypedScalarProjection,
    },
    PowerResultDomain, PowerScalarControl, ProviderElementType, ProviderImagePowerCall,
    ProviderLayout, ProviderOperationIdentity, ProviderRepresentation, ProviderResidency,
    ProviderStorage, RealPowerOperand,
};
use runmat_types::NumericClass;

pub(super) struct ImagePowerPlans {
    pub square: SelectedImagePower,
    pub final_power: SelectedImagePower,
}

pub(super) struct SelectedImagePower {
    symbolic: PowerSymbolicPlan,
    physical: SelectedPowerPhysical,
}

enum SelectedPowerPhysical {
    Exact,
    Typed(Box<PowerTypedScalarProjection>),
}

impl std::ops::Deref for SelectedImagePower {
    type Target = PowerSymbolicPlan;

    fn deref(&self) -> &Self::Target {
        &self.symbolic
    }
}

impl SelectedImagePower {
    pub(super) fn output_requested_bytes(&self) -> Option<u64> {
        match &self.physical {
            SelectedPowerPhysical::Exact => {
                Some(self.symbolic.exact_physical_proposal()?.output_buffer_bytes)
            }
            SelectedPowerPhysical::Typed(projection) => {
                Some(projection.real_output_requested_bytes())
            }
        }
    }

    fn physical_upper_bound(&self) -> Option<super::resources::ImageStagePhysicalUpperBound> {
        let mut bound = super::resources::ImageStagePhysicalUpperBound::default();
        match &self.physical {
            SelectedPowerPhysical::Exact => {
                let physical = self.symbolic.exact_physical_proposal()?;
                bound.add(
                    physical.requested_incremental_peak_bytes()?,
                    physical.dispatches,
                    Some(physical.queue_submissions),
                )?;
            }
            SelectedPowerPhysical::Typed(projection) => {
                let resources = projection.resource_bounds()?;
                let runmat_execution::ResourceByteBound::Known(peak) = resources.peak_bytes else {
                    return None;
                };
                bound.add(peak, resources.dispatches, Some(resources.queue_slots))?;
            }
        }
        Some(bound)
    }

    fn registered_output_count(&self) -> Option<usize> {
        match &self.physical {
            SelectedPowerPhysical::Exact => Some(1),
            SelectedPowerPhysical::Typed(projection) => projection.registered_output_count(),
        }
    }

    fn provider_host_workspace_bytes(&self) -> Option<u64> {
        match &self.physical {
            SelectedPowerPhysical::Exact => Some(0),
            SelectedPowerPhysical::Typed(projection) => {
                Some(projection.resource_bounds()?.host_workspace_bytes)
            }
        }
    }
}

impl ImagePowerPlans {
    pub(super) fn physical_upper_bound(
        &self,
    ) -> Option<super::resources::ImageStagePhysicalUpperBound> {
        // The selected route must also have a finite registration inventory;
        // provider metadata admission consumes this same count before work.
        self.registered_output_count()?;
        let mut bound = super::resources::ImageStagePhysicalUpperBound::default();
        for plan in [&self.square, &self.final_power] {
            bound.combine(plan.physical_upper_bound()?)?;
        }
        Some(bound)
    }

    pub(super) fn registered_output_count(&self) -> Option<usize> {
        self.square
            .registered_output_count()?
            .checked_add(self.final_power.registered_output_count()?)
    }

    pub(super) fn provider_host_workspace_bytes(&self) -> Option<u64> {
        self.square
            .provider_host_workspace_bytes()?
            .checked_add(self.final_power.provider_host_workspace_bytes()?)
    }
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
    let selected =
        |symbolic: PowerSymbolicPlan| -> ProviderExecutionResult<Option<SelectedImagePower>> {
            let physical = match symbolic.physical_recipe().route {
                PowerPhysicalRoute::ExactSingleBase => {
                    if symbolic.exact_physical_proposal().is_none() {
                        return Ok(None);
                    }
                    SelectedPowerPhysical::Exact
                }
                PowerPhysicalRoute::TypedDomain => {
                    let Some(projection) = PowerTypedScalarProjection::plan(provider, &symbolic)?
                    else {
                        return Ok(None);
                    };
                    SelectedPowerPhysical::Typed(Box::new(projection))
                }
                PowerPhysicalRoute::LegacyProvenReal => return Ok(None),
            };
            Ok(Some(SelectedImagePower { symbolic, physical }))
        };
    let (Some(square), Some(final_power)) = (selected(square)?, selected(final_power)?) else {
        return Ok(None);
    };
    Ok(Some(ImagePowerPlans {
        square,
        final_power,
    }))
}

impl ImageNormalizeStageSession {
    pub(super) async fn execute_stage_power(
        &self,
        base: &GpuTensorHandle,
        selected: Option<(&ProviderImagePowerCall, &SelectedImagePower)>,
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
        let executed = match selected.map(|(_, plan)| &plan.physical) {
            Some(SelectedPowerPhysical::Typed(projection)) => {
                execute_resident_scalar_power_projected_admitted(
                    prepared,
                    projection,
                    None,
                    self.provider_metadata_permit.as_ref().map(|permit| {
                        permit.as_ref()
                            as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
                    }),
                )
                .await
            }
            _ => {
                execute_resident_scalar_power_admitted(
                    prepared,
                    self.provider_metadata_permit.as_ref().map(|permit| {
                        permit.as_ref()
                            as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
                    }),
                )
                .await
            }
        }
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
