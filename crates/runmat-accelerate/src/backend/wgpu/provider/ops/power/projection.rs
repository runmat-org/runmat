//! Future dense sources use the same flag/reduction/evaluation calculators as
//! live paired execution. This module never registers or allocates a buffer.
use super::super::reduction::FrozenAnySnapshot;
use super::*;
use runmat_accelerate_api::{
    power_service::{PowerPairPhysicalProjection, PowerProbePhysicalProposal},
    NumericElementType, PowerInputDescription, ProviderElementType, ProviderExecutionError,
    ProviderExecutionResult, ProviderLayout, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};

impl WgpuProvider {
    pub(crate) fn propose_power_pair_geometry_exec(
        &self,
        inputs: &[ProviderRepresentation; 2],
        descriptions: &[PowerInputDescription; 2],
        frozen: Option<&PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<Option<PowerPairPhysicalProjection>> {
        let input_precision = match inputs[0].element_type {
            ProviderElementType::F32 => NumericPrecision::F32,
            ProviderElementType::F64 => NumericPrecision::F64,
            _ => return Ok(None),
        };
        if inputs[1].element_type != inputs[0].element_type
            || (input_precision != self.precision
                && !(input_precision == NumericPrecision::F32
                    && self.precision == NumericPrecision::F64))
        {
            return Ok(None);
        }
        let mut shapes = Vec::with_capacity(2);
        for (input, description) in inputs.iter().zip(descriptions) {
            if input.storage != ProviderStorage::DenseReal
                || input.layout != ProviderLayout::ColumnMajorContiguous
                || input.residency != ProviderResidency::Device
            {
                return Ok(None);
            }
            let shape = input
                .shape
                .iter()
                .copied()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(ProviderExecutionError::from_source)?;
            match description {
                PowerInputDescription::PlannedDense { requested_bytes } => {
                    if self.plan_dense_real_output_buffer_bytes_exec(input)?
                        != Some(*requested_bytes)
                    {
                        return Ok(None);
                    }
                }
                PowerInputDescription::Live { physical, view } => {
                    view.validate(physical)
                        .map_err(ProviderExecutionError::from_source)?;
                    let element = match input_precision {
                        NumericPrecision::F32 => NumericElementType::F32,
                        NumericPrecision::F64 => NumericElementType::F64,
                    };
                    if view.shape() != shape
                        || view.is_logical()
                        || view.layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
                        || physical.element_type() != element
                        || physical.storage() != GpuTensorStorage::Real
                    {
                        return Ok(None);
                    }
                }
            }
            shapes.push(shape);
        }
        let shape = runmat_types::geometry::broadcast_concrete_shape(&shapes[0], &shapes[1])
            .map_err(ProviderExecutionError::from_source)?;
        let len = if shape.contains(&0) {
            0
        } else {
            shape
                .iter()
                .try_fold(1usize, |count, &extent| count.checked_mul(extent))
                .ok_or_else(|| anyhow!("planned POWER output extent overflow"))?
        };
        if len == 0 {
            return Ok(None);
        }
        let snapshot =
            frozen.map_or_else(FrozenAnySnapshot::capture, |physical| FrozenAnySnapshot {
                enabled: physical.snapshot_copy,
            });
        let probe = self.project_power_probe_geometry(input_precision, &shape, len, snapshot)?;
        let Some(real) = self.prepare_power_evaluation_geometry(
            input_precision,
            shape.clone(),
            len,
            PowerResultDomain::Real,
        )?
        else {
            return Ok(None);
        };
        let Some(complex) = self.prepare_power_evaluation_geometry(
            input_precision,
            shape,
            len,
            PowerResultDomain::Complex,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(PowerPairPhysicalProjection {
            probe,
            real: real.physical()?,
            complex: complex.physical()?,
        }))
    }
}
