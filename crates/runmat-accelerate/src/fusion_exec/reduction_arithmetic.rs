//! Request-local arithmetic admission for the selected legacy reduction
//! producer. This runs before uploads or kernel submission and can suspend for
//! operation-owned domain proof.

use super::{admission::AdmittedBindings, FusionExecutionRequest};
use crate::fusion::reduction_emission::ReductionProducerEmission;
use crate::fusion::semantics::ResolvedProgram;
use crate::fusion::FusionGroupPlan;
use runmat_accelerate_api::{ProviderExecutionError, ProviderPrecision};

#[path = "reduction_program/legacy_producer.rs"]
mod legacy_producer;

fn has_required_arithmetic(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
) -> bool {
    (0..request.plan.operations.len()).any(|index| {
        bindings.semantic(index).is_some_and(|snapshot| {
            snapshot
                .declaration
                .semantics
                .is_some_and(|semantics| semantics.required_arithmetic.is_some())
        })
    })
}

/// Only consumed producer leaves need the typed constant owner. Reduction
/// controls and unused cached values do not manufacture shader dependencies.
fn has_captured_operands(producer: &FusionGroupPlan) -> bool {
    producer
        .output
        .is_some_and(|id| producer.const_values.contains_key(&id))
        || producer.operations.iter().any(|operation| {
            let inputs = match operation {
                crate::fusion::FusionOp::Primitive { inputs, .. }
                | crate::fusion::FusionOp::Builtin { inputs, .. } => inputs,
            };
            inputs
                .iter()
                .any(|id| producer.const_values.contains_key(id))
        })
}

pub(super) enum PreparedReductionArithmetic {
    Unneeded,
    /// Authenticated arithmetic requires the projected checked recipe. The
    /// legacy raw-source ABI must never stage its metadata controls as data.
    TypedOnly,
    Selected(Box<SelectedReductionArithmetic>),
}

pub(super) struct SelectedReductionArithmetic {
    source: FusionGroupPlan,
    program: ResolvedProgram,
    emission: ReductionProducerEmission,
    precision: ProviderPrecision,
}

impl PreparedReductionArithmetic {
    pub(super) fn requires_typed(&self) -> bool {
        matches!(self, Self::TypedOnly)
    }
    pub(super) fn validate(
        &self,
        plan: &FusionGroupPlan,
        inputs: &[runmat_value::Value],
        precision: ProviderPrecision,
        reduce_len: usize,
        num_slices: usize,
    ) -> anyhow::Result<()> {
        if let Self::Selected(selected) = self {
            let SelectedReductionArithmetic {
                source,
                program,
                emission,
                precision: selected_precision,
            } = selected.as_ref();
            if *selected_precision != precision {
                anyhow::bail!("selected reduction precision changed after admission");
            }
            let (current, _) = legacy_producer::project(plan)?;
            program.require_source_plan(&current)?;
            program.require_source_plan(source)?;
            emission.require(
                source
                    .output
                    .ok_or_else(|| anyhow::anyhow!("selected reduction producer has no root"))?,
                precision,
                program,
                source,
                inputs,
            )?;
            emission.require_geometry(
                plan.legacy_reduction_axis()
                    .ok_or_else(|| anyhow::anyhow!("selected reduction axis unavailable"))?,
                reduce_len,
                num_slices,
            )?;
        }
        Ok(())
    }
    pub(super) fn emission(&self) -> Option<&ReductionProducerEmission> {
        match self {
            Self::Unneeded | Self::TypedOnly => None,
            Self::Selected(selected) => Some(&selected.emission),
        }
    }
}

fn selected_source(
    producer: FusionGroupPlan,
    program: ResolvedProgram,
    precision: ProviderPrecision,
    inputs: &[runmat_value::Value],
) -> Result<Option<PreparedReductionArithmetic>, ProviderExecutionError> {
    let unavailable = |error: anyhow::Error| {
        if inputs
            .iter()
            .any(runmat_runtime::dispatcher::value_contains_gpu)
        {
            Err(ProviderExecutionError::from(error))
        } else {
            Ok(None)
        }
    };
    for operation in &producer.operations {
        let output = match operation {
            crate::fusion::FusionOp::Primitive { output, .. }
            | crate::fusion::FusionOp::Builtin { output, .. } => *output,
        };
        if let Some(output) = output {
            // The legacy raw reduction frame stages every source leaf as
            // data. Certified metadata controls require the checked recipe.
            if program.power_arithmetic(output).is_some_and(|receipt| {
                matches!(
                    receipt.mode,
                    runmat_accelerate_api::RealPowerArithmeticMode::MixedDoubleThenSingle { .. }
                )
            }) {
                if program.exact_single_base_power(output).is_none() {
                    return unavailable(anyhow::anyhow!(
                        "selected reduction has no certified mixed POWER arithmetic"
                    ));
                }
                // Validate the full checked producer and physical projection
                // in the shared typed recipe owner, never in the legacy frame.
                return Ok(Some(PreparedReductionArithmetic::TypedOnly));
            }
            if let Err(error) = program.require_ordinary_arithmetic(output, precision) {
                return unavailable(error);
            }
        }
    }
    let emission = match ReductionProducerEmission::new(&producer, inputs, &program, precision) {
        Ok(emission) => emission,
        Err(error) => return unavailable(error),
    };
    Ok(Some(PreparedReductionArithmetic::Selected(Box::new(
        SelectedReductionArithmetic {
            source: producer,
            program,
            emission,
            precision,
        },
    ))))
}

pub(super) async fn admitted(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    precision: ProviderPrecision,
) -> Result<Option<PreparedReductionArithmetic>, ProviderExecutionError> {
    // Typed reduction mapping can derive its root from the selected terminal
    // even when no legacy axis/root description exists. Preserve that route;
    // ordinary typed recipes prepare their own scalar leaves as data.
    if request.plan.reduction_data.is_none() && !has_required_arithmetic(request, bindings) {
        return Ok(Some(PreparedReductionArithmetic::Unneeded));
    }
    let (producer, indices) = legacy_producer::project(request.plan)?;
    let snapshots: Vec<_> = indices
        .iter()
        .map(|&index| bindings.semantic(index))
        .collect();
    if !has_captured_operands(&producer)
        && !snapshots.iter().flatten().any(|snapshot| {
            snapshot
                .declaration
                .semantics
                .is_some_and(|semantics| semantics.required_arithmetic.is_some())
        })
    {
        return Ok(Some(PreparedReductionArithmetic::Unneeded));
    }
    // Async admission owns one observed proof pass, including rejected host
    // scans. A static prepass would repeat that work without accounting for it.
    let values: Vec<_> = request.inputs.iter().collect();
    let program = match ResolvedProgram::resolve_captured(
        &producer,
        &values,
        precision,
        request.placement,
        &snapshots,
    )
    .await
    {
        Ok(program) => program,
        Err(error) => {
            return match error.into_execution_or_structural() {
                Ok(error)
                    if request
                        .inputs
                        .iter()
                        .any(runmat_runtime::dispatcher::value_contains_gpu) =>
                {
                    Err(ProviderExecutionError::from(error))
                }
                Ok(_) => Ok(None),
                Err(error) => Err(error),
            }
        }
    };
    selected_source(producer, program, precision, &request.inputs)
}

pub(super) fn admitted_static(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    precision: ProviderPrecision,
) -> Result<Option<PreparedReductionArithmetic>, ProviderExecutionError> {
    let (producer, indices) = legacy_producer::project(request.plan)?;
    let snapshots: Vec<_> = indices
        .iter()
        .map(|&index| bindings.semantic(index))
        .collect();
    let values: Vec<_> = request.inputs.iter().collect();
    let program =
        match ResolvedProgram::resolve_static_captured(&producer, &values, precision, &snapshots) {
            Ok(Some(program)) => program,
            Ok(None) => return Ok(None),
            Err(error) => {
                return match error.into_execution_or_structural() {
                    Ok(error)
                        if request
                            .inputs
                            .iter()
                            .any(runmat_runtime::dispatcher::value_contains_gpu) =>
                    {
                        Err(ProviderExecutionError::from(error))
                    }
                    Ok(_) => Ok(None),
                    Err(error) => Err(error),
                }
            }
        };
    selected_source(producer, program, precision, &request.inputs)
}

#[cfg(test)]
#[path = "reduction_arithmetic/tests.rs"]
mod tests;
