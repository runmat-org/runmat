//! Uniform data ABI selection and legacy-compatible dispatch. Source arithmetic
//! is still admitted by the typed producer, never by this representation choice.
use super::*;
use crate::fusion::{semantics::ResolvedProgram, typed_shader::GeneratedElementwiseShader};
use runmat_accelerate_api::{
    ProviderFusedElementwiseAbi as Abi, ProviderFusedElementwiseRequest as Request,
};
use runmat_runtime::builtins::common::spec::ScalarType;
use runmat_types::NumericClass;

#[derive(Clone, Copy)]
pub(super) enum ShaderBodyPreparation {
    Unbounded,
    Counted {
        single: Option<usize>,
        provider_default: Option<usize>,
    },
}

impl ShaderBodyPreparation {
    pub(super) fn single_capacity(self) -> Option<usize> {
        match self {
            Self::Unbounded => None,
            Self::Counted { single, .. } => single,
        }
    }

    pub(super) fn default_capacity(self) -> Option<usize> {
        match self {
            Self::Unbounded => None,
            Self::Counted {
                provider_default, ..
            } => provider_default,
        }
    }
}

/// Only a typed missing Single recipe can use the already-admitted default
/// recipe. In finite mode every other failure invalidates that preparation.
fn single_shader_may_use_default(
    error: &anyhow::Error,
    preparation: ShaderBodyPreparation,
) -> bool {
    if error
        .downcast_ref::<crate::fusion::typed_shader::CountedGeometryViolation>()
        .is_some()
        || error.downcast_ref::<std::fmt::Error>().is_some()
        || error
            .downcast_ref::<std::collections::TryReserveError>()
            .is_some()
        || matches!(
            error.downcast_ref::<runmat_runtime::builtins::common::spec::FusionError>(),
            Some(runmat_runtime::builtins::common::spec::FusionError::ExpressionTooLarge)
        )
    {
        return false;
    }
    match preparation {
        ShaderBodyPreparation::Counted { .. } => error
            .downcast_ref::<crate::fusion::typed_producer::CountedRecipeUnavailable>()
            .is_some(),
        ShaderBodyPreparation::Unbounded => true,
    }
}

pub(super) fn may_try_single(
    plan: &FusionGroupPlan,
    outputs: &[ValueId],
    program: &ResolvedProgram,
    provider: &dyn AccelProvider,
    inputs: &[Value],
    indices: &[usize],
) -> Result<bool> {
    let abi = Abi::BroadcastReal {
        precision: ProviderPrecision::F32,
    };
    if provider.precision() != ProviderPrecision::F64
        || !provider.supports_fused_elementwise_abi(abi)
        || !outputs
            .iter()
            .all(|id| program.floating_class(*id).ok() == Some(NumericClass::Single))
        || !plan.operations.iter().all(|operation| {
            let output = match operation {
                crate::fusion::FusionOp::Primitive { output, .. }
                | crate::fusion::FusionOp::Builtin { output, .. } => *output,
            };
            output.is_some_and(|id| program.floating_class(id).ok() == Some(NumericClass::Single))
        })
        || !program.physical_inputs_are_single(indices)
    {
        return Ok(false);
    }
    Ok(indices.iter().all(|&index| match inputs.get(index) {
        Some(Value::Tensor(tensor)) => tensor.numeric_dtype() == NumericDType::F32,
        Some(Value::GpuTensor(handle)) => {
            handle.physical().element_type() == runmat_accelerate_api::NumericElementType::F32
                && handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
        }
        _ => false,
    }))
}

pub(super) fn physical_indices(
    plan: &FusionGroupPlan,
    outputs: &[ValueId],
    program: &ResolvedProgram,
    precision: ProviderPrecision,
) -> Result<Vec<usize>> {
    let controls = program.embedded_controls_sorted_for_precision(plan, outputs, precision)?;
    let mut indices = Vec::new();
    indices.try_reserve_exact(plan.inputs.len())?;
    indices.extend(
        plan.inputs
            .iter()
            .enumerate()
            .filter_map(|(index, id)| (controls.binary_search(id).is_err()).then_some(index)),
    );
    Ok(indices)
}

pub(super) fn select_shader(
    plan: &FusionGroupPlan,
    outputs: &[ValueId],
    program: &ResolvedProgram,
    provider: &dyn AccelProvider,
    inputs: &[Value],
    preparation: ShaderBodyPreparation,
) -> Result<(
    GeneratedElementwiseShader,
    Option<runmat_accelerate_api::ProviderFusedElementwiseWorkload>,
)> {
    let abi = Abi::BroadcastReal {
        precision: ProviderPrecision::F32,
    };
    let indices = physical_indices(plan, outputs, program, ProviderPrecision::F32)?;
    if may_try_single(plan, outputs, program, provider, inputs, &indices)? {
        match plan.prepare_typed_shader_with_inputs_bounded(
            outputs,
            program,
            ScalarType::F32,
            &indices,
            preparation.single_capacity(),
        ) {
            Ok(shader) => {
                let work = describe_inputs(abi, &indices, inputs)?;
                return Ok((shader, Some(work)));
            }
            Err(error) if single_shader_may_use_default(&error, preparation) => {}
            Err(error) => return Err(error),
        }
    }
    let scalar = match provider.precision() {
        ProviderPrecision::F32 => ScalarType::F32,
        ProviderPrecision::F64 => ScalarType::F64,
    };
    if matches!(
        preparation,
        ShaderBodyPreparation::Counted {
            provider_default: None,
            ..
        }
    ) {
        return Err(crate::fusion::typed_producer::CountedRecipeUnavailable(
            "default fusion ABI has no counted scalar recipe",
        )
        .into());
    }
    let indices = physical_indices(plan, outputs, program, provider.precision())?;
    Ok((
        plan.prepare_typed_shader_with_inputs_bounded(
            outputs,
            program,
            scalar,
            &indices,
            preparation.default_capacity(),
        )?,
        None,
    ))
}

pub(super) fn describe_inputs(
    abi: Abi,
    indices: &[usize],
    inputs: &[Value],
) -> Result<runmat_accelerate_api::ProviderFusedElementwiseWorkload> {
    let (elements, maximum) = indices
        .iter()
        .try_fold((0_u64, 0_u64), |(count, maximum), &index| {
            let extra = match inputs.get(index) {
                Some(Value::Tensor(tensor)) if tensor.numeric_dtype() == NumericDType::F32 => {
                    if tensor.as_f32_slice().is_none() {
                        u64::try_from(tensor.len()).ok()?
                    } else {
                        0
                    }
                }
                Some(Value::GpuTensor(handle))
                    if handle.physical().element_type()
                        == runmat_accelerate_api::NumericElementType::F32 =>
                {
                    0
                }
                _ => return None,
            };
            Some((count.checked_add(extra)?, maximum.max(extra)))
        })
        .ok_or_else(|| anyhow!("typed fusion input preparation changed or overflowed"))?;
    let work = runmat_accelerate_api::ProviderFusedElementwiseWorkload {
        abi,
        host_materialization_elements: elements,
        host_workspace_bytes: maximum
            .checked_mul(4)
            .ok_or_else(|| anyhow!("typed fusion host workspace overflows"))?,
    };
    work.checked_host_workspace_bytes()
        .ok_or_else(|| anyhow!("typed fusion host workspace overflows"))?;
    Ok(work)
}

pub(super) fn validate_inputs(
    plan: &FusionGroupPlan,
    program: &ResolvedProgram,
    indices: &[usize],
    inputs: &[Value],
    work: runmat_accelerate_api::ProviderFusedElementwiseWorkload,
) -> Result<()> {
    if describe_inputs(work.abi, indices, inputs)? != work {
        return Err(anyhow!("typed fusion host view preparation changed"));
    }
    for &index in indices {
        let value = inputs
            .get(index)
            .ok_or_else(|| anyhow!("typed fusion input disappeared"))?;
        let actual = runmat_runtime::value_fact::value_fact(value);
        let expected = program.fact(plan.inputs[index])?;
        if actual.shape != expected.shape
            || actual
                .numeric()
                .is_none_or(|fact| fact.class != NumericClass::Single)
        {
            return Err(anyhow!("typed fusion input class or shape changed"));
        }
        if let Value::GpuTensor(handle) = value {
            handle.allocation_lease().validate_live()?;
            handle.view().validate(handle.physical())?;
        }
        if let Value::Tensor(tensor) = value {
            tensor.validate_shape().map_err(anyhow::Error::msg)?;
        }
    }
    Ok(())
}

pub(super) async fn dispatch_single(
    provider: &dyn AccelProvider,
    typed: Option<runmat_accelerate_api::ProviderFusedElementwiseWorkload>,
    request: Request<'_>,
) -> ExecutionResult<GpuTensorHandle> {
    if let Some(work) = typed {
        if work.abi != request.abi {
            return Err(anyhow!("fallback shader ABI changed").into());
        }
        provider.fused_elementwise_typed(request).await
    } else {
        let [class] = request.output_classes else {
            return Err(anyhow!("single fusion requires one output").into());
        };
        provider
            .fused_elementwise(
                request.shader,
                request.requirements,
                request.inputs,
                request.output_shape,
                request.len,
                *class,
            )
            .await
    }
}

pub(super) async fn dispatch_multi(
    provider: &dyn AccelProvider,
    typed: Option<runmat_accelerate_api::ProviderFusedElementwiseWorkload>,
    request: Request<'_>,
) -> ExecutionResult<Vec<GpuTensorHandle>> {
    if let Some(work) = typed {
        if work.abi != request.abi {
            return Err(anyhow!("multi shader ABI changed").into());
        }
        provider.fused_elementwise_multi_typed(request).await
    } else {
        provider
            .fused_elementwise_multi(
                request.shader,
                request.requirements,
                request.inputs,
                request.output_shape,
                request.len,
                request.output_classes,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::{single_shader_may_use_default, ShaderBodyPreparation};

    #[test]
    fn counted_single_route_only_falls_back_for_missing_recipe() {
        let counted = ShaderBodyPreparation::Counted {
            single: Some(32),
            provider_default: Some(64),
        };
        let unavailable =
            crate::fusion::typed_producer::CountedRecipeUnavailable("single recipe unavailable");
        assert!(single_shader_may_use_default(&unavailable.into(), counted));
        for error in [
            crate::fusion::typed_shader::CountedGeometryViolation::BodyCapacity {
                admitted: 4,
                actual: 5,
            },
            crate::fusion::typed_shader::CountedGeometryViolation::ShellCapacity {
                admitted: 8,
                actual: 9,
            },
            crate::fusion::typed_shader::CountedGeometryViolation::SourceLength {
                expected: 12,
                actual: 11,
            },
        ] {
            assert!(!single_shader_may_use_default(&error.into(), counted));
        }
        assert!(!single_shader_may_use_default(
            &anyhow::anyhow!("unclassified finite preparation failure"),
            counted,
        ));
        assert!(single_shader_may_use_default(
            &anyhow::anyhow!("legacy structural single route"),
            ShaderBodyPreparation::Unbounded,
        ));
    }
}
