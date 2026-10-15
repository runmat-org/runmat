//! Uniform data ABI selection and legacy-compatible dispatch. Source arithmetic
//! is still admitted by the typed producer, never by this representation choice.
use super::*;
use crate::fusion::{semantics::ResolvedProgram, typed_shader::GeneratedElementwiseShader};
use runmat_accelerate_api::{
    ProviderFusedElementwiseAbi as Abi, ProviderFusedElementwiseRequest as Request,
};
use runmat_runtime::builtins::common::spec::ScalarType;
use runmat_types::NumericClass;

pub(super) fn select_shader(
    plan: &FusionGroupPlan,
    outputs: &[ValueId],
    program: &ResolvedProgram,
    classes: &[NumericClass],
    provider: &dyn AccelProvider,
    inputs: &[Value],
) -> Result<(
    GeneratedElementwiseShader,
    Option<runmat_accelerate_api::ProviderFusedElementwiseWorkload>,
)> {
    let abi = Abi::BroadcastReal {
        precision: ProviderPrecision::F32,
    };
    let controls = program.embedded_scalar_controls(plan, outputs);
    let indices = plan
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(index, id)| (!controls.contains(id)).then_some(index))
        .collect::<Vec<_>>();
    let authentic_single = indices.iter().all(|&index| match inputs.get(index) {
        Some(Value::Tensor(tensor)) => tensor.numeric_dtype() == NumericDType::F32,
        Some(Value::GpuTensor(handle)) => {
            handle.physical().element_type() == runmat_accelerate_api::NumericElementType::F32
                && handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
        }
        _ => false,
    });
    if provider.precision() == ProviderPrecision::F64
        && provider.supports_fused_elementwise_abi(abi)
        && classes.iter().all(|class| *class == NumericClass::Single)
        && program.physical_inputs_are_single(&indices)
        && authentic_single
    {
        if let Ok(shader) =
            plan.prepare_typed_shader_with_inputs(outputs, program, ScalarType::F32, &indices)
        {
            let work = describe_inputs(abi, &indices, inputs)?;
            return Ok((shader, Some(work)));
        }
    }
    let scalar = match provider.precision() {
        ProviderPrecision::F32 => ScalarType::F32,
        ProviderPrecision::F64 => ScalarType::F64,
    };
    Ok((plan.prepare_typed_shader(outputs, program, scalar)?, None))
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
