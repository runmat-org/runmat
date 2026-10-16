//! Exact-owner, protected publication of host-computed eigen factors.

use super::*;

pub(super) fn upload_eig_eval_to_owner(
    eval: EigEval,
    source: &GpuTensorHandle,
    vector_output: bool,
    operands: &[GpuTensorHandle],
) -> BuiltinResult<EigEval> {
    // Tensor/complex carriers clone their shared host buffers. Preserve one
    // complete host envelope so an automatic unsupported transfer cannot
    // publish a mixed resident/host eigen bundle after earlier uploads.
    let host_fallback = eval.clone();
    let mut published = Vec::with_capacity(if eval.left.is_some() { 4 } else { 3 });
    let eigenvalues = publish_factor(eval.eigenvalues, source, operands, &mut published)?;
    if !matches!(&eigenvalues, Value::GpuTensor(_)) {
        return Ok(host_fallback);
    }
    let diagonal_matrix = publish_factor(eval.diagonal_matrix, source, operands, &mut published)?;
    if !matches!(&diagonal_matrix, Value::GpuTensor(_)) {
        return Ok(host_fallback);
    }
    let right = publish_factor(eval.right, source, operands, &mut published)?;
    if !matches!(&right, Value::GpuTensor(_)) {
        return Ok(host_fallback);
    }
    let left = eval
        .left
        .map(|value| publish_factor(value, source, operands, &mut published))
        .transpose()?;
    if left
        .as_ref()
        .is_some_and(|value| !matches!(value, Value::GpuTensor(_)))
    {
        return Ok(host_fallback);
    }
    // The public vector option projects the same eigenvalue factor; it does
    // not require a second allocation with an unrelated identity or class.
    let diagonal_output = if vector_output {
        eigenvalues.clone()
    } else {
        diagonal_matrix.clone()
    };
    Ok(EigEval {
        eigenvalues,
        diagonal_matrix,
        diagonal_output,
        right,
        left,
    })
}

fn publish_factor(
    value: Value,
    source: &GpuTensorHandle,
    operands: &[GpuTensorHandle],
    previous: &mut Vec<GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let intent_sources = operands.iter().collect::<Vec<_>>();
    let protected = intent_sources
        .iter()
        .copied()
        .chain(previous.iter())
        .collect::<Vec<_>>();
    let restored = gpu_helpers::restore_class_preserving_value_with_operands(
        source,
        value,
        BUILTIN_NAME,
        gpu_helpers::RestoreOperandContext::new(&protected, &intent_sources),
    )
    .map_err(with_eig_context)?;
    if let Value::GpuTensor(handle) = &restored {
        previous.push(handle.clone());
    }
    Ok(restored)
}
