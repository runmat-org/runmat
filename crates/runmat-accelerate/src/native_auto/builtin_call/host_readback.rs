//! Invocation-private automatic resident preparation, using the same physical
//! source and resource owner as VM overlays. No workspace or VM right is made.
use super::*;
use crate::fusion_exec::host_readback::{CpuHostResources, CpuReadbackPlan, Source};
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{
    selected_fusion_binding, FusionHostResourceInput, FusionHostResourceQuery,
};
use runmat_runtime::context::preparation::AdmittedBuiltinPreparation;
use runmat_types::{CallRequest, NumericClass, NumericDomain, ValueKindFact};

pub(super) struct Admission {
    pub plan: CpuReadbackPlan,
    pub resources: CpuHostResources,
    pub identity: Digest,
}

pub(super) fn admit(
    auto: &NativeAutoOffload,
    context: &RuntimeContext,
    request: &AdmittedBuiltinPreparation,
) -> Option<Admission> {
    let selected = request.admission();
    if !runmat_runtime::context::preparation::PreparedHostExecution::supports_form(request.form()) {
        return None;
    }
    // Extra controls require their own operation-owned inference/resource join;
    // never infer an omitted option or prototype from the data-only projection.
    if selected.input_count != request.operands().len()
        || selected.requested_outputs.is_some_and(|n| n > 1)
    {
        return None;
    }
    let declaration = selected_fusion_binding(
        selected.binding.builtin.name,
        Some(selected.binding),
        Some(context),
    )?
    .canonical?;
    let callback = declaration.host_resources?;
    let mut sources: Vec<Source> = Vec::new();
    let mut facts = Vec::with_capacity(request.operands().len());
    for (offset, value) in request.operands().iter().enumerate() {
        let mut fact = runmat_runtime::value_fact::value_fact(value);
        if let Value::GpuTensor(handle) = value {
            if let Some(source) = sources
                .iter_mut()
                .find(|source| source.matches_handle(handle))
            {
                source.offsets.push(offset);
            } else {
                sources.push(Source::admit(handle, auto.provider, vec![offset], vec![])?);
            }
            fact = sources
                .iter()
                .find(|source| source.matches_handle(handle))?
                .host_fact(value)?;
        } else if runmat_runtime::dispatcher::value_contains_gpu(value) {
            return None;
        }
        match &fact.kind {
            ValueKindFact::Numeric(numeric)
                if matches!(numeric.class, NumericClass::Single | NumericClass::Double)
                    && numeric.domain == NumericDomain::Real => {}
            _ => return None,
        }
        facts.push(fact);
    }
    if sources.is_empty() {
        return None;
    }
    let entry = runmat_builtins::builtin_catalog_entry_by_name(selected.binding.builtin.name)?;
    let inferred = runmat_builtins::infer_catalog_call(
        entry,
        &CallRequest {
            arguments: facts.clone(),
            literals: Default::default(),
            outputs: Default::default(),
        },
    );
    if !inferred.diagnostics.is_empty() || inferred.outputs.len() != 1 {
        return None;
    }
    let output = inferred.outputs.into_iter().next()?;
    let all_facts = facts
        .iter()
        .chain(std::iter::once(&output))
        .collect::<Vec<_>>();
    let shapes = all_facts
        .iter()
        .map(|fact| {
            fact.shape
                .known_dims()?
                .into_iter()
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()?;
    let classes = all_facts
        .iter()
        .map(|fact| match &fact.kind {
            ValueKindFact::Numeric(numeric)
                if matches!(numeric.class, NumericClass::Single | NumericClass::Double)
                    && numeric.domain == NumericDomain::Real =>
            {
                Some(numeric.class)
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let inputs = shapes[..facts.len()]
        .iter()
        .zip(&classes)
        .map(|(shape, &class)| FusionHostResourceInput { shape, class })
        .collect::<Vec<_>>();
    let bound = callback(&FusionHostResourceQuery {
        inputs: &inputs,
        output: FusionHostResourceInput {
            shape: shapes.last()?,
            class: *classes.last()?,
        },
    })?;
    let operand_metadata = shapes[..facts.len()]
        .iter()
        .try_fold(0_u64, |total, shape| {
            total.checked_add(
                u64::try_from(shape.len())
                    .ok()?
                    .checked_mul(std::mem::size_of::<usize>() as u64)?,
            )
        })?;
    let resources = crate::fusion_exec::host_readback::resources(
        &sources,
        bound.retained_bytes,
        bound.scratch_bytes,
        operand_metadata,
    )?;
    let identity = Digest::sha256_serialized(&(
        selected.binding,
        selected.form,
        selected.signature,
        selected.input_count,
        selected.requested_outputs,
        &facts,
        &output,
        resources.resources,
    ))
    .ok()?;
    let resident_offsets = sources
        .iter()
        .flat_map(|source| source.offsets.iter().copied())
        .collect();
    Some(Admission {
        plan: CpuReadbackPlan {
            sources,
            stack_len: request.operands().len(),
            resident_offsets,
        },
        resources,
        identity,
    })
}
