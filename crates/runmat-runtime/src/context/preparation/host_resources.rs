//! Canonical allocation envelope of an admitted implementation. These are
//! original native host facts, not provider-effective casts or execution proof.
use super::{AdmittedBuiltinPreparation, RuntimeContext};
use crate::builtins::common::spec::{
    selected_fusion_binding, FusionHostOutputDomain, FusionHostResourceBound,
    FusionHostResourceInput, FusionHostResourceQuery,
};
use runmat_types::{CallRequest, NumericClass, NumericDomain, NumericFact, ValueKindFact};
use runmat_value::Value;

pub struct AdmittedHostResources {
    pub bound: FusionHostResourceBound,
    /// Exact temporary storage for a body that consumes the canonical
    /// preallocated numerical output. Legacy bodies retain `bound.scratch_bytes`.
    pub prepared_scratch_bytes: u64,
    pub output_shape: Vec<usize>,
    pub output_class: NumericClass,
    pub output_domain: FusionHostOutputDomain,
    pub operand_metadata: u64,
    pub operand_shapes: Vec<Vec<usize>>,
    pub identity: runmat_execution::Digest,
    pub operation: runmat_builtins::BuiltinPreparationOperation,
}

/// Missing canonical allocation contracts remain unavailable. In particular,
/// an unknown output domain uses the numerical owner's explicit real/complex
/// envelope; it never becomes a certificate for real provider arithmetic.
pub(super) fn bound(
    request: &AdmittedBuiltinPreparation,
    context: Option<&RuntimeContext>,
) -> Option<AdmittedHostResources> {
    let selected = request.admission();
    if selected.input_count != request.operands().len()
        || selected.requested_outputs.is_some_and(|n| n > 1)
    {
        return None;
    }
    let declaration = selected_fusion_binding(
        selected.binding.builtin.name,
        Some(selected.binding),
        context,
    )?
    .canonical?;
    let callback = declaration.host_resources?;
    let logical_arithmetic = matches!(
        request.form().operation,
        runmat_builtins::BuiltinPreparationOperation::ElementwiseBinary
    );
    let input_numeric = |fact: &runmat_types::ValueFact| {
        fact.numeric().or_else(|| {
            (logical_arithmetic && fact.kind == ValueKindFact::Logical).then_some(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        })
    };
    let facts = request
        .operands()
        .iter()
        .map(|value| {
            let mut fact = crate::value_fact::value_fact(value);
            if let Value::GpuTensor(handle) = value {
                handle.allocation_lease().validate_live().ok()?;
                handle.view().validate(handle.physical()).ok()?;
                let resident_logical = logical_arithmetic
                    && runmat_accelerate_api::handle_is_logical(handle)
                    && handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
                    && matches!(
                        handle.physical().element_type(),
                        runmat_accelerate_api::NumericElementType::U8
                            | runmat_accelerate_api::NumericElementType::F32
                            | runmat_accelerate_api::NumericElementType::F64
                    )
                    && matches!(fact.kind, ValueKindFact::Logical);
                if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
                    || !(resident_logical
                        || (!handle.view().is_logical()
                            && matches!(
                                handle.physical().element_type(),
                                runmat_accelerate_api::NumericElementType::F32
                                    | runmat_accelerate_api::NumericElementType::F64
                            )))
                {
                    return None;
                }
                if !resident_logical {
                    let class = NumericClass::from_class_identity(
                        &handle.physical().element_type().class_identity(),
                    )?;
                    if !matches!(&fact.kind, ValueKindFact::Numeric(numeric)
                    if numeric.class == class && numeric.domain == NumericDomain::Real)
                    {
                        return None;
                    }
                }
                fact.storage = runmat_types::StorageFact::Dense;
                fact = runmat_builtins::gathered_value_fact(&fact);
                fact.alias = runmat_types::AliasFact::Shared;
            } else if crate::dispatcher::value_contains_gpu(value) {
                return None;
            }
            let numeric = input_numeric(&fact)?;
            (numeric.domain == NumericDomain::Real
                && matches!(numeric.class, NumericClass::Single | NumericClass::Double))
            .then_some(fact)
        })
        .collect::<Option<Vec<_>>>()?;
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
    let dims = |fact: &runmat_types::ValueFact| {
        fact.shape
            .known_dims()?
            .into_iter()
            .collect::<Option<Vec<_>>>()
    };
    let shapes = facts.iter().map(dims).collect::<Option<Vec<_>>>()?;
    let classes = facts
        .iter()
        .map(|fact| input_numeric(fact).map(|numeric| numeric.class))
        .collect::<Option<Vec<_>>>()?;
    let (output_shape, output_class, output_domain) = if matches!(
        request.form().operation,
        runmat_builtins::BuiltinPreparationOperation::RealUnary(_)
            | runmat_builtins::BuiltinPreparationOperation::RealPower
    ) {
        let envelope = crate::builtins::math::host_resources::promoting_output(
            request.form().operation,
            &facts,
            &output,
        )?;
        (envelope.shape, envelope.class, envelope.domain)
    } else {
        let ValueKindFact::Numeric(numeric) = &output.kind else {
            return None;
        };
        if numeric.domain != NumericDomain::Real
            || !matches!(numeric.class, NumericClass::Single | NumericClass::Double)
        {
            return None;
        }
        (
            dims(&output)?,
            numeric.class,
            FusionHostOutputDomain::Exact(numeric.domain),
        )
    };
    let inputs = shapes
        .iter()
        .zip(&classes)
        .map(|(shape, &class)| FusionHostResourceInput { shape, class })
        .collect::<Vec<_>>();
    let query = FusionHostResourceQuery {
        inputs: &inputs,
        output: FusionHostResourceInput {
            shape: &output_shape,
            class: output_class,
        },
        output_domain,
    };
    let bound = callback(&query)?;
    let prepared_scratch_bytes = crate::builtins::math::host_resources::prepared_scratch_bytes(
        request.form().operation,
        &query,
    )?;
    let operand_metadata = shapes.iter().try_fold(0_u64, |total, shape| {
        total.checked_add(
            u64::try_from(shape.len())
                .ok()?
                .checked_mul(std::mem::size_of::<usize>() as u64)?,
        )
    })?;
    let identity = runmat_execution::Digest::sha256_serialized(&(
        selected.binding,
        selected.form,
        selected.signature,
        selected.input_count,
        selected.requested_outputs,
        &facts,
        &output,
        (&output_shape, output_class, output_domain),
        bound.retained_bytes,
        bound.scratch_bytes,
        prepared_scratch_bytes,
    ))
    .ok()?;
    Some(AdmittedHostResources {
        bound,
        prepared_scratch_bytes,
        output_shape,
        output_class,
        output_domain,
        operand_metadata,
        operand_shapes: shapes,
        identity,
        operation: request.form().operation,
    })
}
