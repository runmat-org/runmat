use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_unary, transpose_shape, AliasFact, CallInference, CallRequest, ContiguityFact,
    DynamicReason, EffectKind, EffectSet, InferenceDiagnostic, LayoutFact, MutationFact,
    OperatorKind, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

fn transpose_container(input: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) -> ValueFact {
    if input.shape.known_dims().is_some_and(|dims| {
        dims.iter()
            .skip(2)
            .any(|extent| extent.is_some_and(|n| n != 1))
    }) {
        diagnostics.push(InferenceDiagnostic::error(
            "RM-TYPE-MATRIX-RANK",
            "ctranspose operand must be two-dimensional",
        ));
        return ValueFact::unknown(DynamicReason::ConflictingControlFlow);
    }
    let mut output = input.clone();
    output.shape = transpose_shape(&input.shape);
    let scalar_identity = matches!(
        &input.kind,
        ValueKindFact::String | ValueKindFact::Struct(_)
    ) && matches!(&input.shape, ShapeFact::Scalar)
        && input.storage == StorageFact::Scalar;
    if scalar_identity {
        output.shape = input.shape.clone();
    }
    if output.shape.element_count() != Some(1) {
        match &mut output.kind {
            ValueKindFact::Cell(cell) => {
                cell.elements.clear();
                cell.elements_complete = false;
            }
            ValueKindFact::Struct(structure) => {
                structure.elements.clear();
                structure.elements_complete = false;
            }
            _ => {}
        }
    }
    let new_outer_array = matches!(&input.kind, ValueKindFact::Cell(_))
        || (matches!(
            &input.kind,
            ValueKindFact::String | ValueKindFact::Struct(_)
        ) && input.storage == StorageFact::Dense
            && !(matches!(&input.kind, ValueKindFact::String)
                && matches!(&input.shape, ShapeFact::Shaped { dims } if dims.is_empty())));
    if new_outer_array {
        output.layout = LayoutFact::ColumnMajor;
        output.contiguity = ContiguityFact::Contiguous;
        output.view = ViewFact::Materialized;
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
    } else if !scalar_identity {
        // A partially known representation, including a rank-zero StringArray,
        // cannot inherit its receiver's identity or prove a new allocation.
        output.layout = LayoutFact::Unknown;
        output.contiguity = ContiguityFact::Unknown;
        output.view = ViewFact::Unknown;
        output.alias = AliasFact::Unknown;
    }
    output.residency = ResidencyFact::Host;
    output
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CTRANSPOSE-ARITY",
            "ctranspose requires exactly one input",
            request.arguments.len().min(1),
        ));
    }
    let Some(input) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let dynamic_callback = matches!(
        &input.kind,
        ValueKindFact::Object(_) | ValueKindFact::Unknown
    );
    let output = match &input.kind {
        ValueKindFact::Object(_) | ValueKindFact::Unknown => {
            ValueFact::unknown(DynamicReason::DynamicDispatch)
        }
        ValueKindFact::Cell(_) | ValueKindFact::Struct(_) | ValueKindFact::String => {
            transpose_container(input, &mut diagnostics)
        }
        _ => {
            let inferred = infer_unary(OperatorKind::ConjugateTranspose, input);
            diagnostics.extend(inferred.diagnostics);
            let mut output = inferred.fact;
            if input.storage == StorageFact::Sparse
                && matches!(
                    &output.kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Logical
                )
            {
                // The shared unary rule scalarizes one-element storage, but the
                // runtime SparseTensor branch returns sparse storage at 1×1.
                output.storage = StorageFact::Sparse;
            }
            if input.shape.element_count() == Some(1) {
                // Scalar and singleton storage can be returned unchanged.
                output.view = ViewFact::Unknown;
                output.alias = AliasFact::Unknown;
            }
            if !matches!(input.residency, ResidencyFact::Host) {
                // An unsupported provider hook may gather and return a host value;
                // a successful hook may publish a lazy view or an aliased handle.
                output.residency = ResidencyFact::Unknown;
                output.layout = LayoutFact::Unknown;
                output.contiguity = ContiguityFact::Unknown;
                output.view = ViewFact::Unknown;
                output.alias = AliasFact::Unknown;
            }
            output
        }
    };
    let mut inference = finish_fixed(entry, request, output, diagnostics);
    if !dynamic_callback {
        inference.effects = EffectSet([EffectKind::MaySuspend, EffectKind::MayThrow].into());
    }
    inference
}
