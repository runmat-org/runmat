use crate::{BuiltinCatalogEntry, RelationalOperator};
use runmat_types::{
    broadcast_shape, AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason,
    LayoutFact, MutationFact, ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};

#[cfg(test)]
mod tests;

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    _operator: RelationalOperator,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(super::super::argument_error(
            "RM-CATALOG-RELATIONAL-ARITY",
            format!("{} requires exactly two inputs", entry.identity.name),
            request.arguments.len().min(1),
        ));
    }
    let (Some(left), Some(right)) = (request.arguments.first(), request.arguments.get(1)) else {
        return super::super::finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let symbolic = matches!(left.kind, ValueKindFact::Symbolic)
        || matches!(right.kind, ValueKindFact::Symbolic);
    let kind = if matches!(left.kind, ValueKindFact::Unknown)
        || matches!(right.kind, ValueKindFact::Unknown)
    {
        ValueKindFact::Unknown
    } else if symbolic {
        ValueKindFact::Symbolic
    } else {
        ValueKindFact::Logical
    };
    // Runtime text normalization treats a character operand as one string when
    // paired with string storage, not as a vector of numeric code points.
    let left_shape = if matches!(left.kind, ValueKindFact::Character)
        && matches!(right.kind, ValueKindFact::String)
    {
        ShapeFact::Scalar
    } else {
        left.shape.clone()
    };
    let right_shape = if matches!(right.kind, ValueKindFact::Character)
        && matches!(left.kind, ValueKindFact::String)
    {
        ShapeFact::Scalar
    } else {
        right.shape.clone()
    };
    let shape = match broadcast_shape(&left_shape, &right_shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            ShapeFact::Unknown
        }
    };
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let output = ValueFact {
        kind,
        shape,
        storage,
        layout: LayoutFact::ColumnMajor,
        contiguity: ContiguityFact::Contiguous,
        view: ViewFact::Materialized,
        residency: if left.residency == runmat_types::ResidencyFact::Host
            && right.residency == runmat_types::ResidencyFact::Host
        {
            runmat_types::ResidencyFact::Host
        } else {
            runmat_types::ResidencyFact::Unknown
        },
        alias: AliasFact::Unique,
        mutation: MutationFact::ValueSemantics,
        certainty: if matches!(left.kind, ValueKindFact::Unknown)
            || matches!(right.kind, ValueKindFact::Unknown)
        {
            runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
        } else {
            runmat_types::CertaintyFact::Proven
        },
        invalidation: Default::default(),
    };
    super::super::finish_fixed(entry, request, output, diagnostics)
}
