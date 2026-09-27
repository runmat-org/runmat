use runmat_types::{
    AliasFact, ContiguityFact, DynamicReason, LayoutFact, MutationFact, StorageFact, ValueFact,
    ViewFact,
};

#[cfg(test)]
mod tests;

pub(crate) fn materialize(output: &mut ValueFact) {
    output.storage = if output.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.view = ViewFact::Materialized;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
}

/// Representation of a floating result whose runtime scalar carrier is double-only.
/// `preserve_sparse` also retains uncertainty when sparse input has not been ruled out.
pub(in crate::catalog::inference) fn materialize_floating_result(
    output: &mut ValueFact,
    preserve_sparse: bool,
) {
    use runmat_types::{NumericClass, ResidencyFact};
    let Some(numeric) = output.numeric() else {
        return;
    };
    if preserve_sparse && output.storage == StorageFact::Sparse {
        output.view = ViewFact::Materialized;
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
        return;
    }
    output.storage = if output.residency != ResidencyFact::Host
        || (preserve_sparse && output.storage == StorageFact::Unknown)
    {
        StorageFact::Unknown
    } else if numeric.class == NumericClass::Single {
        StorageFact::Dense
    } else {
        match output.shape.element_count() {
            Some(1) => StorageFact::Scalar,
            Some(_) => StorageFact::Dense,
            None => StorageFact::Unknown,
        }
    };
    if output.residency == ResidencyFact::Host {
        output.layout = LayoutFact::ColumnMajor;
        output.contiguity = ContiguityFact::Contiguous;
    } else {
        output.layout = LayoutFact::Unknown;
        output.contiguity = ContiguityFact::Unknown;
    }
    output.view = ViewFact::Materialized;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
}

pub(in crate::catalog::inference) fn materialize_preserving_sparse_storage(output: &mut ValueFact) {
    if matches!(output.storage, StorageFact::Sparse) {
        output.view = ViewFact::Materialized;
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
    } else {
        materialize(output);
    }
}

pub(in crate::catalog::inference) fn preserve_shape_as_dynamic(output: &mut ValueFact) {
    let shape = output.shape.clone();
    *output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = shape;
}
