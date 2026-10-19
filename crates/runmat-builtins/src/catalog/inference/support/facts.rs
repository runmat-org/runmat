use runmat_types::{
    AliasFact, ContiguityFact, DynamicReason, LayoutFact, MutationFact, StorageFact, ValueFact,
    ViewFact,
};

mod floating;
#[cfg(test)]
mod tests;

pub(in crate::catalog::inference) use floating::materialize_floating_result;

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
