use runmat_types::{
    AliasFact, ContiguityFact, DynamicReason, LayoutFact, MutationFact, StorageFact, ValueFact,
    ViewFact,
};

pub(super) fn materialize_output(output: &mut ValueFact) {
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

pub(super) fn materialize_output_preserving_storage(output: &mut ValueFact) {
    if matches!(output.storage, StorageFact::Sparse) {
        output.view = ViewFact::Materialized;
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
    } else {
        materialize_output(output);
    }
}

pub(super) fn preserve_shape_on_dynamic_input(output: &mut ValueFact) {
    let shape = output.shape.clone();
    *output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = shape;
}
