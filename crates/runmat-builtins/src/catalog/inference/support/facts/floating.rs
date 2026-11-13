use runmat_types::{
    AliasFact, ContiguityFact, LayoutFact, MutationFact, NumericClass, ResidencyFact, StorageFact,
    ValueFact, ViewFact,
};

/// Representation of a floating result whose runtime scalar carrier is double-only.
/// `preserve_sparse` also retains uncertainty when sparse input has not been ruled out.
pub(in crate::catalog::inference) fn materialize_floating_result(
    output: &mut ValueFact,
    preserve_sparse: bool,
) {
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
