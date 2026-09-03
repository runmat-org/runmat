use crate::{
    catalog::inference::{finish_fixed, numeric_kind},
    BuiltinCatalogEntry,
};
use runmat_types::{
    AliasFact, CallInference, CallRequest, ContiguityFact, LayoutFact, MutationFact, NumericClass,
    NumericDomain, ResidencyFact, StorageFact, ValueFact, ViewFact,
};

pub(super) fn finish_materialized_real(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    input: &ValueFact,
    output_class: NumericClass,
    diagnostics: Vec<runmat_types::InferenceDiagnostic>,
) -> CallInference {
    let mut output = input.clone();
    output.kind = numeric_kind(output_class, NumericDomain::Real);
    output.storage = if input.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.residency = match input.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        ResidencyFact::Device { .. } => input.residency.clone(),
        _ => ResidencyFact::Unknown,
    };
    output.alias = AliasFact::Unique;
    output.view = ViewFact::Materialized;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}
