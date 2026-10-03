use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut output = ValueFact::scalar(ValueKindFact::String);
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM.GPU_INFO.TOO_MANY_INPUTS",
            "gpuInfo accepts no inputs",
            0,
        )]
    };
    finish_fixed(entry, request, output, diagnostics)
}
