use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

use super::contract::GPU_INFO_ERROR_TOO_MANY_INPUTS;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut output = ValueFact::scalar(ValueKindFact::String);
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            GPU_INFO_ERROR_TOO_MANY_INPUTS.code,
            "gpuInfo accepts no inputs",
            0,
        )]
    };
    finish_fixed(entry, request, output, diagnostics)
}
