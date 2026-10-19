mod counts;
mod grouped_apply;
mod index_labels;
mod sorted_groups;

use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact};

fn infer(
    name: &str,
    arguments: Vec<ValueFact>,
    outputs: RequestedOutputCount,
) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name(name).expect("catalog entry"),
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(outputs),
        },
    )
}
