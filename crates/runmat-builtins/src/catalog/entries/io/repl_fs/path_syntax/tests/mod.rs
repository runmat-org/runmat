mod catalog;
mod inference;

use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount, ValueFact};

fn request(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> runmat_types::CallRequest {
    runmat_types::CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(outputs),
    }
}
