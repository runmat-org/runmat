use crate::{BuiltinCatalogEntry, BuiltinDistributedPolicy};
use runmat_types::{CallRequest, InferenceDiagnostic, ValueKindFact};

pub(super) fn validate(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let BuiltinDistributedPolicy::MapUnaryConstrained(contract) = entry.placement.distributed
    else {
        return;
    };
    for (index, argument) in request.arguments.iter().enumerate() {
        let ValueKindFact::Distributed(distributed) = &argument.kind else {
            continue;
        };
        let ValueKindFact::Numeric(numeric) = distributed.value.kind else {
            continue;
        };
        if !contract.numeric_classes.contains(&numeric.class) {
            diagnostics.push(super::super::argument_error(
                "RM-CATALOG-DISTRIBUTED-NUMERIC-CLASS",
                format!(
                    "{} does not support distributed {} input",
                    entry.identity.name,
                    numeric.class.class_name()
                ),
                index,
            ));
        }
    }
}
