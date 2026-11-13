//! Static result facts for the four names that read the current SPMD rank or size.

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-CATALOG-SPMD-CONTEXT-ARITY",
            format!("{} accepts no inputs", entry.identity.name),
            0,
        )]
    };
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::parallel::context::{
        LABINDEX_CATALOG_ENTRY, NUMLABS_CATALOG_ENTRY, SPMD_INDEX_CATALOG_ENTRY,
        SPMD_SIZE_CATALOG_ENTRY,
    };
    use runmat_types::{
        LiteralContext, OutputSelection, RequestedOutputCount, StorageFact, ValueKindFact,
    };

    fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        }
    }

    #[test]
    fn each_context_name_has_the_same_exact_double_scalar_result() {
        for entry in [
            &SPMD_INDEX_CATALOG_ENTRY,
            &SPMD_SIZE_CATALOG_ENTRY,
            &LABINDEX_CATALOG_ENTRY,
            &NUMLABS_CATALOG_ENTRY,
        ] {
            let inference = infer(&request(Vec::new(), 1), entry);
            assert!(inference.diagnostics.is_empty(), "{}", entry.identity.name);
            assert_eq!(inference.outputs.len(), 1);
            assert_eq!(inference.outputs[0].storage, StorageFact::Scalar);
            assert_eq!(
                inference.outputs[0].kind,
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })
            );
        }
    }

    #[test]
    fn input_and_output_bounds_are_checked_for_each_name() {
        for entry in [
            &SPMD_INDEX_CATALOG_ENTRY,
            &SPMD_SIZE_CATALOG_ENTRY,
            &LABINDEX_CATALOG_ENTRY,
            &NUMLABS_CATALOG_ENTRY,
        ] {
            assert!(infer(&request(Vec::new(), 0), entry).outputs.is_empty());
            let invalid_input = infer(
                &request(
                    vec![ValueFact::unknown(
                        runmat_types::DynamicReason::RuntimeValue,
                    )],
                    1,
                ),
                entry,
            );
            assert!(invalid_input
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "RM-CATALOG-SPMD-CONTEXT-ARITY"));
            let invalid_output = infer(&request(Vec::new(), 2), entry);
            assert!(
                !invalid_output.diagnostics.is_empty(),
                "{}",
                entry.identity.name
            );
        }
    }
}
