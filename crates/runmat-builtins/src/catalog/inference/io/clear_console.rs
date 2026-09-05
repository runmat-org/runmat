use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, NumericClass, NumericDomain, ShapeFact};

use super::super::{argument_error, finish_fixed, numeric_kind};

pub(in crate::catalog::inference) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-CATALOG-CLC-ARITY",
            "clc accepts no input arguments",
            0,
        )]
    };
    let mut output =
        runmat_types::ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real));
    output.shape = ShapeFact::from(vec![Some(0), Some(0)]);
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
    };

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn infers_empty_double_matrix_and_rejects_arguments() {
        let entry = crate::builtin_catalog_entry_by_name("clc").expect("clc catalog entry");
        let inferred = infer(&request(Vec::new()), entry);
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(0), Some(0)])
        );
        assert!(inferred.diagnostics.is_empty());

        let invalid = infer(
            &request(vec![ValueFact::scalar(ValueKindFact::Logical)]),
            entry,
        );
        assert_eq!(invalid.diagnostics.len(), 1);
        assert_eq!(invalid.diagnostics[0].argument, Some(0));
    }
}
