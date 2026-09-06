use crate::catalog::entries::io::repl_fs::search_path;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, NumericClass, NumericDomain, NumericFact,
    ValueFact, ValueKindFact,
};

pub(in crate::catalog::entries::io::repl_fs) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SAVEPATH-ARITY",
            "savepath accepts at most one filename",
            1,
        ));
    }
    if let Some(filename) = request.arguments.first() {
        if !search_path::input::supports(filename, search_path::input::Policy::Generate) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SAVEPATH-FILENAME",
                "savepath expects a character row, string scalar, or numeric character-code row",
                0,
            ));
        }
    }

    let mut contract = CallContract::fixed(vec![
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        search_path::result::character_row(),
        search_path::result::character_row(),
    ]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    };

    fn request(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(outputs),
        }
    }

    #[test]
    fn infers_status_and_optional_diagnostic_character_rows() {
        let entry = crate::builtin_catalog_entry_by_name("savepath").expect("savepath entry");
        let inferred = infer(
            &request(
                vec![ValueFact::scalar(ValueKindFact::String)],
                RequestedOutputCount::Exactly(3),
            ),
            entry,
        );
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].numeric().unwrap().class,
            NumericClass::Double
        );
        assert_eq!(inferred.outputs[1].kind, ValueKindFact::Character);
        assert_eq!(
            inferred.outputs[2].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }

    #[test]
    fn rejects_known_invalid_inputs_and_excess_arity() {
        let entry = crate::builtin_catalog_entry_by_name("savepath").expect("savepath entry");
        let matrix = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        );
        let inferred = infer(
            &request(
                vec![matrix, ValueFact::scalar(ValueKindFact::String)],
                RequestedOutputCount::One,
            ),
            entry,
        );
        assert!(inferred.diagnostics.len() >= 2);
    }
}
