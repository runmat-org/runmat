use crate::catalog::inference::{argument_error, finish_fixed, literal_text, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-RANDPERM-ARITY",
            "randperm requires a population size",
            0,
        ));
    }
    let mut k_index = None;
    let mut like = None;
    let mut index = 1;
    while index < request.arguments.len() {
        match request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        {
            Some(text) if text.eq_ignore_ascii_case("like") => {
                like = Some(index + 1);
                index += 2;
            }
            Some(text) if text.eq_ignore_ascii_case("double") => index += 1,
            Some(text) if text.eq_ignore_ascii_case("single") => {
                diagnostics.push(argument_error(
                    "RM-RANDPERM-SINGLE",
                    "single precision permutation output is not implemented",
                    index,
                ));
                index += 1;
            }
            Some(_) => {
                diagnostics.push(argument_error(
                    "RM-RANDPERM-OPTION",
                    "unrecognized permutation option",
                    index,
                ));
                index += 1;
            }
            None => {
                k_index = Some(index);
                index += 1;
            }
        }
    }
    let n = super::super::shape_facts::scalar_dimension(request, 0);
    let k = k_index.map_or(n, |index| {
        super::super::shape_facts::scalar_dimension(request, index)
    });
    if n.zip(k).is_some_and(|(n, k)| k > n) {
        diagnostics.push(argument_error(
            "RM-RANDPERM-K",
            "K must not exceed N",
            k_index.unwrap_or(0),
        ));
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
    output.shape = ShapeFact::from(vec![Some(1), k]);
    output.storage = StorageFact::Dense;
    if let Some(prototype) = like.and_then(|index| request.arguments.get(index)) {
        if matches!(
            prototype.kind,
            ValueKindFact::Logical | ValueKindFact::Character | ValueKindFact::String
        ) {
            diagnostics.push(argument_error(
                "RM-RANDPERM-LIKE",
                "prototype must be numeric",
                like.unwrap_or(0),
            ));
        }
        output.residency = match prototype.residency {
            ResidencyFact::Host => ResidencyFact::Host,
            _ => ResidencyFact::Unknown,
        };
    }
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection};

    #[test]
    fn unknown_explicit_k_does_not_inherit_known_population_extent() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real)),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::new(vec![LiteralValue::Number(7.0), LiteralValue::Unknown]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::RANDPERM_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
}
