use crate::catalog::inference::{default_double_scalar, finish_fixed_outputs, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, ResidencyFact, ShapeFact, StorageFact, ValueFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let count = request.arguments.len();
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.residency = ResidencyFact::Host;
    if count <= 2 {
        output.kind = request
            .arguments
            .first()
            .map(|fact| fact.kind.clone())
            .unwrap_or(output.kind);
        output.shape = ShapeFact::Ranked { rank: 2 };
        output.storage = StorageFact::Dense;
    } else {
        output.kind = request
            .arguments
            .get(if count == 3 { 2 } else { 0 })
            .map(|fact| fact.kind.clone())
            .unwrap_or(output.kind);
        output.shape = if count == 3 {
            request
                .arguments
                .get(2)
                .map(|fact| fact.shape.clone())
                .unwrap_or(ShapeFact::Unknown)
        } else {
            ShapeFact::from(
                (2..4)
                    .map(|index| {
                        request
                            .literals
                            .literal_args
                            .get(index)
                            .and_then(|literal| {
                                if matches!(
                                    literal,
                                    runmat_types::LiteralValue::Vector(_)
                                        | runmat_types::LiteralValue::Matrix(_)
                                ) {
                                    None
                                } else {
                                    request
                                        .literals
                                        .numeric_dims_from(index)
                                        .first()
                                        .copied()
                                        .flatten()
                                }
                            })
                    })
                    .collect::<Vec<_>>(),
            )
        };
        output.storage = StorageFact::Sparse;
    }
    if count != 1
        || request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count <= 1)
    {
        return finish_fixed_outputs(entry, request, vec![output], Vec::new());
    }
    let mut id = ValueFact::unknown(DynamicReason::RuntimeValue);
    id.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
    id.shape = ShapeFact::from(vec![None, Some(1)]);
    id.storage = StorageFact::Dense;
    id.residency = ResidencyFact::Host;
    // The extraction form pads requested outputs beyond [B,d] with empties.
    // Other forms produce only one output and retain the fixed contract.
    let mut contract = CallContract::fixed(vec![output, id]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(default_double_scalar()));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    infer_call(&contract, request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount};

    #[test]
    fn extraction_pads_only_the_one_argument_form() {
        let matrix = ValueFact::unknown(DynamicReason::RuntimeValue);
        let mut request = CallRequest {
            arguments: vec![matrix.clone()],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        };
        let result = infer(&request, &super::super::SPDIAGS_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs.len(), 3);
        assert_eq!(result.outputs[2], default_double_scalar());

        request.arguments.push(matrix);
        request.literals.literal_args.push(LiteralValue::Unknown);
        let result = infer(&request, &super::super::SPDIAGS_CATALOG_ENTRY);
        assert!(!result.diagnostics.is_empty());
    }

    #[test]
    fn construction_size_arguments_are_not_taken_from_vector_data_or_offsets() {
        let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        let size = ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real));
        let request = CallRequest {
            arguments: vec![unknown.clone(), unknown, size.clone(), size],
            literals: LiteralContext::new(vec![
                LiteralValue::Vector(vec![LiteralValue::Number(11.0), LiteralValue::Number(12.0)]),
                LiteralValue::Vector(vec![LiteralValue::Number(-1.0), LiteralValue::Number(1.0)]),
                LiteralValue::Number(2.0),
                LiteralValue::Number(3.0),
            ]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::SPDIAGS_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}
