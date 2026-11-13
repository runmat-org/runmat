use super::super::super::argument_error;
use super::common::{double, finish};
use super::sub2ind::{invalid_size_literal, known_size_cell_count, valid_real_numeric, valid_size};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallContract, CallInference, CallRequest, DynamicReason, ResidencyFact, ShapeFact, StorageFact,
};

pub(super) fn infer_ind2sub(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-IND2SUB-ARITY",
            "ind2sub requires a size vector and linear indices",
            request.arguments.len().min(2),
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|size| !valid_size(size))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-IND2SUB-SIZE-TYPE",
            "ind2sub size must be a real numeric vector or numeric cell vector",
            0,
        ));
    }
    if request.arguments.first().is_some_and(|size| {
        size.shape.element_count() == Some(1)
            || known_size_cell_count(size).is_some_and(|count| count < 2)
    }) || invalid_size_literal(request.literals.literal_args.first(), 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-IND2SUB-SIZE",
            "ind2sub requires at least two positive dimensions",
            0,
        ));
    }
    let index = request.arguments.get(1);
    if let Some(index) = index {
        if !valid_real_numeric(index) {
            diagnostics.push(argument_error(
                "RM-CATALOG-IND2SUB-INDEX",
                "ind2sub indices must be real numeric or logical",
                1,
            ));
        }
    }
    let index_shape = index.map_or(ShapeFact::Unknown, |index| index.shape.clone());
    let subscript = double(index_shape, index);
    let count = request.outputs.requested.known_count();
    let mut contract = if let Some(count) = count {
        CallContract::fixed(vec![subscript.clone(); count])
    } else {
        let mut dynamic = CallContract::dynamic(DynamicReason::RuntimeValue);
        dynamic.outputs = vec![subscript.clone()];
        dynamic.variadic_output = Some(Box::new(subscript));
        dynamic
    };
    if !matches!(
        request.arguments.first().map(|v| &v.residency),
        Some(ResidencyFact::Host)
    ) {
        for output in &mut contract.outputs {
            output.residency = ResidencyFact::Unknown;
            output.storage = StorageFact::Unknown;
        }
    }
    finish(entry, request, contract, diagnostics)
}
