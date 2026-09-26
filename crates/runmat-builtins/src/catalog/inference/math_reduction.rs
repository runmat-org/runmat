use super::{argument_error, finish_fixed};
use crate::{BuiltinCatalogEntry, LogicalReductionKind};
use runmat_types::{DynamicReason, ShapeFact, StorageFact, ValueFact, ValueKindFact};

#[path = "math_reduction/options.rs"]
mod options;

pub(super) fn infer_logical(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
    _kind: LogicalReductionKind,
) -> runmat_types::CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGICAL-REDUCTION-ARITY",
            format!("{} requires an input value", entry.identity.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if !matches!(
        input.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGICAL-REDUCTION-INPUT",
            format!(
                "{} requires numeric, logical, complex, or character input",
                entry.identity.name
            ),
            0,
        ));
    }
    // Options still diagnose with unknown input shape; all still proves scalar.
    let selector = options::analyze(request, &mut diagnostics);
    let shape = reduced_shape(&input.shape, selector);
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    finish_fixed(
        entry,
        request,
        ValueFact::proven(ValueKindFact::Logical, shape, storage),
        diagnostics,
    )
}

fn reduced_shape(input: &ShapeFact, selector: options::Selector) -> ShapeFact {
    use options::Selector;
    if matches!(selector, Selector::All) {
        return ShapeFact::Scalar;
    }
    if matches!(selector, Selector::Unknown) {
        return unknown_output_shape(input);
    }
    let Some(mut dimensions) = input.known_dims() else {
        return ShapeFact::Unknown;
    };
    let selected = match selector {
        Selector::Dimensions(selected) => selected,
        Selector::Default => {
            let mut selected = 0;
            for (index, dimension) in dimensions.iter().enumerate() {
                match dimension {
                    Some(1) => {}
                    Some(_) => {
                        selected = index;
                        break;
                    }
                    // Unknown earlier extents can move the default reduction axis.
                    None => return unknown_output_shape(input),
                }
            }
            vec![selected + 1]
        }
        Selector::All | Selector::Unknown => unreachable!("handled above"),
    };
    for dimension in selected {
        if let Some(extent) = dimensions.get_mut(dimension - 1) {
            *extent = Some(1);
        }
    }
    let shape = ShapeFact::from(dimensions);
    if shape.element_count() == Some(1) {
        ShapeFact::Scalar
    } else {
        shape
    }
}

fn unknown_output_shape(input: &ShapeFact) -> ShapeFact {
    // Unknown options can select all, and scalar reduction results have rank two.
    if input.element_count() == Some(1) {
        return ShapeFact::Scalar;
    }
    if input.rank() == Some(2) {
        ShapeFact::Ranked { rank: 2 }
    } else {
        ShapeFact::Unknown
    }
}

#[cfg(test)]
#[path = "math_reduction/tests.rs"]
mod tests;
