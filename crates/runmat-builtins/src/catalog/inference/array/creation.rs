use super::super::{
    argument_error, default_double_scalar, finish_fixed, literal_text, numeric_kind,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[cfg(test)]
#[path = "creation/zeros_tests.rs"]
mod zeros_tests;

pub(super) fn infer_zeros(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let literals = &request.literals.literal_args;
    let like_index = literals.iter().position(
        |literal| matches!(literal, LiteralValue::String(value) | LiteralValue::Character(value) | LiteralValue::Keyword(value) if value.eq_ignore_ascii_case("like")),
    );
    let trailing_class = literals
        .last()
        .and_then(literal_text)
        .filter(|value| !value.eq_ignore_ascii_case("like"))
        .filter(|_| like_index.is_none_or(|index| index + 1 != literals.len() - 1));
    let gpu_output = trailing_class
        .as_deref()
        .is_some_and(|class| class.eq_ignore_ascii_case("gpuarray"));
    let element_class = if gpu_output {
        literals
            .get(literals.len().saturating_sub(2))
            .and_then(literal_text)
            .filter(|class| {
                NumericClass::from_class_name(class).is_some()
                    || class.eq_ignore_ascii_case("logical")
            })
    } else {
        trailing_class.clone()
    };
    let dimension_end = like_index.unwrap_or_else(|| {
        literals.len().saturating_sub(
            usize::from(trailing_class.is_some())
                + usize::from(gpu_output && element_class.is_some()),
        )
    });

    let mut output = like_index
        .and_then(|index| request.arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(default_double_scalar);
    // A class creation method may return a different class, shape or storage
    // than its prototype. Only that selected method's inference can establish
    // those facts; copying the prototype would assert an unsupported result.
    if like_index
        .and_then(|index| request.arguments.get(index + 1))
        .is_some_and(|prototype| {
            matches!(
                prototype.kind,
                ValueKindFact::Object(_) | ValueKindFact::Unknown
            )
        })
    {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    }
    if gpu_output && element_class.is_none() && literals.len() >= 2 {
        let preceding = literals.len() - 2;
        if matches!(literals[preceding], LiteralValue::Unknown)
            && matches!(
                request.arguments[preceding].kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
        {
            output.kind = ValueKindFact::Unknown;
        }
    }

    let known_class_controls: Vec<_> = literals
        .iter()
        .enumerate()
        .filter_map(|(index, literal)| {
            if like_index.is_some_and(|like| index == like + 1) {
                return None;
            }
            let class = literal_text(literal)?;
            (NumericClass::from_class_name(&class).is_some()
                || class.eq_ignore_ascii_case("logical")
                || class.eq_ignore_ascii_case("gpuarray"))
            .then_some((index, class))
        })
        .collect();
    if known_class_controls.len() > 1
        && !(known_class_controls.len() == 2
            && known_class_controls[1].0 == literals.len() - 1
            && known_class_controls[1].1.eq_ignore_ascii_case("gpuarray")
            && !known_class_controls[0].1.eq_ignore_ascii_case("gpuarray"))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-ZEROS-CLASS-CONFLICT",
            "class and gpuArray controls must be a single element class followed by gpuArray",
            known_class_controls[1].0,
        ));
    }
    if like_index.is_some() && !known_class_controls.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-ZEROS-CLASS-CONFLICT",
            "like cannot be combined with class or gpuArray controls",
            known_class_controls[0].0,
        ));
    }

    if let Some(class) = element_class.as_deref() {
        if let Some(numeric_class) = NumericClass::from_class_name(class) {
            output.kind = numeric_kind(numeric_class, NumericDomain::Real);
        } else if class.eq_ignore_ascii_case("logical") {
            output.kind = ValueKindFact::Logical;
        } else {
            diagnostics.push(argument_error(
                "RM-CATALOG-ZEROS-CLASS",
                "zeros class specifier is not recognized",
                literals.len().saturating_sub(1),
            ));
        }
    }
    if gpu_output {
        output.residency = ResidencyFact::Device { provider: None };
        output.storage = StorageFact::Dense;
    }

    if dimension_end > 0 {
        output.shape = zeros_shape(request, dimension_end);
        output.storage = StorageFact::Dense;
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn zeros_shape(request: &CallRequest, dimension_end: usize) -> ShapeFact {
    if dimension_end == 1 {
        if let Some(dims) = request.literals.numeric_vector_at(0) {
            return ShapeFact::from(dims);
        }
        if let Some(size) = request.literals.numeric_dims().first().copied().flatten() {
            return ShapeFact::from(vec![Some(size), Some(size)]);
        }
        return match request
            .arguments
            .first()
            .and_then(|argument| argument.shape.rank())
        {
            Some(2) if !request.arguments[0].is_scalar() => {
                let rank = request.arguments[0]
                    .shape
                    .element_count()
                    .unwrap_or(2)
                    .max(2);
                ShapeFact::Ranked { rank }
            }
            _ => ShapeFact::Ranked { rank: 2 },
        };
    }
    ShapeFact::from(
        request
            .literals
            .numeric_dims()
            .into_iter()
            .take(dimension_end)
            .collect::<Vec<_>>(),
    )
}

pub(super) fn infer_full(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = request.arguments.first().cloned().unwrap_or_else(|| {
        diagnostics.push(argument_error(
            "RM-CATALOG-FULL-ARITY",
            "full requires exactly one input",
            0,
        ));
        ValueFact::unknown(DynamicReason::RuntimeValue)
    });
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FULL-ARITY",
            "full accepts exactly one input",
            1,
        ));
    }
    match output.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Character => {
            if matches!(output.storage, StorageFact::Sparse) {
                output.storage = StorageFact::Dense;
            }
        }
        ValueKindFact::Unknown => {}
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FULL-INPUT",
                "full requires a numeric, logical, or character array",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
