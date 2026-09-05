use super::super::super::{argument_error, finish_fixed, preserved_binary_residency};
use super::{class, shape};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
use runmat_types::{
    DynamicReason, NumericDomain, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-COMPLEX-ARITY",
            "complex requires one or two inputs",
            request.arguments.len().min(1),
        ));
    }
    let Some(first) = request.arguments.first() else {
        return finish_fixed(entry, request, dynamic(), diagnostics);
    };
    if matches!(first.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-COMPLEX-SPARSE",
            "complex does not currently accept sparse input",
            0,
        ));
        return finish_fixed(entry, request, unsupported(), diagnostics);
    }
    let Some(second) = request.arguments.get(1) else {
        let output = unary(first, &mut diagnostics);
        return finish_fixed(entry, request, output, diagnostics);
    };
    if matches!(second.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-COMPLEX-SPARSE",
            "complex does not currently accept sparse input",
            1,
        ));
        return finish_fixed(entry, request, unsupported(), diagnostics);
    }
    let output = binary(first, second, &mut diagnostics);
    finish_fixed(entry, request, output, diagnostics)
}

fn unary(input: &ValueFact, diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>) -> ValueFact {
    match class::unary(input) {
        class::ComponentClass::Known(class) => {
            let already_complex = matches!(
                input.kind,
                ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex
            );
            if already_complex {
                return input.clone();
            }
            let mut output = input.clone();
            output.kind = class::output_kind(class);
            super::super::super::support::facts::materialize(&mut output);
            output
        }
        class::ComponentClass::Unknown => dynamic_with_shape(input),
        class::ComponentClass::Invalid => {
            diagnostics.push(argument_error(
                "RM-CATALOG-COMPLEX-INPUT",
                "complex requires numeric or logical input",
                0,
            ));
            unsupported()
        }
    }
}

fn binary(
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    let class = match class::binary(left, right) {
        class::ComponentClass::Known(class) => class,
        class::ComponentClass::Unknown => return dynamic(),
        class::ComponentClass::Invalid => {
            diagnostics.push(argument_error(
                "RM-CATALOG-COMPLEX-INPUT",
                "complex requires real numeric components; integer components require matching classes or a scalar double peer",
                invalid_binary_argument(left),
            ));
            return unsupported();
        }
    };
    if class.integer_class().is_some()
        && (!class::valid_integer_component(left, class)
            || !class::valid_integer_component(right, class))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-COMPLEX-INTEGER-PEER",
            "integer components require the same integer class or a scalar double peer",
            usize::from(class::valid_integer_component(left, class)),
        ));
        return unsupported();
    }

    let mut output = ValueFact::proven(
        class::output_kind(class),
        match shape::same_size_or_scalar(left, right) {
            Ok(shape) => shape,
            Err(mut diagnostic) => {
                diagnostic.argument = Some(1);
                diagnostics.push(diagnostic);
                runmat_types::ShapeFact::Unknown
            }
        },
        StorageFact::Dense,
    );
    super::super::super::support::facts::materialize(&mut output);
    output.residency = preserved_binary_residency(&left.residency, &right.residency);
    output
}

fn invalid_binary_argument(left: &ValueFact) -> usize {
    usize::from(
        matches!(
            left.kind,
            ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real
        ) || matches!(left.kind, ValueKindFact::Logical | ValueKindFact::Unknown),
    )
}

fn dynamic() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn unsupported() -> ValueFact {
    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
}

fn dynamic_with_shape(input: &ValueFact) -> ValueFact {
    let mut output = dynamic();
    output.shape = input.shape.clone();
    output.residency = match input.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        _ => ResidencyFact::Unknown,
    };
    output
}
