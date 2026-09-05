use super::class;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, DynamicReason, NumericDomain, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, finish_fixed, numeric_kind, preserved_binary_residency};
use super::super::binary_containers::{self, Operation};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ATAN2-ARITY",
            "atan2 requires exactly two inputs",
            request.arguments.len().min(1),
        ));
    }
    let Some(left) = request.arguments.first() else {
        return finish_fixed(entry, request, dynamic(), diagnostics);
    };
    let Some(right) = request.arguments.get(1) else {
        return finish_fixed(entry, request, dynamic(), diagnostics);
    };
    if matches!(left.storage, StorageFact::Sparse) || matches!(right.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ATAN2-SPARSE",
            "atan2 does not accept sparse input",
            usize::from(!matches!(left.storage, StorageFact::Sparse)),
        ));
        return finish_fixed(entry, request, unsupported(), diagnostics);
    }
    if let Some(output) = binary_containers::infer(Operation::Atan2, left, right, &mut diagnostics)
    {
        return finish_fixed(entry, request, output, diagnostics);
    }
    if matches!(left.kind, ValueKindFact::Object(_))
        || matches!(right.kind, ValueKindFact::Object(_))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-ATAN2-OBJECT",
            "atan2 accepts only table and timetable objects",
            usize::from(matches!(left.kind, ValueKindFact::Object(_))),
        ));
        return finish_fixed(entry, request, unsupported(), diagnostics);
    }

    let Some(left_numeric) = numeric_input(left, 0, &mut diagnostics) else {
        return finish_fixed(entry, request, dynamic(), diagnostics);
    };
    let Some(right_numeric) = numeric_input(right, 1, &mut diagnostics) else {
        return finish_fixed(entry, request, dynamic(), diagnostics);
    };
    if left_numeric.domain == NumericDomain::Complex
        || right_numeric.domain == NumericDomain::Complex
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-ATAN2-COMPLEX",
            "atan2 requires real operands",
            usize::from(right_numeric.domain == NumericDomain::Complex),
        ));
        return finish_fixed(entry, request, unsupported(), diagnostics);
    }

    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = numeric_kind(
        class::output(left_numeric, right_numeric),
        NumericDomain::Real,
    );
    output.shape = match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            ShapeFact::Unknown
        }
    };
    super::super::super::support::facts::materialize(&mut output);
    output.residency = preserved_binary_residency(&left.residency, &right.residency);
    finish_fixed(entry, request, output, diagnostics)
}

fn numeric_input(
    input: &ValueFact,
    argument: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<runmat_types::NumericFact> {
    let numeric = class::input(&input.kind);
    if numeric.is_none() && !matches!(input.kind, ValueKindFact::Unknown) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ATAN2-INPUT",
            "atan2 requires real single, double, table, or timetable input; RunMat mode also accepts integer, logical, and character input",
            argument,
        ));
    }
    numeric
}

fn dynamic() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn unsupported() -> ValueFact {
    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
}
