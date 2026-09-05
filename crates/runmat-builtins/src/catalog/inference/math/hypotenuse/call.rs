use super::class;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, DynamicReason, NumericDomain, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, finish_fixed, numeric_kind, preserved_binary_residency};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-HYPOT-ARITY",
            "hypot requires exactly two inputs",
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
            "RM-CATALOG-HYPOT-SPARSE",
            "hypot does not currently accept sparse input",
            usize::from(!matches!(left.storage, StorageFact::Sparse)),
        ));
        return finish_fixed(entry, request, unsupported(), diagnostics);
    }

    let Some(left_numeric) = class::input(&left.kind) else {
        report_invalid_input(left, 0, &mut diagnostics);
        return finish_fixed(entry, request, unsupported(), diagnostics);
    };
    let Some(right_numeric) = class::input(&right.kind) else {
        report_invalid_input(right, 1, &mut diagnostics);
        return finish_fixed(entry, request, unsupported(), diagnostics);
    };

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

fn report_invalid_input(
    input: &ValueFact,
    argument: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    if !matches!(input.kind, ValueKindFact::Unknown) {
        diagnostics.push(argument_error(
            "RM-CATALOG-HYPOT-INPUT",
            "hypot requires single, double, or complex-floating input; RunMat mode also accepts integer, logical, and character input",
            argument,
        ));
    }
}

fn dynamic() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn unsupported() -> ValueFact {
    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
}
