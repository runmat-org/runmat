use super::class;
use crate::{BuiltinCatalogEntry, RemainderFunction};
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, DynamicReason, NumericDomain, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, finish_fixed, numeric_kind, preserved_binary_residency};
use super::super::binary_containers::{self, Operation};

pub(super) fn infer(
    function: RemainderFunction,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let name = name(function);
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-REMAINDER-ARITY",
            format!("{name} requires exactly two inputs"),
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
            "RM-CATALOG-REMAINDER-SPARSE",
            format!("{name} does not currently accept sparse input"),
            usize::from(!matches!(left.storage, StorageFact::Sparse)),
        ));
        return finish_fixed(entry, request, unsupported(), diagnostics);
    }
    if let Some(output) = binary_containers::infer(
        Operation::Remainder(function),
        left,
        right,
        &mut diagnostics,
    ) {
        return finish_fixed(entry, request, output, diagnostics);
    }

    let Some(left_numeric) = numeric_input(left, name, 0, &mut diagnostics) else {
        return finish_fixed(entry, request, dynamic(), diagnostics);
    };
    let Some(right_numeric) = numeric_input(right, name, 1, &mut diagnostics) else {
        return finish_fixed(entry, request, dynamic(), diagnostics);
    };
    if left_numeric.domain == NumericDomain::Complex
        || right_numeric.domain == NumericDomain::Complex
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-REMAINDER-COMPLEX",
            format!("{name} requires real operands"),
            usize::from(right_numeric.domain == NumericDomain::Complex),
        ));
        return finish_fixed(entry, request, unsupported(), diagnostics);
    }
    let Some(output_class) = class::output(left, left_numeric, right, right_numeric) else {
        diagnostics.push(argument_error(
            "RM-CATALOG-REMAINDER-INTEGER-CLASS",
            format!("{name} requires integer operands to share a class; the other operand may be scalar double"),
            1,
        ));
        return finish_fixed(entry, request, unsupported(), diagnostics);
    };

    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = numeric_kind(output_class, NumericDomain::Real);
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
    name: &str,
    argument: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<runmat_types::NumericFact> {
    let numeric = class::input(&input.kind);
    if numeric.is_none() && !matches!(input.kind, ValueKindFact::Unknown) {
        diagnostics.push(argument_error(
            "RM-CATALOG-REMAINDER-INPUT",
            format!("{name} requires real numeric, logical, character, table, timetable, or duration input"),
            argument,
        ));
    }
    numeric
}

fn name(function: RemainderFunction) -> &'static str {
    match function {
        RemainderFunction::Modulus => "mod",
        RemainderFunction::Remainder => "rem",
    }
}

fn dynamic() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn unsupported() -> ValueFact {
    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
}
