use super::{argument_error, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason, LayoutFact, MutationFact,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
    ViewFact,
};

pub(super) fn infer_phase_angle(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-ARITY",
            "angle requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-ARITY",
            "angle accepts exactly one input",
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PHASE-ANGLE-SPARSE",
            "angle does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            numeric.domain = NumericDomain::Real;
            materialize_output(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-PHASE-ANGLE-INPUT",
                "angle requires real or complex single- or double-precision input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_signum(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-SIGNUM-ARITY",
            "sign requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SIGNUM-ARITY",
            "sign accepts exactly one input",
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SIGNUM-SPARSE",
            "sign does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    let mut changes_class = false;
    match &output.kind {
        ValueKindFact::Numeric(NumericFact { class, domain })
            if domain == &NumericDomain::Complex
                && !matches!(class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SIGNUM-COMPLEX-INTEGER",
                "sign does not accept complex fixed-width integer input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(_) => materialize_output(&mut output),
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_class = true;
            materialize_output(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SIGNUM-INPUT",
                "sign requires numeric, logical, or character input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    if changes_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn materialize_output(output: &mut ValueFact) {
    output.storage = if output.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.view = ViewFact::Materialized;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
}

fn preserve_shape_on_dynamic_input(output: &mut ValueFact) {
    let shape = output.shape.clone();
    *output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = shape;
}
