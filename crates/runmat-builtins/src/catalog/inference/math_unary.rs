use super::{argument_error, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason, LayoutFact, MutationFact,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
    ViewFact,
};

pub(super) fn infer_exp(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_exponential(request, entry, ExponentialKind::Exp)
}

pub(super) fn infer_expm1(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_exponential(request, entry, ExponentialKind::Expm1)
}

#[derive(Debug, Clone, Copy)]
enum ExponentialKind {
    Exp,
    Expm1,
}

impl ExponentialKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Exp => "exp",
            Self::Expm1 => "expm1",
        }
    }

    const fn accepts_symbolic(self) -> bool {
        matches!(self, Self::Exp)
    }

    const fn preserves_sparse_zeros(self) -> bool {
        matches!(self, Self::Expm1)
    }
}

fn infer_exponential(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: ExponentialKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXPONENTIAL-ARITY",
            format!("{} requires exactly one input", operation.name()),
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
            "RM-CATALOG-EXPONENTIAL-ARITY",
            format!("{} accepts exactly one input", operation.name()),
            1,
        ));
    }

    let mut output = input.clone();
    let mut changes_numeric_class = false;
    match &mut output.kind {
        ValueKindFact::Numeric(numeric) => {
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-EXPONENTIAL-COMPLEX-INTEGER",
                    format!(
                        "{} does not accept complex fixed-width integer input",
                        operation.name()
                    ),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                return finish_fixed(entry, request, output, diagnostics);
            }
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
                changes_numeric_class = true;
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_numeric_class = true;
        }
        ValueKindFact::Symbolic if operation.accepts_symbolic() => {
            return finish_fixed(entry, request, output, diagnostics);
        }
        ValueKindFact::Object(object) => {
            let tabular = object
                .runtime_class
                .as_ref()
                .and_then(|class| class.display_name())
                .is_some_and(|class| matches!(class.as_str(), "table" | "timetable"));
            if tabular {
                object.properties.clear();
                object.properties_complete = false;
                output.alias = AliasFact::Unique;
                output.mutation = MutationFact::ValueSemantics;
            } else {
                preserve_shape_on_dynamic_input(&mut output);
            }
            return finish_fixed(entry, request, output, diagnostics);
        }
        ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
            return finish_fixed(entry, request, output, diagnostics);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-EXPONENTIAL-INPUT",
                format!(
                    "{} requires numeric, logical, character, or supported tabular input{}",
                    operation.name(),
                    if operation.accepts_symbolic() {
                        ", or a symbolic expression"
                    } else {
                        ""
                    }
                ),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            return finish_fixed(entry, request, output, diagnostics);
        }
    }

    if matches!(output.storage, StorageFact::Sparse) {
        output.residency = ResidencyFact::Host;
        if !operation.preserves_sparse_zeros() {
            output.storage = StorageFact::Dense;
        }
    } else if changes_numeric_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
    materialize_output_preserving_storage(&mut output);
    finish_fixed(entry, request, output, diagnostics)
}

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

fn materialize_output_preserving_storage(output: &mut ValueFact) {
    if matches!(output.storage, StorageFact::Sparse) {
        output.view = ViewFact::Materialized;
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
    } else {
        materialize_output(output);
    }
}

fn preserve_shape_on_dynamic_input(output: &mut ValueFact) {
    let shape = output.shape.clone();
    *output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = shape;
}
