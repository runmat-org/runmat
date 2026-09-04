use super::super::super::support::facts::{
    materialize_preserving_sparse_storage, preserve_shape_as_dynamic,
};
use super::super::super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, ExponentialKind};
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, NumericClass,
    NumericDomain, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

impl ExponentialKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Natural => "exp",
            Self::MinusOne => "expm1",
        }
    }

    const fn accepts_symbolic(self) -> bool {
        matches!(self, Self::Natural)
    }

    const fn preserves_sparse_zeros(self) -> bool {
        matches!(self, Self::MinusOne)
    }
}

pub(super) fn infer(
    operation: ExponentialKind,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
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
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(runmat_types::standard::TABLE)
                    || class.is(runmat_types::standard::TIMETABLE)
            }) {
                object.properties.clear();
                object.properties_complete = false;
                output.alias = AliasFact::Unique;
                output.mutation = MutationFact::ValueSemantics;
            } else {
                preserve_shape_as_dynamic(&mut output);
            }
            return finish_fixed(entry, request, output, diagnostics);
        }
        ValueKindFact::Unknown => {
            preserve_shape_as_dynamic(&mut output);
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
    materialize_preserving_sparse_storage(&mut output);
    finish_fixed(entry, request, output, diagnostics)
}
