#[cfg(test)]
mod tests;

use crate::catalog::inference::support::facts::{materialize, preserve_shape_as_dynamic};
use crate::catalog::inference::{finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let prepared = match super::input::prepare(request, "sqrt") {
        Ok(prepared) => prepared,
        Err(diagnostics) => {
            return finish_fixed(
                entry,
                request,
                ValueFact::unknown(DynamicReason::RuntimeValue),
                diagnostics,
            )
        }
    };
    let mut diagnostics = prepared.diagnostics;
    if matches!(prepared.fact.storage, StorageFact::Sparse) {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-SQRT-SPARSE",
            "sqrt does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }
    let mut output = prepared.fact.clone();
    infer_output(
        prepared.fact,
        prepared.literal,
        &mut output,
        &mut diagnostics,
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn infer_output(
    input: &ValueFact,
    literal: Option<super::literal::RootLiteralDomain>,
    output: &mut ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => reject(
            output,
            diagnostics,
            "sqrt does not accept complex fixed-width integer input",
        ),
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => set_numeric_output(output, *class, NumericDomain::Complex),
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                NumericClass::Double
            };
            let domain = literal
                .map(super::literal::RootLiteralDomain::principal_output)
                .or_else(|| is_unsigned(*class).then_some(NumericDomain::Real));
            if let Some(domain) = domain {
                set_numeric_output(output, output_class, domain);
            } else {
                preserve_shape_as_dynamic(output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            set_numeric_output(output, NumericClass::Double, NumericDomain::Real)
        }
        ValueKindFact::Symbolic => {}
        ValueKindFact::Unknown => preserve_shape_as_dynamic(output),
        _ => reject(
            output,
            diagnostics,
            "sqrt requires numeric, logical, character, or symbolic input",
        ),
    }
}

fn set_numeric_output(output: &mut ValueFact, class: NumericClass, domain: NumericDomain) {
    output.kind = numeric_kind(class, domain);
    materialize(output);
    if matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
}

fn reject(
    output: &mut ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
    message: &'static str,
) {
    diagnostics.push(crate::catalog::inference::argument_error(
        "RM-CATALOG-SQRT-INPUT",
        message,
        0,
    ));
    *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
}

const fn is_unsigned(class: NumericClass) -> bool {
    matches!(
        class,
        NumericClass::UInt8 | NumericClass::UInt16 | NumericClass::UInt32 | NumericClass::UInt64
    )
}
