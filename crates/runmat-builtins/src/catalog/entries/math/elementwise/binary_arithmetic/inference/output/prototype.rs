//! Output prototypes refine result domain and placement independently of operands.
use runmat_types::{NumericDomain, ResidencyFact, ValueFact, ValueKindFact};

use super::{argument_error, DiagnosticCollector};

pub(super) fn apply_prototype(
    prototype: Option<&ValueFact>,
    output: &mut ValueFact,
    diagnostics: &mut DiagnosticCollector,
) {
    let Some(prototype) = prototype else { return };
    output.residency = match &prototype.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        ResidencyFact::Device { provider } => ResidencyFact::Device {
            provider: provider.clone(),
        },
        _ => ResidencyFact::Unknown,
    };
    if matches!(
        prototype.numeric().map(|numeric| numeric.domain),
        Some(NumericDomain::Complex)
    ) {
        if let ValueKindFact::Numeric(numeric) = &mut output.kind {
            numeric.domain = NumericDomain::Complex;
        }
    } else if !matches!(
        prototype.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(|| argument_error(
            "RM-CATALOG-BINARY-ARITHMETIC-PROTOTYPE",
            "the 'like' prototype must be numeric, logical, character, or provider-resident numeric data",
            3,
        ));
    }
}
