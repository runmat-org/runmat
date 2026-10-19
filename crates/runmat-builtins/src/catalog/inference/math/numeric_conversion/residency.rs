use super::super::super::argument_error;
use runmat_types::{
    InferenceDiagnostic, NumericDomain, NumericFact, ResidencyFact, ValueFact, ValueKindFact,
};

pub(super) fn apply_default_conversion_residency(output: &mut ValueFact) {
    if matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
}

pub(super) fn apply_like_prototype_residency(
    output: &mut ValueFact,
    prototype: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if prototype.storage == runmat_types::StorageFact::Sparse {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
            "sparse numeric conversion prototypes are not supported",
            2,
        ));
        output.residency = ResidencyFact::Unknown;
        output.storage = runmat_types::StorageFact::Unknown;
        return;
    }
    match prototype.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical => {
            output.residency = match &prototype.residency {
                ResidencyFact::Host => ResidencyFact::Host,
                ResidencyFact::Device { provider } => ResidencyFact::Device {
                    provider: provider.clone(),
                },
                ResidencyFact::Unknown => ResidencyFact::Unknown,
                ResidencyFact::Remote { .. } => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
                        "a remote value cannot be used as a numeric conversion prototype",
                        2,
                    ));
                    ResidencyFact::Unknown
                }
            };
        }
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        }) => {
            output.residency = match &prototype.residency {
                ResidencyFact::Device { provider } => ResidencyFact::Device {
                    provider: provider.clone(),
                },
                ResidencyFact::Host | ResidencyFact::Remote { .. } => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
                        "complex host numeric conversion prototypes are not supported",
                        2,
                    ));
                    ResidencyFact::Unknown
                }
                ResidencyFact::Unknown => ResidencyFact::Unknown,
            };
        }
        ValueKindFact::Unknown => output.residency = ResidencyFact::Unknown,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
                "numeric conversion prototypes must be real numeric or logical values",
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }
    output.storage = match &output.residency {
        ResidencyFact::Device { .. } => runmat_types::StorageFact::Opaque,
        ResidencyFact::Host => output.storage,
        _ => runmat_types::StorageFact::Unknown,
    };
}
