use super::support::facts::{
    materialize, materialize_preserving_sparse_storage, preserve_shape_as_dynamic,
};
use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, RootKind};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_root(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: RootKind,
) -> CallInference {
    let name = match kind {
        RootKind::Principal => "sqrt",
        RootKind::RealOnly => "realsqrt",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROOT-ARITY",
            format!("{name} requires exactly one input"),
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
            "RM-CATALOG-ROOT-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }

    if kind == RootKind::Principal && matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
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

    let literal = request
        .literals
        .literal_args
        .first()
        .and_then(root_literal_domain);
    let mut output = input.clone();
    match kind {
        RootKind::Principal => infer_principal_root(input, literal, &mut output, &mut diagnostics),
        RootKind::RealOnly => infer_real_root(input, literal, &mut output, &mut diagnostics),
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn infer_principal_root(
    input: &ValueFact,
    literal: Option<RootLiteralDomain>,
    output: &mut ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SQRT-COMPLEX-INTEGER",
                "sqrt does not accept complex fixed-width integer input",
                0,
            ));
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            output.kind = numeric_kind(*class, NumericDomain::Complex);
            materialize(output);
            dynamic_device_residency(output);
        }
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
                .map(RootLiteralDomain::principal_output)
                .or_else(|| {
                    matches!(
                        class,
                        NumericClass::UInt8
                            | NumericClass::UInt16
                            | NumericClass::UInt32
                            | NumericClass::UInt64
                    )
                    .then_some(NumericDomain::Real)
                });
            if let Some(domain) = domain {
                output.kind = numeric_kind(output_class, domain);
                materialize(output);
                dynamic_device_residency(output);
            } else {
                preserve_shape_as_dynamic(output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize(output);
            dynamic_device_residency(output);
        }
        ValueKindFact::Symbolic => {}
        ValueKindFact::Unknown => preserve_shape_as_dynamic(output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SQRT-INPUT",
                "sqrt requires numeric, logical, character, or symbolic input",
                0,
            ));
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
}

fn infer_real_root(
    input: &ValueFact,
    literal: Option<RootLiteralDomain>,
    output: &mut ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double | NumericClass::Single,
            domain: NumericDomain::Real,
        }) => {
            if matches!(
                literal,
                Some(RootLiteralDomain::Negative | RootLiteralDomain::Complex)
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REALSQRT-DOMAIN",
                    "realsqrt input must be real and nonnegative",
                    0,
                ));
            }
            materialize_preserving_sparse_storage(output);
            if matches!(output.storage, StorageFact::Sparse) {
                output.residency = ResidencyFact::Host;
            } else {
                dynamic_device_residency(output);
            }
        }
        ValueKindFact::Unknown => preserve_shape_as_dynamic(output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-REALSQRT-INPUT",
                "realsqrt requires real single or double input",
                0,
            ));
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
}

fn dynamic_device_residency(output: &mut ValueFact) {
    if matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootLiteralDomain {
    Nonnegative,
    Negative,
    Complex,
}

impl RootLiteralDomain {
    const fn principal_output(self) -> NumericDomain {
        match self {
            Self::Nonnegative => NumericDomain::Real,
            Self::Negative | Self::Complex => NumericDomain::Complex,
        }
    }
}

fn root_literal_domain(literal: &LiteralValue) -> Option<RootLiteralDomain> {
    match literal {
        LiteralValue::Number(value) => Some(root_real_domain(*value)),
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => {
            text.parse().ok().map(root_real_domain)
        }
        LiteralValue::Complex { .. } => Some(RootLiteralDomain::Complex),
        LiteralValue::Bool(_) | LiteralValue::Character(_) | LiteralValue::Empty => {
            Some(RootLiteralDomain::Nonnegative)
        }
        LiteralValue::Vector(values) => root_literal_sequence_domain(values.iter()),
        LiteralValue::Matrix(rows) => root_literal_sequence_domain(rows.iter().flatten()),
        LiteralValue::String(_)
        | LiteralValue::Keyword(_)
        | LiteralValue::Symbolic(_)
        | LiteralValue::Unknown => None,
    }
}

fn root_literal_sequence_domain<'a>(
    values: impl Iterator<Item = &'a LiteralValue>,
) -> Option<RootLiteralDomain> {
    let mut domain = RootLiteralDomain::Nonnegative;
    for value in values {
        match root_literal_domain(value)? {
            RootLiteralDomain::Complex => return Some(RootLiteralDomain::Complex),
            RootLiteralDomain::Negative => domain = RootLiteralDomain::Negative,
            RootLiteralDomain::Nonnegative => {}
        }
    }
    Some(domain)
}

fn root_real_domain(value: f64) -> RootLiteralDomain {
    if value < 0.0 {
        RootLiteralDomain::Negative
    } else {
        RootLiteralDomain::Nonnegative
    }
}
