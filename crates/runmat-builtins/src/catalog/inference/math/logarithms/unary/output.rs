use super::super::super::super::{argument_error, numeric_kind};
use super::policy::UnaryLogarithmPolicy;
use runmat_types::{
    AliasFact, DynamicReason, MutationFact, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ValueFact, ValueKindFact,
};

pub(super) struct InferredOutput {
    pub(super) value: ValueFact,
    pub(super) materialize: bool,
    pub(super) preserve_dynamic_shape: bool,
}

pub(super) fn infer(
    policy: UnaryLogarithmPolicy,
    input: &ValueFact,
    literal_domain: Option<NumericDomain>,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> InferredOutput {
    let mut value = input.clone();
    let mut changes_class = false;
    let (materialize, preserve_dynamic_shape) = match &input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) if !matches!(class, NumericClass::Double | NumericClass::Single) => {
            diagnostics.push(argument_error(
                policy.complex_integer_diagnostic(),
                format!(
                    "{} does not accept complex fixed-width integer input",
                    policy.name()
                ),
                0,
            ));
            value = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            (false, false)
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }) => {
            value.kind = numeric_kind(*class, NumericDomain::Complex);
            (true, false)
        }
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => {
            let output_class = if matches!(class, NumericClass::Double | NumericClass::Single) {
                *class
            } else {
                changes_class = true;
                NumericClass::Double
            };
            let domain = literal_domain.or_else(|| {
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
                value.kind = numeric_kind(output_class, domain);
                (true, false)
            } else {
                (false, true)
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            value.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            changes_class = true;
            (true, false)
        }
        ValueKindFact::Symbolic if policy.accepts_symbolic() => (true, false),
        ValueKindFact::Object(object)
            if policy.accepts_tabular()
                && object
                    .runtime_class
                    .as_ref()
                    .is_some_and(runmat_types::standard::is_tabular) =>
        {
            let ValueKindFact::Object(object) = &mut value.kind else {
                unreachable!("tabular branch preserves object facts")
            };
            object.properties.clear();
            object.properties_complete = false;
            value.alias = AliasFact::Unique;
            value.mutation = MutationFact::ValueSemantics;
            (false, false)
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown => (false, true),
        _ => {
            diagnostics.push(argument_error(
                policy.input_diagnostic(),
                format!(
                    "{} requires a supported numeric or container input",
                    policy.name()
                ),
                0,
            ));
            value = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            (false, false)
        }
    };
    if changes_class && matches!(value.residency, ResidencyFact::Device { .. }) {
        value.residency = ResidencyFact::Unknown;
    }
    InferredOutput {
        value,
        materialize,
        preserve_dynamic_shape,
    }
}
