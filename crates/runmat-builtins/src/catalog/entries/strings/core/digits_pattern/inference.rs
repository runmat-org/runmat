use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        super::super::pattern::facts::pattern_fact(),
        super::super::pattern::facts::bounded_count_diagnostics(request, "digitsPattern", true),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };
    #[test]
    fn exact_wide_bounds_and_reversed_range() {
        let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        let mut request = CallRequest {
            arguments: vec![scalar.clone(), scalar],
            literals: LiteralContext::new(vec![
                LiteralValue::Integer {
                    text: "9007199254740993".into(),
                    class: NumericClass::UInt64,
                },
                LiteralValue::Integer {
                    text: "9007199254740994".into(),
                    class: NumericClass::UInt64,
                },
            ]),
            outputs: OutputSelection::default(),
        };
        let entry = &super::super::DIGITS_PATTERN_CATALOG_ENTRY;
        assert!(infer(&request, entry).diagnostics.is_empty());
        request.literals.literal_args.swap(0, 1);
        assert!(infer(&request, entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-digitsPattern-RANGE"));
        request.arguments[0] = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        assert!(infer(&request, entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-digitsPattern-COUNT"));
    }

    #[test]
    fn known_resident_count_and_excess_outputs_are_diagnosed() {
        let mut resident = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt32,
            domain: NumericDomain::Real,
        }));
        resident.residency = ResidencyFact::Device { provider: None };
        let request = CallRequest {
            arguments: vec![resident],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        let diagnostics = infer(&request, &super::super::DIGITS_PATTERN_CATALOG_ENTRY).diagnostics;
        assert!(diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-digitsPattern-COUNT"));
        assert!(diagnostics.iter().any(|d| d.code == "RM-TYPE-CALL-ARITY"));
    }
}
