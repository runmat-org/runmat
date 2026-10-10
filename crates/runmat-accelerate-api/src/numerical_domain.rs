//! Request-owned numerical evidence; absent serialized evidence stays unknown.
use crate::PowerResultDomain;
use runmat_types::{RealUnaryClassification, RealUnaryPredicate};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundDigitsMode {
    Decimals,
    Significant,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation", content = "domain")]
pub enum NumericalDomainEvidence {
    #[default]
    Unknown,
    /// A selected real operation proves every element is nonnegative (either
    /// signed zero included) or NaN. This is request evidence, not a claim of
    /// provider arithmetic equivalence.
    NonnegativeOrNaN,
    RealPower(PowerResultDomain),
    RealUnary {
        predicate: RealUnaryPredicate,
        classification: RealUnaryClassification,
    },
    /// Parsed numerical strategy retained by the admitted domain invocation.
    RoundDigits {
        digits: i32,
        mode: RoundDigitsMode,
    },
    FloatingRemainder(runmat_types::FloatingRemainderKind),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floating_remainder_strategies_round_trip() {
        for kind in [
            runmat_types::FloatingRemainderKind::Modulus,
            runmat_types::FloatingRemainderKind::Remainder,
        ] {
            let evidence = NumericalDomainEvidence::FloatingRemainder(kind);
            assert_eq!(
                serde_json::from_str::<NumericalDomainEvidence>(
                    &serde_json::to_string(&evidence).unwrap()
                )
                .unwrap(),
                evidence
            );
        }
    }
    #[test]
    fn parsed_rounding_strategies_round_trip_without_becoming_tensor_arguments() {
        for mode in [RoundDigitsMode::Decimals, RoundDigitsMode::Significant] {
            let evidence = NumericalDomainEvidence::RoundDigits {
                digits: i32::MAX,
                mode,
            };
            let encoded = serde_json::to_string(&evidence).unwrap();
            assert_eq!(
                serde_json::from_str::<NumericalDomainEvidence>(&encoded).unwrap(),
                evidence
            );
        }
    }
    #[test]
    fn unary_evidence_round_trips_and_old_workloads_stay_unknown() {
        let old: crate::ProviderWorkload =
            serde_json::from_str(r#"{"elements":4,"flops":null,"batch":null}"#).unwrap();
        assert_eq!(old.numerical_domain, NumericalDomainEvidence::Unknown);
        for predicate in [
            RealUnaryPredicate::ClosedUnitInterval,
            RealUnaryPredicate::AtLeastOne,
            RealUnaryPredicate::Nonnegative,
            RealUnaryPredicate::AtLeastMinusOne,
        ] {
            for classification in [
                RealUnaryClassification::Inside,
                RealUnaryClassification::Outside,
                RealUnaryClassification::Indeterminate,
            ] {
                let evidence = NumericalDomainEvidence::RealUnary {
                    predicate,
                    classification,
                };
                assert_eq!(
                    serde_json::from_str::<NumericalDomainEvidence>(
                        &serde_json::to_string(&evidence).unwrap()
                    )
                    .unwrap(),
                    evidence
                );
            }
        }
    }
}
