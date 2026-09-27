//! Request-owned numerical evidence; absent serialized evidence stays unknown.
use crate::PowerResultDomain;
use runmat_types::{RealUnaryClassification, RealUnaryPredicate};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation", content = "domain")]
pub enum NumericalDomainEvidence {
    #[default]
    Unknown,
    RealPower(PowerResultDomain),
    RealUnary {
        predicate: RealUnaryPredicate,
        classification: RealUnaryClassification,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
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
