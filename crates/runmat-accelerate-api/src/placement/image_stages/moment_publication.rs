//! Selected public moment boundaries, including casts between nested calls.
//! This is immutable logical work, not physical admission or sign evidence.
use std::collections::HashSet;

use runmat_types::{NumericClass, NumericDomain, StorageFact, ValueFact};
use serde::{Deserialize, Serialize};

use crate::ProviderPrecision;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImageMomentConversion {
    pub source_input: u64,
    pub source_output: u64,
    pub output_fact: ValueFact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImageMomentPublication {
    /// Exact operand of the public reduction call or raw SQRT.
    pub source_input: u64,
    pub raw_output: u64,
    pub raw_fact: ValueFact,
    /// Includes same-class operations. Empty means explicitly proven identity.
    pub conversions: Vec<ProviderImageMomentConversion>,
    pub public_output: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderImageMomentProgram {
    pub mu: Vec<ProviderImageMomentPublication>,
    pub variance: Vec<ProviderImageMomentPublication>,
    pub sigma: ProviderImageMomentPublication,
}

/// The existing canonical fact is the class/geometry authority. Unknown,
/// opaque and sparse carriers cannot become dense because a sign is known.
pub fn moment_precision(fact: &ValueFact) -> Option<ProviderPrecision> {
    let numeric = fact.numeric()?;
    if numeric.domain != NumericDomain::Real
        || !matches!(fact.storage, StorageFact::Dense | StorageFact::Scalar)
        || fact
            .shape
            .element_count()
            .is_none_or(|elements| elements == 0)
        || (fact.storage == StorageFact::Scalar
            && (fact.shape.element_count() != Some(1) || numeric.class != NumericClass::Double))
    {
        return None;
    }
    match numeric.class {
        NumericClass::Single => Some(ProviderPrecision::F32),
        NumericClass::Double => Some(ProviderPrecision::F64),
        _ => None,
    }
}

impl ProviderImageMomentPublication {
    pub fn public_fact(&self) -> &ValueFact {
        self.conversions
            .last()
            .map_or(&self.raw_fact, |call| &call.output_fact)
    }

    fn validate_into(&self, produced: &mut HashSet<u64>) -> bool {
        if self.source_input == self.raw_output
            || !produced.insert(self.raw_output)
            || moment_precision(&self.raw_fact).is_none()
        {
            return false;
        }
        let mut current = self.raw_output;
        for call in &self.conversions {
            if call.source_input != current
                || call.source_output == self.source_input
                || !produced.insert(call.source_output)
                || call.output_fact.shape != self.raw_fact.shape
                || moment_precision(&call.output_fact).is_none()
            {
                return false;
            }
            current = call.source_output;
        }
        self.public_output == current
    }
}

impl ProviderImageMomentProgram {
    pub fn source_outputs(&self) -> impl Iterator<Item = u64> + '_ {
        self.mu
            .iter()
            .chain(&self.variance)
            .chain([&self.sigma])
            .flat_map(|publication| {
                std::iter::once(publication.raw_output).chain(
                    publication
                        .conversions
                        .iter()
                        .map(|call| call.source_output),
                )
            })
    }

    pub fn validate(&self) -> bool {
        let mut produced = HashSet::new();
        for route in [&self.mu, &self.variance] {
            if !(1..=2).contains(&route.len()) {
                return false;
            }
            for (index, publication) in route.iter().enumerate() {
                if (index > 0 && publication.source_input != route[index - 1].public_output)
                    || !publication.validate_into(&mut produced)
                {
                    return false;
                }
            }
        }
        self.sigma.validate_into(&mut produced)
    }

    pub fn conversion_calls(&self) -> usize {
        self.mu
            .iter()
            .chain(&self.variance)
            .chain([&self.sigma])
            .map(|publication| publication.conversions.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DimensionFact, NumericFact, ShapeFact, ValueKindFact};

    fn fact(class: NumericClass) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Shaped {
                dims: vec![
                    DimensionFact::Known(2),
                    DimensionFact::Known(1),
                    DimensionFact::Known(1),
                ],
            },
            StorageFact::Dense,
        )
    }
    fn publication(raw: u64, input: u64, outputs: &[u64]) -> ProviderImageMomentPublication {
        let mut current = raw;
        let conversions = outputs
            .iter()
            .map(|&output| {
                let call = ProviderImageMomentConversion {
                    source_input: current,
                    source_output: output,
                    output_fact: fact(NumericClass::Single),
                };
                current = output;
                call
            })
            .collect();
        ProviderImageMomentPublication {
            source_input: input,
            raw_output: raw,
            raw_fact: fact(NumericClass::Single),
            conversions,
            public_output: current,
        }
    }
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn zero_one_many_same_class_and_exact_chain_negatives() {
        let program = ProviderImageMomentProgram {
            mu: vec![publication(18, 2, &[19])],
            variance: vec![publication(27, 25, &[])],
            sigma: publication(29, 28, &[30, 31]),
        };
        assert!(program.validate());
        assert_eq!(program.conversion_calls(), 3);
        let mut bad = program.clone();
        bad.sigma.conversions.swap(0, 1);
        assert!(!bad.validate());
        let mut bad = program.clone();
        bad.mu[0].public_output = 18;
        assert!(!bad.validate());
        let mut bad = program.clone();
        bad.mu[0].conversions[0].source_input = 17;
        assert!(!bad.validate());
        let mut bad = program.clone();
        bad.mu[0].conversions[0].output_fact = fact(NumericClass::Int32);
        assert!(!bad.validate());
        let mut bad = program.clone();
        bad.mu[0].raw_fact.storage = StorageFact::Unknown;
        assert!(!bad.validate());
        let mut bad = program.clone();
        bad.mu[0].raw_fact.storage = StorageFact::Scalar;
        assert!(!bad.validate());
        let mut bad = program.clone();
        bad.sigma.raw_output = 18;
        assert!(!bad.validate());
        let mut bad = program.clone();
        bad.sigma.conversions[0].output_fact.shape = ShapeFact::Scalar;
        assert!(!bad.validate());
        let mut nested = program;
        nested.mu.push(publication(21, 19, &[22]));
        assert!(nested.validate());
        nested.mu[1].source_input = 18;
        assert!(!nested.validate());
    }
}
