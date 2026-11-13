//! Native output geometry/class envelopes of domain-promoting operation owners.
//! Unknown catalog kind stays unknown: this bounds both possible allocations.
use crate::builtins::common::spec::FusionHostOutputDomain;
use runmat_builtins::BuiltinPreparationOperation;
use runmat_types::{NumericClass, NumericDomain, ValueFact, ValueKindFact};

pub struct PromotingOutput {
    pub shape: Vec<usize>,
    pub class: NumericClass,
    pub domain: FusionHostOutputDomain,
}

pub fn promoting_output(
    operation: BuiltinPreparationOperation,
    inputs: &[ValueFact],
    inferred: &ValueFact,
) -> Option<PromotingOutput> {
    let native = |fact: &ValueFact| {
        let numeric = fact.numeric()?;
        (numeric.domain == NumericDomain::Real
            && matches!(numeric.class, NumericClass::Single | NumericClass::Double))
        .then_some(numeric.class)
    };
    let dims = |fact: &ValueFact| {
        fact.shape
            .known_dims()?
            .into_iter()
            .collect::<Option<Vec<_>>>()
    };
    let (shape, class) = match (operation, inputs) {
        (BuiltinPreparationOperation::RealUnary(_), [input]) => (dims(input)?, native(input)?),
        (BuiltinPreparationOperation::RealPower, [left, right]) => {
            let left_shape = dims(left)?;
            let right_shape = dims(right)?;
            let plan =
                crate::builtins::common::broadcast::BroadcastPlan::new(&left_shape, &right_shape)
                    .ok()?;
            let mode = runmat_accelerate_api::RealPowerArithmeticMode::for_classes(
                native(left)?,
                native(right)?,
            )?;
            (plan.output_shape().to_vec(), mode.output_class())
        }
        _ => return None,
    };
    if dims(inferred)? != shape {
        return None;
    }
    let domain = match &inferred.kind {
        ValueKindFact::Unknown => FusionHostOutputDomain::RealOrComplex,
        ValueKindFact::Numeric(numeric) if numeric.class == class => {
            FusionHostOutputDomain::Exact(numeric.domain)
        }
        _ => return None,
    };
    Some(PromotingOutput {
        shape,
        class,
        domain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{CallRequest, NumericFact};
    use runmat_value::{Tensor, Value};

    #[test]
    fn canonical_unknown_promotion_has_an_envelope_not_a_real_fact() {
        let single = crate::value_fact::value_fact(&Value::Tensor(
            Tensor::from_f32(vec![0.5, -0.5], vec![1, 2]).unwrap(),
        ));
        let double = crate::value_fact::value_fact(&Value::Tensor(
            Tensor::new(vec![0.5, 2.0], vec![1, 2]).unwrap(),
        ));
        for (entry, operation, inputs) in [
            (
                &runmat_builtins::LOG_CATALOG_ENTRY,
                BuiltinPreparationOperation::RealUnary(
                    runmat_types::RealUnaryPredicate::Nonnegative,
                ),
                vec![single.clone()],
            ),
            (
                &runmat_builtins::POWER_CATALOG_ENTRY,
                BuiltinPreparationOperation::RealPower,
                vec![single.clone(), double],
            ),
        ] {
            let inference = runmat_builtins::infer_catalog_call(
                entry,
                &CallRequest {
                    arguments: inputs.clone(),
                    literals: Default::default(),
                    outputs: Default::default(),
                },
            );
            assert!(inference.diagnostics.is_empty());
            let [output] = inference.outputs.as_slice() else {
                panic!("one canonical output");
            };
            assert!(matches!(output.kind, ValueKindFact::Unknown));
            let envelope = promoting_output(operation, &inputs, output).unwrap();
            assert_eq!(envelope.class, NumericClass::Single);
            assert_eq!(envelope.shape, [1, 2]);
            assert_eq!(envelope.domain, FusionHostOutputDomain::RealOrComplex);
            let mut contradicted = output.clone();
            contradicted.kind = ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            });
            assert!(promoting_output(operation, &inputs, &contradicted).is_none());
        }
    }
}
