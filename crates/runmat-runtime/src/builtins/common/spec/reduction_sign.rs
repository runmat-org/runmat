//! Optional sign projection from a canonical reduction owner's already
//! defined preparation and ordered stage resolution. No second parser.

use super::{FusionReductionContext, FusionReductionControls, FusionReductionDeclaration};
use runmat_types::{NumericClass, NumericDomain, StorageFact, ValueFact};
use std::{future::Future, pin::Pin};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionReductionSignProjection {
    pub class: NumericClass,
    pub storage: StorageFact,
}

pub type FusionReductionSignFuture<'a> =
    Pin<Box<dyn Future<Output = crate::BuiltinResult<Option<FusionReductionSignProjection>>> + 'a>>;

impl FusionReductionDeclaration {
    pub fn prove_nonnegative_or_nan<'a>(
        &'a self,
        context: FusionReductionContext<'a>,
        output: &'a ValueFact,
    ) -> FusionReductionSignFuture<'a> {
        Box::pin(async move {
            if context.real_floating_input()?.is_none() {
                return Ok(None);
            }
            let Some(prepared) = (self.prepare)(FusionReductionControls {
                values: context.controls,
                requested_outputs: context.requested_outputs,
            })
            .await?
            else {
                return Ok(None);
            };
            let Some(stages) = prepared.resolve_stages(context.input)? else {
                return Ok(None);
            };
            let Some(storage) = prepared.nonnegative_output_storage(context.input, output, &stages)
            else {
                return Ok(None);
            };
            let Some(last) = stages.last() else {
                return Ok(None);
            };
            if !matches!(
                last.output_class,
                NumericClass::Single | NumericClass::Double
            ) || output.numeric().is_some_and(|numeric| {
                numeric.domain != NumericDomain::Real || numeric.class != last.output_class
            }) {
                return Ok(None);
            }
            if !matches!(storage, StorageFact::Scalar | StorageFact::Dense)
                || matches!(output.storage, StorageFact::Sparse)
            {
                return Ok(None);
            }
            Ok(Some(FusionReductionSignProjection {
                class: last.output_class,
                storage,
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::spec::PreparedFusionReduction;
    use crate::value_fact::value_fact;
    use runmat_value::Value;

    struct OpaqueReduction;
    impl PreparedFusionReduction for OpaqueReduction {
        fn resolve(
            &self,
            _input: &ValueFact,
        ) -> crate::BuiltinResult<Option<super::super::FusionReductionContract>> {
            Ok(None)
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn unspecified_reduction_cannot_certify_dense_from_sign() {
        let fact = value_fact(&Value::Num(1.0));
        assert_eq!(
            OpaqueReduction.nonnegative_output_storage(&fact, &fact, &[]),
            None
        );
    }
}
