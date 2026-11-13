//! Descriptive source-to-binding projection. Actual buffers are authenticated
//! later by whole-set input preflight and the submitting provider, not here.
use super::{inputs::Inputs, ProviderPrecision, ResolvedProgram};
use crate::fusion::typed_producer::ExpressionSlots;
use crate::{fusion::FusionOp, graph::ValueId};
use runmat_types::NumericClass;

pub(super) struct Projection {
    pub controls: Vec<ValueId>,
    pub precision: ProviderPrecision,
    pub word_storage: bool,
}

impl Projection {
    pub fn derive(
        mapped: &Inputs<'_>,
        program: &ResolvedProgram,
        owner: ProviderPrecision,
    ) -> anyhow::Result<Self> {
        // A word recipe may omit only authenticated metadata controls. All
        // actual Double data/interior nodes still require their native owner.
        let mut controls = program.embedded_controls_sorted_for_precision(
            &mapped.producer,
            &[mapped.root],
            ProviderPrecision::F32,
        )?;
        // Preserve all genuine Double data/interior operations. The buffer
        // choice cannot grant arithmetic admission or narrow a source fact.
        let single_operations = mapped.producer.operations.iter().all(|operation| {
            let output = match operation {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => *output,
            };
            output.is_some_and(|id| program.floating_class(id).ok() == Some(NumericClass::Single))
        });
        let single_data = mapped
            .data
            .iter()
            .filter(|input| !controls.contains(&input.value_id))
            .all(|input| {
                runmat_runtime::value_fact::value_fact(input.value)
                    .numeric()
                    .is_some_and(|fact| fact.class == NumericClass::Single)
            });
        let precision = if single_operations && single_data {
            ProviderPrecision::F32
        } else {
            owner
        };
        if precision != ProviderPrecision::F32 {
            controls = program.embedded_controls_sorted_for_precision(
                &mapped.producer,
                &[mapped.root],
                precision,
            )?;
        }
        let word_storage = precision == ProviderPrecision::F32
            && crate::fusion::typed_producer::TypedProducer::requires_binary32_word_transport(
                &mapped.producer,
                program,
                Some(&[mapped.root]),
            )?;
        Ok(Self {
            controls,
            precision,
            word_storage,
        })
    }

    /// Called only after the selected producer's counted phase is admitted.
    pub fn expressions(
        &self,
        mapped: &Inputs<'_>,
    ) -> anyhow::Result<(ExpressionSlots, ExpressionSlots)> {
        // This method runs inside the caller's pre-admitted counted producer
        // phase. Fragment allocations use the same exact formatter but do not
        // reserve those already-funded bytes a second time.
        let mut expressions = ExpressionSlots::new(&mapped.producer)?;
        let mut words = ExpressionSlots::new(&mapped.producer)?;
        for (slot, input) in mapped
            .data
            .iter()
            .filter(|input| !self.controls.contains(&input.value_id))
            .enumerate()
        {
            let load = crate::fusion::typed_producer::checked_format(format_args!(
                "input{slot}.data[i{slot}]"
            ))?;
            if self.word_storage {
                words.insert(input.value_id, load.clone())?;
                expressions.insert(
                    input.value_id,
                    crate::fusion::typed_producer::checked_format(format_args!(
                        "bitcast<f32>({load})"
                    ))?,
                )?;
            } else {
                expressions.insert(input.value_id, load)?;
            }
        }
        for &id in &self.controls {
            expressions.insert_if_missing(id)?;
        }
        Ok((expressions, words))
    }
}
