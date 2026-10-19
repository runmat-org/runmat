//! Descriptive source-to-binding projection. Actual buffers are authenticated
//! later by whole-set input preflight and the submitting provider, not here.
use super::{inputs::Inputs, ProviderPrecision, ResolvedProgram};
use crate::{fusion::FusionOp, graph::ValueId};
use runmat_types::NumericClass;
use std::collections::{HashMap, HashSet};

pub(super) struct Projection {
    pub controls: HashSet<ValueId>,
    pub precision: ProviderPrecision,
    pub expressions: HashMap<ValueId, String>,
    pub words: HashMap<ValueId, String>,
}

impl Projection {
    pub fn derive(
        mapped: &Inputs<'_>,
        program: &ResolvedProgram,
        owner: ProviderPrecision,
    ) -> anyhow::Result<Self> {
        // A word recipe may omit only authenticated metadata controls. All
        // actual Double data/interior nodes still require their native owner.
        let mut controls = program.embedded_controls_for_precision(
            &mapped.producer,
            &[mapped.root],
            ProviderPrecision::F32,
        );
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
            controls = program.embedded_controls_for_precision(
                &mapped.producer,
                &[mapped.root],
                precision,
            );
        }
        let word_storage = precision == ProviderPrecision::F32
            && crate::fusion::typed_producer::TypedProducer::requires_binary32_word_transport(
                &mapped.producer,
                program,
                Some(&[mapped.root]),
            )?;
        let words = mapped
            .data
            .iter()
            .filter(|input| !controls.contains(&input.value_id))
            .enumerate()
            .filter(|_| word_storage)
            .map(|(slot, input)| (input.value_id, format!("input{slot}.data[i{slot}]")))
            .collect();
        let expressions = mapped
            .data
            .iter()
            .filter(|input| !controls.contains(&input.value_id))
            .enumerate()
            .map(|(slot, input)| {
                let load = format!("input{slot}.data[i{slot}]");
                (
                    input.value_id,
                    if word_storage {
                        format!("bitcast<f32>({load})")
                    } else {
                        load
                    },
                )
            })
            // Dependency markers only: checked POWER embeds original bits and
            // never reads or casts these strings into a physical expression.
            .chain(controls.iter().map(|&id| (id, String::new())))
            .collect();
        Ok(Self {
            controls,
            precision,
            expressions,
            words,
        })
    }
}
