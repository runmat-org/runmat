//! Ordinary MixedDoubleThenSingle scalar Add, composed from two or three existing
//! privately prepared real recipes. No image/source permission is established.
use super::*;

#[path = "mixed_add/resources.rs"]
mod resources;
pub use resources::{
    ProviderMixedRealAddObserved, ProviderMixedRealAddResources,
    ProviderMixedRealAddStepObservations,
};
#[path = "mixed_add/control.rs"]
mod control;
#[path = "mixed_add/cleanup.rs"]
mod cleanup;
#[cfg(test)]
#[path = "mixed_add/tests.rs"]
mod tests;

#[derive(Clone, Copy)]
enum Step {
    Widen,
    Add,
    Narrow,
}
const STEPS: [Step; 3] = [Step::Widen, Step::Add, Step::Narrow];
const DOUBLE_SOURCE_STEPS: [Step; 2] = [Step::Add, Step::Narrow];

#[derive(Clone)]
enum PreparedInput {
    Widen(ProviderRealElementwiseRecipe),
    DoubleSource,
}

pub struct ProviderExecutedMixedRealAdd {
    pub output: GpuTensorHandle,
    pub observed: ProviderMixedRealAddObserved,
}

#[derive(Clone)]
pub struct ProviderMixedRealAddRecipe {
    owner: &'static dyn AccelProvider,
    original: ProviderRealElementwiseRequest,
    input: PreparedInput,
    add: ProviderRealElementwiseRecipe,
    narrow: ProviderRealElementwiseRecipe,
    resources: ProviderMixedRealAddResources,
}

impl ProviderMixedRealAddRecipe {
    pub fn plan(
        owner: &'static dyn AccelProvider,
        original: ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<Self>> {
        let ProviderRealElementwiseOperation::ScalarAdd {
            control,
            mode: runmat_types::RealFloatingAddMode::MixedDoubleThenSingle,
        } = original.operation
        else {
            return Ok(None);
        };
        let Some((route, native_control)) =
            control::native_double_control(original.source.element_type, control)
        else {
            return Ok(None);
        };
        let input = match route {
            control::InputRoute::SingleSource => {
                let Some(widen) = ProviderRealElementwiseRecipe::plan(
                    owner,
                    ProviderRealElementwiseRequest {
                        operation: ProviderRealElementwiseOperation::FloatingCast {
                            target: ProviderPrecision::F64,
                        },
                        source: original.source.clone(),
                        source_buffer_bytes: original.source_buffer_bytes,
                    },
                )?
                else {
                    return Ok(None);
                };
                PreparedInput::Widen(widen)
            }
            control::InputRoute::DoubleSource => PreparedInput::DoubleSource,
        };
        let (add_source, add_source_bytes) = match &input {
            PreparedInput::Widen(widen) => {
                (widen.output().clone(), widen.physical().output_buffer_bytes)
            }
            PreparedInput::DoubleSource => (original.source.clone(), original.source_buffer_bytes),
        };
        let Some(add) = ProviderRealElementwiseRecipe::plan(
            owner,
            ProviderRealElementwiseRequest {
                operation: ProviderRealElementwiseOperation::ScalarAdd {
                    control: native_control,
                    mode: runmat_types::RealFloatingAddMode::Double,
                },
                source: add_source,
                source_buffer_bytes: add_source_bytes,
            },
        )?
        else {
            return Ok(None);
        };
        let Some(narrow) = ProviderRealElementwiseRecipe::plan(
            owner,
            ProviderRealElementwiseRequest {
                operation: ProviderRealElementwiseOperation::FloatingCast {
                    target: ProviderPrecision::F32,
                },
                source: add.output().clone(),
                source_buffer_bytes: add.physical().output_buffer_bytes,
            },
        )?
        else {
            return Ok(None);
        };
        let resources = match &input {
            PreparedInput::Widen(widen) => ProviderMixedRealAddResources::planned([
                widen.physical(),
                add.physical(),
                narrow.physical(),
            ]),
            PreparedInput::DoubleSource => {
                ProviderMixedRealAddResources::planned([add.physical(), narrow.physical()])
            }
        };
        let Some(resources) = resources else {
            return Ok(None);
        };
        Ok(Some(Self {
            owner,
            original,
            input,
            add,
            narrow,
            resources,
        }))
    }

    fn recipe(&self, step: Step) -> &ProviderRealElementwiseRecipe {
        match step {
            Step::Widen => match &self.input {
                PreparedInput::Widen(widen) => widen,
                PreparedInput::DoubleSource => {
                    unreachable!("private Double source route has no Widen step")
                }
            },
            Step::Add => &self.add,
            Step::Narrow => &self.narrow,
        }
    }
    fn steps(&self) -> &'static [Step] {
        match &self.input {
            PreparedInput::Widen(_) => &STEPS,
            PreparedInput::DoubleSource => &DOUBLE_SOURCE_STEPS,
        }
    }
    pub fn request(&self) -> &ProviderRealElementwiseRequest {
        &self.original
    }
    pub fn output(&self) -> &ProviderRepresentation {
        self.narrow.output()
    }
    pub fn resources(&self) -> &ProviderMixedRealAddResources {
        &self.resources
    }
    pub fn identity(&self) -> anyhow::Result<runmat_execution::Digest> {
        let ProviderRealElementwiseOperation::ScalarAdd {
            control,
            mode: runmat_types::RealFloatingAddMode::MixedDoubleThenSingle,
        } = self.original.operation
        else {
            unreachable!("private mixed recipe mode");
        };
        Ok(runmat_execution::Digest::sha256(serde_json::to_vec(&(
            "ordinary-mixed-double-then-single-add-v2",
            control,
            &self.original.source,
            self.original.source_buffer_bytes,
            self.steps()
                .iter()
                .map(|step| self.recipe(*step).identity())
                .collect::<anyhow::Result<Vec<_>>>()?,
            &self.resources,
        ))?))
    }
    pub async fn execute(
        &self,
        input: &GpuTensorHandle,
    ) -> ProviderExecutionResult<ProviderExecutedMixedRealAdd> {
        self.execute_protected(input, &[]).await
    }

    pub async fn execute_protected(
        &self,
        input: &GpuTensorHandle,
        borrowed: &[&GpuTensorHandle],
    ) -> ProviderExecutionResult<ProviderExecutedMixedRealAdd> {
        // Re-admit the entire sequence before its first effect; each hook then
        // rederives its own live owner/class/view/extent and physical proposal.
        let Some(current) = Self::plan(self.owner, self.original.clone())? else {
            return Err(crate::unsupported_provider_operation(
                "mixed real Add recipes unavailable before execution",
            )
            .into());
        };
        if current.resources != self.resources
            || current.identity().map_err(ProviderExecutionError::from)?
                != self.identity().map_err(ProviderExecutionError::from)?
        {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "mixed real Add prepared sequence changed"
            )));
        }
        let mut owned = Vec::with_capacity(self.steps().len());
        let mut receipts = Vec::with_capacity(self.steps().len());
        for &step in self.steps() {
            let result = {
                let source = owned.last().unwrap_or(input);
                let protected = std::iter::once(input)
                    .chain(borrowed.iter().copied())
                    .chain(owned.iter())
                    .collect::<Vec<_>>();
                self.recipe(step)
                    .execute_protected(source, &protected)
                    .await
            };
            match result {
                Ok(result) => {
                    owned.push(result.output);
                    receipts.push(result.observed);
                }
                Err(error) => {
                    let error = if owned.is_empty() {
                        error
                    } else {
                        error.after_effect()
                    };
                    return Err(cleanup::failure(error, owned));
                }
            }
        }
        // Every Double intermediate remains owned through the final carrier,
        // source/borrowed validation and complete observed-resource check.
        let observed = ProviderMixedRealAddObserved::from_steps(receipts, self.steps())
            .filter(|observed| observed.resources == self.resources);
        let Some(observed) = observed else {
            return Err(cleanup::failure(
                ProviderExecutionError::from(anyhow::anyhow!(
                    "mixed real Add observations differ from prepared resources"
                ))
                .after_effect(),
                owned,
            ));
        };
        let output = owned.pop().expect("executed private mixed steps");
        let errors = cleanup::retire(owned);
        if !errors.is_empty() {
            return Err(cleanup::failed_completion(output, errors));
        }
        Ok(ProviderExecutedMixedRealAdd { output, observed })
    }
}
