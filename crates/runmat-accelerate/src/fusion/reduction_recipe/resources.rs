//! Physical first-party metadata blocks retained by the immutable recipe.
//! Shader text owns separate exact claims; backend/compiler internals are not
//! included in this allocation recipe.
use super::*;
use crate::wgsl::host_buffers::{claim_bytes, Claim};
use runmat_accelerate_api::scalar_program::{EmittedScalarImplementation, ScalarNode};
use runmat_execution::resource::ProviderHostMetadataAdmission;

#[derive(Default)]
struct Bytes(usize);
impl Bytes {
    fn vector<T>(&mut self, count: usize) -> Result<()> {
        let bytes = std::alloc::Layout::array::<T>(count)?.size();
        self.0 = self
            .0
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("recipe metadata extent overflow"))?;
        Ok(())
    }
    fn contract(&mut self, contract: &FusionReductionContract) -> Result<()> {
        self.vector::<usize>(contract.input_shape.len())?;
        self.vector::<usize>(contract.output_shape.len())?;
        self.vector::<usize>(contract.axes.len())
    }
}

pub(super) struct Funding {
    pub(super) retained: Claim,
    pub(super) _scratch: Claim,
}

pub(super) fn prepare(
    program: &ResolvedReductionProgram<'_>,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> Result<Funding> {
    let contract = program.contract();
    let mut retained = Bytes::default();
    retained.vector::<ReductionInputLayout>(program.data_inputs().len())?;
    retained.vector::<ReductionInputBinding>(program.data_inputs().len())?;
    let mut scratch = 0usize;
    for input in program.data_inputs() {
        let (rank, fact_rank, origin) = match input.value {
            runmat_value::Value::Num(_) => (2, 0, 0),
            runmat_value::Value::Tensor(tensor) => (tensor.shape().len(), tensor.shape().len(), 0),
            runmat_value::Value::GpuTensor(handle) => {
                (handle.shape().len(), handle.shape().len(), 17)
            }
            _ => return Err(anyhow!("unsupported recipe data operand")),
        };
        // Original layout and the descriptor's independently owned clone.
        for _ in 0..2 {
            retained.vector::<usize>(rank)?;
            retained.vector::<u32>(contract.input_shape.len())?;
            retained.vector::<u32>(contract.input_shape.len())?;
        }
        let fact = std::alloc::Layout::array::<runmat_types::DimensionFact>(fact_rank)?
            .size()
            .checked_add(origin)
            .ok_or_else(|| anyhow!("recipe fact extent overflow"))?;
        // Inputs are constructed serially; only one transient ValueFact lives.
        scratch = scratch.max(fact);
    }
    // ReductionRecipe and physical descriptor each own their contract clone.
    retained.contract(contract)?;
    retained.contract(contract)?;
    retained.vector::<usize>(contract.input_shape.len())?;
    retained.vector::<usize>(contract.output_shape.len())?;
    retained.vector::<EmittedScalarImplementation>(program.producer_implementations().len())?;
    if let Some(scalar) = program.checked_scalar() {
        retained.vector::<ScalarNode>(scalar.wire().nodes.len())?;
        for node in &scalar.wire().nodes {
            if let ScalarNode::Apply { operands, .. } = node {
                retained.vector::<u32>(operands.len())?;
            }
        }
        retained.vector::<u32>(program.data_inputs().len())?;
    }
    // Both blocks are admitted before any metadata construction. The retained
    // claim moves after ALL recipe fields; scratch retires after construction.
    Ok(Funding {
        retained: claim_bytes(metadata, retained.0)?,
        _scratch: claim_bytes(metadata, scratch)?,
    })
}
