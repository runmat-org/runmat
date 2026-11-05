//! Canonical CFG liveness and immutable, exactly admitted retirement metadata.
mod admission;
mod compute;
pub use admission::NativeLivenessAdmission;

use runmat_execution::resource::ExactShared;
use runmat_runtime::context::host_allocation::HostAllocationError;
use runmat_value::{exact_capacity_vec, ExactHostAllocationClaim};
use std::alloc::Layout;

use super::{NativeAssembly, NativeBlockId, NativeMirSite, NativeValueId};
use crate::{NativeCodegenError, NativeCodegenResult};

#[derive(Clone, Copy, Debug)]
pub struct NativeFunctionLiveness {
    pub function: runmat_types::FunctionId,
    pub first_value: usize,
    pub values: usize,
    pub first_block: usize,
    pub blocks: usize,
    pub first_site: usize,
    pub sites: usize,
    pub words: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeBlockLiveness {
    pub block: NativeBlockId,
    entry: usize,
    exit: usize,
}

#[derive(Clone, Debug)]
pub struct NativeRetirementSite {
    pub site: NativeMirSite,
    first_retired: usize,
    retired: usize,
}

/// Shared by the immutable executor and its active frames. Flat metadata
/// contains only SSA identities and bitsets, never Values or copied roots.
pub struct NativeProgramLiveness {
    functions: Vec<NativeFunctionLiveness>,
    blocks: Vec<NativeBlockLiveness>,
    values: Vec<NativeValueId>,
    bits: Vec<u64>,
    sites: Vec<NativeRetirementSite>,
    retired: Vec<NativeValueId>,
    admission: NativeLivenessAdmission,
    // ExactShared releases its header before dropping this payload. The Vec
    // backings drop before this original claim refunds their combined grant.
    _claim: Option<ExactHostAllocationClaim>,
}

impl NativeProgramLiveness {
    pub fn admission(&self) -> &NativeLivenessAdmission {
        &self.admission
    }

    pub fn function(&self, function: runmat_types::FunctionId) -> Option<&NativeFunctionLiveness> {
        self.functions
            .binary_search_by_key(&function, |row| row.function)
            .ok()
            .map(|index| &self.functions[index])
    }

    pub fn live_on_entry(
        &self,
        function: runmat_types::FunctionId,
        block: NativeBlockId,
        value: NativeValueId,
    ) -> bool {
        let Some(function) = self.function(function) else {
            return false;
        };
        let blocks = &self.blocks[function.first_block..function.first_block + function.blocks];
        let Ok(index) = blocks.binary_search_by_key(&block, |row| row.block) else {
            return false;
        };
        let block = &blocks[index];
        let ids = &self.values[function.first_value..function.first_value + function.values];
        ids.binary_search(&value)
            .ok()
            .is_some_and(|id| self.bits[block.entry + id / 64] & (1u64 << (id % 64)) != 0)
    }

    pub fn retired_after(
        &self,
        function: runmat_types::FunctionId,
        site: &NativeMirSite,
    ) -> &[NativeValueId] {
        let Some(function) = self.function(function) else {
            return &[];
        };
        let sites = &self.sites[function.first_site..function.first_site + function.sites];
        let Ok(index) = sites.binary_search_by(|row| row.site.cmp(site)) else {
            return &[];
        };
        let row = &sites[index];
        &self.retired[row.first_retired..row.first_retired + row.retired]
    }
}

/// A checked upper bound for all retained arrays/header and the one temporary
/// fixed-point workspace. No site-by-value root/payload inventory is retained.
#[derive(Clone, Copy, Debug)]
pub struct NativeLivenessPlan {
    pub retained_bytes: usize,
    pub workspace_bytes: usize,
    functions: usize,
    blocks: usize,
    definitions: usize,
    sites: usize,
    references: usize,
    words: usize,
    workspace_words: usize,
}

fn overflow() -> NativeCodegenError {
    NativeCodegenError::new(
        "native.liveness.geometry",
        "native liveness metadata geometry overflowed",
    )
    .with_allocation(HostAllocationError::Overflow)
}

fn add(left: usize, right: usize) -> NativeCodegenResult<usize> {
    left.checked_add(right).ok_or_else(overflow)
}
fn mul(left: usize, right: usize) -> NativeCodegenResult<usize> {
    left.checked_mul(right).ok_or_else(overflow)
}
fn bytes<T>(count: usize) -> NativeCodegenResult<usize> {
    Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| overflow())
}

fn capacity<T>(count: usize) -> NativeCodegenResult<Vec<T>> {
    exact_capacity_vec(count).map_err(|error| {
        NativeCodegenError::new(
            "native.liveness.allocation",
            "native liveness allocation failed",
        )
        .with_allocation(HostAllocationError::Numeric(error))
    })
}

impl NativeLivenessPlan {
    pub fn for_assembly(assembly: &NativeAssembly) -> NativeCodegenResult<Self> {
        let mut plan = Self {
            retained_bytes: 0,
            workspace_bytes: 0,
            functions: assembly.functions.len(),
            blocks: 0,
            definitions: 0,
            sites: 0,
            references: 0,
            words: 0,
            workspace_words: 0,
        };
        for function in &assembly.functions {
            let mut definitions = function.blocks.len(); // effect-epoch parameters
            let mut references = 0;
            let mut sites = function.blocks.len();
            for block in &function.blocks {
                definitions = add(definitions, block.parameters.len())?;
                sites = add(sites, block.instructions.len())?;
                for instruction in &block.instructions {
                    definitions = add(
                        definitions,
                        add(
                            instruction.outputs.len(),
                            usize::from(instruction.effect_epoch_output.is_some()),
                        )?,
                    )?;
                    compute::instruction_uses(block, instruction, |_: NativeValueId| {
                        references = add(references, 1)?;
                        Ok(())
                    })?;
                }
                compute::terminator_uses(block, |_: NativeValueId| {
                    references = add(references, 1)?;
                    Ok(())
                })?;
            }
            let words = add(definitions, 63)? / 64;
            let block_words = mul(function.blocks.len(), words)?;
            let graph_words = add(function.blocks.len(), 63)? / 64;
            // use/definition sets, reverse scan masks, exceptional-edge matrix,
            // and two reachability worksets reused across all functions.
            let scratch = add(
                add(mul(block_words, 2)?, mul(words, 2)?)?,
                add(
                    mul(function.blocks.len(), graph_words)?,
                    mul(graph_words, 2)?,
                )?,
            )?;
            plan.workspace_words = plan.workspace_words.max(scratch);
            plan.words = add(plan.words, mul(block_words, 2)?)?;
            plan.blocks = add(plan.blocks, function.blocks.len())?;
            plan.definitions = add(plan.definitions, definitions)?;
            plan.references = add(plan.references, add(definitions, references)?)?;
            plan.sites = add(plan.sites, sites)?;
        }
        let mut retained = ExactShared::<NativeProgramLiveness>::requested_bytes();
        for block in [
            bytes::<NativeFunctionLiveness>(plan.functions)?,
            bytes::<NativeBlockLiveness>(plan.blocks)?,
            bytes::<NativeValueId>(plan.definitions)?,
            bytes::<u64>(plan.words)?,
            bytes::<NativeRetirementSite>(plan.sites)?,
            bytes::<NativeValueId>(plan.references)?,
        ] {
            retained = add(retained, block)?;
        }
        plan.retained_bytes = retained;
        plan.workspace_bytes = bytes::<u64>(plan.workspace_words)?;
        Ok(plan)
    }
}

pub fn analyze_liveness(
    assembly: &NativeAssembly,
    admission: NativeLivenessAdmission,
) -> NativeCodegenResult<ExactShared<NativeProgramLiveness>> {
    assembly.verify()?;
    let plan = NativeLivenessPlan::for_assembly(assembly)?;
    // Both grants precede every backing allocation. The original owner itself
    // accounts its canonical claim metadata; no ledger funds its own creation.
    let claim = admission.claim(plan.retained_bytes)?;
    let workspace_claim = admission.claim(plan.workspace_bytes)?;
    let mut workspace = capacity::<u64>(plan.workspace_words)?;
    workspace.resize(plan.workspace_words, 0);
    let mut value = NativeProgramLiveness {
        functions: capacity(plan.functions)?,
        blocks: capacity(plan.blocks)?,
        values: capacity(plan.definitions)?,
        bits: capacity(plan.words)?,
        sites: capacity(plan.sites)?,
        retired: capacity(plan.references)?,
        admission,
        _claim: claim,
    };
    for function in &assembly.functions {
        compute::function(function, &mut value, &mut workspace)?;
    }
    drop(workspace);
    drop(workspace_claim);
    value.admission.check_current()?;
    ExactShared::try_new(value).map_err(|_| {
        NativeCodegenError::new(
            "native.liveness.allocation",
            "native liveness owner allocation failed",
        )
        .with_allocation(HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::AllocationFailed,
        ))
    })
}
