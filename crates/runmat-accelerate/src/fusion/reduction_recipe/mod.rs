//! Immutable physical reduction recipes, not execution or placement authority.
//! Both strategies share producer indexing and state semantics. A future backend
//! must authenticate operands and implement every declared adapter before use.
use crate::fusion_exec::ResolvedReductionProgram;
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::{
    GpuPhysicalDescriptor, GpuTensorLayout, ProviderPrecision, ProviderResidency,
};
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{FusionReductionContract, FusionReductionOperation};
use runmat_types::NumericClass;
use serde::Serialize;

mod indexing;
mod state;
mod shader;

/// Frozen dimensions/limits supplied by the provider owner, not ambient knobs.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ReductionRecipeConfig {
    pub workgroup_size: u32,
    pub chunk_elements: u32,
    pub max_workgroup_size: u32,
    pub max_workgroup_storage_bytes: u64,
    pub max_dispatch_dimension: u32,
    pub max_storage_binding_bytes: u64,
    pub max_rank: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReductionInputLayout {
    pub value_id: crate::graph::ValueId,
    pub request_slot: Option<usize>,
    pub source_precision: ProviderPrecision,
    pub bound_precision: ProviderPrecision,
    pub source_layout: GpuTensorLayout,
    /// Actual allocation facts, not view-sized storage. The WGPU owner records
    /// buffer.size() as requested_bytes; full-buffer bindings require this
    /// extent even when the logical view or adapted destination is smaller.
    pub source_physical: Option<GpuPhysicalDescriptor>,
    pub residency: ProviderResidency,
    pub shape: Vec<usize>,
    pub broadcast_shape: Vec<u32>,
    pub strides: Vec<u32>,
    pub source_payload_bytes: u64,
    pub bound_payload_bytes: u64,
}
impl ReductionInputLayout {
    pub fn requires_cast(&self) -> bool {
        self.source_precision != self.bound_precision
    }
    pub fn requires_materialization(&self) -> bool {
        self.requires_cast() && self.source_layout != GpuTensorLayout::Contiguous
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ReductionPartialLayout {
    pub precision: ProviderPrecision,
    pub alignment: u32,
    pub stride: u32,
    pub count_offset: u32,
    pub flags_offset: u32,
}
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ReductionDispatch {
    pub workgroups: [u32; 3],
    pub logical_groups: u32,
}
#[derive(Debug, Clone, Serialize)]
pub struct ReductionGeometry {
    pub reduce_elements: u32,
    pub slices: u32,
    pub chunks: u32,
    pub partial_elements: u32,
    pub single: ReductionDispatch,
    pub pass1: ReductionDispatch,
    pub pass2: ReductionDispatch,
}
#[derive(Debug, Clone, Serialize)]
pub struct ReductionRecipeWork {
    /// Producer work is the existing operation-owned prior, not measured time.
    pub producer_scalar_work_units: u64,
    pub output_payload_bytes: u64,
    pub output_requested_bytes: u64,
    pub partial_payload_bytes: u64,
    pub partial_requested_bytes: u64,
    pub workgroup_bytes: u64,
    pub single_dispatches: u32,
    pub two_pass_dispatches: u32,
    pub uniform_bytes: u64,
}

pub struct ReductionRecipe {
    identity: Digest,
    config: ReductionRecipeConfig,
    inputs: Vec<ReductionInputLayout>,
    contract: FusionReductionContract,
    geometry: ReductionGeometry,
    partial: ReductionPartialLayout,
    work: ReductionRecipeWork,
    single: String,
    pass1: String,
    pass2: String,
}
impl ReductionRecipe {
    pub fn identity(&self) -> Digest {
        self.identity
    }
    pub fn config(&self) -> ReductionRecipeConfig {
        self.config
    }
    pub fn inputs(&self) -> &[ReductionInputLayout] {
        &self.inputs
    }
    pub fn contract(&self) -> &FusionReductionContract {
        &self.contract
    }
    pub fn geometry(&self) -> &ReductionGeometry {
        &self.geometry
    }
    pub fn partial_layout(&self) -> ReductionPartialLayout {
        self.partial
    }
    pub fn work(&self) -> &ReductionRecipeWork {
        &self.work
    }
    pub fn single_source(&self) -> &str {
        &self.single
    }
    pub fn pass1_source(&self) -> &str {
        &self.pass1
    }
    pub fn pass2_source(&self) -> &str {
        &self.pass2
    }
}

pub fn build(
    program: &ResolvedReductionProgram<'_>,
    config: ReductionRecipeConfig,
) -> Result<ReductionRecipe> {
    ensure!(
        config.workgroup_size.is_power_of_two()
            && config.workgroup_size <= config.max_workgroup_size,
        "invalid reduction workgroup size"
    );
    ensure!(config.chunk_elements != 0, "zero reduction chunk size");
    let contract = program.contract();
    let precision = precision(if contract.identity {
        contract.input_class
    } else {
        contract.accumulator_class
    })?;
    let output_precision = precision_of_output(contract)?;
    ensure!(
        program.execution_precision() == ProviderPrecision::F64
            || (precision == ProviderPrecision::F32 && output_precision == ProviderPrecision::F32),
        "provider cannot represent reduction arithmetic"
    );
    let inputs = indexing::inputs(program, config)?;
    let geometry = indexing::geometry(contract, config)?;
    let partial = state::layout(precision);
    let workgroup_bytes = if contract.identity {
        0
    } else {
        u64::from(config.workgroup_size)
            .checked_mul(u64::from(partial.stride))
            .ok_or_else(|| anyhow!("reduction workgroup size overflow"))?
    };
    ensure!(
        workgroup_bytes <= config.max_workgroup_storage_bytes,
        "reduction workgroup storage limit"
    );
    let output_payload_bytes = u64::from(geometry.slices) * bytes(output_precision);
    let partial_payload_bytes = u64::from(geometry.partial_elements) * u64::from(partial.stride);
    ensure!(
        output_payload_bytes.max(4) <= config.max_storage_binding_bytes
            && partial_payload_bytes.max(4) <= config.max_storage_binding_bytes,
        "reduction storage binding limit"
    );
    let work = ReductionRecipeWork {
        producer_scalar_work_units: program.producer_work_units(),
        output_payload_bytes,
        output_requested_bytes: output_payload_bytes.max(4),
        partial_payload_bytes,
        partial_requested_bytes: if geometry.partial_elements == 0 {
            0
        } else {
            partial_payload_bytes.max(4)
        },
        workgroup_bytes,
        single_dispatches: u32::from(geometry.slices != 0),
        two_pass_dispatches: if geometry.slices == 0 { 0 } else { 2 },
        uniform_bytes: 0, // All recipe geometry is specialized in source.
    };
    let sources = shader::sources(program, &inputs, &geometry, partial, config)?;
    let identity = Digest::sha256(serde_json::to_vec(&(
        "typed-reduction-physical-v1",
        program.identity(),
        config,
        &inputs,
        contract,
        &geometry,
        partial,
        &work,
        &sources,
    ))?);
    Ok(ReductionRecipe {
        identity,
        config,
        inputs,
        contract: contract.clone(),
        geometry,
        partial,
        work,
        single: sources.0,
        pass1: sources.1,
        pass2: sources.2,
    })
}

fn precision(class: NumericClass) -> Result<ProviderPrecision> {
    match class {
        NumericClass::Single => Ok(ProviderPrecision::F32),
        NumericClass::Double => Ok(ProviderPrecision::F64),
        _ => Err(anyhow!("non-floating reduction class")),
    }
}
fn precision_of_output(contract: &FusionReductionContract) -> Result<ProviderPrecision> {
    precision(contract.output_class)
}
fn bytes(precision: ProviderPrecision) -> u64 {
    match precision {
        ProviderPrecision::F32 => 4,
        ProviderPrecision::F64 => 8,
    }
}
fn scalar(precision: ProviderPrecision) -> &'static str {
    match precision {
        ProviderPrecision::F32 => "f32",
        ProviderPrecision::F64 => "f64",
    }
}
fn is_extrema(operation: FusionReductionOperation) -> bool {
    matches!(
        operation,
        FusionReductionOperation::Minimum | FusionReductionOperation::Maximum
    )
}
