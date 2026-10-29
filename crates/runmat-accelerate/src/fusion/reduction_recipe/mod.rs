//! Immutable physical reduction recipes, not execution or placement authority.
//! Both strategies share producer indexing and state semantics. A future backend
//! must authenticate operands and implement every declared adapter before use.
use crate::fusion_exec::ResolvedReductionProgram;
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::reduction::{
    ProviderReductionRecipe, ReductionInputBinding, ReductionRecipeDescriptor,
    ReductionShaderSources,
};
pub use runmat_accelerate_api::reduction::{
    ReductionDispatch, ReductionGeometry, ReductionPartialLayout, ReductionRecipeConfig,
    ReductionRecipeWork,
};
use runmat_accelerate_api::{
    GpuPhysicalDescriptor, GpuTensorLayout, ProviderPrecision, ProviderResidency,
};
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{FusionReductionContract, FusionReductionOperation};
use runmat_types::NumericClass;
use serde::Serialize;

mod indexing;
mod resources;
mod shader;
mod state;
mod writer;

/// Check that all shader stages lower from the carried scalar program and
/// reduction contract. This does not authenticate the scalar program against
/// the admitted source plan; submission must do that before provider effects.
pub fn validate_lowered_sources(
    recipe: &ProviderReductionRecipe,
    metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> Result<()> {
    shader::validate_lowered_sources(recipe, metadata)
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

pub struct ReductionRecipe {
    inputs: Vec<ReductionInputLayout>,
    contract: FusionReductionContract,
    physical: ProviderReductionRecipe,
    // Physical sources and metadata deallocate before their backing claims.
    _source_claims: [crate::wgsl::host_buffers::Claim; 3],
    _metadata_claim: crate::wgsl::host_buffers::Claim,
}
impl ReductionRecipe {
    pub fn identity(&self) -> Digest {
        self.physical.identity()
    }
    pub fn config(&self) -> ReductionRecipeConfig {
        self.physical.descriptor().config
    }
    pub fn inputs(&self) -> &[ReductionInputLayout] {
        &self.inputs
    }
    pub fn contract(&self) -> &FusionReductionContract {
        &self.contract
    }
    pub fn geometry(&self) -> &ReductionGeometry {
        &self.physical.descriptor().geometry
    }
    pub fn partial_layout(&self) -> ReductionPartialLayout {
        self.physical.descriptor().partial
    }
    pub fn work(&self) -> &ReductionRecipeWork {
        &self.physical.descriptor().work
    }
    pub fn single_source(&self) -> &str {
        &self.physical.sources().single
    }
    pub fn pass1_source(&self) -> &str {
        &self.physical.sources().pass1
    }
    pub fn pass2_source(&self) -> &str {
        &self.physical.sources().pass2
    }
    pub fn physical(&self) -> &ProviderReductionRecipe {
        &self.physical
    }
}

pub fn build(
    program: &ResolvedReductionProgram<'_>,
    config: ReductionRecipeConfig,
    metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
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
        program.owner_precision() == ProviderPrecision::F64
            || (precision == ProviderPrecision::F32 && output_precision == ProviderPrecision::F32),
        "provider cannot represent reduction arithmetic"
    );
    let funding = resources::prepare(program, metadata)?;
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
        output_payload_bytes.max(4) <= config.max_storage_binding_bytes,
        "reduction output storage binding limit"
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
    let descriptor = ReductionRecipeDescriptor {
        contract: contract.clone(),
        execution_precision: program.execution_precision(),
        producer_transport: program.producer_transport(),
        producer_implementations: program.producer_implementations().to_vec(),
        config,
        inputs: inputs
            .iter()
            .map(|input| ReductionInputBinding {
                precision: input.bound_precision,
                shape: input.shape.clone(),
                layout: if input.requires_cast() {
                    GpuTensorLayout::Contiguous
                } else {
                    input.source_layout
                },
                broadcast_shape: input.broadcast_shape.clone(),
                strides: input.strides.clone(),
                payload_bytes: input.bound_payload_bytes,
            })
            .collect(),
        iteration_shape: contract.input_shape.clone(),
        output_shape: contract.output_shape.clone(),
        output_precision,
        geometry,
        partial,
        work,
    };
    // Include the original-to-prepared mapping as well as captured semantic
    // identity. Physical source ownership stays upstream of the provider ABI.
    let semantic_identity = Digest::sha256_serialized(&(program.identity(), &inputs, contract))?;
    let sources = shader::sources(program, &descriptor, semantic_identity, metadata)?;
    let (single, single_claim) = sources.0.into_parts();
    let (pass1, pass1_claim) = sources.1.into_parts();
    let (pass2, pass2_claim) = sources.2.into_parts();
    let sources = ReductionShaderSources {
        single,
        pass1,
        pass2,
    };
    let physical = if let Some(scalar) = program.checked_scalar() {
        ProviderReductionRecipe::new_checked(
            semantic_identity,
            descriptor,
            sources,
            scalar.clone(),
            program
                .data_inputs()
                .iter()
                .map(|input| input.value_id)
                .collect(),
        )?
    } else {
        ProviderReductionRecipe::new(semantic_identity, descriptor, sources)?
    };
    if physical.checked_scalar().is_some() {
        validate_lowered_sources(&physical, metadata)?;
    }
    Ok(ReductionRecipe {
        inputs,
        contract: contract.clone(),
        physical,
        _source_claims: [single_claim, pass1_claim, pass2_claim],
        _metadata_claim: funding.retained,
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
