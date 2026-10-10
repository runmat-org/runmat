//! Immutable request-local semantic receipt; this module submits no kernels and
//! is not an alternative placement scheduler or a provider capability promise.
use super::{AdmittedBindings, FusionExecutionRequest};
use crate::fusion::{semantics::ResolvedProgram, typed_producer::TypedProducer};
use crate::graph::ValueId;
use runmat_accelerate_api::scalar_program::ScalarProgram;
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult, ProviderPrecision};
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{
    FusionReductionContract, FusionReductionControls, ScalarType,
};
use runmat_types::NumericClass;
use runmat_value::Value;
use std::collections::HashMap;

#[path = "reduction_program/inputs.rs"]
mod inputs;

/// A physical data operand retains its exact original carrier and view. Layout
/// strides must later be obtained from the provider's existing operand-layout
/// authority, not manufactured from this mapping.
pub struct ReductionDataInput<'a> {
    pub value_id: ValueId,
    pub request_slot: Option<usize>,
    pub value: &'a Value,
}

pub struct ResolvedReductionProgram<'a> {
    data: Vec<ReductionDataInput<'a>>,
    contract: FusionReductionContract,
    body: String,
    preamble: String,
    expression: String,
    identity: Digest,
    data_classes: Vec<NumericClass>,
    execution_precision: ProviderPrecision,
    producer_work_units: u64,
    checked_scalar: Option<ScalarProgram>,
    host_bindings: Option<Vec<runmat_builtins::BuiltinBindingIdentity>>,
    host_producer_shapes: Option<Vec<Vec<usize>>>,
    host_producer_operand_shapes: Option<Vec<[Vec<usize>; 2]>>,
}
impl<'a> ResolvedReductionProgram<'a> {
    pub fn data_inputs(&self) -> &[ReductionDataInput<'a>] {
        &self.data
    }
    pub fn contract(&self) -> &FusionReductionContract {
        &self.contract
    }
    pub fn producer_body(&self) -> &str {
        &self.body
    }
    pub fn producer_preamble(&self) -> &str {
        &self.preamble
    }
    pub fn producer_expression(&self) -> &str {
        &self.expression
    }
    pub fn identity(&self) -> Digest {
        self.identity
    }
    /// Effective leaf classes after the existing precision policy. Physical
    /// source carriers remain unchanged; a recipe must declare any adapter.
    pub fn data_classes(&self) -> &[NumericClass] {
        &self.data_classes
    }
    pub fn execution_precision(&self) -> ProviderPrecision {
        self.execution_precision
    }
    pub fn producer_work_units(&self) -> u64 {
        self.producer_work_units
    }
    pub fn checked_scalar(&self) -> Option<&ScalarProgram> {
        self.checked_scalar.as_ref()
    }
    pub fn host_bindings(&self) -> Option<&[runmat_builtins::BuiltinBindingIdentity]> {
        self.host_bindings.as_deref()
    }
    pub fn host_producer_shapes(&self) -> Option<&[Vec<usize>]> {
        self.host_producer_shapes.as_deref()
    }
    pub fn host_producer_operand_shapes(&self) -> Option<&[[Vec<usize>; 2]]> {
        self.host_producer_operand_shapes.as_deref()
    }
}

/// Resolve one actual invocation, capturing selected binding authority before
/// any operation-owned domain admission. None is a missing semantic proof;
/// actual owner parser/probe errors retain their original source.
pub async fn resolve_reduction_program<'a>(
    request: &'a FusionExecutionRequest<'_>,
    precision: ProviderPrecision,
) -> ProviderExecutionResult<Option<ResolvedReductionProgram<'a>>> {
    let Some(mapped) = inputs::map(request) else {
        return Ok(None);
    };
    let Some(bindings) = AdmittedBindings::new(request) else {
        return Ok(None);
    };
    // Primitive VM instructions also need their actual runtime-selected
    // canonical implementation. Capture before any owner probe suspends.
    let host_bindings = bindings.canonical_bindings(request.runtime.as_ref());
    let Some(declaration) = bindings.reduction(mapped.terminal) else {
        return Ok(None);
    };
    // Capture the original operation's selected authority, not a fresh lookup
    // in the pruned program after an earlier admission has suspended.
    let snapshots: Vec<_> = mapped
        .producer_indices
        .iter()
        .map(|&index| bindings.semantic(index))
        .collect();
    for (operation, snapshot) in mapped.producer.operations.iter().zip(&snapshots) {
        if matches!(operation, crate::fusion::FusionOp::Builtin { .. }) && snapshot.is_none() {
            return Ok(None);
        }
        if let Some(snapshot) = snapshot {
            let entry = snapshot.entry;
            if entry.contract.purity != runmat_builtins::BuiltinPurity::Pure
                || entry.contract.workspace_effect.is_some()
                || entry.contract.environment_effect.is_some()
            {
                return Ok(None);
            }
        }
    }
    let Some(prepared) = (declaration.prepare)(FusionReductionControls {
        values: &mapped.controls,
        requested_outputs: 1,
    })
    .await
    .map_err(ProviderExecutionError::from_source)?
    else {
        return Ok(None);
    };
    let values: Vec<_> = mapped.data.iter().map(|input| input.value).collect();
    let program = match ResolvedProgram::resolve_captured(
        &mapped.producer,
        &values,
        precision,
        request.placement,
        &snapshots,
    )
    .await
    {
        Ok(program) => program,
        Err(error) => {
            return match error.into_execution_or_structural() {
                Ok(_) => Ok(None),
                Err(failure) => Err(failure),
            }
        }
    };
    let fact = match program.fact(mapped.root) {
        Ok(fact) => fact,
        Err(_) => return Ok(None),
    };
    let Some(contract) = prepared
        .resolve(fact)
        .map_err(ProviderExecutionError::from_source)?
    else {
        return Ok(None);
    };
    if contract.operation != declaration.operation {
        return Err(anyhow::anyhow!(
            "selected reduction declaration changed its operation during resolution"
        )
        .into());
    }
    // Source-derived Double arithmetic is not implicitly licensed by the
    // existing policy for downcasting Double INPUTS. Preserve that gap honestly.
    if precision == ProviderPrecision::F32
        && (contract.output_class == NumericClass::Double
            || (!contract.identity && contract.accumulator_class == NumericClass::Double))
    {
        return Ok(None);
    }
    let scalar = match precision {
        ProviderPrecision::F32 => ScalarType::F32,
        ProviderPrecision::F64 => ScalarType::F64,
    };
    let expressions: HashMap<_, _> = mapped
        .data
        .iter()
        .enumerate()
        .map(|(slot, input)| (input.value_id, format!("input{slot}.data[i{slot}]")))
        .collect();
    let producer = match TypedProducer::emit(
        &mapped.producer,
        &program,
        scalar,
        expressions,
        Some(&[mapped.root]),
    ) {
        Ok(producer) => producer,
        Err(_) => return Ok(None),
    };
    // Only the checked stream actually used for WGSL emission can authorize
    // portable provider execution. A legacy-only lowering remains unbound.
    let checked_scalar = producer.checked_program().cloned();
    let expression = producer
        .expression(mapped.root)
        .map_err(ProviderExecutionError::from)?
        .to_owned();
    let preamble = producer
        .preamble(&program)
        .map_err(ProviderExecutionError::from)?;
    let mapping: Vec<_> = mapped
        .data
        .iter()
        .map(|input| (input.value_id, input.request_slot))
        .collect();
    let semantic = Digest::sha256(
        serde_json::to_vec(&(
            "typed-reduction-program-v1",
            program.cache_identity()?,
            declaration.revision,
            &contract,
            mapping,
            &producer.body,
            &preamble,
            &expression,
            checked_scalar.as_ref().map(ScalarProgram::identity),
        ))
        .map_err(ProviderExecutionError::from_source)?,
    );
    let identity = bindings.identity(Some(semantic))?;
    let data_classes = mapped
        .data
        .iter()
        .map(|input| program.floating_class(input.value_id))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let elements =
        runmat_runtime::builtins::common::spec::checked_reduction_elements(&contract.input_shape)
            .and_then(|elements| u64::try_from(elements).ok())
            .ok_or_else(|| anyhow::anyhow!("reduction producer cardinality overflow"))?;
    let producer_work_units = program.scalar_work_units(&mapped.producer, elements)?;
    // These are the actual invocation's checked producer output facts, in
    // executable operation order. Unknown extents do not qualify a resident
    // host route. The source program's cache identity above already includes
    // these facts; this projection is not a new layout authority.
    let host_producer_shapes = mapped
        .producer
        .operations
        .iter()
        .map(|operation| {
            let output = match operation {
                crate::fusion::FusionOp::Primitive { output, .. }
                | crate::fusion::FusionOp::Builtin { output, .. } => (*output)?,
            };
            program
                .fact(output)
                .ok()?
                .shape
                .known_dims()?
                .into_iter()
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>();
    // The ordinary Plus owner consumes two operand tensors for each producer.
    // Retain exact source-derived shapes to account the possible COW copies
    // while VM/private values and invocation arguments still own the payload.
    let host_producer_operand_shapes = mapped
        .producer
        .operations
        .iter()
        .map(|operation| {
            let inputs = match operation {
                crate::fusion::FusionOp::Primitive { inputs, .. }
                | crate::fusion::FusionOp::Builtin { inputs, .. } => inputs,
            };
            let [left, right] = inputs.as_slice() else {
                return None;
            };
            let shape = |id| {
                program
                    .fact(id)
                    .ok()?
                    .shape
                    .known_dims()?
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
            };
            Some([shape(*left)?, shape(*right)?])
        })
        .collect::<Option<Vec<_>>>();
    Ok(Some(ResolvedReductionProgram {
        data: mapped.data,
        contract,
        body: producer.body,
        preamble,
        expression,
        identity,
        data_classes,
        execution_precision: precision,
        producer_work_units,
        checked_scalar,
        host_bindings,
        host_producer_shapes,
        host_producer_operand_shapes,
    }))
}
