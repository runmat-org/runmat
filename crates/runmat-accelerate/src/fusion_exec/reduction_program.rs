//! Immutable request-local semantic receipt; this module submits no kernels and
//! is not an alternative placement scheduler or a provider capability promise.
use super::{AdmittedBindings, FusionExecutionRequest};
use crate::fusion::{
    semantics::{FusionAdmissionFailure, ResolvedProgram},
    typed_producer::TypedProducer,
};
use crate::graph::ValueId;
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
            return match error.downcast::<FusionAdmissionFailure>() {
                Ok(failure) => Err(failure.0.into()),
                Err(_) => Ok(None),
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
    }))
}
