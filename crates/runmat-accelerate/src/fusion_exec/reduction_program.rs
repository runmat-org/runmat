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
    FusionReductionContext, FusionReductionContract, ScalarType,
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
    // Preflight pure canonical producer declarations before resolve can probe.
    for operation in &mapped.producer.operations {
        if let crate::fusion::FusionOp::Builtin { name, .. } = operation {
            let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(name) else {
                return Ok(None);
            };
            if entry.contract.purity != runmat_builtins::BuiltinPurity::Pure
                || entry.contract.workspace_effect.is_some()
                || entry.contract.environment_effect.is_some()
            {
                return Ok(None);
            }
        }
    }
    let values: Vec<_> = mapped.data.iter().map(|input| input.value).collect();
    let program = match ResolvedProgram::resolve_borrowed(
        &mapped.producer,
        &values,
        precision,
        request.placement,
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
    let Some(contract) = (declaration.resolve)(FusionReductionContext {
        input: fact,
        controls: &mapped.controls,
        requested_outputs: 1,
    })
    .await
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
    Ok(Some(ResolvedReductionProgram {
        data: mapped.data,
        contract,
        body: producer.body,
        preamble,
        expression,
        identity,
    }))
}
