//! Immutable request-local semantic receipt; this module submits no kernels and
//! is not an alternative placement scheduler or a provider capability promise.
use super::{AdmittedBindings, FusionExecutionRequest};
use crate::fusion::{semantics::ResolvedProgram, typed_producer::TypedProducer};
use crate::graph::ValueId;
use futures::FutureExt;
use runmat_accelerate_api::scalar_program::{
    EmittedScalarImplementation, ScalarProgram, ScalarTransport,
};
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult, ProviderPrecision};
use runmat_execution::Digest;
use runmat_runtime::builtins::common::spec::{
    FusionReductionContract, FusionReductionControls, ScalarType,
};
use runmat_types::NumericClass;
use runmat_value::Value;

#[path = "reduction_program/host_control.rs"]
mod host_control;
#[path = "reduction_program/host_work.rs"]
mod host_work;
#[path = "reduction_program/inputs.rs"]
mod inputs;
#[path = "reduction_program/physical.rs"]
mod physical;

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
    word_expression: Option<String>,
    identity: Digest,
    data_classes: Vec<NumericClass>,
    execution_precision: ProviderPrecision,
    producer_transport: ScalarTransport,
    producer_implementations: Vec<EmittedScalarImplementation>,
    embedded_controls: Vec<(ValueId, u64)>,
    owner_precision: ProviderPrecision,
    producer_work_units: u64,
    host_total_work_units: Option<u64>,
    checked_scalar: Option<ScalarProgram>,
    host_bindings: Option<Vec<runmat_builtins::BuiltinBindingIdentity>>,
    host_producer_shapes: Option<Vec<Vec<usize>>>,
    host_producer_operand_shapes: Option<Vec<[Vec<usize>; 2]>>,
    // All returned first-party metadata fields above deallocate before their
    // captured Runtime control funding. No device payload is charged here.
    _host_control: [crate::wgsl::host_buffers::Claim; 4],
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
    pub fn producer_word_expression(&self) -> Option<&str> {
        self.word_expression.as_deref()
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
    pub fn producer_transport(&self) -> ScalarTransport {
        self.producer_transport
    }
    pub fn producer_implementations(&self) -> &[EmittedScalarImplementation] {
        &self.producer_implementations
    }
    pub(super) fn require_current_controls(
        &self,
        request: &FusionExecutionRequest<'_>,
    ) -> anyhow::Result<()> {
        use runmat_runtime::builtins::common::spec::FusionScalarControl;
        for &(id, bits) in &self.embedded_controls {
            let value = request
                .plan
                .inputs
                .iter()
                .position(|&candidate| candidate == id)
                .and_then(|slot| request.inputs.get(slot))
                .or_else(|| request.plan.const_values.get(&id));
            anyhow::ensure!(
                crate::fusion::semantics::exact_scalar_control(value)
                    == Some(FusionScalarControl::DoubleBits(bits))
                    && value.is_some_and(|value| runmat_runtime::value_fact::value_fact(value)
                        .shape
                        .element_count()
                        == Some(1)),
                "selected reduction original scalar control changed"
            );
        }
        Ok(())
    }
    pub fn owner_precision(&self) -> ProviderPrecision {
        self.owner_precision
    }
    pub fn producer_work_units(&self) -> u64 {
        self.producer_work_units
    }
    /// Original-class CPU producer plus terminal work. Missing price does not
    /// revoke an independently admitted ordinary host execution route.
    pub fn host_total_work_units(&self) -> Option<u64> {
        self.host_total_work_units
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
    resolve(request, precision, false).await
}

pub(super) async fn resolve_reduction_program_static<'a>(
    request: &'a FusionExecutionRequest<'_>,
    precision: ProviderPrecision,
) -> ProviderExecutionResult<Option<ResolvedReductionProgram<'a>>> {
    resolve(request, precision, true).await
}

async fn resolve<'a>(
    request: &'a FusionExecutionRequest<'_>,
    precision: ProviderPrecision,
    static_only: bool,
) -> ProviderExecutionResult<Option<ResolvedReductionProgram<'a>>> {
    let host_control =
        runmat_runtime::context::host_allocation::ProviderHostControlAdmission::capture(
            request.runtime.as_ref(),
        )
        .map_err(ProviderExecutionError::from_source)?;
    let metadata = host_control.as_ref().map(|admission| {
        admission as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
    });
    let Some(mapped) = inputs::map(request, metadata)? else {
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
    let Some(resources) = (declaration.resources)(
        runmat_runtime::builtins::common::spec::FusionReductionPreparationQuery {
            controls: FusionReductionControls {
                values: &mapped.controls,
                requested_outputs: 1,
            },
            input_rank_upper: host_control::input_rank_upper(&mapped),
        },
    ) else {
        return Ok(None);
    };
    let prepare_scratch = host_control::claim(metadata, resources.prepare_scratch_bytes)?;
    let prepared_retained = host_control::claim(metadata, resources.prepared_retained_bytes)?;
    let preparation = (declaration.prepare)(FusionReductionControls {
        values: &mapped.controls,
        requested_outputs: 1,
    });
    // Host-only controls were checked by map. Static entrypoints never block
    // or probe a device; a suspending parser simply has no static receipt.
    let preparation = if static_only {
        let Some(result) = preparation.now_or_never() else {
            return Ok(None);
        };
        result
    } else {
        preparation.await
    };
    let Some(prepared) = preparation.map_err(ProviderExecutionError::from_source)? else {
        return Ok(None);
    };
    // The opaque future Box has been consumed/destroyed by await/now_or_never.
    // The separate retained record follows its Prepared Box through all calls.
    drop(prepare_scratch);
    let prepared = host_control::PreparedControl::new(prepared, prepared_retained);
    let values: Vec<_> = mapped.data.iter().map(|input| input.value).collect();
    // Source facts/domain tables and the optional host-static pricing program
    // overlap. Their canonical counted owner is admitted before either runs.
    let _semantic_claim = if metadata.is_some() {
        let geometry =
            crate::fusion::semantics::preparation_resources::PureSourceGeometry::admit_borrowed(
                &mapped.producer,
                &values,
            )
            .map_err(ProviderExecutionError::from_source)?;
        let bytes = geometry
            .overlapping_preparation_bytes(None)
            .map_err(ProviderExecutionError::from_source)?;
        crate::wgsl::host_buffers::claim_bytes(
            metadata,
            usize::try_from(bytes).map_err(ProviderExecutionError::from_source)?,
        )
        .map_err(ProviderExecutionError::from_source)?
    } else {
        None
    };
    let pricing_resolve_bytes = resources
        .resolve_scratch_bytes
        .checked_add(resources.contract_retained_bytes)
        .ok_or_else(|| anyhow::anyhow!("reduction pricing control size overflow"))?;
    let pricing_resolve_claim = host_control::claim(metadata, pricing_resolve_bytes)?;
    let host_total_work_units = host_work::derive(
        &mapped,
        &values,
        precision,
        &snapshots,
        prepared.as_ref(),
        declaration.operation,
    );
    // Host pricing returns only a scalar count; its temporary contract is gone.
    drop(pricing_resolve_claim);
    let resolved = if static_only {
        ResolvedProgram::resolve_static_captured(&mapped.producer, &values, precision, &snapshots)
    } else {
        ResolvedProgram::resolve_captured(
            &mapped.producer,
            &values,
            precision,
            request.placement,
            &snapshots,
        )
        .await
        .map(Some)
    };
    let program = match resolved {
        Ok(Some(program)) => program,
        Ok(None) => return Ok(None),
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
    let resolve_scratch = host_control::claim(metadata, resources.resolve_scratch_bytes)?;
    let contract_retained = host_control::claim(metadata, resources.contract_retained_bytes)?;
    let Some(contract) = prepared
        .resolve(fact)
        .map_err(ProviderExecutionError::from_source)?
    else {
        return Ok(None);
    };
    drop(resolve_scratch);
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
    let projection = physical::Projection::derive(&mapped, &program, precision)?;
    let controls = &projection.controls;
    let producer_precision = projection.precision;
    let word_storage = projection.word_storage;
    let scalar = match producer_precision {
        ProviderPrecision::F32 => ScalarType::F32,
        ProviderPrecision::F64 => ScalarType::F64,
    };
    let producer_transport = if !word_storage {
        ScalarTransport::NativeFloat
    } else {
        ScalarTransport::Binary32Words
    };
    let producer_bound = if metadata.is_some() {
        Some(mapped.producer.measured_typed_producer_preparation_bytes(
            mapped.root,
            &program,
            scalar,
            mapped.data.len(),
        )?)
    } else {
        None
    };
    let producer_claim = host_control::claim(
        metadata,
        producer_bound.map_or(0, |bound| bound.requested_bytes),
    )?;
    let (expressions, words) = projection.expressions(&mapped)?;
    let producer = match TypedProducer::emit_indexed_with_words_bounded(
        &mapped.producer,
        &program,
        scalar,
        expressions,
        Some(&[mapped.root]),
        words,
        producer_bound.map(|bound| bound.body_bytes),
    ) {
        Ok(producer) => producer,
        Err(error) if metadata.is_some() => return Err(error.into()),
        Err(_) => return Ok(None),
    };
    // Only the checked stream actually used for WGSL emission can authorize
    // portable provider execution. A legacy-only lowering remains unbound.
    let checked_scalar = producer.checked_program().cloned();
    let widening = producer_transport == ScalarTransport::Binary32Words
        && !contract.identity
        && contract.accumulator_class == NumericClass::Double;
    let implementation_capacity = producer
        .implementations()
        .len()
        .checked_add(usize::from(widening))
        .ok_or_else(|| anyhow::anyhow!("reduction implementation capacity overflow"))?;
    let mut producer_implementations = runmat_value::exact_capacity_vec(implementation_capacity)
        .map_err(ProviderExecutionError::from_source)?;
    producer_implementations.extend_from_slice(producer.implementations());
    if widening {
        producer_implementations.push(crate::wgsl::ieee_words::binary32_widening_implementation(
            mapped.root,
        ));
    }
    let mut embedded_controls = runmat_value::exact_capacity_vec(mapped.data.len())
        .map_err(ProviderExecutionError::from_source)?;
    for input in mapped
        .data
        .iter()
        .filter(|input| controls.contains(&input.value_id))
    {
        let bits = program
            .embedded_scalar_control_bits(input.value_id)
            .ok_or_else(|| {
                anyhow::anyhow!("omitted reduction scalar control lost original words")
            })?;
        embedded_controls.push((input.value_id, bits));
    }
    let expression = producer
        .expression(mapped.root)
        .map_err(ProviderExecutionError::from)?
        .to_owned();
    let preamble = producer
        .preamble(&program)
        .map_err(ProviderExecutionError::from)?;
    let word_expression = producer.word_expression(mapped.root).map(str::to_owned);
    let mapping = host_control::DataMapping {
        data: &mapped.data,
        controls,
    };
    let semantic = Digest::sha256_serialized(&(
        "typed-reduction-program-v3",
        program.cache_identity()?,
        declaration.revision,
        &contract,
        mapping,
        &producer.body,
        &preamble,
        &expression,
        &word_expression,
        checked_scalar.as_ref().map(ScalarProgram::identity),
        producer_transport,
        &producer_implementations,
        host_total_work_units,
    ))
    .map_err(ProviderExecutionError::from_source)?;
    let identity = bindings.identity(Some(semantic))?;
    let mut data_classes = runmat_value::exact_capacity_vec(mapped.data.len())
        .map_err(ProviderExecutionError::from_source)?;
    for input in mapped
        .data
        .iter()
        .filter(|input| !controls.contains(&input.value_id))
    {
        data_classes.push(program.floating_class(input.value_id)?);
    }
    let elements =
        runmat_runtime::builtins::common::spec::checked_reduction_elements(&contract.input_shape)
            .and_then(|elements| u64::try_from(elements).ok())
            .ok_or_else(|| anyhow::anyhow!("reduction producer cardinality overflow"))?;
    let producer_work_units = program.physical_scalar_work_units(
        &mapped.producer,
        elements,
        &producer_implementations,
    )?;
    // These are the actual invocation's checked producer output facts, in
    // executable operation order. Unknown extents do not qualify a resident
    // host route. The source program's cache identity above already includes
    // these facts; this projection is not a new layout authority.
    let host_producer_shapes = host_control::producer_shapes(&mapped.producer, &program)?;
    // The ordinary Plus owner consumes two operand tensors for each producer.
    // Retain exact source-derived shapes to account the possible COW copies
    // while VM/private values and invocation arguments still own the payload.
    let host_producer_operand_shapes =
        host_control::producer_operand_shapes(&mapped.producer, &program)?;
    // Keep the original mapped allocation and its existing mapping claim;
    // removing metadata controls requires no second data carrier/backing.
    let mut data = mapped.data;
    data.retain(|input| !controls.contains(&input.value_id));
    Ok(Some(ResolvedReductionProgram {
        data,
        contract,
        body: producer.body,
        preamble,
        expression,
        word_expression,
        identity,
        data_classes,
        execution_precision: producer_precision,
        producer_transport,
        producer_implementations,
        embedded_controls,
        owner_precision: precision,
        producer_work_units,
        host_total_work_units,
        checked_scalar,
        host_bindings,
        host_producer_shapes,
        host_producer_operand_shapes,
        _host_control: [
            mapped._mapping_claim,
            None,
            producer_claim,
            contract_retained,
        ],
    }))
}
