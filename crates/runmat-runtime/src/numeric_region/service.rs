//! Owned native numeric-region requests. Issuance is a trusted Rust executor
//! integration boundary, not a serialized plan or a VM completion grant.
use super::{workload, NumericRegionProgram};
use crate::context::{RuntimeContext, RuntimeContextLocalId};
use crate::execution::RootedValues;
use crate::native::NativeSiteRequest;
use crate::{runtime_error::semantic_error, RuntimeError};
pub use runmat_accelerate_api::observation::SourceRegionRouteAudit;
use runmat_execution::{cancellation::CancellationToken, Digest};
use runmat_types::RegionId;
use runmat_value::{OutputListStorage, Value};

mod candidates;
#[cfg(test)]
mod tests;
pub use candidates::{NativeNumericRegionCpuCandidates, NativeNumericRegionCpuKind};

/// Closed implementation capabilities, not permission to reinterpret classes.
/// The CPU evaluator remains the narrower DenseF64 subset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericRegionCapability {
    DenseF64,
    /// Real Single/Double host and resident operands. Provider type/broadcast
    /// admission remains at the actual typed fusion producer.
    RealFloating,
}

/// Linear publication authority retained by the native site's continuation.
/// The issuing executor has verified the program and the exact region/site;
/// Runtime additionally binds the request to its owning execution generation.
pub struct NativeNumericRegionRight {
    owner: RuntimeContextLocalId,
    revision: Option<Digest>,
    cancellation: CancellationToken,
    region: RegionId,
    site: NativeSiteRequest,
    started: runmat_time::Instant,
}

/// Borrow-free selection facet of one issued native right. It cannot publish
/// outputs or complete an observation; those consume the original right.
#[derive(Clone)]
pub struct NativeNumericRegionSelectionAuthority {
    owner: RuntimeContextLocalId,
    revision: Option<Digest>,
    cancellation: CancellationToken,
    region: RegionId,
    site: NativeSiteRequest,
}
impl NativeNumericRegionSelectionAuthority {
    pub fn is_current(&self, runtime: &RuntimeContext) -> bool {
        self.owner == runtime.local_identity()
            && runtime
                .program_revision()
                .map(|revision| revision.identity_digest())
                .transpose()
                .is_ok_and(|revision| revision == self.revision)
            && self.cancellation.same_generation(&runtime.cancellation())
            && !runtime.is_cancelled()
    }
    pub fn identity(&self) -> (RegionId, NativeSiteRequest) {
        (self.region, self.site)
    }
}

impl NativeNumericRegionRight {
    pub fn selection_authority(&self) -> NativeNumericRegionSelectionAuthority {
        NativeNumericRegionSelectionAuthority {
            owner: self.owner,
            revision: self.revision,
            cancellation: self.cancellation.clone(),
            region: self.region,
            site: self.site,
        }
    }
    pub fn is_current(&self, runtime: &RuntimeContext) -> bool {
        self.owner == runtime.local_identity()
            && runtime
                .program_revision()
                .map(|revision| revision.identity_digest())
                .transpose()
                .is_ok_and(|revision| self.revision == revision)
            && self.cancellation.same_generation(&runtime.cancellation())
            && !runtime.is_cancelled()
    }

    pub fn matches(&self, region: RegionId, site: NativeSiteRequest) -> bool {
        self.region == region && self.site == site
    }

    pub fn identity(&self) -> (RegionId, NativeSiteRequest) {
        (self.region, self.site)
    }

    pub fn started(&self) -> runmat_time::Instant {
        self.started
    }
}

/// Inputs retain the original numeric payload/shape allocations and claims.
/// No arena reference or borrowed Value crosses the asynchronous boundary.
pub struct NativeNumericRegionRequest {
    program: NativeNumericRegionProgramOwner,
    inputs: RootedValues,
    right: NativeNumericRegionRight,
}

/// Program/candidate backing retires before the receipt that admitted it,
/// including cancellation and error unwinding across a provider future.
pub struct NativeNumericRegionProgramOwner {
    program: NumericRegionProgram,
    cpu: Option<NativeNumericRegionCpuCandidates>,
    input_count: usize,
    operation_sites: Option<[Option<NativeSiteRequest>; super::MAX_NODES]>,
    metadata: Option<crate::context::host_allocation::HostAllocationReceipt>,
}
impl NativeNumericRegionProgramOwner {
    pub fn input_count(&self) -> usize {
        self.input_count
    }
    pub fn program(&self) -> &NumericRegionProgram {
        &self.program
    }
    pub fn cpu_candidates(&self) -> Option<&NativeNumericRegionCpuCandidates> {
        self.cpu.as_ref()
    }
    pub fn has_native_operation_sites(&self) -> bool {
        self.operation_sites.is_some()
    }
    pub fn native_operation_count(&self) -> usize {
        self.program
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    node,
                    super::NumericRegionNode::Unary { .. }
                        | super::NumericRegionNode::Binary { .. }
                )
            })
            .count()
    }
    pub fn native_host_operations<'a>(
        &'a self,
        canonical: &'a [crate::context::VmFusionHostOperation],
    ) -> Option<impl Iterator<Item = crate::context::NativeRegionHostOperation> + 'a> {
        let sites = self.operation_sites.as_ref()?;
        if canonical.len() != self.native_operation_count() {
            return None;
        }
        let substitution = move |node: usize| match self.program.nodes.get(node) {
            Some(super::NumericRegionNode::Input(input)) => Some(*input),
            _ => None,
        };
        Some(
            self.program
                .nodes
                .iter()
                .enumerate()
                .filter_map(move |(node, operation)| {
                    let (operator, input_substitutions) = match operation {
                        super::NumericRegionNode::Unary { operation, input } => (
                            match operation {
                                super::NumericUnaryOperation::Plus => {
                                    runmat_types::OperatorKind::UnaryPlus
                                }
                                super::NumericUnaryOperation::Minus => {
                                    runmat_types::OperatorKind::UnaryMinus
                                }
                            },
                            [substitution(*input), None],
                        ),
                        super::NumericRegionNode::Binary {
                            operation,
                            left,
                            right,
                        } => (
                            match operation {
                                super::NumericBinaryOperation::Add => {
                                    runmat_types::OperatorKind::Add
                                }
                                super::NumericBinaryOperation::Subtract => {
                                    runmat_types::OperatorKind::Subtract
                                }
                                super::NumericBinaryOperation::Multiply => {
                                    runmat_types::OperatorKind::ElementwiseMultiply
                                }
                                super::NumericBinaryOperation::Divide => {
                                    runmat_types::OperatorKind::ElementwiseDivide
                                }
                                super::NumericBinaryOperation::LeftDivide => {
                                    runmat_types::OperatorKind::ElementwiseLeftDivide
                                }
                            },
                            [substitution(*left), substitution(*right)],
                        ),
                        _ => return None,
                    };
                    Some((
                        sites[node].expect("validated native arithmetic site"),
                        operator,
                        input_substitutions,
                    ))
                })
                .zip(canonical.iter().copied())
                .map(|((site, operator, input_substitutions), operation)| {
                    crate::context::NativeRegionHostOperation {
                        site,
                        operator,
                        operation,
                        input_substitutions,
                    }
                }),
        )
    }
    pub fn checkpoint(&self, runtime: &RuntimeContext) -> Result<(), RuntimeError> {
        self.metadata.as_ref().map_or(Ok(()), |receipt| {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation_error(runtime, source))
        })
    }
}

impl NativeNumericRegionRequest {
    /// Called by the verified native region owner, after resolving all input
    /// generations and before installing the canonical hostcall suspension.
    /// Ineligibility is established before preparation or provider effects.
    pub fn capture_verified(
        runtime: &RuntimeContext,
        region: RegionId,
        site: NativeSiteRequest,
        program: &NumericRegionProgram,
        inputs: &[&Value],
    ) -> Result<Option<Self>, RuntimeError> {
        let started = runmat_time::Instant::now();
        if runtime.is_cancelled() {
            return Ok(None);
        }
        if !inputs.iter().all(|input| real_floating_input(input)) {
            return Ok(None);
        }
        let cpu_workload = workload(inputs);
        program
            .validate(inputs.len())
            .map_err(|message| error(message))?;
        use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
        let bytes = program
            .nodes
            .len()
            .checked_mul(std::mem::size_of::<super::NumericRegionNode>())
            .and_then(|bytes| {
                program
                    .outputs
                    .len()
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|outputs| bytes.checked_add(outputs))
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .and_then(|bytes| {
                bytes.checked_add(NativeNumericRegionCpuCandidates::identity_storage_bytes(
                    region,
                ))
            })
            .ok_or_else(|| error("numeric program metadata size overflow"))?;
        let metadata = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(bytes, 0)
                .map_err(|source| allocation_error(runtime, source))?,
        )
        .map_err(|source| allocation_error(runtime, source))?;
        let program = program.clone();
        let arithmetic_operations = program
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    node,
                    super::NumericRegionNode::Unary { .. }
                        | super::NumericRegionNode::Binary { .. }
                )
            })
            .count();
        let cpu = cpu_workload
            .map(|workload| {
                NativeNumericRegionCpuCandidates::for_verified_program(
                    region,
                    u32::try_from(arithmetic_operations)
                        .map_err(|_| error("numeric operation count overflow"))?,
                    workload,
                    program.outputs.len(),
                )
                .ok_or_else(|| error("numeric CPU resource quote overflow"))
            })
            .transpose()?;
        let inputs = capture_numeric_inputs(runtime, inputs)?;
        Ok(Some(Self {
            program: NativeNumericRegionProgramOwner {
                program,
                cpu,
                input_count: inputs.len(),
                operation_sites: None,
                metadata,
            },
            inputs,
            right: NativeNumericRegionRight {
                owner: runtime.local_identity(),
                revision: runtime
                    .program_revision()
                    .map(|revision| revision.identity_digest())
                    .transpose()
                    .map_err(|source| {
                        crate::runtime_error::semantic_error_with_source(
                            "RunMat:Native:NumericRegion",
                            "native region program revision is invalid",
                            source,
                        )
                    })?,
                cancellation: runtime.cancellation(),
                region,
                site,
                started,
            },
        }))
    }

    /// Bind the actual native arithmetic occurrences before the request is
    /// suspended. Numeric program validation bounds this inline transcript;
    /// no per-execution site vector or bytecode coordinates are allocated.
    pub fn with_native_operation_sites(
        mut self,
        sites: impl IntoIterator<Item = NativeSiteRequest>,
    ) -> Result<Self, RuntimeError> {
        if self.program.input_count > super::MAX_NODES {
            return Err(error(
                "native site input transcript exceeds the verified numeric node bound",
            ));
        }
        let mut sites = sites.into_iter();
        let mut transcript = [None; super::MAX_NODES];
        for (index, node) in self.program.program.nodes.iter().enumerate() {
            if !matches!(
                node,
                super::NumericRegionNode::Unary { .. } | super::NumericRegionNode::Binary { .. }
            ) {
                continue;
            }
            let site = sites
                .next()
                .ok_or_else(|| error("native arithmetic site transcript is incomplete"))?;
            if site.function != self.right.site.function
                || site.phase != crate::native::NativeSitePhase::RVALUE
            {
                return Err(error(
                    "native arithmetic site transcript has a different source owner",
                ));
            }
            transcript[index] = Some(site);
        }
        if sites.next().is_some() {
            return Err(error(
                "native arithmetic site transcript exceeds its verified program",
            ));
        }
        self.program.operation_sites = Some(transcript);
        Ok(self)
    }

    pub fn into_parts(
        self,
    ) -> (
        NativeNumericRegionProgramOwner,
        RootedValues,
        NativeNumericRegionRight,
    ) {
        (self.program, self.inputs, self.right)
    }
}

/// The backend retains its real selected provider receipt until the native
/// owner has adopted every output. Drop is abandonment, never success.
pub trait NativeNumericRegionPublication {
    fn complete(self: Box<Self>, runtime: &RuntimeContext, right: NativeNumericRegionRight);
}

/// The payload allocation must retire before its funding refunds. This
/// inline handoff owner keeps the claim outside the Box on both abandonment
/// and consuming completion; it adds no allocation or publication authority.
pub struct NativeNumericRegionPublicationOwner {
    payload: Box<dyn NativeNumericRegionPublication>,
    claim: Option<runmat_value::ExactHostAllocationClaim>,
}

impl NativeNumericRegionPublicationOwner {
    /// The producer admits the exact payload layout before constructing its
    /// Box. The claim is optional only for ordinary unbounded allocation.
    pub fn new(
        payload: Box<dyn NativeNumericRegionPublication>,
        claim: Option<runmat_value::ExactHostAllocationClaim>,
    ) -> Self {
        Self { payload, claim }
    }

    pub fn complete(self, runtime: &RuntimeContext, right: NativeNumericRegionRight) {
        let Self { payload, claim } = self;
        payload.complete(runtime, right);
        // The consuming trait call has returned and deallocated its Box.
        drop(claim);
    }
}

pub trait NativeNumericRegionCpuObservation {
    fn finish(
        self: Box<Self>,
        runtime: &RuntimeContext,
        right: NativeNumericRegionRight,
        succeeded: bool,
    );
}

/// The original native MIR sites execute themselves. The provider adapter
/// owns only their physical audit and post-publication completion sample.
pub trait NativeOriginalRegionObservation {
    fn audit(&self) -> Option<SourceRegionRouteAudit>;
    fn complete(
        self: Box<Self>,
        runtime: RuntimeContext,
        right: NativeNumericRegionRight,
    ) -> Result<crate::execution::RuntimeContinuationFuture<'static, ()>, RuntimeError>;
}

pub fn with_native_region_poll<T>(
    audit: Option<SourceRegionRouteAudit>,
    poll: impl FnOnce() -> T,
) -> T {
    runmat_accelerate_api::observation::with_source_region_audit(audit, poll)
}

pub fn set_native_region_for_current_poll(audit: Option<SourceRegionRouteAudit>) {
    runmat_accelerate_api::observation::set_source_region_for_current_poll(audit);
}

/// Claimed private inputs beside the native frame, never public bindings.
pub struct NativeNumericRegionHostExecution {
    inputs: RootedValues,
    route: crate::context::NativeRegionHostRoute,
}
impl NativeNumericRegionHostExecution {
    pub fn new(inputs: RootedValues, route: crate::context::NativeRegionHostRoute) -> Self {
        Self { inputs, route }
    }
    pub fn route(&self) -> &crate::context::NativeRegionHostRoute {
        &self.route
    }
    pub fn input(&self, index: usize) -> Option<&Value> {
        self.inputs.values().get(index)
    }
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }
}

pub enum NativeNumericRegionOutcome {
    /// Only a pre-effect structural/placement decline permits original native
    /// execution. Provider errors after preparation are returned as errors.
    Ineligible,
    Original {
        right: NativeNumericRegionRight,
        observation: Option<Box<dyn NativeOriginalRegionObservation>>,
    },
    NativeCpu {
        kind: NativeNumericRegionCpuKind,
        right: NativeNumericRegionRight,
        observation: Option<Box<dyn NativeNumericRegionCpuObservation>>,
        host: Option<NativeNumericRegionHostExecution>,
    },
    Completed {
        outputs: RootedValues,
        right: NativeNumericRegionRight,
        publication: Option<NativeNumericRegionPublicationOwner>,
    },
}

fn error(message: impl Into<String>) -> RuntimeError {
    semantic_error("RunMat:Native:NumericRegion", message.into())
}

fn allocation_error(
    runtime: &RuntimeContext,
    source: crate::context::host_allocation::HostAllocationError,
) -> RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        Some(runtime),
        "RunMat:Native:NumericRegion",
        "numeric region allocation admission failed",
        source,
    )
}

fn capture_numeric_inputs(
    runtime: &RuntimeContext,
    inputs: &[&Value],
) -> Result<RootedValues, RuntimeError> {
    root_values(runtime, inputs.len(), |index| match inputs[index] {
        Value::Num(value) => Ok(Value::Num(*value)),
        Value::Tensor(tensor)
            if matches!(
                tensor.numeric_dtype(),
                runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
            ) =>
        {
            Ok(Value::Tensor(tensor.clone()))
        }
        Value::GpuTensor(handle) if real_floating_input(inputs[index]) => {
            Ok(Value::GpuTensor(handle.clone()))
        }
        _ => Err(error("numeric input representation changed before capture")),
    })
}

fn real_floating_input(value: &Value) -> bool {
    match value {
        Value::Num(_) => true,
        Value::Tensor(tensor) => matches!(
            tensor.numeric_dtype(),
            runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
        ),
        Value::GpuTensor(handle) => {
            handle.allocation_lease().validate_live().is_ok()
                && handle.view().validate(handle.physical()).is_ok()
                && !handle.view().is_logical()
                && handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
                && handle.view().layout() == runmat_accelerate_api::GpuTensorLayout::Contiguous
                && matches!(
                    handle.physical().element_type(),
                    runmat_accelerate_api::NumericElementType::F32
                        | runmat_accelerate_api::NumericElementType::F64
                )
        }
        _ => false,
    }
}

/// Admit final physical slot backing before taking any produced Value from
/// the backend's transaction. Input and output payload owners keep their own
/// existing claims; this funds only the new collection and external roots.
pub fn root_values(
    runtime: &RuntimeContext,
    width: usize,
    initialize: impl FnMut(usize) -> Result<Value, RuntimeError>,
) -> Result<RootedValues, RuntimeError> {
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPlan, HostAllocationReceipt,
    };
    let allocation_error = |source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:Native:NumericRegion",
            "numeric input capture admission failed",
            source,
        )
    };
    let bytes = OutputListStorage::requested_bytes(width)
        .map_err(|source| allocation_error(source.into()))?;
    let bytes = u64::try_from(bytes).map_err(|_| error("numeric input capture size overflow"))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(bytes, usize::from(width != 0))
        .map_err(allocation_error)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(allocation_error)?,
    )
    .map_err(allocation_error)?;
    let storage = if let Some(receipt) = receipt {
        let permit = receipt.output_permit(runtime).map_err(allocation_error)?;
        let storage =
            OutputListStorage::try_initialize(width, &permit, initialize).map_err(|source| {
                match source {
                    runmat_value::OutputListInitializationError::Allocation(source) => {
                        allocation_error(source.into())
                    }
                    runmat_value::OutputListInitializationError::Element(source) => source,
                }
            })?;
        receipt.checkpoint(runtime).map_err(allocation_error)?;
        storage
    } else {
        OutputListStorage::unbounded((0..width).map(initialize).collect::<Result<Vec<_>, _>>()?)
    };
    RootedValues::from_output_storage_in_context(storage, runtime)
}
