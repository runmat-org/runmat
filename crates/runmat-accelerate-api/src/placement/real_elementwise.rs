//! One ordinary real-elementwise physical recipe. Numerical/source admission
//! remains separate; an operation description or wire value is not authority.
#[path = "real_elementwise/mixed_add.rs"]
mod mixed_add;
use crate::{
    scalar_program::ScalarLiteral, AccelProvider, GpuTensorHandle, GpuTensorLayout,
    GpuTensorStorage, NumericElementType, ProviderElementType, ProviderExecutionCompletionError,
    ProviderExecutionError, ProviderExecutionResult, ProviderLayout, ProviderPrecision,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};
pub use mixed_add::{
    ProviderExecutedMixedRealAdd, ProviderMixedRealAddObserved, ProviderMixedRealAddRecipe,
    ProviderMixedRealAddResources, ProviderMixedRealAddStepObservations,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderRealElementwiseOperation {
    /// Exact Double scalar exponent on a real F32 base; source control bits
    /// remain part of the recipe identity and cannot stand in for Single.
    ExactSingleBasePower {
        control: crate::PowerScalarControl,
    },
    ScalarAdd {
        control: ScalarLiteral,
        mode: runmat_types::RealFloatingAddMode,
    },
    /// Physical native-class multiplication only. Selected arithmetic
    /// equivalence remains an independent admission.
    ScalarMul {
        control: ScalarLiteral,
    },
    /// Physical native-class maximum only. This does not certify the
    /// selected NaN, comparison, or signed-zero policy.
    ScalarMax {
        control: ScalarLiteral,
    },
    Sqrt,
    /// IEEE spacing at each real element's magnitude.
    Spacing,
    /// Native real exp(x)-1, without widening Single storage.
    Expm1,
    /// Ordinary real unary execution only. The recipe supplies physical work,
    /// not the operation owner's separate numerical-domain admission.
    UnaryMath(runmat_types::ScalarUnaryMath),
    FloatingCast {
        target: ProviderPrecision,
    },
}

/// Same-class ordinary casts publish the authenticated source allocation.
/// This is not permission for arithmetic operations to alias their inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ProviderRealElementwiseOutputRoute {
    FreshAllocation,
    SameClassFloatingCastAlias,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderRealElementwiseRequest {
    pub operation: ProviderRealElementwiseOperation,
    pub source: ProviderRepresentation,
    pub source_buffer_bytes: u64,
}

/// Backend description only. Callers cannot turn this into an executable
/// recipe; the privately minted recipe retains its actual provider instance.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ProviderRealElementwisePhysicalProposal {
    pub output_route: ProviderRealElementwiseOutputRoute,
    pub output_buffer_bytes: u64,
    pub workgroup_size: u32,
    pub chunk_capacity: u64,
    pub uniform_allocations: u32,
    pub uniform_buffer_bytes: u64,
    pub dispatches: u32,
    pub queue_submissions: u32,
    pub pipeline_key: Option<u64>,
}

impl ProviderRealElementwisePhysicalProposal {
    /// Output and all potentially retained uniforms; excludes existing source
    /// storage, cache residency, and independently budgeted observation fences.
    pub fn requested_incremental_peak_bytes(&self) -> Option<u64> {
        let fresh_output = match self.output_route {
            ProviderRealElementwiseOutputRoute::FreshAllocation => self.output_buffer_bytes,
            ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias => 0,
        };
        fresh_output.checked_add(self.uniform_buffer_bytes)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderRealElementwiseObserved {
    pub output_buffer_bytes: Option<u64>,
    pub uniform_allocations: u32,
    pub uniform_buffer_bytes: u64,
    pub dispatches: u32,
    pub queue_submissions: u32,
}

pub struct ProviderExecutedRealElementwise {
    pub output: GpuTensorHandle,
    pub observed: ProviderRealElementwiseObserved,
}

#[derive(Clone)]
pub struct ProviderRealElementwiseRecipe {
    owner: &'static dyn AccelProvider,
    request: ProviderRealElementwiseRequest,
    output: ProviderRepresentation,
    physical: ProviderRealElementwisePhysicalProposal,
}

impl ProviderRealElementwiseRecipe {
    pub fn plan(
        owner: &'static dyn AccelProvider,
        request: ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<Self>> {
        if !native_request(&request) {
            return Ok(None);
        }
        let Some(physical) = owner.propose_real_elementwise(&request)? else {
            return Ok(None);
        };
        let (output, route) = output_projection(&request);
        if physical.workgroup_size == 0
            || physical.chunk_capacity == 0
            || physical.output_buffer_bytes < output.checked_byte_len().unwrap_or(u64::MAX)
            || physical.output_route != route
            || !valid_route(&request, &physical)
            || physical.requested_incremental_peak_bytes().is_none()
        {
            return Ok(None);
        }
        Ok(Some(Self {
            owner,
            request,
            output,
            physical,
        }))
    }

    pub fn request(&self) -> &ProviderRealElementwiseRequest {
        &self.request
    }
    pub fn physical(&self) -> &ProviderRealElementwisePhysicalProposal {
        &self.physical
    }
    pub fn output(&self) -> &ProviderRepresentation {
        &self.output
    }
    pub fn is_owned_by(&self, owner: &dyn AccelProvider) -> bool {
        crate::same_provider_instance(self.owner, owner)
    }
    pub fn identity(&self) -> anyhow::Result<runmat_execution::Digest> {
        let (operation, control, add_mode, target) = match self.request.operation {
            ProviderRealElementwiseOperation::ExactSingleBasePower { control } => (
                5,
                Some(ScalarLiteral::F64Bits(match control {
                    crate::PowerScalarControl::DoubleBits(bits) => bits,
                    _ => unreachable!("certified Double control"),
                })),
                None,
                None,
            ),
            ProviderRealElementwiseOperation::ScalarAdd { control, mode } => (
                0_u8,
                Some(control),
                Some(match mode {
                    runmat_types::RealFloatingAddMode::Single => 0_u8,
                    runmat_types::RealFloatingAddMode::Double => 1,
                    runmat_types::RealFloatingAddMode::MixedDoubleThenSingle => 2,
                }),
                None,
            ),
            ProviderRealElementwiseOperation::Sqrt => (1, None, None, None),
            ProviderRealElementwiseOperation::Spacing => (6, None, None, None),
            ProviderRealElementwiseOperation::Expm1 => (7, None, None, None),
            ProviderRealElementwiseOperation::FloatingCast { target } => {
                (2, None, None, Some(target))
            }
            ProviderRealElementwiseOperation::ScalarMul { control } => {
                (3, Some(control), None, None)
            }
            ProviderRealElementwiseOperation::ScalarMax { control } => {
                (4, Some(control), None, None)
            }
            ProviderRealElementwiseOperation::UnaryMath(_) => (5, None, None, None),
        };
        let unary = match self.request.operation {
            ProviderRealElementwiseOperation::UnaryMath(operation) => Some(operation),
            _ => None,
        };
        Ok(runmat_execution::Digest::sha256(serde_json::to_vec(&(
            "ordinary-real-elementwise-v4",
            &self.request.source,
            self.request.source_buffer_bytes,
            control,
            operation,
            add_mode,
            target,
            unary,
            &self.output,
            &self.physical,
        ))?))
    }

    /// Input lifetime spans the provider await. The provider rederives its
    /// physical recipe before effect; this does not establish numerical-domain
    /// or selected transcript authority for the caller.
    pub async fn execute(
        &self,
        input: &GpuTensorHandle,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        self.execute_protected(input, &[]).await
    }

    /// Protect every borrowed allocation before a provider can return a
    /// malformed output. The source alone is insufficient when surrounding
    /// selected execution retains other public outputs or scratch handles.
    pub async fn execute_protected(
        &self,
        input: &GpuTensorHandle,
        borrowed: &[&GpuTensorHandle],
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        self.execute_protected_admitted(input, borrowed, None).await
    }

    pub async fn execute_protected_admitted(
        &self,
        input: &GpuTensorHandle,
        borrowed: &[&GpuTensorHandle],
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedRealElementwise> {
        for handle in borrowed {
            handle
                .view()
                .validate(handle.physical())
                .map_err(ProviderExecutionError::from_source)?;
            handle
                .allocation_lease()
                .validate_live()
                .map_err(ProviderExecutionError::from_source)?;
        }
        if !live_matches(
            self.owner,
            input,
            &self.request.source,
            self.request.source_buffer_bytes,
        ) || self.owner.propose_real_elementwise(&self.request)?.as_ref() != Some(&self.physical)
        {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "real elementwise source or physical recipe changed"
            )));
        }
        // A provider hook may suspend or fail after mutating registry metadata.
        // Keep restoration live across suspension, including cancellation.
        let metadata_guards = std::iter::once(input)
            .chain(borrowed.iter().copied())
            .map(crate::ResidencyRestoreGuard::new)
            .collect::<Vec<_>>();
        let result = self
            .owner
            .execute_real_elementwise_recipe(input, self, admission)
            .await;
        for guard in &metadata_guards {
            guard.restore();
        }
        drop(metadata_guards);
        let result = result?;
        let output_ok = live_matches(
            self.owner,
            &result.output,
            &self.output,
            self.physical.output_buffer_bytes,
        );
        let aliases_source = result.output.shares_allocation_with(input);
        let alias_ok = match self.physical.output_route {
            ProviderRealElementwiseOutputRoute::FreshAllocation => !std::iter::once(input)
                .chain(borrowed.iter().copied())
                .any(|handle| result.output.shares_allocation_with(handle)),
            ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias => {
                aliases_source
                    && result.output.view() == input.view()
                    && result.output.physical() == input.physical()
            }
        };
        let observed = &result.observed;
        if !output_ok
            || !live_matches(
                self.owner,
                input,
                &self.request.source,
                self.request.source_buffer_bytes,
            )
            || !alias_ok
            || borrowed
                .iter()
                .any(|handle| handle.allocation_lease().validate_live().is_err())
            || observed.output_buffer_bytes != Some(self.physical.output_buffer_bytes)
            || observed.uniform_allocations != self.physical.uniform_allocations
            || observed.uniform_buffer_bytes != self.physical.uniform_buffer_bytes
            || observed.dispatches != self.physical.dispatches
            || observed.queue_submissions != self.physical.queue_submissions
        {
            let error = ProviderExecutionError::from(anyhow::anyhow!(
                "real elementwise observed execution differs from recipe"
            ))
            .after_effect();
            // Same-allocation outputs are borrowed aliases, not retireable raw
            // allocations. Never retire the authenticated source on rejection.
            if std::iter::once(input)
                .chain(borrowed.iter().copied())
                .any(|handle| result.output.shares_allocation_with(handle))
            {
                drop(result.output);
                return Err(error);
            }
            let cleanup = result
                .output
                .release()
                .err()
                .into_iter()
                .collect::<Vec<_>>();
            return if cleanup.is_empty() {
                Err(error)
            } else {
                Err(
                    ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                        Some(error),
                        cleanup,
                    ))
                    .after_effect(),
                )
            };
        }
        Ok(result)
    }
}

fn native_request(request: &ProviderRealElementwiseRequest) -> bool {
    let source = &request.source;
    if source.storage != ProviderStorage::DenseReal
        || !(source.layout == ProviderLayout::ColumnMajorContiguous
            || (matches!(
                request.operation,
                ProviderRealElementwiseOperation::ExactSingleBasePower { .. }
            ) && source.layout == ProviderLayout::Strided
                && source.shape.len() == 2))
        || source.residency != ProviderResidency::Device
        || !matches!(
            source.element_type,
            ProviderElementType::F32 | ProviderElementType::F64
        )
        || source.checked_element_count().is_none_or(|count| {
            count == 0
                && !matches!(
                    request.operation,
                    ProviderRealElementwiseOperation::ExactSingleBasePower { .. }
                )
        })
        || source
            .checked_byte_len()
            .is_none_or(|bytes| bytes > request.source_buffer_bytes)
    {
        return false;
    }
    match request.operation {
        ProviderRealElementwiseOperation::ExactSingleBasePower { control } => {
            source.element_type == ProviderElementType::F32
                && crate::ExactSingleBasePower::for_control(control).is_some()
        }
        ProviderRealElementwiseOperation::FloatingCast { .. } => true,
        ProviderRealElementwiseOperation::Sqrt
        | ProviderRealElementwiseOperation::Spacing
        | ProviderRealElementwiseOperation::Expm1
        | ProviderRealElementwiseOperation::UnaryMath(_) => true,
        ProviderRealElementwiseOperation::ScalarMul { control }
        | ProviderRealElementwiseOperation::ScalarMax { control } => matches!(
            (source.element_type, control),
            (ProviderElementType::F32, ScalarLiteral::F32Bits(_))
                | (ProviderElementType::F64, ScalarLiteral::F64Bits(_))
        ),
        ProviderRealElementwiseOperation::ScalarAdd { control, mode } => matches!(
            (source.element_type, control, mode),
            (
                ProviderElementType::F32,
                ScalarLiteral::F32Bits(_),
                runmat_types::RealFloatingAddMode::Single
            ) | (
                ProviderElementType::F64,
                ScalarLiteral::F64Bits(_),
                runmat_types::RealFloatingAddMode::Double
            )
        ),
    }
}

fn output_projection(
    request: &ProviderRealElementwiseRequest,
) -> (ProviderRepresentation, ProviderRealElementwiseOutputRoute) {
    let mut output = request.source.clone();
    let route = match request.operation {
        ProviderRealElementwiseOperation::FloatingCast { target } => {
            output.element_type = match target {
                ProviderPrecision::F32 => ProviderElementType::F32,
                ProviderPrecision::F64 => ProviderElementType::F64,
            };
            if output.element_type == request.source.element_type {
                ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias
            } else {
                ProviderRealElementwiseOutputRoute::FreshAllocation
            }
        }
        ProviderRealElementwiseOperation::ScalarAdd { .. }
        | ProviderRealElementwiseOperation::ExactSingleBasePower { .. }
        | ProviderRealElementwiseOperation::ScalarMul { .. }
        | ProviderRealElementwiseOperation::ScalarMax { .. }
        | ProviderRealElementwiseOperation::Sqrt
        | ProviderRealElementwiseOperation::Spacing
        | ProviderRealElementwiseOperation::Expm1
        | ProviderRealElementwiseOperation::UnaryMath(_) => {
            output.layout = ProviderLayout::ColumnMajorContiguous;
            ProviderRealElementwiseOutputRoute::FreshAllocation
        }
    };
    (output, route)
}

fn valid_route(
    request: &ProviderRealElementwiseRequest,
    physical: &ProviderRealElementwisePhysicalProposal,
) -> bool {
    match physical.output_route {
        ProviderRealElementwiseOutputRoute::FreshAllocation => {
            physical.pipeline_key.is_some()
                || (request.source.checked_element_count() == Some(0)
                    && physical.uniform_allocations == 0
                    && physical.uniform_buffer_bytes == 0
                    && physical.dispatches == 0
                    && physical.queue_submissions == 0)
        }
        ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias => {
            physical.output_buffer_bytes == request.source_buffer_bytes
                && physical.uniform_allocations == 0
                && physical.uniform_buffer_bytes == 0
                && physical.dispatches == 0
                && physical.queue_submissions == 0
                && physical.pipeline_key.is_none()
        }
    }
}

fn live_matches(
    owner: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    representation: &ProviderRepresentation,
    bytes: u64,
) -> bool {
    let expected = match representation.element_type {
        ProviderElementType::F32 => NumericElementType::F32,
        ProviderElementType::F64 => NumericElementType::F64,
        _ => return false,
    };
    handle.allocation_lease().validate_live().is_ok()
        && handle.view().validate(handle.physical()).is_ok()
        && crate::provider_for_handle(handle)
            .is_some_and(|actual| crate::same_provider_instance(actual, owner))
        && handle.device_id() == owner.device_id()
        && handle.physical().element_type() == expected
        && handle.physical().storage() == GpuTensorStorage::Real
        && !handle.view().is_logical()
        && matches!(
            (representation.layout, handle.view().layout()),
            (
                ProviderLayout::ColumnMajorContiguous,
                GpuTensorLayout::Contiguous
            ) | (ProviderLayout::Strided, GpuTensorLayout::Transpose(_))
        )
        && handle.physical().requested_bytes() == bytes
        && handle
            .shape()
            .iter()
            .copied()
            .map(|n| u64::try_from(n).ok())
            .collect::<Option<Vec<_>>>()
            .as_ref()
            == Some(&representation.shape)
}

#[cfg(test)]
#[path = "real_elementwise/tests.rs"]
mod tests;
