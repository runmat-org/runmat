mod equation;
use runmat_execution::resource::ExactShared;
use runmat_gc_api::{Trace, Tracer};
use std::sync::OnceLock;

use super::optimization_model::storage::{Backing, MetadataRef};
use super::optimization_model::{
    EquationProblemData, OptimizationModelPayload, OptimizationProblemData,
};
use super::warm_start::ActiveSetWarmStartData;
use super::{
    optimization_variable::OptimizationVariablePayload, ContainersMapStorage, GraphicsReceiverRef,
    MemmapReceiverAuthority, MemmapReceiverState, NumericFactorData, ObjectInstance,
};
use crate::optimization_model::PreparedOptimizationNative;
use crate::{ExactArrayClaimPromotionError, SharedStorageIdentity};
use crate::{ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError};
use crate::{ForeignRef, Value};
#[path = "native/warm_start.rs"]
mod warm_start;

#[derive(Debug)]
struct SharedNativePayload<T> {
    value: T,
    // The native DTO and ExactShared header retire before this aggregate
    // physical claim refunds.
    claim: OnceLock<ExactHostAllocationClaim>,
}

#[derive(Debug)]
pub(super) struct SharedNativeState<T>(ExactShared<SharedNativePayload<T>>);

impl<T> Clone for SharedNativeState<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: PartialEq> PartialEq for SharedNativeState<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.get().value == other.0.get().value
    }
}

impl<T> SharedNativeState<T> {
    fn new(value: T) -> Self {
        Self(ExactShared::new(SharedNativePayload {
            value,
            claim: OnceLock::new(),
        }))
    }
    fn get(&self) -> &T {
        &self.0.get().value
    }
    fn try_new_exact(
        value: T,
        retained_value_bytes: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let bytes = ExactShared::<SharedNativePayload<T>>::requested_bytes()
            .checked_add(retained_value_bytes)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let claim = permit.reserve_exact_claim(bytes)?;
        let claims = OnceLock::new();
        let _ = claims.set(claim);
        ExactShared::try_new(SharedNativePayload {
            value,
            claim: claims,
        })
        .map(Self)
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }
    fn get_mut_unbounded(&mut self) -> Option<&mut T> {
        if !self.0.is_unique() {
            return None;
        }
        let payload = self.0.get_mut_unique();
        payload.claim.get().is_none().then_some(&mut payload.value)
    }
    fn has_retained_claim(&self) -> bool {
        self.0.get().claim.get().is_some()
    }
    fn try_admit_exact(
        &self,
        retained_value_bytes: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactHostNumericAllocationError> {
        if self.has_retained_claim() {
            return Ok(());
        }
        let bytes = ExactShared::<SharedNativePayload<T>>::requested_bytes()
            .checked_add(retained_value_bytes)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let claim = permit.reserve_exact_claim(bytes)?;
        let _ = self.0.get().claim.set(claim);
        Ok(())
    }
}
impl<T> Backing for SharedNativeState<T> {
    fn identity(&self) -> Option<SharedStorageIdentity> {
        Some(SharedStorageIdentity::of(self.0.get()))
    }
    fn quote(&self) -> Result<Option<usize>, ExactArrayClaimPromotionError> {
        Ok((!self.has_retained_claim())
            .then_some(ExactShared::<SharedNativePayload<T>>::requested_bytes()))
    }
    fn promote(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactArrayClaimPromotionError> {
        self.try_admit_exact(0, permit)
            .map_err(ExactArrayClaimPromotionError::Allocation)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum NativeObjectPayload {
    ContainersMap(SharedNativeState<ContainersMapStorage>),
    ForeignHostControl(ForeignRef),
    DatasetAuthority(ForeignRef),
    MatFileAuthority(ForeignRef),
    MemmapReceiver(SharedNativeState<MemmapReceiverState>),
    TcpAuthority(ForeignRef),
    GraphicsReceiver(GraphicsReceiverRef),
    NumericFactor(SharedNativeState<NumericFactorData>),
    OptimizationVariable(SharedNativeState<OptimizationVariablePayload>),
    OptimizationModel(SharedNativeState<OptimizationModelPayload>),
    OptimizationProblem(SharedNativeState<OptimizationProblemData>),
    ActiveSetWarmStart(SharedNativeState<ActiveSetWarmStartData>),
    EquationProblem(SharedNativeState<EquationProblemData>),
}

impl NativeObjectPayload {
    pub(in crate::object) fn optimization_native_requested_bytes() -> usize {
        ExactShared::<SharedNativePayload<OptimizationVariablePayload>>::requested_bytes()
    }
    pub(in crate::object) fn optimization_native_exact(
        input: PreparedOptimizationNative,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        SharedNativeState::try_new_exact(input.0.into_payload(), 0, permit)
            .map(Self::OptimizationVariable)
    }
    pub(super) fn numeric_factor_requested_bytes() -> usize {
        ExactShared::<SharedNativePayload<NumericFactorData>>::requested_bytes()
    }
    pub(super) fn numeric_factor_exact(
        value: NumericFactorData,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        SharedNativeState::try_new_exact(value, 0, permit).map(Self::NumericFactor)
    }
    pub(super) fn numeric_factor_unbounded(value: NumericFactorData) -> Self {
        Self::NumericFactor(SharedNativeState::new(value))
    }
    pub(super) fn memmap_receiver_unbounded(value: MemmapReceiverState) -> Self {
        Self::MemmapReceiver(SharedNativeState::new(value))
    }
    pub(super) fn containers_map_requested_bytes(
        value: &ContainersMapStorage,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        ExactShared::<SharedNativePayload<ContainersMapStorage>>::requested_bytes()
            .checked_add(value.retained_physical_bytes()?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub(super) fn memmap_receiver_requested_bytes(
        value: &MemmapReceiverState,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        ExactShared::<SharedNativePayload<MemmapReceiverState>>::requested_bytes()
            .checked_add(value.retained_physical_bytes()?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    pub(super) fn containers_map_exact(
        value: ContainersMapStorage,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let bytes = value.retained_physical_bytes()?;
        SharedNativeState::try_new_exact(value, bytes, permit).map(Self::ContainersMap)
    }

    pub(super) fn memmap_receiver_exact(
        value: MemmapReceiverState,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let bytes = value.retained_physical_bytes()?;
        SharedNativeState::try_new_exact(value, bytes, permit).map(Self::MemmapReceiver)
    }

    pub(super) fn clone_unbounded(&self) -> Self {
        match self {
            Self::ContainersMap(value) => {
                Self::ContainersMap(SharedNativeState::new(value.get().clone()))
            }
            Self::MemmapReceiver(value) => {
                Self::MemmapReceiver(SharedNativeState::new(value.get().clone()))
            }
            Self::ForeignHostControl(value) => Self::ForeignHostControl(value.clone()),
            Self::DatasetAuthority(value) => Self::DatasetAuthority(value.clone()),
            Self::MatFileAuthority(value) => Self::MatFileAuthority(value.clone()),
            Self::TcpAuthority(value) => Self::TcpAuthority(value.clone()),
            Self::GraphicsReceiver(value) => Self::GraphicsReceiver(value.clone()),
            Self::NumericFactor(value) => Self::NumericFactor(value.clone()),
            Self::OptimizationVariable(value) => Self::OptimizationVariable(value.clone()),
            Self::OptimizationModel(value) => Self::OptimizationModel(value.clone()),
            Self::OptimizationProblem(value) => Self::OptimizationProblem(value.clone()),
            Self::ActiveSetWarmStart(value) => Self::ActiveSetWarmStart(value.clone()),
            Self::EquationProblem(value) => Self::EquationProblem(value.clone()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectPersistencePayloadKind {
    PlainValue,
    SelectedClass,
    MixedSelectedNative,
    ContainersMap,
    ForeignHostControl,
    DatasetAuthority,
    MatFileAuthority,
    MemmapReceiver,
    TcpAuthority,
    GraphicsReceiver,
    NumericFactor,
    OptimizationVariable,
    OptimizationModel,
}

impl Trace for NativeObjectPayload {
    fn trace(&self, tracer: &mut dyn Tracer) {
        match self {
            Self::ContainersMap(storage) => storage.get().trace(tracer),
            Self::MemmapReceiver(receiver) => {
                if let MemmapReceiverAuthority::Detached(snapshot) = &receiver.get().authority {
                    snapshot.trace(tracer);
                }
            }
            Self::ForeignHostControl(_)
            | Self::DatasetAuthority(_)
            | Self::MatFileAuthority(_)
            | Self::TcpAuthority(_)
            | Self::GraphicsReceiver(_) => {}
            Self::NumericFactor(_) => {}
            Self::OptimizationVariable(value) => value.get().trace(tracer),
            Self::OptimizationModel(value) => value.get().trace(tracer),
            Self::OptimizationProblem(value) => value.get().trace(tracer),
            Self::ActiveSetWarmStart(value) => value.get().trace(tracer),
            Self::EquationProblem(value) => value.get().trace(tracer),
        }
    }
}

impl ObjectInstance {
    pub fn optimization_native_storage_identity(&self) -> Option<SharedStorageIdentity> {
        match self.generation().native_payload.as_ref()? {
            NativeObjectPayload::OptimizationVariable(value) => value.identity(),
            _ => None,
        }
    }
    pub(in crate::object) fn optimization_metadata_header(&self) -> Option<MetadataRef<'_>> {
        match self.generation().native_payload.as_ref()? {
            NativeObjectPayload::OptimizationVariable(value) => Some(MetadataRef::new(value)),
            _ => None,
        }
    }
    pub(super) fn native_is_admitted_for_copy(&self) -> bool {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::ContainersMap(storage)) => storage.has_retained_claim(),
            Some(NativeObjectPayload::MemmapReceiver(storage)) => storage.has_retained_claim(),
            Some(NativeObjectPayload::NumericFactor(storage)) => storage.has_retained_claim(),
            Some(NativeObjectPayload::OptimizationVariable(storage)) => {
                storage.has_retained_claim()
            }
            Some(NativeObjectPayload::OptimizationModel(storage)) => storage.has_retained_claim(),
            Some(NativeObjectPayload::OptimizationProblem(storage)) => storage.has_retained_claim(),
            Some(NativeObjectPayload::ActiveSetWarmStart(storage)) => storage.has_retained_claim(),
            Some(NativeObjectPayload::EquationProblem(storage)) => storage.has_retained_claim(),
            _ => true,
        }
    }

    pub(super) fn native_unclaimed_quote(
        &self,
    ) -> Result<(usize, usize), ExactHostNumericAllocationError> {
        let Some(native) = self.generation().native_payload.as_ref() else {
            return Ok((0, 0));
        };
        match native {
            NativeObjectPayload::EquationProblem(storage) if !storage.has_retained_claim() => {
                Err(ExactHostNumericAllocationError::UnsupportedClass)
            }
            NativeObjectPayload::ActiveSetWarmStart(_) => {
                Err(ExactHostNumericAllocationError::UnsupportedClass)
            }
            NativeObjectPayload::OptimizationProblem(storage) if !storage.has_retained_claim() => {
                Err(ExactHostNumericAllocationError::UnsupportedClass)
            }
            NativeObjectPayload::OptimizationModel(storage) if !storage.has_retained_claim() => {
                Err(ExactHostNumericAllocationError::UnsupportedClass)
            }
            NativeObjectPayload::OptimizationVariable(storage) if !storage.has_retained_claim() => {
                Err(ExactHostNumericAllocationError::UnsupportedClass)
            }
            NativeObjectPayload::NumericFactor(storage) if !storage.has_retained_claim() => {
                Ok((NativeObjectPayload::numeric_factor_requested_bytes(), 1))
            }
            NativeObjectPayload::ContainersMap(storage) if !storage.has_retained_claim() => Ok((
                NativeObjectPayload::containers_map_requested_bytes(storage.get())?,
                1,
            )),
            NativeObjectPayload::MemmapReceiver(storage) if !storage.has_retained_claim() => Ok((
                NativeObjectPayload::memmap_receiver_requested_bytes(storage.get())?,
                1,
            )),
            _ => Ok((0, 0)),
        }
    }

    pub(super) fn try_admit_native_exact(
        &self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<(), ExactHostNumericAllocationError> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::OptimizationVariable(storage))
                if storage.has_retained_claim() =>
            {
                Ok(())
            }
            Some(
                NativeObjectPayload::OptimizationVariable(_)
                | NativeObjectPayload::OptimizationModel(_)
                | NativeObjectPayload::OptimizationProblem(_)
                | NativeObjectPayload::ActiveSetWarmStart(_)
                | NativeObjectPayload::EquationProblem(_),
            ) => Err(ExactHostNumericAllocationError::UnsupportedClass),
            Some(NativeObjectPayload::NumericFactor(storage)) => storage.try_admit_exact(0, permit),
            Some(NativeObjectPayload::ContainersMap(storage)) => {
                storage.try_admit_exact(storage.get().retained_physical_bytes()?, permit)
            }
            Some(NativeObjectPayload::MemmapReceiver(storage)) => {
                storage.try_admit_exact(storage.get().retained_physical_bytes()?, permit)
            }
            _ => Ok(()),
        }
    }

    pub fn persistence_payload_kind(&self) -> ObjectPersistencePayloadKind {
        let payload = self.generation();
        if payload.selected_class.is_some() && payload.native_payload.is_some() {
            return ObjectPersistencePayloadKind::MixedSelectedNative;
        }
        if payload.selected_class.is_some() {
            return ObjectPersistencePayloadKind::SelectedClass;
        }
        match payload.native_payload.as_ref() {
            None => ObjectPersistencePayloadKind::PlainValue,
            Some(NativeObjectPayload::ContainersMap(_)) => {
                ObjectPersistencePayloadKind::ContainersMap
            }
            Some(NativeObjectPayload::ForeignHostControl(_)) => {
                ObjectPersistencePayloadKind::ForeignHostControl
            }
            Some(NativeObjectPayload::DatasetAuthority(_)) => {
                ObjectPersistencePayloadKind::DatasetAuthority
            }
            Some(NativeObjectPayload::MatFileAuthority(_)) => {
                ObjectPersistencePayloadKind::MatFileAuthority
            }
            Some(NativeObjectPayload::MemmapReceiver(_)) => {
                ObjectPersistencePayloadKind::MemmapReceiver
            }
            Some(NativeObjectPayload::TcpAuthority(_)) => {
                ObjectPersistencePayloadKind::TcpAuthority
            }
            Some(NativeObjectPayload::GraphicsReceiver(_)) => {
                ObjectPersistencePayloadKind::GraphicsReceiver
            }
            Some(NativeObjectPayload::NumericFactor(_)) => {
                ObjectPersistencePayloadKind::NumericFactor
            }
            Some(NativeObjectPayload::OptimizationVariable(_)) => {
                ObjectPersistencePayloadKind::OptimizationVariable
            }
            Some(
                NativeObjectPayload::OptimizationModel(_)
                | NativeObjectPayload::OptimizationProblem(_)
                | NativeObjectPayload::ActiveSetWarmStart(_)
                | NativeObjectPayload::EquationProblem(_),
            ) => ObjectPersistencePayloadKind::OptimizationModel,
        }
    }

    fn with_native(mut self, native: NativeObjectPayload) -> Self {
        let payload = self
            .generation_mut_unique()
            .expect("native payload attachment requires a unique object generation");
        assert!(
            payload.payload_claim.is_none() && !payload.class_name.has_retained_claim(),
            "native payload attachment requires an explicitly unbounded object generation"
        );
        payload.native_payload = Some(native);
        self
    }

    pub fn with_containers_map_payload(self, storage: ContainersMapStorage) -> Self {
        self.with_native(NativeObjectPayload::ContainersMap(SharedNativeState::new(
            storage,
        )))
    }
    /// Runtime must admit the complete definition/reference storage before
    /// calling this unmanaged factory. Exact adoption is deliberately refused
    /// until every retained child and metadata allocation has an owner.
    pub fn with_optimization_variable_payload(self, data: OptimizationVariablePayload) -> Self {
        self.with_native(NativeObjectPayload::OptimizationVariable(
            SharedNativeState::new(data),
        ))
    }
    /// Runtime validates every child edge and admits the complete native model
    /// before this ordinary factory; exact finite adoption remains unsupported.
    pub fn with_optimization_problem_payload(self, data: OptimizationProblemData) -> Self {
        self.with_native(NativeObjectPayload::OptimizationProblem(
            SharedNativeState::new(data),
        ))
    }
    pub fn optimization_problem_payload(&self) -> Option<&OptimizationProblemData> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::OptimizationProblem(value)) => Some(value.get()),
            _ => None,
        }
    }
    pub fn with_optimization_model_payload(self, data: OptimizationModelPayload) -> Self {
        self.with_native(NativeObjectPayload::OptimizationModel(
            SharedNativeState::new(data),
        ))
    }
    pub fn optimization_model_payload(&self) -> Option<&OptimizationModelPayload> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::OptimizationModel(value)) => Some(value.get()),
            _ => None,
        }
    }
    pub fn optimization_model_shape(&self) -> Option<&crate::SharedShape> {
        if self.selected_class().is_some() {
            return None;
        }
        match self.optimization_model_payload() {
            Some(OptimizationModelPayload::Expression(value))
                if self
                    .class_name()
                    .is(runmat_types::standard::OPTIMIZATION_EXPRESSION) =>
            {
                Some(&value.shape)
            }
            Some(OptimizationModelPayload::Constraint(value))
                if self
                    .class_name()
                    .is(runmat_types::standard::OPTIMIZATION_CONSTRAINT)
                    || self
                        .class_name()
                        .is(runmat_types::standard::OPTIMIZATION_EQUALITY)
                    || self
                        .class_name()
                        .is(runmat_types::standard::OPTIMIZATION_INEQUALITY) =>
            {
                Some(&value.shape)
            }
            _ => None,
        }
    }
    pub fn optimization_variable_shape(&self) -> Option<&crate::SharedShape> {
        if !self
            .class_name()
            .is(runmat_types::standard::OPTIMIZATION_VARIABLE)
            || self.selected_class().is_some()
        {
            return None;
        }
        match self.optimization_variable_payload() {
            Some(OptimizationVariablePayload::Reference(value)) => Some(value.shape()),
            _ => None,
        }
    }
    pub fn optimization_variable_payload(&self) -> Option<&OptimizationVariablePayload> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::OptimizationVariable(value)) => Some(value.get()),
            _ => None,
        }
    }
    pub fn with_numeric_factor_data(self, data: NumericFactorData) -> Self {
        self.with_native(NativeObjectPayload::numeric_factor_unbounded(data))
    }
    pub fn numeric_factor_data(&self) -> Option<&NumericFactorData> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::NumericFactor(value)) => Some(value.get()),
            _ => None,
        }
    }
    pub fn with_foreign_host_control(self, reference: ForeignRef) -> Self {
        self.with_native(NativeObjectPayload::ForeignHostControl(reference))
    }
    pub fn with_dataset_authority(self, reference: ForeignRef) -> Self {
        self.with_native(NativeObjectPayload::DatasetAuthority(reference))
    }
    pub fn with_matfile_authority(self, reference: ForeignRef) -> Self {
        self.with_native(NativeObjectPayload::MatFileAuthority(reference))
    }
    pub fn with_tcp_authority(self, reference: ForeignRef) -> Self {
        self.with_native(NativeObjectPayload::TcpAuthority(reference))
    }
    pub fn with_graphics_receiver(self, receiver: GraphicsReceiverRef) -> Self {
        self.with_native(NativeObjectPayload::GraphicsReceiver(receiver))
    }
    pub fn with_memmap_receiver(self, receiver: MemmapReceiverState) -> Self {
        self.with_native(NativeObjectPayload::MemmapReceiver(SharedNativeState::new(
            receiver,
        )))
    }

    pub fn foreign_host_control(&self) -> Option<&ForeignRef> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::ForeignHostControl(value)) => Some(value),
            _ => None,
        }
    }
    pub fn dataset_authority(&self) -> Option<&ForeignRef> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::DatasetAuthority(value)) => Some(value),
            _ => None,
        }
    }
    pub fn matfile_authority(&self) -> Option<&ForeignRef> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::MatFileAuthority(value)) => Some(value),
            _ => None,
        }
    }
    pub fn tcp_authority(&self) -> Option<&ForeignRef> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::TcpAuthority(value)) => Some(value),
            _ => None,
        }
    }
    pub fn graphics_receiver(&self) -> Option<&GraphicsReceiverRef> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::GraphicsReceiver(value)) => Some(value),
            _ => None,
        }
    }
    pub fn memmap_receiver(&self) -> Option<&MemmapReceiverState> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::MemmapReceiver(value)) => Some(value.get()),
            _ => None,
        }
    }
    pub fn memmap_receiver_mut(&mut self) -> Option<&mut MemmapReceiverState> {
        let payload = self.generation_mut_unique()?;
        if payload.payload_claim.is_some() || payload.class_name.has_retained_claim() {
            return None;
        }
        match payload.native_payload.as_mut() {
            Some(NativeObjectPayload::MemmapReceiver(value)) => value.get_mut_unbounded(),
            _ => None,
        }
    }
    pub fn containers_map_payload(&self) -> Option<&ContainersMapStorage> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::ContainersMap(value)) => Some(value.get()),
            _ => None,
        }
    }
    pub fn containers_map_payload_mut(&mut self) -> Option<&mut ContainersMapStorage> {
        let payload = self.generation_mut_unique()?;
        if payload.payload_claim.is_some() || payload.class_name.has_retained_claim() {
            return None;
        }
        match payload.native_payload.as_mut() {
            Some(NativeObjectPayload::ContainersMap(value)) => value.get_mut_unbounded(),
            _ => None,
        }
    }
    pub fn has_native_payload(&self) -> bool {
        self.generation().native_payload.is_some()
    }

    pub fn for_each_native_value<'a>(&'a self, mut visitor: impl FnMut(&'a Value)) {
        if let Some(warm) = self.active_set_warm_start_payload() {
            visitor(warm.x());
            visitor(warm.options());
            return;
        }
        if let Some(variable) = self.optimization_variable_payload() {
            let mut index = 0;
            while let Some(value) = variable.value_at(index) {
                visitor(value);
                index += 1;
            }
            return;
        }
        if let Some(problem) = self.equation_problem_payload() {
            let mut index = 0;
            while let Some(value) = problem.value_at(index) {
                visitor(value);
                index += 1;
            }
            return;
        }
        if let Some(problem) = self.optimization_problem_payload() {
            let mut index = 0;
            while let Some(value) = problem.value_at(index) {
                visitor(value);
                index += 1;
            }
            return;
        }
        if let Some(model) = self.optimization_model_payload() {
            let mut index = 0;
            while let Some(value) = model.value_at(index) {
                visitor(value);
                index += 1;
            }
            return;
        }
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::ContainersMap(storage)) => {
                storage.get().for_each_value(&mut visitor)
            }
            Some(NativeObjectPayload::MemmapReceiver(receiver)) => {
                if let MemmapReceiverAuthority::Detached(snapshot) = &receiver.get().authority {
                    visitor(snapshot);
                }
            }
            _ => {}
        }
    }

    /// Direct native GC edges participate in the same iterative Value tracer
    /// as native child Values; no fabricated HandleObject wrapper is needed.
    pub(crate) fn trace_native_handle_at(&self, index: usize) -> Option<runmat_gc_api::GcHandle> {
        if let Some(model) = self.optimization_model_payload() {
            return model.handle_at(index);
        }
        if index != 0 {
            return None;
        }
        match self.optimization_variable_payload() {
            Some(OptimizationVariablePayload::Reference(value)) => Some(value.definition()),
            _ => None,
        }
    }
    pub(crate) fn trace_native_value_at(&self, index: usize) -> Option<&Value> {
        if let Some(warm) = self.active_set_warm_start_payload() {
            return warm.value_at(index);
        }
        if let Some(variable) = self.optimization_variable_payload() {
            return variable.value_at(index);
        }
        if let Some(problem) = self.equation_problem_payload() {
            return problem.value_at(index);
        }
        if let Some(problem) = self.optimization_problem_payload() {
            return problem.value_at(index);
        }

        if let Some(model) = self.optimization_model_payload() {
            return model.value_at(index);
        }
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::ContainersMap(storage)) => {
                storage.get().trace_value_at(index)
            }
            Some(NativeObjectPayload::MemmapReceiver(receiver)) if index == 0 => {
                match &receiver.get().authority {
                    MemmapReceiverAuthority::Detached(value) => Some(value),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub(crate) fn trace_native_payload(&self, tracer: &mut dyn Tracer) {
        if let Some(payload) = &self.generation().native_payload {
            payload.trace(tracer);
        }
    }
}
