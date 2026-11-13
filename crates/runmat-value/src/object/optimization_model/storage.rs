//! Prepared variable metadata: owned publication and existing aliases share
//! one sealed source view and cursor without boxing or cloning the payload.
use crate::optimization_variable::OptimizationVariablePayload;
use crate::{
    ExactArrayClaimPromotionError, ExactArrayStorage, ExactHostAllocationPermit, ObjectInstance,
    SharedShape, SharedStorageIdentity, Utf8Text, Value,
};
use runmat_gc_api::GcHandle;
mod backing;
mod cursor;
pub(in crate::object) use backing::{Backing, MetadataRef};
pub use cursor::OptimizationNativeAdmission;

/// The one owned payload moves into the actual admitted native header.
pub struct OptimizationNativeInput(pub(in crate::object) OptimizationVariablePayload);
/// An O(1) alias, retained only while its original metadata is inspected.
pub struct ExistingOptimizationNativeInput(ObjectInstance);
pub struct PreparedOptimizationNative(pub(in crate::object) OptimizationNativeInput);
#[derive(Clone, Copy, Debug)]
pub struct OptimizationMetadataQuote {
    pub owner: Option<SharedStorageIdentity>,
    pub bytes: Option<usize>,
}
#[derive(Clone, Copy, Debug)]
pub struct OptimizationNativeEdge {
    pub handle: GcHandle,
}
mod sealed {
    pub trait Sealed {}
    impl Sealed for super::OptimizationNativeInput {}
    impl Sealed for super::ExistingOptimizationNativeInput {}
}
/// Closed, borrowed metadata projection. It issues no class/allocation proof.
pub trait OptimizationNativeSource: sealed::Sealed {
    fn variable(&self) -> Option<&OptimizationVariablePayload>;
    fn existing_object(&self) -> Option<&ObjectInstance>;
    fn value_at(&self, index: usize) -> Option<&Value> {
        self.variable()?.value_at(index)
    }
    fn handle_at(&self, index: usize) -> Option<GcHandle> {
        match self.variable()? {
            OptimizationVariablePayload::Reference(value) if index == 0 => Some(value.definition()),
            _ => None,
        }
    }
    fn edge_at(&self, index: usize) -> Option<OptimizationNativeEdge> {
        self.handle_at(index)
            .map(|handle| OptimizationNativeEdge { handle })
    }
}
impl OptimizationNativeInput {
    pub fn new(value: OptimizationVariablePayload) -> Self {
        Self(value)
    }
    pub fn into_payload(self) -> OptimizationVariablePayload {
        self.0
    }
    pub fn native_header_requested_bytes() -> usize {
        crate::object::native::NativeObjectPayload::optimization_native_requested_bytes()
    }
    pub fn begin_admission(self) -> OptimizationNativeAdmission<Self> {
        OptimizationNativeAdmission::new(self)
    }
}
impl ExistingOptimizationNativeInput {
    pub fn new(object: ObjectInstance) -> Self {
        Self(object)
    }
    pub fn begin_admission(self) -> OptimizationNativeAdmission<Self> {
        OptimizationNativeAdmission::new(self)
    }
}
impl OptimizationNativeSource for OptimizationNativeInput {
    fn variable(&self) -> Option<&OptimizationVariablePayload> {
        Some(&self.0)
    }
    fn existing_object(&self) -> Option<&ObjectInstance> {
        None
    }
}
impl OptimizationNativeSource for ExistingOptimizationNativeInput {
    fn variable(&self) -> Option<&OptimizationVariablePayload> {
        self.0.optimization_variable_payload()
    }
    fn existing_object(&self) -> Option<&ObjectInstance> {
        Some(&self.0)
    }
}
