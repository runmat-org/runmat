mod array;
mod builder;
mod containers_map;
mod dynamic_property;
mod graphics_callback;
mod graphics_receiver;
mod graphics_scene;
mod handle;
mod instance;
mod listener;
mod memmap;
mod native;
mod numeric_factor;
mod payload;
mod selected_class;

pub use array::{
    ObjectArray, ObjectArrayAdmission, ObjectArrayBuildError, ObjectArrayBuilder,
    ObjectArrayElementError, ObjectArrayQuote, ValidatedObjectArrayParts,
};
pub use builder::{
    ExactObjectBuilder, ExactObjectQuote, ObjectClassPreparationError, ObjectEditInput,
    ObjectEditPreparationError, PreparedObjectClassName, UnboundedObjectBuilder,
};
pub use containers_map::{
    ContainersMapEntry, ContainersMapKey, ContainersMapKeyType, ContainersMapStorage,
    ContainersMapValueClass, ContainersMapValueType,
};
pub use dynamic_property::{DynamicPropertyAdmissionError, DynamicPropertyDef};
pub use graphics_callback::{
    GraphicsCallbackDisposition, GraphicsCallbackHostFailure, GraphicsCallbackIngressFailure,
    GraphicsCallbackKind, GraphicsCallbackResult, GraphicsCallbackRolePayload,
};
pub use graphics_receiver::{GraphicsReceiverKind, GraphicsReceiverRef};
pub use graphics_scene::GraphicsSceneLease;
pub use handle::HandleRef;
pub use instance::ObjectInstance;
pub use listener::Listener;
pub use memmap::{
    MemmapElementType, MemmapFieldState, MemmapFormatPresentationState, MemmapReceiverAuthority,
    MemmapReceiverState,
};
use native::NativeObjectPayload;
pub use native::ObjectPersistencePayloadKind;
pub use numeric_factor::{
    DenseOrthogonalFactorData, FactorPermutation, NativeDenseFactorMatrix,
    NumericFactorBufferAdoption, NumericFactorCandidate, NumericFactorData, NumericFactorKind,
    NumericFactorValidation, OrthogonalFactorData,
};
use payload::{ObjectPayload, SharedObjectPayload};
pub use selected_class::{
    SelectedObjectClassFacts, SelectedObjectMethod, SelectedReceiverClassDeclaration,
    SelectedReceiverDeclarations, SelectedReceiverMethodDeclaration,
    SelectedReceiverPropertyDeclaration,
};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
