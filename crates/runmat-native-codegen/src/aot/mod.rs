mod object;
mod product;

pub use object::{emit_relocatable_object, emit_relocatable_object_with_data};
pub use product::{
    NativeObjectData, NativeObjectDataDescriptor, NativeObjectFormat, NativeObjectFunction,
    NativeObjectManifest, NativeOptimization, RelocatableNativeObject,
    NATIVE_OBJECT_SCHEMA_VERSION,
};
