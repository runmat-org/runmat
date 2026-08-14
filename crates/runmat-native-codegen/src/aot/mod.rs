mod launcher;
mod object;
mod product;

pub use object::{emit_relocatable_object, emit_relocatable_object_with_data};
pub use product::{
    embedded_blob, NativeObjectData, NativeObjectDataDescriptor, NativeObjectFormat,
    NativeObjectFunction, NativeObjectManifest, NativeOptimization, RelocatableNativeObject,
    AOT_ENTRY_SYMBOL, AOT_NATIVE_IR_SYMBOL, AOT_PROGRAM_SYMBOL, AOT_RESUME_POINTS_SYMBOL,
    AOT_RUNTIME_MAIN_SYMBOL, NATIVE_OBJECT_SCHEMA_VERSION,
};
