#[cfg(feature = "compiler")]
mod launcher;
#[cfg(feature = "compiler")]
mod object;
mod product;

#[cfg(feature = "compiler")]
pub use object::{
    emit_relocatable_object, emit_relocatable_object_for_runtime, emit_relocatable_object_with_data,
};
pub use product::{
    embedded_blob, AotBuiltinBinding, AotProgramFunction, AotProgramManifest,
    AotRuntimeBindingMode, NativeObjectData, NativeObjectDataDescriptor, NativeObjectFormat,
    NativeObjectFunction, NativeObjectManifest, NativeOptimization, RelocatableNativeObject,
    AOT_ENTRY_SYMBOL, AOT_NATIVE_IR_SYMBOL, AOT_PROGRAM_MANIFEST_SCHEMA_VERSION,
    AOT_PROGRAM_SYMBOL, AOT_RESUME_POINTS_SYMBOL, AOT_RUNTIME_MAIN_SYMBOL,
    NATIVE_OBJECT_SCHEMA_VERSION,
};
