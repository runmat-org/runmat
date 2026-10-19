#[cfg(feature = "compiler")]
mod launcher;
#[cfg(feature = "compiler")]
mod object;
// Object v13 and launcher v8 require explicit product mode and the optional
// linked source-service factory pointer. Older manifests,
// including the former pure-v6 migration, cannot infer a host-unselected grant.
mod product;
mod selected;
pub use selected::{validate_native_product, LinkedNativeProduct, SelectedNativeProduct};

#[cfg(feature = "compiler")]
pub use object::{
    emit_relocatable_object, emit_relocatable_object_for_runtime,
    emit_relocatable_object_with_data, emit_selected_relocatable_object,
};
pub use product::{
    embedded_blob, native_object_format_for_target, AotBuiltinBinding, AotProgramFunction,
    AotProgramManifest, AotRuntimeBindingMode, NativeObjectData, NativeObjectDataDescriptor,
    NativeObjectFormat, NativeObjectFunction, NativeObjectManifest, NativeOptimization,
    RelocatableNativeObject, AOT_ENTRY_SYMBOL, AOT_JAVA_ARTIFACTS_SYMBOL, AOT_MEX_ARTIFACTS_SYMBOL,
    AOT_NATIVE_INTERFACES_SYMBOL, AOT_NATIVE_IR_SYMBOL, AOT_PROGRAM_MANIFEST_SCHEMA_VERSION,
    AOT_PROGRAM_SYMBOL, AOT_PYTHON_ARTIFACTS_SYMBOL, AOT_RESUME_POINTS_SYMBOL,
    AOT_RUNTIME_MAIN_SYMBOL, AOT_SOURCE_SERVICE_FACTORY_SYMBOL, NATIVE_OBJECT_SCHEMA_VERSION,
};
