#[cfg(feature = "compiler")]
mod launcher;
#[cfg(feature = "compiler")]
mod object;
// Object v15 retains explicit product mode and supports a typed paired-test
// launcher. Older manifests cannot infer a host-unselected grant or test ABI.
mod paired;
mod product;
mod selected;
pub use paired::validate_paired_product_header;
pub use selected::{validate_native_product, LinkedNativeProduct, SelectedNativeProduct};

#[cfg(feature = "compiler")]
pub use object::{
    emit_relocatable_object, emit_relocatable_object_for_runtime,
    emit_relocatable_object_with_data, emit_selected_paired_relocatable_object,
    emit_selected_relocatable_object,
};
pub use product::{
    embedded_blob, native_object_format_for_target, AotBuiltinBinding, AotProgramFunction,
    AotProgramManifest, AotRuntimeBindingMode, NativeObjectData, NativeObjectDataDescriptor,
    NativeObjectFormat, NativeObjectFunction, NativeObjectManifest, NativeOptimization,
    RelocatableNativeObject, AOT_CLOCK_SERVICE_FACTORY_SYMBOL, AOT_ENTRY_SYMBOL,
    AOT_HOSTED_DYNAMIC_FACTORY_SYMBOL, AOT_HOST_CALLBACK_FACTORY_SYMBOL, AOT_JAVA_ARTIFACTS_SYMBOL,
    AOT_LOCAL_SERVICE_FACTORY_SYMBOL, AOT_MEX_ARTIFACTS_SYMBOL, AOT_NATIVE_INTERFACES_SYMBOL,
    AOT_NATIVE_IR_SYMBOL, AOT_NETWORK_SERVICE_FACTORY_SYMBOL, AOT_PAIRED_PRODUCT_SYMBOL,
    AOT_PROGRAM_MANIFEST_SCHEMA_VERSION, AOT_PROGRAM_SYMBOL, AOT_PYTHON_ARTIFACTS_SYMBOL,
    AOT_RESUME_POINTS_SYMBOL, AOT_RUNTIME_MAIN_SYMBOL, AOT_RUNTIME_TEST_MAIN_SYMBOL,
    AOT_SOURCE_SERVICE_FACTORY_SYMBOL, AOT_UI_SERVICE_FACTORY_SYMBOL, NATIVE_OBJECT_SCHEMA_VERSION,
};
