use runmat_runtime::native::NativeAbiLayout;

/// Return the runtime-owned layout. Codegen consumes this contract and never
/// defines a second frame, call, value, root, or exit representation.
pub fn runtime_native_layout() -> NativeAbiLayout {
    runmat_runtime::native::native_abi_layout()
}
