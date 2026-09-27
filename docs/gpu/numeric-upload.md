# Numeric upload ownership

`AccelProvider::upload_numeric` remains the validated numeric upload entrypoint for callers without borrowed resident operands. Ownership-sensitive internal code calls the non-overridable API function `runmat_accelerate_api::upload_numeric_protecting(provider, view, protected_handles)` directly and includes every borrowed input and output prototype that the provider could return as an alias.

The shared boundary validates the host request before invoking the provider, then validates the returned device, shape, element type, storage, semantic annotations, and borrowed-handle identity. A borrowed alias is rejected without releasing it. Borrowed logical, class, and transpose annotations are restored on success, error, and unwinding. Fresh numeric uploads cannot carry conflicting logical/class annotations or a transpose view. A fresh rejected allocation is released through its known exact owner, or through the invoked provider when no owner is known. An unrelated selected provider is never used for cleanup. The validated entrypoints do not return malformed handles.

## Provider implementation migration

Providers that previously overrode `upload_numeric` must rename that override to `upload_numeric_raw`. Keep the native transfer implementation, including single, complex, and exact integer handling, in that raw hook. Do not replace native transfers with conversion through legacy double storage. In-process, WGPU, and CUDA providers use this contract.

The raw hook returns its produced handle to the shared validator. It must not perform generic descriptor-rejection cleanup because it does not know which handles the caller borrowed. A failure inside allocation or transfer remains the provider's responsibility: release only allocations owned by that invocation. Ordinary callers should not invoke the raw hook directly.

The default raw hook adapts legacy real-double and real-integer uploads. It rejects representations those hooks cannot express. An out-of-tree native provider must migrate its override explicitly; retaining an override of the old validated method does not supply the new protected function with native transfer support. Rust permits overriding default trait methods, so implementors must not override the validated convenience method. Such an override cannot bypass the protected function used internally.

## Error handling

`InvalidNumericUploadResult` identifies a malformed result or borrowed alias. `is_invalid_numeric_upload_result` recognizes that error through an `anyhow::Error`. It is an execution failure, not an unsupported-operation signal, and must not be swallowed by a host fallback. Existing provider transfer errors retain their own error type and cause. The typed tensor-upload adapter preserves those errors through automatic placement; integer promotion falls back only for `UnsupportedProviderOperation`. The default unimplemented integer upload reports that typed unsupported condition, while allocation and execution failures remain failures.
