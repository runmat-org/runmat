use crate::{NativeCodegenError, NativeCodegenResult};

pub use runmat_execution::host::NativeAbiBinding;

pub fn current_native_abi_binding() -> NativeAbiBinding {
    NativeAbiBinding {
        schema_version: runmat_runtime::native::NATIVE_ABI_SCHEMA_VERSION,
        encoded_version: runmat_runtime::native::NATIVE_ABI_VERSION.encoded(),
        contract_fingerprint: runmat_runtime::native::native_abi_contract_fingerprint(),
        layout_fingerprint: runmat_runtime::native::native_abi_layout_fingerprint(),
    }
}

pub fn validate_current_native_abi(binding: &NativeAbiBinding) -> NativeCodegenResult<()> {
    if binding
        .require_exact(&current_native_abi_binding())
        .is_err()
    {
        return Err(NativeCodegenError::new(
            "native.abi.binding",
            "native ABI version or target layout does not match this compiler",
        ));
    }
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn portable_abi_wire_is_unchanged_but_decoding_is_not_host_admission() {
        let binding = current_native_abi_binding();
        let expected = format!(
            "{{\"schema_version\":{},\"encoded_version\":{},\"contract_fingerprint\":\"{}\",\"layout_fingerprint\":\"{}\"}}",
            binding.schema_version, binding.encoded_version,
            binding.contract_fingerprint, binding.layout_fingerprint,
        );
        assert_eq!(serde_json::to_vec(&binding).unwrap(), expected.as_bytes());
        let mut foreign: NativeAbiBinding = serde_json::from_str(&expected).unwrap();
        foreign.layout_fingerprint = runmat_execution::Digest::sha256(b"a distinct target probe");
        foreign.validate().unwrap();
        assert!(validate_current_native_abi(&foreign).is_err());
        let mut target = crate::NativeTarget::current();
        let crate::NativeTarget::Native {
            target: selected, ..
        } = &mut target
        else {
            panic!("native ABI fixture requires a native target")
        };
        selected.target.abi = foreign.identity().unwrap();
        assert!(
            selected.validate().is_ok(),
            "portable target metadata alone is not native ABI admission"
        );
        assert!(target.validate().is_err());
        assert!(target.require_host_executable().is_err());
    }
}
