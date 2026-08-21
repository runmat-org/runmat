use crate::{NativeCodegenError, NativeCodegenResult};

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeAbiBinding {
    pub schema_version: u16,
    pub encoded_version: u32,
    pub contract_fingerprint: runmat_execution::Digest,
    pub layout_fingerprint: runmat_execution::Digest,
}

impl NativeAbiBinding {
    pub fn current() -> Self {
        Self {
            schema_version: runmat_runtime::native::NATIVE_ABI_SCHEMA_VERSION,
            encoded_version: runmat_runtime::native::NATIVE_ABI_VERSION.encoded(),
            contract_fingerprint: runmat_runtime::native::native_abi_contract_fingerprint(),
            layout_fingerprint: runmat_runtime::native::native_abi_layout_fingerprint(),
        }
    }

    pub fn validate(&self) -> NativeCodegenResult<()> {
        let current = Self::current();
        if self != &current {
            return Err(NativeCodegenError::new(
                "native.abi.binding",
                "native ABI version or target layout does not match this compiler",
            ));
        }
        Ok(())
    }
}
