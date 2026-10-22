//! Canonical target-built ABI probe. Decoding proves structure, not provenance
//! or permission to execute. Runtime owns production; packs authenticate bytes.
use super::{
    NativeAbiBinding, NativeArchitecture, NativeEndianness, NativeEnvironment,
    NativeOperatingSystem,
};
use crate::{ContractError, Digest};
#[path = "native_probe_codec.rs"]
mod codec;

pub const NATIVE_ABI_PROBE_SCHEMA_VERSION: u16 = 2;
pub const NATIVE_ABI_PROBE_SYMBOL: &str = "RUNMAT_NATIVE_ABI_PROBE_V2";
pub const NATIVE_ABI_CONTRACT_CAPACITY: usize = 512;
const MAGIC: [u8; 8] = *b"RMABIPR2";

macro_rules! layout_types {
    ($($name:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        #[repr(usize)]
        pub enum NativeAbiType { $($name),+ }
        impl NativeAbiType { pub const ALL: &'static [Self] = &[$(Self::$name),+]; }
    };
}
layout_types!(
    HostStatus,
    Value,
    Root,
    RootSet,
    Source,
    Utf8,
    SourceMapEntry,
    SourceMapView,
    Resume,
    Frame,
    Call,
    Host,
    Exception,
    Cancellation,
    Suspension,
    Deoptimization,
    Exit,
    Safepoint,
    SitePhase,
    SiteRequest,
    SiteOutcomeKind,
    SiteOutcome,
    RuntimeBuiltinBinding,
    BuiltinBindingIdentity
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum NativeAbiField {
    CallHost,
    CallFrame,
    FrameResume,
    ResumeBlock,
    HostContext,
    HostExecuteSite,
    RequestFunction,
    RequestBlock,
    RequestPosition,
    RequestPhase,
    RequestOrdinal,
    RequestReserved,
    OutcomeKind,
    OutcomeEdge,
    OutcomeFlags,
    OutcomeReserved,
    BindingIdentity,
    BindingImplementation,
    BindingContinuationLayout,
    BindingProvenance,
}
impl NativeAbiField {
    pub const ALL: &'static [Self] = &[
        Self::CallHost,
        Self::CallFrame,
        Self::FrameResume,
        Self::ResumeBlock,
        Self::HostContext,
        Self::HostExecuteSite,
        Self::RequestFunction,
        Self::RequestBlock,
        Self::RequestPosition,
        Self::RequestPhase,
        Self::RequestOrdinal,
        Self::RequestReserved,
        Self::OutcomeKind,
        Self::OutcomeEdge,
        Self::OutcomeFlags,
        Self::OutcomeReserved,
        Self::BindingIdentity,
        Self::BindingImplementation,
        Self::BindingContinuationLayout,
        Self::BindingProvenance,
    ];
    fn storage(
        self,
        pointer_bytes: u32,
        layouts: &[NativeProbeTypeLayout; NativeAbiType::ALL.len()],
    ) -> (NativeAbiType, NativeProbeTypeLayout) {
        use NativeAbiField::*;
        let (owner, width) = match self {
            CallHost | CallFrame => (NativeAbiType::Call, pointer_bytes),
            FrameResume => (NativeAbiType::Frame, pointer_bytes),
            ResumeBlock => (NativeAbiType::Resume, 4),
            HostContext | HostExecuteSite => (NativeAbiType::Host, pointer_bytes),
            RequestFunction | RequestBlock | RequestPosition | RequestPhase | RequestOrdinal
            | RequestReserved => (NativeAbiType::SiteRequest, 4),
            OutcomeKind | OutcomeEdge | OutcomeFlags | OutcomeReserved => {
                (NativeAbiType::SiteOutcome, 4)
            }
            BindingIdentity => {
                return (
                    NativeAbiType::RuntimeBuiltinBinding,
                    layouts[NativeAbiType::BuiltinBindingIdentity as usize],
                )
            }
            BindingImplementation | BindingContinuationLayout | BindingProvenance => {
                (NativeAbiType::RuntimeBuiltinBinding, pointer_bytes)
            }
        };
        (
            owner,
            NativeProbeTypeLayout {
                size: width,
                alignment: width,
            },
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeProbeTypeLayout {
    pub size: u32,
    pub alignment: u32,
}

pub const NATIVE_ABI_PROBE_BYTES: usize = 26
    + NATIVE_ABI_CONTRACT_CAPACITY
    + NativeAbiType::ALL.len() * 8
    + NativeAbiField::ALL.len() * 4;

/// Explicit physical facts supplied by the target compiler, not an ISA choice.
pub struct NativeProbeTarget {
    pub architecture: NativeArchitecture,
    pub operating_system: NativeOperatingSystem,
    pub environment: NativeEnvironment,
    pub endianness: NativeEndianness,
    pub pointer_width: u16,
}
pub struct NativeProbeContract<'a> {
    pub schema_version: u16,
    pub encoded_version: u32,
    pub preimage: &'a [u8],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAbiProbe {
    pub architecture: NativeArchitecture,
    pub operating_system: NativeOperatingSystem,
    pub environment: NativeEnvironment,
    pub endianness: NativeEndianness,
    pub pointer_width: u16,
    pub abi_schema_version: u16,
    pub abi_encoded_version: u32,
    contract_length: u16,
    contract: [u8; NATIVE_ABI_CONTRACT_CAPACITY],
    pub layouts: [NativeProbeTypeLayout; NativeAbiType::ALL.len()],
    pub offsets: [u32; NativeAbiField::ALL.len()],
}

impl NativeAbiProbe {
    pub const fn from_target_layout(
        target: NativeProbeTarget,
        abi: NativeProbeContract<'_>,
        layouts: [NativeProbeTypeLayout; NativeAbiType::ALL.len()],
        offsets: [u32; NativeAbiField::ALL.len()],
    ) -> Self {
        let contract_preimage = abi.preimage;
        assert!(
            !contract_preimage.is_empty()
                && contract_preimage.len() <= NATIVE_ABI_CONTRACT_CAPACITY
        );
        let mut contract = [0; NATIVE_ABI_CONTRACT_CAPACITY];
        let mut i = 0;
        while i < contract_preimage.len() {
            contract[i] = contract_preimage[i];
            i += 1;
        }
        Self {
            architecture: target.architecture,
            operating_system: target.operating_system,
            environment: target.environment,
            endianness: target.endianness,
            pointer_width: target.pointer_width,
            abi_schema_version: abi.schema_version,
            abi_encoded_version: abi.encoded_version,
            contract_length: contract_preimage.len() as u16,
            contract,
            layouts,
            offsets,
        }
    }
    pub fn layout(&self, ty: NativeAbiType) -> NativeProbeTypeLayout {
        self.layouts[ty as usize]
    }
    pub fn offset(&self, field: NativeAbiField) -> u32 {
        self.offsets[field as usize]
    }
    pub fn validate(&self) -> Result<(), ContractError> {
        let invalid = || {
            ContractError::invalid(
                "native ABI probe",
                "invalid or unsupported target layout record",
            )
        };
        if self.pointer_width != 64
            || self.endianness != NativeEndianness::Little
            || self.abi_schema_version == 0
            || self.abi_encoded_version == 0
            || self.contract_length == 0
            || usize::from(self.contract_length) > self.contract.len()
            || self.contract[usize::from(self.contract_length)..]
                .iter()
                .any(|byte| *byte != 0)
            || !matches!(
                (self.operating_system, self.environment),
                (
                    NativeOperatingSystem::Linux,
                    NativeEnvironment::Gnu | NativeEnvironment::Musl
                ) | (NativeOperatingSystem::Macos, NativeEnvironment::Darwin)
                    | (NativeOperatingSystem::Windows, NativeEnvironment::Msvc)
            )
        {
            return Err(invalid());
        }
        for layout in self.layouts {
            if layout.size == 0
                || layout.size > 65536
                || !layout.alignment.is_power_of_two()
                || layout.alignment > layout.size
                || layout.size % layout.alignment != 0
            {
                return Err(invalid());
            }
        }
        for field in NativeAbiField::ALL {
            let (owner, storage) = field.storage(u32::from(self.pointer_width / 8), &self.layouts);
            let width = storage.size;
            let offset = self.offset(*field);
            if !offset.is_multiple_of(storage.alignment)
                || self.layout(owner).alignment < storage.alignment
                || offset
                    .checked_add(width)
                    .is_none_or(|end| end > self.layout(owner).size)
            {
                return Err(invalid());
            }
            for previous in NativeAbiField::ALL.iter().take(*field as usize) {
                let (previous_owner, previous_storage) =
                    previous.storage(u32::from(self.pointer_width / 8), &self.layouts);
                let previous_width = previous_storage.size;
                let previous_offset = self.offset(*previous);
                if owner == previous_owner
                    && offset < previous_offset + previous_width
                    && previous_offset < offset + width
                {
                    return Err(invalid());
                }
            }
        }
        Ok(())
    }
    pub fn binding(&self) -> Result<NativeAbiBinding, ContractError> {
        self.validate()?;
        Ok(NativeAbiBinding {
            schema_version: self.abi_schema_version,
            encoded_version: self.abi_encoded_version,
            contract_fingerprint: Digest::sha256(
                &self.contract[..usize::from(self.contract_length)],
            ),
            layout_fingerprint: Digest::sha256(self.canonical_bytes()),
        })
    }
}
