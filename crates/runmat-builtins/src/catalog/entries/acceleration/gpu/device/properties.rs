//! Canonical GPUDevice class/member identities. Executable property dispatch
//! carries this enum; display spelling is never its semantic authority.
pub const GPU_DEVICE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("parallel.gpu.GPUDevice");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuDeviceProperty {
    Name,
    Index,
    TotalMemory,
    AvailableMemory,
    SupportsDouble,
    DeviceAvailable,
    DeviceSelected,
}
impl GpuDeviceProperty {
    pub const ALL: [Self; 7] = [
        Self::Name,
        Self::Index,
        Self::TotalMemory,
        Self::AvailableMemory,
        Self::SupportsDouble,
        Self::DeviceAvailable,
        Self::DeviceSelected,
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Name => "Name",
            Self::Index => "Index",
            Self::TotalMemory => "TotalMemory",
            Self::AvailableMemory => "AvailableMemory",
            Self::SupportsDouble => "SupportsDouble",
            Self::DeviceAvailable => "DeviceAvailable",
            Self::DeviceSelected => "DeviceSelected",
        }
    }
    pub const fn binding(self) -> &'static str {
        match self {
            Self::Name => "__runmat_gpudevice_name",
            Self::Index => "__runmat_gpudevice_index",
            Self::TotalMemory => "__runmat_gpudevice_total_memory",
            Self::AvailableMemory => "__runmat_gpudevice_available_memory",
            Self::SupportsDouble => "__runmat_gpudevice_supports_double",
            Self::DeviceAvailable => "__runmat_gpudevice_available",
            Self::DeviceSelected => "__runmat_gpudevice_selected",
        }
    }
    pub fn fact(self) -> runmat_types::ValueFact {
        use runmat_types::*;
        let mut fact = match self {
            Self::Name => ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(1), None]),
                StorageFact::Dense,
            ),
            Self::SupportsDouble | Self::DeviceAvailable | Self::DeviceSelected => {
                ValueFact::scalar(ValueKindFact::Logical)
            }
            _ => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        };
        fact.residency = ResidencyFact::Host;
        fact
    }
}
