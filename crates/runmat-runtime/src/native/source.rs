/// Stable source location. It contains no host path or Rust string layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeSourceLocation {
    pub source: u32,
    pub reserved: u32,
    pub start: u64,
    pub end: u64,
}

/// Borrowed UTF-8 bytes owned by the executable product.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeUtf8 {
    pub bytes: *const u8,
    pub len: usize,
}

impl Default for NativeUtf8 {
    fn default() -> Self {
        Self {
            bytes: std::ptr::null(),
            len: 0,
        }
    }
}

impl NativeUtf8 {
    pub fn validate(self) -> Result<(), super::NativeAbiError> {
        super::validation::validate_slice("native.utf8", self.bytes, self.len)
    }
}

impl NativeSourceLocation {
    pub fn validate(self) -> Result<(), super::NativeAbiError> {
        if self.reserved != 0 {
            return Err(super::NativeAbiError::new(
                "native.source_location.reserved",
                "reserved bits must be zero for this ABI revision",
            ));
        }
        if self.end < self.start {
            return Err(super::NativeAbiError::new(
                "native.source_location",
                "source range end precedes its start",
            ));
        }
        Ok(())
    }
}

impl NativeSourceMapEntry {
    pub fn validate(self) -> Result<(), super::NativeAbiError> {
        if self.reserved != 0 {
            return Err(super::NativeAbiError::new(
                "native.source_map_entry.reserved",
                "reserved bits must be zero for this ABI revision",
            ));
        }
        self.owner_identity.validate()?;
        self.relative_path.validate()?;
        self.display_name.validate()
    }
}

/// Path-independent source-map projection for diagnostics and stack traces.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeSourceMapEntry {
    pub source: u32,
    pub reserved: u32,
    pub owner_identity: NativeUtf8,
    pub relative_path: NativeUtf8,
    pub display_name: NativeUtf8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeSourceMapView {
    pub entries: *const NativeSourceMapEntry,
    pub count: usize,
}

impl Default for NativeSourceMapView {
    fn default() -> Self {
        Self {
            entries: std::ptr::null(),
            count: 0,
        }
    }
}
