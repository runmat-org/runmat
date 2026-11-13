#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RunMatUtf8View {
    pub data: *const u8,
    pub length: usize,
}

impl RunMatUtf8View {
    pub const EMPTY: Self = Self {
        data: core::ptr::null(),
        length: 0,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RunMatErrorView {
    pub identifier: RunMatUtf8View,
    pub message: RunMatUtf8View,
}
