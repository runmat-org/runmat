#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RunMatForeignHandle {
    pub host: u64,
    pub resource: u64,
    pub generation: u64,
}

impl RunMatForeignHandle {
    pub const INVALID: Self = Self {
        host: 0,
        resource: 0,
        generation: 0,
    };

    pub const fn is_valid(self) -> bool {
        self.host != 0 && self.resource != 0 && self.generation != 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RunMatValueHandle {
    pub resource: u64,
    pub generation: u64,
}

impl RunMatValueHandle {
    pub const INVALID: Self = Self {
        resource: 0,
        generation: 0,
    };

    pub const fn is_valid(self) -> bool {
        self.resource != 0 && self.generation != 0
    }
}
