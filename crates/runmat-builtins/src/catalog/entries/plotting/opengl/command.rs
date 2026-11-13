//! Public option spelling is decoded once into a typed compatibility command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenGlCommand {
    Report,
    Hardware,
    HardwareBasic,
    Software,
    AutoSelect,
    NeverSelect,
    Save,
}
impl OpenGlCommand {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text.trim().to_ascii_lowercase().as_str() {
            "info" | "data" => Self::Report,
            "hardware" => Self::Hardware,
            "hardwarebasic" => Self::HardwareBasic,
            "software" => Self::Software,
            "autoselect" => Self::AutoSelect,
            "neverselect" => Self::NeverSelect,
            "save" => Self::Save,
            _ => return None,
        })
    }
    pub fn status(self) -> Option<&'static str> {
        match self {
            Self::Hardware | Self::AutoSelect => Some("hardware"),
            Self::HardwareBasic => Some("hardwarebasic"),
            Self::Software => Some("software"),
            Self::NeverSelect => Some("neverselect"),
            Self::Report | Self::Save => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenGlSaveMode {
    Hardware,
    HardwareBasic,
    Software,
    AutoSelect,
    None,
}
impl OpenGlSaveMode {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text.trim().to_ascii_lowercase().as_str() {
            "hardware" => Self::Hardware,
            "hardwarebasic" => Self::HardwareBasic,
            "software" => Self::Software,
            "autoselect" => Self::AutoSelect,
            "none" => Self::None,
            _ => return None,
        })
    }
}
