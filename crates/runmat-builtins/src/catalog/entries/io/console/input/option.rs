#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputStringMode {
    RawText,
}

impl InputStringMode {
    pub fn decode(text: &str) -> Option<Self> {
        (text == "s").then_some(Self::RawText)
    }
}
