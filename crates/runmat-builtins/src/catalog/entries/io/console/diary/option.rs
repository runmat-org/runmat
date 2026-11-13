//! The public diary state/filename spelling is classified once at the boundary.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiaryCommand {
    On,
    Off,
    Filename(String),
}

impl DiaryCommand {
    pub fn decode(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        if trimmed.eq_ignore_ascii_case("on") {
            Some(Self::On)
        } else if trimmed.eq_ignore_ascii_case("off") {
            Some(Self::Off)
        } else if trimmed.is_empty() {
            None
        } else {
            Some(Self::Filename(trimmed.to_owned()))
        }
    }
}
