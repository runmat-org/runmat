/// Closed MAT object constructor control. Spelling is decoded once at the
/// public boundary and is not used as an internal execution identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatFileOptionName {
    Writable,
}

impl MatFileOptionName {
    pub fn parse(text: &str) -> Option<Self> {
        text.eq_ignore_ascii_case("Writable")
            .then_some(Self::Writable)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatLoadOption {
    Mat,
    Regexp,
}

impl MatLoadOption {
    pub fn parse(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("-mat") {
            Some(Self::Mat)
        } else if text.eq_ignore_ascii_case("-regexp") {
            Some(Self::Regexp)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatSaveOption {
    Append,
    Struct,
    Regexp,
}

impl MatSaveOption {
    pub fn parse(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("-append") {
            Some(Self::Append)
        } else if text.eq_ignore_ascii_case("-struct") {
            Some(Self::Struct)
        } else if text.eq_ignore_ascii_case("-regexp") {
            Some(Self::Regexp)
        } else {
            None
        }
    }
}
