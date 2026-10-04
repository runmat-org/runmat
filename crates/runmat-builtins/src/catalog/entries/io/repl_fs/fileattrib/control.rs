//! One public flag decoder shared by runtime admission and static inference.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileAttributeControl {
    SetReadOnly(bool),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileAttributeControlError {
    ReadPermissionUnavailable,
    UnsupportedFlag(String),
}

impl FileAttributeControl {
    pub fn parse(text: &str) -> Result<Self, FileAttributeControlError> {
        let normalized = text.to_ascii_lowercase();
        match normalized.as_str() {
            "+w" | "w" => Ok(Self::SetReadOnly(false)),
            "-w" => Ok(Self::SetReadOnly(true)),
            "+r" | "r" | "-r" => Err(FileAttributeControlError::ReadPermissionUnavailable),
            _ => Err(FileAttributeControlError::UnsupportedFlag(normalized)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_and_unavailable_flags_share_one_case_insensitive_boundary() {
        assert_eq!(
            FileAttributeControl::parse("+W"),
            Ok(FileAttributeControl::SetReadOnly(false))
        );
        assert_eq!(
            FileAttributeControl::parse("W"),
            Ok(FileAttributeControl::SetReadOnly(false))
        );
        assert_eq!(
            FileAttributeControl::parse("-w"),
            Ok(FileAttributeControl::SetReadOnly(true))
        );
        assert_eq!(
            FileAttributeControl::parse("+R"),
            Err(FileAttributeControlError::ReadPermissionUnavailable)
        );
        assert_eq!(
            FileAttributeControl::parse("-R"),
            Err(FileAttributeControlError::ReadPermissionUnavailable)
        );
        assert_eq!(
            FileAttributeControl::parse("+x"),
            Err(FileAttributeControlError::UnsupportedFlag("+x".into()))
        );
    }
}
