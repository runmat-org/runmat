#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilewriteOptionName {
    Encoding,
    WriteMode,
}

impl FilewriteOptionName {
    pub fn from_label(label: &str) -> Option<Self> {
        if label.eq_ignore_ascii_case("encoding") {
            Some(Self::Encoding)
        } else if label.eq_ignore_ascii_case("writemode") {
            Some(Self::WriteMode)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilewriteEncoding {
    Auto,
    Utf8,
    Ascii,
    Latin1,
    Raw,
}

impl FilewriteEncoding {
    pub fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" | "system" | "native" => Some(Self::Auto),
            "utf-8" | "utf8" | "utf_8" | "unicode" => Some(Self::Utf8),
            "ascii" | "us-ascii" | "us_ascii" | "usascii" => Some(Self::Ascii),
            "latin1" | "latin-1" | "latin_1" | "iso-8859-1" | "iso8859-1" | "iso88591" => {
                Some(Self::Latin1)
            }
            "raw" | "bytes" | "byte" | "binary" => Some(Self::Raw),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilewriteMode {
    Overwrite,
    Append,
}

impl FilewriteMode {
    pub fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "overwrite" | "replace" | "truncate" => Some(Self::Overwrite),
            "append" | "add" => Some(Self::Append),
            _ => None,
        }
    }
}
