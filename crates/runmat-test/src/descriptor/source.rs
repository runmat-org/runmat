use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SourceDescriptor {
    pub owner_identity: String,
    pub relative_path: String,
    pub semantic_path: String,
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_byte: u32,
    pub end_byte: u32,
    pub start_line: u32,
    pub start_column: u32,
    pub end_line: u32,
    pub end_column: u32,
}

impl SourceSpan {
    pub fn is_valid(self) -> bool {
        self.start_byte <= self.end_byte
            && (self.start_line, self.start_column) <= (self.end_line, self.end_column)
    }
}
