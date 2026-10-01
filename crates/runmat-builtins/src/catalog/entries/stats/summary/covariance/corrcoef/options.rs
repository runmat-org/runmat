#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CorrcoefOptionKey {
    Rows,
    Alpha,
}

impl CorrcoefOptionKey {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "rows" => Some(Self::Rows),
            "alpha" => Some(Self::Alpha),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CorrcoefRowsOption {
    All,
    Complete,
    Pairwise,
}

impl CorrcoefRowsOption {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "all" => Some(Self::All),
            "complete" | "completecase" | "completecases" => Some(Self::Complete),
            "pairwise" | "pairwisecomplete" | "pairwisecompletecase" | "pairwisecompletecases" => {
                Some(Self::Pairwise)
            }
            _ => None,
        }
    }
}
