#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CoveragePlan {
    sites: Vec<CoverageSite>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoverageSite {
    pub semantic_path: String,
    pub source_id: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}

impl CoveragePlan {
    pub fn sites(&self) -> &[CoverageSite] {
        &self.sites
    }
}
