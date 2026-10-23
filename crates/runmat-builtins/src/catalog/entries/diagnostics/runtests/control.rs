/// The supported in-program test controls, shared by static and runtime admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntestsOption {
    IncludeSubfolders,
    UseParallel,
    BaseFolder,
    Name,
    ProcedureName,
    OutputDetail,
    LoggingLevel,
    Tag,
    Coverage,
}

impl RuntestsOption {
    pub fn parse(name: &str) -> Option<Self> {
        let normalized: String = name
            .chars()
            .filter(|ch| !ch.is_ascii_whitespace() && *ch != '_' && *ch != '-')
            .flat_map(char::to_lowercase)
            .collect();
        Some(match normalized.as_str() {
            "includesubfolders" => Self::IncludeSubfolders,
            "useparallel" => Self::UseParallel,
            "basefolder" => Self::BaseFolder,
            "name" => Self::Name,
            "procedurename" => Self::ProcedureName,
            "outputdetail" => Self::OutputDetail,
            "logginglevel" => Self::LoggingLevel,
            "tag" => Self::Tag,
            "coverage" => Self::Coverage,
            _ => return None,
        })
    }
}
