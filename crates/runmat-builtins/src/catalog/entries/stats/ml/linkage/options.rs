#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkageMethod {
    Single,
    Complete,
    Average,
    Weighted,
    Centroid,
    Median,
    Ward,
}

impl LinkageMethod {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "single" | "nearest" => Some(Self::Single),
            "complete" | "farthest" => Some(Self::Complete),
            "average" | "upgma" => Some(Self::Average),
            "weighted" | "wpgma" => Some(Self::Weighted),
            "centroid" | "upgmc" => Some(Self::Centroid),
            "median" | "wpgmc" => Some(Self::Median),
            "ward" | "wardlinkage" => Some(Self::Ward),
            _ => None,
        }
    }

    pub const fn requires_euclidean_distances(self) -> bool {
        matches!(self, Self::Centroid | Self::Median | Self::Ward)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkageOption {
    SaveMemory,
}

impl LinkageOption {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        keyword
            .eq_ignore_ascii_case("savememory")
            .then_some(Self::SaveMemory)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaveMemorySetting {
    On,
    Off,
}

impl SaveMemorySetting {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        if keyword.eq_ignore_ascii_case("on") {
            Some(Self::On)
        } else if keyword.eq_ignore_ascii_case("off") {
            Some(Self::Off)
        } else {
            None
        }
    }
}
