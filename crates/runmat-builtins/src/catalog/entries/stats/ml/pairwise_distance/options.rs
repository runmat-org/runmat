//! Pairwise-distance selection identities; names are decoded only at the public boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceSelectionKind {
    Smallest,
    Largest,
}

impl DistanceSelectionKind {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        if keyword.eq_ignore_ascii_case("smallest") {
            Some(Self::Smallest)
        } else if keyword.eq_ignore_ascii_case("largest") {
            Some(Self::Largest)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceControlKind {
    CacheSize,
}

impl DistanceControlKind {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        keyword
            .eq_ignore_ascii_case("cachesize")
            .then_some(Self::CacheSize)
    }
}

/// The named metric is decoded once from public text, before metric-specific controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistanceMetricKind {
    Euclidean,
    SquareEuclidean,
    Cityblock,
    Chebychev,
    Minkowski,
    Seuclidean,
    Mahalanobis,
    Cosine,
    Correlation,
    Hamming,
    Jaccard,
    Spearman,
}

impl DistanceMetricKind {
    pub const fn accepts_parameter(self) -> bool {
        matches!(self, Self::Minkowski | Self::Seuclidean | Self::Mahalanobis)
    }

    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "euclidean" => Some(Self::Euclidean),
            "squaredeuclidean" | "sqeuclidean" | "squared euclidean" => Some(Self::SquareEuclidean),
            "cityblock" | "city block" | "manhattan" => Some(Self::Cityblock),
            "chebychev" | "chebyshev" | "cheby" => Some(Self::Chebychev),
            "minkowski" => Some(Self::Minkowski),
            "seuclidean" | "standardizedeuclidean" | "standardized euclidean" => {
                Some(Self::Seuclidean)
            }
            "mahalanobis" | "mahal" => Some(Self::Mahalanobis),
            "cosine" => Some(Self::Cosine),
            "correlation" => Some(Self::Correlation),
            "hamming" => Some(Self::Hamming),
            "jaccard" => Some(Self::Jaccard),
            "spearman" => Some(Self::Spearman),
            _ => None,
        }
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::SquareEuclidean => "squaredeuclidean",
            Self::Cityblock => "cityblock",
            Self::Chebychev => "chebychev",
            Self::Minkowski => "minkowski",
            Self::Seuclidean => "seuclidean",
            Self::Mahalanobis => "mahalanobis",
            Self::Cosine => "cosine",
            Self::Correlation => "correlation",
            Self::Hamming => "hamming",
            Self::Jaccard => "jaccard",
            Self::Spearman => "spearman",
        }
    }
}
