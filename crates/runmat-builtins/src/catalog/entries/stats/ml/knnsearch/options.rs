use super::super::pairwise_distance::DistanceMetricKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnnOption {
    K,
    IncludeTies,
    SortIndices,
    Distance,
    P,
    Cov,
    Scale,
    NsMethod,
    CacheSize,
    BucketSize,
}

impl KnnOption {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "k" => Some(Self::K),
            "includeties" => Some(Self::IncludeTies),
            "sortindices" => Some(Self::SortIndices),
            "distance" => Some(Self::Distance),
            "p" => Some(Self::P),
            "cov" => Some(Self::Cov),
            "scale" => Some(Self::Scale),
            "nsmethod" => Some(Self::NsMethod),
            "cachesize" => Some(Self::CacheSize),
            "bucketsize" => Some(Self::BucketSize),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnnSearchMethod {
    Exhaustive,
    KdTree,
}

impl KnnSearchMethod {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        if keyword.eq_ignore_ascii_case("exhaustive") {
            Some(Self::Exhaustive)
        } else if keyword.eq_ignore_ascii_case("kdtree") {
            Some(Self::KdTree)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnnCacheSizeKeyword {
    Maximal,
}

impl KnnCacheSizeKeyword {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        keyword
            .eq_ignore_ascii_case("maximal")
            .then_some(Self::Maximal)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnnMetric {
    pub kind: DistanceMetricKind,
    pub fast_alias: bool,
}

impl KnnMetric {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        let (kind, fast_alias) = if keyword.eq_ignore_ascii_case("fasteuclidean") {
            (DistanceMetricKind::Euclidean, true)
        } else if keyword.eq_ignore_ascii_case("fastseuclidean") {
            (DistanceMetricKind::Seuclidean, true)
        } else {
            (DistanceMetricKind::from_keyword(keyword)?, false)
        };
        Some(Self { kind, fast_alias })
    }

    pub const fn allows_kdtree(self) -> bool {
        matches!(
            self.kind,
            DistanceMetricKind::Euclidean
                | DistanceMetricKind::Cityblock
                | DistanceMetricKind::Chebychev
                | DistanceMetricKind::Minkowski
        )
    }
}
