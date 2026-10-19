//! Public t-SNE spellings decoded into typed controls at the call boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TsneOption {
    Algorithm,
    CacheSize,
    Distance,
    Exaggeration,
    NumDimensions,
    NumPcaComponents,
    Perplexity,
    Standardize,
    InitialY,
    LearnRate,
    NumPrint,
    Options,
    Theta,
    Verbose,
}

impl TsneOption {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "algorithm" => Some(Self::Algorithm),
            "cachesize" => Some(Self::CacheSize),
            "distance" => Some(Self::Distance),
            "exaggeration" => Some(Self::Exaggeration),
            "numdimensions" => Some(Self::NumDimensions),
            "numpcacomponents" => Some(Self::NumPcaComponents),
            "perplexity" => Some(Self::Perplexity),
            "standardize" => Some(Self::Standardize),
            "initialy" => Some(Self::InitialY),
            "learnrate" => Some(Self::LearnRate),
            "numprint" => Some(Self::NumPrint),
            "options" => Some(Self::Options),
            "theta" => Some(Self::Theta),
            "verbose" => Some(Self::Verbose),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TsneAlgorithm {
    Exact,
    BarnesHut,
}

impl TsneAlgorithm {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "exact" => Some(Self::Exact),
            "barneshut" => Some(Self::BarnesHut),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TsneDistance {
    Euclidean,
    Seuclidean,
    Cityblock,
    Chebychev,
    Minkowski,
    Mahalanobis,
    Cosine,
    Correlation,
    Spearman,
    Hamming,
    Jaccard,
}

impl TsneDistance {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "euclidean" | "fasteuclidean" => Some(Self::Euclidean),
            "seuclidean" | "fastseuclidean" => Some(Self::Seuclidean),
            "cityblock" => Some(Self::Cityblock),
            "chebychev" => Some(Self::Chebychev),
            "minkowski" => Some(Self::Minkowski),
            "mahalanobis" => Some(Self::Mahalanobis),
            "cosine" => Some(Self::Cosine),
            "correlation" => Some(Self::Correlation),
            "spearman" => Some(Self::Spearman),
            "hamming" => Some(Self::Hamming),
            "jaccard" => Some(Self::Jaccard),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TsneStatsetField {
    MaxIter,
    TolFun,
    Display,
    OutputFcn,
}

impl TsneStatsetField {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "maxiter" => Some(Self::MaxIter),
            "tolfun" => Some(Self::TolFun),
            "display" => Some(Self::Display),
            "outputfcn" => Some(Self::OutputFcn),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TsneDisplay {
    Off,
    Final,
    Iter,
}

impl TsneDisplay {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "final" => Some(Self::Final),
            "iter" => Some(Self::Iter),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TsneCacheSizeKeyword {
    Maximal,
}

impl TsneCacheSizeKeyword {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        keyword
            .eq_ignore_ascii_case("maximal")
            .then_some(Self::Maximal)
    }
}
