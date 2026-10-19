//! Public kmeans spellings decoded once into typed algorithm controls.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KmeansOption {
    Distance,
    EmptyAction,
    MaxIter,
    Replicates,
    Start,
    Display,
    OnlinePhase,
    Options,
}

impl KmeansOption {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "distance" => Some(Self::Distance),
            "emptyaction" => Some(Self::EmptyAction),
            "maxiter" => Some(Self::MaxIter),
            "replicates" => Some(Self::Replicates),
            "start" => Some(Self::Start),
            "display" => Some(Self::Display),
            "onlinephase" => Some(Self::OnlinePhase),
            "options" => Some(Self::Options),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KmeansDistance {
    SqEuclidean,
    Cityblock,
    Cosine,
    Correlation,
    Hamming,
}

impl KmeansDistance {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "sqeuclidean" | "squaredeuclidean" | "squared euclidean" => Some(Self::SqEuclidean),
            "cityblock" | "city block" | "manhattan" => Some(Self::Cityblock),
            "cosine" => Some(Self::Cosine),
            "correlation" => Some(Self::Correlation),
            "hamming" => Some(Self::Hamming),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KmeansEmptyAction {
    Singleton,
    Error,
    Drop,
}

impl KmeansEmptyAction {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "singleton" => Some(Self::Singleton),
            "error" => Some(Self::Error),
            "drop" => Some(Self::Drop),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KmeansStart {
    Plus,
    Sample,
    Uniform,
    Cluster,
}

impl KmeansStart {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "plus" => Some(Self::Plus),
            "sample" => Some(Self::Sample),
            "uniform" => Some(Self::Uniform),
            "cluster" => Some(Self::Cluster),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KmeansDisplay {
    Off,
    Final,
    Iter,
}

impl KmeansDisplay {
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
pub enum KmeansToggle {
    Off,
    On,
}

impl KmeansToggle {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        match keyword.to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "on" => Some(Self::On),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KmeansStatsetField {
    MaxIter,
    Display,
    UseParallel,
    UseSubstreams,
    Streams,
    TolX,
    TolFun,
    Other,
}

impl KmeansStatsetField {
    pub fn from_keyword(keyword: &str) -> Self {
        match keyword.to_ascii_lowercase().as_str() {
            "maxiter" => Self::MaxIter,
            "display" => Self::Display,
            "useparallel" => Self::UseParallel,
            "usesubstreams" => Self::UseSubstreams,
            "streams" => Self::Streams,
            "tolx" => Self::TolX,
            "tolfun" => Self::TolFun,
            _ => Self::Other,
        }
    }
}
