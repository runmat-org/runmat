use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StatsInferenceRule {
    Random(StatsRandomInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StatsRandomInferenceRule {
    Binomial,
    Gamma,
    Weibull,
}
