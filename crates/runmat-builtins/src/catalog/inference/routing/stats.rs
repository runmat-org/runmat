use crate::{BuiltinCatalogEntry, StatsInferenceRule, StatsRandomInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: StatsInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        StatsInferenceRule::Random(StatsRandomInferenceRule::Binomial) => {
            super::super::stats_random::infer_binornd(request, entry)
        }
        StatsInferenceRule::Random(StatsRandomInferenceRule::Gamma) => {
            super::super::stats_random::infer_gamrnd(request, entry)
        }
        StatsInferenceRule::Random(StatsRandomInferenceRule::Weibull) => {
            super::super::stats_random::infer_wblrnd(request, entry)
        }
    }
}
