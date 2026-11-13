use super::*;

pub(super) async fn parse_topk_args(
    kind: TopKKind,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<TopKArgs> {
    if rest.is_empty() {
        return Err(topk_invalid_argument(kind, "k is required"));
    }
    let k = parse_k(kind, &rest[0]).await?;
    let mut dim = None;
    let mut comparison = ComparisonMethod::Auto;
    let mut idx = 1usize;
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx])? {
            if keyword == "comparisonmethod" {
                let Some(value) = rest.get(idx + 1) else {
                    return Err(topk_invalid_argument(
                        kind,
                        "expected a value after 'ComparisonMethod'",
                    ));
                };
                comparison = parse_comparison(kind, value)?;
                idx += 2;
                continue;
            }
        }
        if dim.is_none() {
            let host = gpu_helpers::gather_value_async(&rest[idx])
                .await
                .map_err(|error| topk_invalid_argument(kind, error.message()))?;
            if let Some(parsed) = tensor::dimension_from_value_async(&host, kind.name(), false)
                .await
                .map_err(|message| topk_invalid_argument(kind, message))?
            {
                dim = Some(parsed);
                idx += 1;
                continue;
            }
        }
        return Err(topk_invalid_argument(kind, "unrecognized argument"));
    }
    Ok(TopKArgs { k, dim, comparison })
}

pub(super) async fn parse_k(kind: TopKKind, value: &Value) -> BuiltinResult<usize> {
    // A resident integer selector must cross the owner boundary as native
    // integer storage, not through the generic scalar-double adapter.
    let host = gpu_helpers::gather_value_async(value)
        .await
        .map_err(|error| topk_invalid_argument(kind, error.message()))?;
    if let Some(integer) = tensor::scalar_integer_value(&host) {
        return parse_integer_k(kind, &integer);
    }
    let Some(raw) = tensor::scalar_f64_from_value_async(&host)
        .await
        .map_err(|message| topk_invalid_argument(kind, message))?
    else {
        return Err(topk_invalid_argument(kind, "k must be a numeric scalar"));
    };
    if !raw.is_finite() {
        return Err(topk_invalid_argument(kind, "k must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-6 {
        return Err(topk_invalid_argument(kind, "k must be an integer"));
    }
    if rounded < 0.0 {
        return Err(topk_invalid_argument(kind, "k must be nonnegative"));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(topk_invalid_argument(
            kind,
            "k is outside the supported range",
        ));
    }
    Ok(rounded as usize)
}

pub(super) fn parse_integer_k(kind: TopKKind, value: &IntValue) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        topk_invalid_argument(
            kind,
            "k must be a nonnegative integer in the supported range",
        )
    })
}

pub(super) fn parse_comparison(kind: TopKKind, value: &Value) -> BuiltinResult<ComparisonMethod> {
    let Some(keyword) = keyword_of(value)? else {
        return Err(topk_invalid_argument(
            kind,
            "'ComparisonMethod' expects a string value",
        ));
    };
    match keyword.as_str() {
        "auto" => Ok(ComparisonMethod::Auto),
        "abs" | "magnitude" => Ok(ComparisonMethod::Abs),
        "real" => Ok(ComparisonMethod::Real),
        other => Err(topk_invalid_argument(
            kind,
            format!("unsupported ComparisonMethod '{other}'"),
        )),
    }
}
