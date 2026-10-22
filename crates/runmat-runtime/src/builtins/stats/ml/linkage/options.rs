//! Public linkage options decoded into typed clustering controls.

use runmat_builtins::{DistanceMetricKind, LinkageMethod, LinkageOption, SaveMemorySetting};
use runmat_value::Value;

use super::invalid;
use crate::builtins::common::random_args::keyword_of;
use crate::BuiltinResult;

#[derive(Clone, Debug)]
pub(super) struct LinkageOptions {
    pub(super) method: LinkageMethod,
    pub(super) metric: Option<LinkageMetric>,
    pub(super) savememory: Option<bool>,
}

#[derive(Clone, Debug)]
pub(super) struct LinkageMetric {
    pub(super) kind: DistanceMetricKind,
    pub(super) parameters: Vec<Value>,
}

pub(super) fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<LinkageOptions> {
    let mut method = LinkageMethod::Single;
    let mut metric_args = Vec::new();
    let mut savememory = None;
    let mut idx = 0usize;

    if let Some(first) = args.first() {
        if let Some(text) = keyword_of(first)? {
            if LinkageOption::from_keyword(&text).is_none() {
                method = LinkageMethod::from_keyword(&text).ok_or_else(|| {
                    invalid(format!("linkage: unsupported linkage method '{text}'"))
                })?;
                idx = 1;
            }
        } else {
            return Err(invalid("linkage: method must be text"));
        }
    }

    while idx < args.len() {
        if let Some(keyword) = keyword_of(&args[idx])? {
            if LinkageOption::from_keyword(&keyword) == Some(LinkageOption::SaveMemory) {
                if idx + 1 >= args.len() {
                    return Err(invalid("linkage: SaveMemory must be paired with a value"));
                }
                if savememory.is_some() {
                    return Err(invalid("linkage: SaveMemory can only be specified once"));
                }
                savememory = Some(parse_savememory(&args[idx + 1])?);
                idx += 2;
                continue;
            }
        }

        if let Value::Cell(cell) = &args[idx] {
            if !metric_args.is_empty() {
                return Err(invalid(
                    "linkage: pdist input cell must be the complete distance metric argument",
                ));
            }
            metric_args.extend(cell.data().iter().cloned());
        } else {
            metric_args.push(args[idx].clone());
        }
        idx += 1;
    }

    let metric = if metric_args.is_empty() {
        None
    } else {
        let keyword = keyword_of(&metric_args[0])?
            .ok_or_else(|| invalid("linkage: distance metric must be text"))?;
        let kind = DistanceMetricKind::from_keyword(&keyword)
            .ok_or_else(|| invalid(format!("linkage: unsupported distance metric '{keyword}'")))?;
        Some(LinkageMetric {
            kind,
            parameters: metric_args.into_iter().skip(1).collect(),
        })
    };
    Ok(LinkageOptions {
        method,
        metric,
        savememory,
    })
}

fn parse_savememory(value: &Value) -> BuiltinResult<bool> {
    if let Some(text) = keyword_of(value)? {
        match SaveMemorySetting::from_keyword(&text) {
            Some(SaveMemorySetting::On) => return Ok(true),
            Some(SaveMemorySetting::Off) => return Ok(false),
            None => {
                return Err(invalid(format!(
                    "linkage: SaveMemory value must be 'on' or 'off', got '{text}'"
                )))
            }
        }
    }
    Err(invalid("linkage: SaveMemory value must be 'on' or 'off'"))
}

pub(super) fn validate_savememory_for_matrix(options: &LinkageOptions) -> BuiltinResult<()> {
    if options.savememory != Some(true) {
        return Ok(());
    }
    if !options.method.requires_euclidean_distances() {
        return Err(invalid(
            "linkage: SaveMemory 'on' is only valid for centroid, median, or ward linkage",
        ));
    }
    if options.metric.as_ref().is_some_and(|metric| {
        metric.kind != DistanceMetricKind::Euclidean || !metric.parameters.is_empty()
    }) {
        return Err(invalid(
            "linkage: SaveMemory 'on' requires the Euclidean distance metric",
        ));
    }
    Ok(())
}
