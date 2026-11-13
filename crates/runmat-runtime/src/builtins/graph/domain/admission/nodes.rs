//! Node selector admission before graph materialization and owner downloads.

use super::super::parsing::{node_index_from_one_based, numeric_vector, parse_string_vector};
use super::super::{gather_if_needed_async, graph_error, BuiltinResult, GraphData, Value};
use super::is_host_value;

pub(crate) enum NodeSelector {
    Numeric(Vec<usize>),
    Named(Vec<String>),
    Resident(Value),
}

fn parse_host_selector(
    value: &Value,
    scalar: bool,
    name: &'static str,
) -> BuiltinResult<NodeSelector> {
    let selector = match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
            NodeSelector::Named(
                parse_string_vector(value).map_err(|error| graph_error(name, error))?,
            )
        }
        _ => NodeSelector::Numeric(numeric_vector(value, name)?),
    };
    if scalar {
        let len = match &selector {
            NodeSelector::Named(names) => names.len(),
            NodeSelector::Numeric(indices) => indices.len(),
            NodeSelector::Resident(_) => unreachable!(),
        };
        if len != 1 {
            let message = if matches!(selector, NodeSelector::Named(_)) {
                format!("{name}: expected one node")
            } else {
                format!("{name}: expected scalar node count")
            };
            return Err(graph_error(name, message));
        }
    }
    Ok(selector)
}

fn admit_selector(value: Value, scalar: bool, name: &'static str) -> BuiltinResult<NodeSelector> {
    if is_host_value(&value) {
        parse_host_selector(&value, scalar, name)
    } else {
        Ok(NodeSelector::Resident(value))
    }
}

pub(in crate::builtins::graph::domain) fn admit_scalar_selectors<I>(
    values: I,
    name: &'static str,
) -> BuiltinResult<Vec<NodeSelector>>
where
    I: IntoIterator<Item = Value>,
    I::IntoIter: ExactSizeIterator,
{
    values
        .into_iter()
        .map(|value| admit_selector(value, true, name))
        .collect()
}

pub(in crate::builtins::graph::domain) fn admit_distance_selectors<I>(
    values: I,
) -> BuiltinResult<Vec<NodeSelector>>
where
    I: IntoIterator<Item = Value>,
    I::IntoIter: ExactSizeIterator,
{
    let values = values.into_iter();
    if values.len() > 2 {
        return Err(graph_error(
            "distances",
            "distances: expected at most source and target node lists",
        ));
    }
    values
        .map(|value| admit_selector(value, false, "distances"))
        .collect()
}

async fn resolve_selector(
    graph: &GraphData,
    selector: NodeSelector,
    scalar: bool,
    name: &'static str,
) -> BuiltinResult<Vec<usize>> {
    let admitted = match selector {
        NodeSelector::Resident(value) => {
            let gathered = gather_if_needed_async(&value).await?;
            parse_host_selector(&gathered, scalar, name)?
        }
        host => host,
    };
    match admitted {
        NodeSelector::Numeric(indices) => indices
            .into_iter()
            .map(|index| node_index_from_one_based(index, graph.node_count, name))
            .collect(),
        NodeSelector::Named(labels) => {
            let Some(names) = &graph.node_names else {
                return Err(graph_error(
                    name,
                    format!("{name}: string node references require named graph nodes"),
                ));
            };
            labels
                .into_iter()
                .map(|label| {
                    names
                        .iter()
                        .position(|candidate| candidate == &label)
                        .ok_or_else(|| graph_error(name, format!("{name}: unknown node '{label}'")))
                })
                .collect()
        }
        NodeSelector::Resident(_) => unreachable!(),
    }
}

pub(in crate::builtins::graph::domain) async fn resolve_scalar_selectors(
    graph: &GraphData,
    selectors: Vec<NodeSelector>,
    name: &'static str,
) -> BuiltinResult<Vec<usize>> {
    let mut nodes = Vec::with_capacity(selectors.len());
    for selector in selectors {
        nodes.push(resolve_selector(graph, selector, true, name).await?[0]);
    }
    Ok(nodes)
}

pub(in crate::builtins::graph::domain) async fn resolve_distance_selectors(
    graph: &GraphData,
    selectors: Vec<NodeSelector>,
) -> BuiltinResult<(Vec<usize>, Vec<usize>)> {
    let mut selectors = selectors.into_iter();
    let sources = match selectors.next() {
        Some(selector) => resolve_selector(graph, selector, false, "distances").await?,
        None => (0..graph.node_count).collect(),
    };
    let targets = match selectors.next() {
        Some(selector) => resolve_selector(graph, selector, false, "distances").await?,
        None => (0..graph.node_count).collect(),
    };
    Ok((sources, targets))
}
