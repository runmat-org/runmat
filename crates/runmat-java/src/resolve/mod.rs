use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JavaArgumentType {
    Null,
    Boolean,
    Byte,
    Short,
    Int,
    Long,
    Float,
    Double,
    Char,
    String,
    Object(String),
    Array(Box<JavaArgumentType>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JavaParameterType {
    Boolean,
    Byte,
    Short,
    Int,
    Long,
    Float,
    Double,
    Char,
    String,
    Object(String),
    Array(Box<JavaParameterType>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JavaCallableCandidate {
    pub identity: String,
    pub parameters: Vec<JavaParameterType>,
    pub varargs: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedOverload {
    pub candidate_index: usize,
    pub identity: String,
    pub conversion_cost: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OverloadError {
    #[error("no Java overload accepts the supplied arguments")]
    NoMatch,
    #[error("Java overload is ambiguous between {0}")]
    Ambiguous(String),
}

pub fn select_overload(
    candidates: &[JavaCallableCandidate],
    arguments: &[JavaArgumentType],
) -> Result<SelectedOverload, OverloadError> {
    let mut feasible = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            candidate_cost(candidate, arguments).map(|cost| SelectedOverload {
                candidate_index: index,
                identity: candidate.identity.clone(),
                conversion_cost: cost,
            })
        })
        .collect::<Vec<_>>();
    feasible.sort_by(|left, right| {
        left.conversion_cost
            .cmp(&right.conversion_cost)
            .then_with(|| left.identity.cmp(&right.identity))
    });
    let Some(best) = feasible.first().cloned() else {
        return Err(OverloadError::NoMatch);
    };
    let tied = feasible
        .iter()
        .take_while(|candidate| candidate.conversion_cost == best.conversion_cost)
        .collect::<Vec<_>>();
    if tied.len() > 1 {
        return Err(OverloadError::Ambiguous(
            tied.iter()
                .map(|candidate| candidate.identity.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    Ok(best)
}

fn candidate_cost(
    candidate: &JavaCallableCandidate,
    arguments: &[JavaArgumentType],
) -> Option<u32> {
    if (!candidate.varargs && candidate.parameters.len() != arguments.len())
        || (candidate.varargs && arguments.len() + 1 < candidate.parameters.len())
    {
        return None;
    }
    let fixed_count = if candidate.varargs {
        candidate.parameters.len().saturating_sub(1)
    } else {
        candidate.parameters.len()
    };
    let mut cost = 0;
    for (argument, parameter) in arguments
        .iter()
        .zip(candidate.parameters.iter())
        .take(fixed_count)
    {
        cost += conversion_cost(argument, parameter)?;
    }
    if candidate.varargs {
        let JavaParameterType::Array(component) = candidate.parameters.last()? else {
            return None;
        };
        cost += 8;
        for argument in &arguments[fixed_count..] {
            cost += conversion_cost(argument, component)?;
        }
    }
    Some(cost)
}

fn conversion_cost(argument: &JavaArgumentType, parameter: &JavaParameterType) -> Option<u32> {
    use JavaArgumentType as A;
    use JavaParameterType as P;
    match (argument, parameter) {
        (A::Boolean, P::Boolean)
        | (A::Byte, P::Byte)
        | (A::Short, P::Short)
        | (A::Int, P::Int)
        | (A::Long, P::Long)
        | (A::Float, P::Float)
        | (A::Double, P::Double)
        | (A::Char, P::Char)
        | (A::String, P::String) => Some(0),
        (A::Null, P::String | P::Object(_) | P::Array(_)) => Some(1),
        (A::String, P::Object(name)) if name == "java.lang.String" => Some(1),
        (A::String, P::Object(name)) if name == "java.lang.Object" => Some(3),
        (A::Object(actual), P::Object(expected)) if actual == expected => Some(0),
        (A::Object(_), P::Object(expected)) if expected == "java.lang.Object" => Some(4),
        (A::Array(actual), P::Array(expected)) => {
            conversion_cost(actual, expected).map(|cost| cost + 1)
        }
        (A::Array(_), P::Object(expected))
            if expected == "java.lang.Object" || expected == "java.lang.Cloneable" =>
        {
            Some(5)
        }
        _ => numeric_widening_cost(argument, parameter),
    }
}

fn numeric_widening_cost(
    argument: &JavaArgumentType,
    parameter: &JavaParameterType,
) -> Option<u32> {
    let argument_rank = numeric_rank_argument(argument)?;
    let parameter_rank = numeric_rank_parameter(parameter)?;
    match argument_rank.cmp(&parameter_rank) {
        Ordering::Equal => Some(0),
        Ordering::Less => Some(u32::from(parameter_rank - argument_rank)),
        Ordering::Greater => None,
    }
}

fn numeric_rank_argument(argument: &JavaArgumentType) -> Option<u8> {
    match argument {
        JavaArgumentType::Byte => Some(1),
        JavaArgumentType::Short | JavaArgumentType::Char => Some(2),
        JavaArgumentType::Int => Some(3),
        JavaArgumentType::Long => Some(4),
        JavaArgumentType::Float => Some(5),
        JavaArgumentType::Double => Some(6),
        _ => None,
    }
}

fn numeric_rank_parameter(parameter: &JavaParameterType) -> Option<u8> {
    match parameter {
        JavaParameterType::Byte => Some(1),
        JavaParameterType::Short => Some(2),
        JavaParameterType::Int => Some(3),
        JavaParameterType::Long => Some(4),
        JavaParameterType::Float => Some(5),
        JavaParameterType::Double => Some(6),
        JavaParameterType::Char => Some(2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_beats_numeric_widening() {
        let candidates = vec![
            JavaCallableCandidate {
                identity: "fixture(int)".into(),
                parameters: vec![JavaParameterType::Int],
                varargs: false,
            },
            JavaCallableCandidate {
                identity: "fixture(long)".into(),
                parameters: vec![JavaParameterType::Long],
                varargs: false,
            },
        ];
        let selected = select_overload(&candidates, &[JavaArgumentType::Int]).unwrap();
        assert_eq!(selected.identity, "fixture(int)");
    }

    #[test]
    fn equal_cost_is_reported_as_ambiguous() {
        let candidates = vec![
            JavaCallableCandidate {
                identity: "fixture(java.lang.String)".into(),
                parameters: vec![JavaParameterType::String],
                varargs: false,
            },
            JavaCallableCandidate {
                identity: "fixture(java.lang.Integer)".into(),
                parameters: vec![JavaParameterType::Object("java.lang.Integer".into())],
                varargs: false,
            },
        ];
        assert!(matches!(
            select_overload(&candidates, &[JavaArgumentType::Null]),
            Err(OverloadError::Ambiguous(_))
        ));
    }
}
