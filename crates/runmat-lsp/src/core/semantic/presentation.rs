use std::fmt::Write;

use runmat_types::{DimensionFact, ShapeFact, ValueFact, ValueKindFact};

pub fn format_fact(fact: Option<&ValueFact>) -> String {
    let Some(fact) = fact else {
        return "unknown".to_string();
    };
    let kind = format_kind(&fact.kind);
    let shape = format_shape(&fact.shape);
    if matches!(fact.shape, ShapeFact::Unknown) {
        kind
    } else {
        format!("{kind} {shape}")
    }
}

pub fn format_variable_hover(
    name: &str,
    kind: &str,
    fact: Option<&ValueFact>,
    global: bool,
) -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "```runmat\n{kind} {name}: {}\n```",
        format_fact(fact)
    );
    if global {
        let _ = writeln!(output, "Global variable available across the workspace.");
    }
    output
}

fn format_kind(kind: &ValueKindFact) -> String {
    match kind {
        ValueKindFact::Never => "never".to_string(),
        ValueKindFact::Unknown => "unknown".to_string(),
        ValueKindFact::Void => "void".to_string(),
        ValueKindFact::Numeric(numeric) => {
            let class = format!("{:?}", numeric.class).to_ascii_lowercase();
            if matches!(numeric.domain, runmat_types::NumericDomain::Complex) {
                format!("complex {class}")
            } else {
                class
            }
        }
        ValueKindFact::Logical => "logical".to_string(),
        ValueKindFact::Character => "character".to_string(),
        ValueKindFact::String => "string".to_string(),
        ValueKindFact::Symbolic => "symbolic".to_string(),
        ValueKindFact::Cell(_) => "cell".to_string(),
        ValueKindFact::Struct(_) => "struct".to_string(),
        ValueKindFact::Object(object) => object.runtime_class.as_ref().map_or_else(
            || "object".to_string(),
            |class| format!("object<{}>", qualified_name(class)),
        ),
        ValueKindFact::ClassReference(class) => class.runtime_class.as_ref().map_or_else(
            || format!("class<{:?}>", class.class),
            |name| format!("class<{}>", qualified_name(name)),
        ),
        ValueKindFact::Callable(_) => "callable".to_string(),
        ValueKindFact::OutputList(_) => "output-list".to_string(),
        ValueKindFact::Exception(_) => "exception".to_string(),
        ValueKindFact::Execution(execution) => match execution {
            runmat_types::ExecutionFact::Future { .. } => "future".to_string(),
            runmat_types::ExecutionFact::Task { .. } => "task".to_string(),
            runmat_types::ExecutionFact::Pool => "pool".to_string(),
            runmat_types::ExecutionFact::Job { .. } => "job".to_string(),
        },
        ValueKindFact::Distributed(_) => "distributed".to_string(),
        ValueKindFact::Foreign(foreign) => foreign.type_name.as_ref().map_or_else(
            || format!("foreign<{}>", foreign.family),
            |name| format!("foreign<{name}>"),
        ),
    }
}

fn qualified_name(name: &runmat_types::QualifiedName) -> String {
    name.display_name().unwrap_or_else(|| "unknown".to_string())
}

fn format_shape(shape: &ShapeFact) -> String {
    match shape {
        ShapeFact::Unknown => "unknown shape".to_string(),
        ShapeFact::Scalar => "[1 x 1]".to_string(),
        ShapeFact::Ranked { rank } => format!("[rank {rank}]"),
        ShapeFact::Shaped { dims } => format!(
            "[{}]",
            dims.iter()
                .map(|dimension| match dimension {
                    DimensionFact::Known(value) => value.to_string(),
                    DimensionFact::Symbolic(symbol) => symbol.0.clone(),
                    DimensionFact::Unknown => "?".to_string(),
                })
                .collect::<Vec<_>>()
                .join(" x ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unknown() -> ValueFact {
        ValueFact::unknown(runmat_types::DynamicReason::Unspecified)
    }

    #[test]
    fn formats_structured_numeric_fact_without_legacy_type() {
        let fact = ValueFact::proven(
            ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            runmat_types::StorageFact::Dense,
        );
        assert_eq!(format_fact(Some(&fact)), "double [2 x 3]");
    }

    #[test]
    fn variable_hover_does_not_expose_internal_execution_facts() {
        let fact = ValueFact::proven(
            ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            runmat_types::StorageFact::Dense,
        );

        let hover = format_variable_hover("x", "workspace", Some(&fact), false);

        assert!(hover.contains("workspace x: double [2 x 3]"));
        assert!(!hover.contains("Storage"));
        assert!(!hover.contains("Layout"));
        assert!(!hover.contains("Residency"));
        assert!(!hover.contains("Certainty"));
        assert!(!hover.contains("Proven"));
        assert!(!hover.contains("Invalidated by"));
    }

    #[test]
    fn formats_every_shared_value_kind_without_collapsing_the_taxonomy() {
        use std::collections::BTreeMap;

        let cases = vec![
            (ValueKindFact::Never, "never"),
            (ValueKindFact::Unknown, "unknown"),
            (ValueKindFact::Void, "void"),
            (
                ValueKindFact::Numeric(runmat_types::NumericFact {
                    class: runmat_types::NumericClass::Single,
                    domain: runmat_types::NumericDomain::Complex,
                }),
                "complex single",
            ),
            (ValueKindFact::Logical, "logical"),
            (ValueKindFact::Character, "character"),
            (ValueKindFact::String, "string"),
            (ValueKindFact::Symbolic, "symbolic"),
            (
                ValueKindFact::Cell(runmat_types::CellFact {
                    element: Box::new(unknown()),
                    elements: vec![unknown()],
                    elements_complete: true,
                }),
                "cell",
            ),
            (
                ValueKindFact::Struct(runmat_types::StructFact {
                    fields: BTreeMap::new(),
                    fields_complete: true,
                }),
                "struct",
            ),
            (
                ValueKindFact::Object(runmat_types::ObjectFact {
                    class: None,
                    runtime_class: None,
                    properties: BTreeMap::new(),
                    properties_complete: false,
                    handle_semantics: None,
                }),
                "object",
            ),
            (
                ValueKindFact::ClassReference(runmat_types::ClassReferenceFact {
                    class: None,
                    runtime_class: None,
                }),
                "class<None>",
            ),
            (
                ValueKindFact::Callable(runmat_types::CallableFact {
                    identity: None,
                    parameters: vec![unknown()],
                    parameters_complete: false,
                    outputs: vec![unknown()],
                    outputs_complete: false,
                    variadic_inputs: true,
                    variadic_outputs: true,
                    captures: Vec::new(),
                    captures_complete: true,
                }),
                "callable",
            ),
            (
                ValueKindFact::OutputList(runmat_types::OutputListFact {
                    outputs: vec![unknown()],
                    variadic: true,
                }),
                "output-list",
            ),
            (
                ValueKindFact::Exception(runmat_types::ExceptionFact {
                    identifier: Some("RunMat:test".to_string()),
                }),
                "exception",
            ),
            (
                ValueKindFact::Execution(runmat_types::ExecutionFact::Future {
                    output: Box::new(unknown()),
                    state: runmat_types::FutureStateFact::Lazy,
                }),
                "future",
            ),
            (
                ValueKindFact::Execution(runmat_types::ExecutionFact::Task {
                    output: Box::new(unknown()),
                    spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
                }),
                "task",
            ),
            (
                ValueKindFact::Execution(runmat_types::ExecutionFact::Pool),
                "pool",
            ),
            (
                ValueKindFact::Execution(runmat_types::ExecutionFact::Job {
                    output: Box::new(unknown()),
                }),
                "job",
            ),
            (
                ValueKindFact::Distributed(runmat_types::DistributedFact {
                    id: runmat_types::DistributedValueId {
                        function: runmat_types::ProgramFunctionId(1),
                        ordinal: 2,
                    },
                    owner: runmat_types::ParallelRegionId(runmat_types::RegionId {
                        function: runmat_types::ProgramFunctionId(1),
                        ordinal: 3,
                    }),
                    scheme: Some(runmat_types::DistributionScheme::Replicated),
                    value: Box::new(unknown()),
                    materializable: true,
                }),
                "distributed",
            ),
            (
                ValueKindFact::Foreign(runmat_types::ForeignFact {
                    family: "java".to_string(),
                    type_name: Some("java.lang.String".to_string()),
                    type_version: None,
                    ownership: runmat_types::ForeignOwnershipFact::Owned,
                    affinity: runmat_types::ForeignAffinityFact::AnyThread,
                    lifetime: runmat_types::ForeignLifetimeFact::Session,
                }),
                "foreign<java.lang.String>",
            ),
        ];

        for (kind, expected) in cases {
            let fact = ValueFact::scalar(kind);
            assert_eq!(format_fact(Some(&fact)), format!("{expected} [1 x 1]"));
        }
        assert_eq!(format_fact(None), "unknown");
    }
}
