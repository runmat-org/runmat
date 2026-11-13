use super::*;

pub(super) struct PreparedOption {
    kind: OptimizableVariableOption,
    value: PreparedValue,
}

pub(super) struct HostOptionFacts {
    pub(super) var_type: Option<VariableType>,
    pub(super) type_deferred: bool,
    pub(super) transform: Option<Transform>,
}

enum PreparedValue {
    Host(ParsedOption),
    Resident(Value),
}

enum ParsedOption {
    Type(VariableType),
    Transform(Transform),
    Optimize(bool),
}

fn variable_type(value: &Value) -> BuiltinResult<VariableType> {
    let text = parse_text_scalar(value, "Type", &ERROR_INVALID_OPTION)?;
    VariableType::from_public_name(&text).ok_or_else(|| {
        error(
            format!("optimizableVariable: unsupported Type '{text}'"),
            &ERROR_INVALID_OPTION,
        )
    })
}

fn transform(value: &Value) -> BuiltinResult<Transform> {
    let text = parse_text_scalar(value, "Transform", &ERROR_INVALID_OPTION)?;
    Transform::from_public_name(&text).ok_or_else(|| {
        error(
            format!("optimizableVariable: unsupported Transform '{text}'"),
            &ERROR_INVALID_OPTION,
        )
    })
}

#[derive(Debug)]
pub(super) struct Options {
    pub(super) var_type: Option<VariableType>,
    pub(super) transform: Transform,
    pub(super) optimize: bool,
}

impl Options {
    pub(super) fn host_facts(args: &[PreparedOption]) -> HostOptionFacts {
        let mut facts = HostOptionFacts {
            var_type: None,
            type_deferred: false,
            transform: None,
        };
        for option in args {
            match (option.kind, &option.value) {
                (
                    OptimizableVariableOption::Type,
                    PreparedValue::Host(ParsedOption::Type(value)),
                ) => {
                    facts.var_type = Some(*value);
                }
                (OptimizableVariableOption::Type, PreparedValue::Resident(_)) => {
                    facts.type_deferred = true;
                }
                (
                    OptimizableVariableOption::Transform,
                    PreparedValue::Host(ParsedOption::Transform(value)),
                ) => facts.transform = Some(*value),
                _ => {}
            }
        }
        facts
    }

    pub(super) fn headers(
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Vec<(OptimizableVariableOption, Value)>> {
        if !args.len().is_multiple_of(2) {
            return Err(error(
                "optimizableVariable: name-value options must be paired",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let mut headers = Vec::with_capacity(args.len() / 2);
        let mut values = args.into_iter();
        while let Some(name) = values.next() {
            let Some(value) = values.next() else {
                return Err(error(
                    "optimizableVariable: name-value options must be paired",
                    &ERROR_INVALID_ARGUMENT,
                ));
            };
            let name = parse_text_scalar(&name, "option name", &ERROR_INVALID_OPTION)?;
            let kind = OptimizableVariableOption::from_public_name(&name).ok_or_else(|| {
                error(
                    format!("optimizableVariable: unknown option '{name}'"),
                    &ERROR_INVALID_OPTION,
                )
            })?;
            headers.push((kind, value));
        }
        Ok(headers)
    }

    pub(super) fn admit(
        args: Vec<(OptimizableVariableOption, Value)>,
    ) -> BuiltinResult<Vec<PreparedOption>> {
        let mut effective = BTreeMap::new();
        for (kind, value) in args {
            effective.insert(kind, value);
        }
        effective
            .into_iter()
            .map(|(kind, value)| {
                let value = if matches!(&value, Value::GpuTensor(_)) {
                    // Only the final effective option is admitted. Resident
                    // numeric storage cannot become a textual Type or Transform.
                    if matches!(
                        kind,
                        OptimizableVariableOption::Type | OptimizableVariableOption::Transform
                    ) {
                        return Err(error(
                            format!(
                                "optimizableVariable: {} must be a text scalar",
                                kind.public_name()
                            ),
                            &ERROR_INVALID_OPTION,
                        ));
                    }
                    PreparedValue::Resident(value)
                } else {
                    PreparedValue::Host(parse_option(kind, &value)?)
                };
                Ok(PreparedOption { kind, value })
            })
            .collect()
    }

    pub(super) async fn materialize(args: Vec<PreparedOption>) -> BuiltinResult<Self> {
        let mut options = Self {
            var_type: None,
            transform: Transform::None,
            optimize: true,
        };
        for PreparedOption { kind, value } in args {
            let parsed = match value {
                PreparedValue::Host(parsed) => parsed,
                PreparedValue::Resident(value) => {
                    let value = gather_if_needed_async(&value)
                        .await
                        .map_err(|err| remap_flow(err, "name-value argument"))?;
                    parse_option(kind, &value)?
                }
            };
            match parsed {
                ParsedOption::Type(var_type) => options.var_type = Some(var_type),
                ParsedOption::Transform(transform) => options.transform = transform,
                ParsedOption::Optimize(optimize) => options.optimize = optimize,
            }
        }
        Ok(options)
    }
}

fn parse_option(kind: OptimizableVariableOption, value: &Value) -> BuiltinResult<ParsedOption> {
    Ok(match kind {
        OptimizableVariableOption::Type => ParsedOption::Type(variable_type(value)?),
        OptimizableVariableOption::Transform => ParsedOption::Transform(transform(value)?),
        OptimizableVariableOption::Optimize => {
            ParsedOption::Optimize(parse_bool(value, "Optimize")?)
        }
    })
}
