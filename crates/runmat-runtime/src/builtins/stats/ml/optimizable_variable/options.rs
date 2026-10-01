use super::*;

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
    pub(super) fn headers(
        args: Vec<Value>,
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

    pub(super) fn parse(args: Vec<(OptimizableVariableOption, Value)>) -> BuiltinResult<Self> {
        let mut options = Self {
            var_type: None,
            transform: Transform::None,
            optimize: true,
        };
        let mut pairs = BTreeMap::new();
        for (kind, value) in args {
            pairs.insert(kind, value);
        }
        for (option, value) in pairs {
            match option {
                OptimizableVariableOption::Type => options.var_type = Some(variable_type(&value)?),
                OptimizableVariableOption::Transform => options.transform = transform(&value)?,
                OptimizableVariableOption::Optimize => {
                    options.optimize = parse_bool(&value, "Optimize")?
                }
            }
        }
        Ok(options)
    }
}
