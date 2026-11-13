use super::*;

pub(super) struct PreparedOption {
    kind: BayesoptOption,
    value: PreparedValue,
}

enum PreparedValue {
    Host { raw: Value, parsed: ParsedOption },
    Resident(Value),
}

enum ParsedOption {
    MaxEvaluations(usize),
    SeedPoints(usize),
    Acquisition(Acquisition),
    ExplorationRatio(f64),
    Deterministic(bool),
    Verbose(bool),
    UseParallel,
    Constraint(Option<Value>),
    EmptyCallback,
    EmptyConditional,
}

#[derive(Clone)]
pub(super) struct BayesoptOptions {
    pub(super) max_evals: usize,
    pub(super) num_seed_points: usize,
    pub(super) acquisition: Acquisition,
    pub(super) exploration_ratio: f64,
    pub(super) deterministic: bool,
    pub(super) verbose: bool,
    pub(super) x_constraint: Option<Value>,
    pub(super) raw_options: StructValue,
}

impl BayesoptOptions {
    pub(super) fn headers(
        args: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Vec<(BayesoptOption, Value)>> {
        if !args.len().is_multiple_of(2) {
            return Err(bayesopt_error(
                "bayesopt: name-value options must be paired",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let mut headers = Vec::with_capacity(args.len() / 2);
        let mut values = args.into_iter();
        while let Some(name) = values.next() {
            let Some(value) = values.next() else {
                return Err(bayesopt_error(
                    "bayesopt: name-value options must be paired",
                    &ERROR_INVALID_ARGUMENT,
                ));
            };
            let name = value_to_text(&name, "option name")?;
            let option = BayesoptOption::from_public_name(&name).ok_or_else(|| {
                bayesopt_error(
                    format!("bayesopt: unsupported option '{name}'"),
                    &ERROR_INVALID_ARGUMENT,
                )
            })?;
            if option.always_unsupported() {
                return Err(bayesopt_error(
                    format!(
                        "bayesopt: {} warm starts are not supported yet",
                        option.public_name()
                    ),
                    &ERROR_UNSUPPORTED,
                ));
            }
            headers.push((option, value));
        }
        Ok(headers)
    }

    pub(super) fn admit(args: Vec<(BayesoptOption, Value)>) -> BuiltinResult<Vec<PreparedOption>> {
        args.into_iter()
            .map(|(kind, value)| {
                let value = if matches!(&value, Value::GpuTensor(_)) {
                    // A resident numeric value cannot decode to a public text
                    // acquisition name. Reject it before another option's
                    // download can obscure the owning argument error.
                    if kind == BayesoptOption::AcquisitionFunctionName {
                        return Err(bayesopt_error(
                            "bayesopt: AcquisitionFunctionName must be a text scalar",
                            &ERROR_INVALID_ARGUMENT,
                        ));
                    }
                    PreparedValue::Resident(value)
                } else {
                    let parsed = parse_option(kind, &value)?;
                    PreparedValue::Host { raw: value, parsed }
                };
                Ok(PreparedOption { kind, value })
            })
            .collect()
    }

    pub(super) async fn materialize(args: Vec<PreparedOption>) -> BuiltinResult<Self> {
        let mut resolved = Vec::with_capacity(args.len());
        for PreparedOption { kind, value } in args {
            match value {
                PreparedValue::Host { raw, parsed } => resolved.push((kind, raw, parsed)),
                PreparedValue::Resident(value) => {
                    let raw = gather_if_needed_async(&value)
                        .await
                        .map_err(|err| map_flow(err, "name-value option"))?;
                    let parsed = parse_option(kind, &raw)?;
                    resolved.push((kind, raw, parsed));
                }
            }
        }
        Self::parse(resolved)
    }

    fn parse(args: Vec<(BayesoptOption, Value, ParsedOption)>) -> BuiltinResult<Self> {
        let mut raw_options = runmat_value::UnboundedStructValueBuilder::new();
        let mut max_evals = DEFAULT_MAX_EVALS;
        let mut num_seed_points = DEFAULT_SEED_POINTS;
        let mut acquisition = Acquisition::ExpectedImprovementPlus;
        let mut exploration_ratio = 0.5;
        let mut deterministic = true;
        let mut verbose = false;
        let mut x_constraint = None;

        for (option, value, parsed) in args {
            let canonical = option.public_name();
            match parsed {
                ParsedOption::MaxEvaluations(n) => max_evals = n,
                ParsedOption::SeedPoints(n) => num_seed_points = n,
                ParsedOption::Acquisition(name) => acquisition = name,
                ParsedOption::ExplorationRatio(ratio) => exploration_ratio = ratio,
                ParsedOption::Deterministic(value) => deterministic = value,
                ParsedOption::Verbose(value) => verbose = value,
                ParsedOption::Constraint(value) => x_constraint = value,
                ParsedOption::UseParallel
                | ParsedOption::EmptyCallback
                | ParsedOption::EmptyConditional => {}
            }
            raw_options.insert(canonical, value);
        }
        if num_seed_points > max_evals {
            num_seed_points = max_evals;
        }
        raw_options.insert("MaxObjectiveEvaluations", Value::Num(max_evals as f64));
        raw_options.insert("NumSeedPoints", Value::Num(num_seed_points as f64));
        raw_options.insert(
            "AcquisitionFunctionName",
            Value::string(acquisition.public_name().to_string()),
        );
        raw_options.insert("ExplorationRatio", Value::Num(exploration_ratio));
        raw_options.insert("IsObjectiveDeterministic", Value::Bool(deterministic));
        raw_options.insert("Verbose", Value::Bool(verbose));
        raw_options.insert("UseParallel", Value::Bool(false));
        let raw_options = raw_options.finish();

        Ok(Self {
            max_evals,
            num_seed_points,
            acquisition,
            exploration_ratio,
            deterministic,
            verbose,
            x_constraint,
            raw_options,
        })
    }
}

fn parse_option(kind: BayesoptOption, value: &Value) -> BuiltinResult<ParsedOption> {
    let canonical = kind.public_name();
    Ok(match kind {
        BayesoptOption::MaxObjectiveEvaluations => {
            ParsedOption::MaxEvaluations(option_usize(value, canonical, 1, MAX_EVALS_LIMIT)?)
        }
        BayesoptOption::NumSeedPoints => {
            ParsedOption::SeedPoints(option_usize(value, canonical, 0, MAX_EVALS_LIMIT)?)
        }
        BayesoptOption::AcquisitionFunctionName => {
            ParsedOption::Acquisition(parse_acquisition(value)?)
        }
        BayesoptOption::ExplorationRatio => {
            ParsedOption::ExplorationRatio(option_unit_f64(value, canonical)?)
        }
        BayesoptOption::IsObjectiveDeterministic => {
            ParsedOption::Deterministic(option_bool(value, canonical)?)
        }
        BayesoptOption::Verbose => ParsedOption::Verbose(option_bool(value, canonical)?),
        BayesoptOption::UseParallel => {
            if option_bool(value, canonical)? {
                return Err(bayesopt_error(
                    "bayesopt: UseParallel=true requires the parallel execution infrastructure",
                    &ERROR_UNSUPPORTED,
                ));
            }
            ParsedOption::UseParallel
        }
        BayesoptOption::XConstraintFcn => {
            if is_empty_option(value) {
                ParsedOption::Constraint(None)
            } else if is_callable(value) {
                ParsedOption::Constraint(Some(value.clone()))
            } else {
                return Err(bayesopt_error(
                    "bayesopt: XConstraintFcn must be empty or a function handle",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
        }
        BayesoptOption::PlotFcn | BayesoptOption::OutputFcn => {
            if !is_empty_option(value) {
                return Err(bayesopt_error(
                    format!("bayesopt: {canonical} callbacks are not supported in this headless runtime slice"),
                    &ERROR_UNSUPPORTED,
                ));
            }
            ParsedOption::EmptyCallback
        }
        BayesoptOption::ConditionalVariableFcn => {
            if !is_empty_option(value) {
                return Err(bayesopt_error(
                    "bayesopt: ConditionalVariableFcn requires dynamic variable activation support",
                    &ERROR_UNSUPPORTED,
                ));
            }
            ParsedOption::EmptyConditional
        }
        BayesoptOption::InitialX
        | BayesoptOption::InitialObjective
        | BayesoptOption::InitialConstraintViolations => {
            return Err(bayesopt_error(
                format!("bayesopt: {canonical} warm starts are not supported yet"),
                &ERROR_UNSUPPORTED,
            ));
        }
    })
}

fn parse_acquisition(value: &Value) -> BuiltinResult<Acquisition> {
    let text = value_to_text(value, "AcquisitionFunctionName")?;
    Acquisition::from_public_name(&text).ok_or_else(|| {
        bayesopt_error(
            format!("bayesopt: unsupported AcquisitionFunctionName '{text}'"),
            &ERROR_INVALID_ARGUMENT,
        )
    })
}
