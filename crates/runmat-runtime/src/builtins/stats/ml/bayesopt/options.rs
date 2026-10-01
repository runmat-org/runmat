use super::*;

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
    pub(super) fn headers(args: Vec<Value>) -> BuiltinResult<Vec<(BayesoptOption, Value)>> {
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

    pub(super) fn parse(args: Vec<(BayesoptOption, Value)>) -> BuiltinResult<Self> {
        let mut raw_options = StructValue::new();
        let mut max_evals = DEFAULT_MAX_EVALS;
        let mut num_seed_points = DEFAULT_SEED_POINTS;
        let mut acquisition = Acquisition::ExpectedImprovementPlus;
        let mut exploration_ratio = 0.5;
        let mut deterministic = true;
        let mut verbose = false;
        let mut x_constraint = None;

        for (option, value) in args {
            let canonical = option.public_name();
            match option {
                BayesoptOption::MaxObjectiveEvaluations => {
                    max_evals = option_usize(&value, &canonical, 1, MAX_EVALS_LIMIT)?
                }
                BayesoptOption::NumSeedPoints => {
                    num_seed_points = option_usize(&value, &canonical, 0, MAX_EVALS_LIMIT)?
                }
                BayesoptOption::AcquisitionFunctionName => {
                    acquisition = parse_acquisition(&value)?;
                }
                BayesoptOption::ExplorationRatio => {
                    exploration_ratio = option_unit_f64(&value, &canonical)?;
                }
                BayesoptOption::IsObjectiveDeterministic => {
                    deterministic = option_bool(&value, &canonical)?;
                }
                BayesoptOption::Verbose => {
                    verbose = option_bool(&value, &canonical)?;
                }
                BayesoptOption::UseParallel => {
                    if option_bool(&value, &canonical)? {
                        return Err(bayesopt_error(
                            "bayesopt: UseParallel=true requires the parallel execution infrastructure",
                            &ERROR_UNSUPPORTED,
                        ));
                    }
                }
                BayesoptOption::XConstraintFcn => {
                    if is_empty_option(&value) {
                        x_constraint = None;
                    } else if is_callable(&value) {
                        x_constraint = Some(value.clone());
                    } else {
                        return Err(bayesopt_error(
                            "bayesopt: XConstraintFcn must be empty or a function handle",
                            &ERROR_INVALID_ARGUMENT,
                        ));
                    }
                }
                BayesoptOption::PlotFcn | BayesoptOption::OutputFcn => {
                    if !is_empty_option(&value) {
                        return Err(bayesopt_error(
                            format!("bayesopt: {canonical} callbacks are not supported in this headless runtime slice"),
                            &ERROR_UNSUPPORTED,
                        ));
                    }
                }
                BayesoptOption::ConditionalVariableFcn => {
                    if !is_empty_option(&value) {
                        return Err(bayesopt_error(
                            "bayesopt: ConditionalVariableFcn requires dynamic variable activation support",
                            &ERROR_UNSUPPORTED,
                        ));
                    }
                }
                BayesoptOption::InitialX
                | BayesoptOption::InitialObjective
                | BayesoptOption::InitialConstraintViolations => {
                    return Err(bayesopt_error(
                        format!("bayesopt: {canonical} warm starts are not supported yet"),
                        &ERROR_UNSUPPORTED,
                    ));
                }
            }
            raw_options.insert(&canonical, value);
        }
        if num_seed_points > max_evals {
            num_seed_points = max_evals;
        }
        raw_options.insert("MaxObjectiveEvaluations", Value::Num(max_evals as f64));
        raw_options.insert("NumSeedPoints", Value::Num(num_seed_points as f64));
        raw_options.insert(
            "AcquisitionFunctionName",
            Value::String(acquisition.public_name().to_string()),
        );
        raw_options.insert("ExplorationRatio", Value::Num(exploration_ratio));
        raw_options.insert("IsObjectiveDeterministic", Value::Bool(deterministic));
        raw_options.insert("Verbose", Value::Bool(verbose));
        raw_options.insert("UseParallel", Value::Bool(false));

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

fn parse_acquisition(value: &Value) -> BuiltinResult<Acquisition> {
    let text = value_to_text(value, "AcquisitionFunctionName")?;
    Acquisition::from_public_name(&text).ok_or_else(|| {
        bayesopt_error(
            format!("bayesopt: unsupported AcquisitionFunctionName '{text}'"),
            &ERROR_INVALID_ARGUMENT,
        )
    })
}
