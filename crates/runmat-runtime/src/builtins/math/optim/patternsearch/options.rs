//! One owned options generation. Callbacks borrow this owner; replacements are
//! admitted/published before the previous generation can retire.
use super::{
    controls::{self, Controls},
    input::empty,
    invalid,
    option_metadata::{self, Policy},
};
use crate::builtins::math::optim::common::{self, ScalarCooperation, ScalarOptionFields};
use crate::{context::RuntimeContext, execution::RootedValue, BuiltinResult};
use runmat_builtins::{
    OptimOptionField as Field, OptimPatternDefault as DefaultValue, OptimSolver,
};
use runmat_value::{StructValue, Value};

pub(super) struct Options {
    source: RootedValue,
    pub controls: Controls,
}
fn index(field: Field) -> usize {
    Field::ALL
        .iter()
        .position(|&candidate| candidate == field)
        .expect("declared option field")
}
fn value_default(field: Field, runtime: Option<&RuntimeContext>) -> BuiltinResult<Value> {
    Ok(
        match DefaultValue::for_field(field)
            .ok_or_else(|| invalid("missing pattern option default"))?
        {
            DefaultValue::Number(value) => Value::Num(value),
            DefaultValue::Logical(value) => Value::Bool(value),
            DefaultValue::Text(value) => Value::from(value),
            DefaultValue::DimensionCount(_) => {
                Value::from(DefaultValue::dimension_expression(field).expect("dimensional default"))
            }
            DefaultValue::Empty => common::output_callback::empty(runtime)?,
        },
    )
}

fn normalize(
    value: Option<&Value>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<RootedValue> {
    let original = match value {
        None => None,
        Some(value) if empty(value) => None,
        Some(Value::Struct(value)) => Some(value),
        _ => return Err(invalid("options must be a scalar structure or numeric []")),
    };
    let mut seen = [false; Field::ALL.len()];
    let mut normalized = runmat_value::UnboundedStructValueBuilder::new();
    if let Some(original) = original {
        for (name, value) in &original.fields {
            if name.eq_ignore_ascii_case("Solver") {
                let solver = crate::builtins::common::validation::value_text_scalar_string(value)
                    .ok_or_else(|| invalid("options.Solver must be scalar text"))?;
                if !solver.is_empty()
                    && OptimSolver::parse(&solver) != Some(OptimSolver::Patternsearch)
                {
                    return Err(invalid("options belong to another solver"));
                }
                continue;
            }
            if empty(value) {
                continue;
            }
            let field = Field::parse_for_solver(name, OptimSolver::Patternsearch)
                .ok_or_else(|| invalid(format!("unsupported option {name}")))?;
            if std::mem::replace(&mut seen[index(field)], true) {
                return Err(invalid(format!("duplicate aliases for {}", field.name())));
            }
            normalized.insert(
                field.name_for_solver(OptimSolver::Patternsearch),
                value.clone(),
            );
        }
    }
    for field in Field::ALL {
        if OptimSolver::Patternsearch.accepts(field) && !seen[index(field)] {
            normalized.insert(
                field.name_for_solver(OptimSolver::Patternsearch),
                value_default(field, runtime)?,
            );
        }
    }
    normalized.insert("Solver", Value::from("patternsearch"));
    RootedValue::in_optional_context(Value::Struct(normalized.finish()), runtime)
}

fn selector_view(prepared: &[Option<Value>; Field::ALL.len()]) -> StructValue {
    let mut values = runmat_value::UnboundedStructValueBuilder::new();
    for field in Field::ALL {
        if let Some(value) = &prepared[index(field)] {
            values.insert(field.name(), value.clone());
        }
    }
    values.finish()
}
async fn gather(value: &Value, cooperation: &mut ScalarCooperation) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::dispatcher::gather_if_needed_in_context_async(value, cooperation.runtime()).await
    } else {
        Ok(value.clone())
    }
}

impl Options {
    pub fn value(&self) -> &Value {
        self.source.value()
    }

    pub fn structure(&self) -> &StructValue {
        let Value::Struct(value) = self.source.value() else {
            unreachable!("prepared option structure")
        };
        value
    }
    pub fn callback(&self, field: Field) -> Option<&Value> {
        self.structure()
            .fields
            .get(field.name_for_solver(OptimSolver::Patternsearch))
            .filter(|value| !empty(value))
    }

    pub async fn prepare(
        value: Option<&Value>,
        n: usize,
        equalities: bool,
        linear: bool,
        nonlinear: bool,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let source = normalize(value, cooperation.runtime())?;
        let Value::Struct(original) = source.value() else {
            unreachable!("normalized options")
        };
        let raw = ScalarOptionFields::from_struct(Some(original));
        let selected_algorithm = controls::algorithm(&raw)?;
        let mut prepared: [Option<Value>; Field::ALL.len()] = std::array::from_fn(|_| None);
        // Pure grammar/callback roles never go through numeric gather.
        for field in [
            Field::Algorithm,
            Field::PollMethod,
            Field::SearchFcn,
            Field::Display,
            Field::OutputFcn,
            Field::PlotFcn,
        ] {
            if field == Field::PollMethod
                && selected_algorithm != runmat_builtins::OptimPatternAlgorithm::Classic
            {
                continue;
            }
            if let Some(value) = raw.exact(field) {
                let value = if field == Field::SearchFcn {
                    super::search::prepare(value, cooperation.runtime())?
                } else {
                    value.clone()
                };
                option_metadata::metadata(field, &value, policy, cooperation.runtime())?;
                prepared[index(field)] = Some(value);
            }
        }
        let pure_view = selector_view(&prepared);
        let selected = ScalarOptionFields::from_struct(Some(&pure_view));
        let selection = controls::select(&selected, n)?;
        if equalities
            && (selection.polling.adaptive()
                || selection.search.is_some_and(|search| {
                    search
                        .poll()
                        .is_some_and(runmat_builtins::OptimPatternPoll::mesh_adaptive)
                }))
        {
            return Err(invalid(
                "MADS polling/search does not support linear equality constraints",
            ));
        }
        if equalities
            && selection.polling.algorithm() == runmat_builtins::OptimPatternAlgorithm::NupsMads
        {
            return Err(controls::unsupported(
                "nups-mads does not support linear equalities",
            ));
        }
        if selection.polling.classic().is_some() && !selection.polling.adaptive() {
            if let Some(value) = raw.exact(Field::PollOrderAlgorithm) {
                option_metadata::metadata(
                    Field::PollOrderAlgorithm,
                    value,
                    policy,
                    cooperation.runtime(),
                )?;
                prepared[index(Field::PollOrderAlgorithm)] = Some(value.clone());
            }
            let view = selector_view(&prepared);
            controls::select(&ScalarOptionFields::from_struct(Some(&view)), n)?;
        }
        let plot = common::output_callback::callback_count(
            selected.exact(Field::PlotFcn),
            common::output_callback::CallbackForm::HandlesOrNames,
        )? > 0;
        // A required provider-valued selector is decoded before its dependent
        // roles are admitted. This never probes a dormant callback/property.
        for field in [Field::UseVectorized, Field::UseParallel, Field::Cache] {
            let view = selector_view(&prepared);
            let selectors = ScalarOptionFields::from_struct(Some(&view));
            if field == Field::UseParallel && controls::switch(&selectors, Field::UseVectorized)? {
                continue;
            }
            if field == Field::Cache && controls::parallel_requested(&selectors)? {
                continue;
            }
            if let Some(value) = raw.exact(field) {
                option_metadata::metadata(field, value, policy, cooperation.runtime())?;
                let value = gather(value, cooperation).await?;
                option_metadata::metadata(field, &value, policy, cooperation.runtime())?;
                prepared[index(field)] = Some(value);
                let view = selector_view(&prepared);
                controls::execution_switch(field, &ScalarOptionFields::from_struct(Some(&view)))?;
            }
        }
        let classic = selection.polling.classic();
        let poll = selection.polling.method();
        let nonuniform = classic.is_none();
        let view = selector_view(&prepared);
        let selectors = ScalarOptionFields::from_struct(Some(&view));
        let vectorized = controls::switch(&selectors, Field::UseVectorized)?;
        let parallel = controls::parallel_requested(&selectors)?;
        let cache = !parallel && controls::switch(&selectors, Field::Cache)?;
        let search = selection
            .search
            .is_some_and(|search| !poll.is_some_and(|poll| search.redundant(poll)));
        let fixed_controls = classic.is_some() && !selection.polling.adaptive()
            || selection
                .search
                .is_some_and(|search| search.poll().is_some_and(|method| !method.mesh_adaptive()));
        let active = |field| match field {
            Field::UseParallel => !vectorized,
            Field::Cache | Field::CacheSize | Field::CacheTol => {
                !parallel && (field == Field::Cache || cache)
            }
            Field::UseCompletePoll | Field::UseCompleteSearch if parallel => false,
            Field::InitialPenalty | Field::PenaltyFactor => nonlinear,
            Field::TolBind => linear,
            Field::MaxMeshSize => fixed_controls || nonuniform,
            Field::MeshExpansionFactor | Field::MeshContractionFactor => fixed_controls,
            Field::UseCompletePoll => classic.is_some(),
            Field::PollMethod | Field::PollOrderAlgorithm | Field::AccelerateMesh => {
                classic.is_some() && !selection.polling.adaptive()
            }
            Field::TolX | Field::FunctionTolerance => !selection.polling.adaptive(),
            Field::MeshRotate => {
                classic.is_some()
                    && !selection.polling.adaptive()
                    && poll.is_some_and(runmat_builtins::OptimPatternPoll::minimal)
                    && !equalities
            }
            Field::ScaleMesh => !equalities || nonuniform,
            Field::UseCompleteSearch => {
                search
                    && !matches!(
                        selection.search,
                        Some(
                            super::search::Search::Custom
                                | super::search::Search::Nelder { .. }
                                | super::search::Search::Rbf
                        )
                    )
            }
            Field::PlotInterval => plot,
            _ => true,
        };
        for field in Field::ALL {
            if OptimSolver::Patternsearch.accepts(field)
                && active(field)
                && prepared[index(field)].is_none()
            {
                if let Some(value) = raw.exact(field) {
                    option_metadata::metadata(field, value, policy, cooperation.runtime())?;
                }
            }
        }
        for field in Field::ALL {
            if OptimSolver::Patternsearch.accepts(field)
                && active(field)
                && prepared[index(field)].is_none()
            {
                if let Some(value) = raw.exact(field) {
                    let value = gather(value, cooperation).await?;
                    option_metadata::metadata(field, &value, policy, cooperation.runtime())?;
                    prepared[index(field)] = Some(value);
                }
            }
        }
        let mut sink = runmat_value::UnboundedStructValueBuilder::new();
        for field in Field::ALL {
            if !OptimSolver::Patternsearch.accepts(field) {
                continue;
            }
            let value = prepared[index(field)].take().unwrap_or_else(|| {
                original
                    .fields
                    .get(field.name_for_solver(OptimSolver::Patternsearch))
                    .expect("normalized complete option generation")
                    .clone()
            });
            sink.insert(field.name_for_solver(OptimSolver::Patternsearch), value);
        }
        sink.insert("Solver", Value::from("patternsearch"));
        let prepared =
            RootedValue::in_optional_context(Value::Struct(sink.finish()), cooperation.runtime())?;
        let Value::Struct(structure) = prepared.value() else {
            unreachable!("published options")
        };
        let fields = ScalarOptionFields::from_struct(Some(structure));
        let controls =
            Controls::read(&fields, n, equalities, linear, nonlinear).map_err(|mut error| {
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error
            })?;
        Ok(Self {
            source: prepared,
            controls,
        })
    }
}
