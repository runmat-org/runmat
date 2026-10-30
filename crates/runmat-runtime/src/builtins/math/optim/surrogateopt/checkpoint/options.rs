//! Resume overlays preserve explicit source roles and total limits. Dormant
//! vectorization is saved without consulting providers or recursive facts.
use super::{
    super::{options::Options, parameters::Selected},
    error,
};
use crate::{builtins::math::optim::common::ScalarCooperation, BuiltinResult};
use runmat_builtins::{OptimDisplay, OptimOptionField as Field, OptimSolver};
use runmat_value::Value;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SavedOptions {
    controls: Selected,
    display: String,
    checkpoint: String,
    vectorized: DormantSwitch,
}

#[derive(serde::Serialize, serde::Deserialize)]
enum DormantSwitch {
    Bool(bool),
    Number(u64),
    Text(String),
    Empty,
    Unavailable,
}
impl DormantSwitch {
    async fn read(
        reader: &mut super::json::Reader,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<Self> {
        if !reader.is(b'{') {
            return reader.metadata(13, c).await;
        }
        reader.token(b'{')?;
        let value = if reader.field_is("Bool") {
            reader.key("Bool", true)?;
            Self::Bool(reader.metadata(5, c).await?)
        } else if reader.field_is("Number") {
            reader.key("Number", true)?;
            Self::Number(reader.metadata(20, c).await?)
        } else {
            reader.key("Text", true)?;
            Self::Text(reader.string(c).await?)
        };
        reader.token(b'}')?;
        Ok(value)
    }
    async fn write(
        &self,
        writer: &mut super::json::Writer,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<()> {
        if let Self::Text(value) = self {
            writer.token(b"{")?;
            writer.key("Text", true)?;
            writer.string(value, c).await?;
            writer.token(b"}")
        } else {
            writer.metadata(self, c).await
        }
    }
    fn capture(value: &Value) -> Self {
        match value {
            Value::Bool(value) => Self::Bool(*value),
            Value::Num(value) => Self::Number(value.to_bits()),
            Value::Tensor(value) if value.is_empty() => Self::Empty,
            Value::Tensor(value)
                if value.len() == 1 && value.numeric_dtype() == runmat_value::NumericDType::F64 =>
            {
                value
                    .as_f64_slice()
                    .and_then(|values| values.first())
                    .map_or(Self::Unavailable, |value| Self::Number(value.to_bits()))
            }
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                crate::builtins::common::validation::value_text_scalar_string(value)
                    .map_or(Self::Unavailable, Self::Text)
            }
            Value::LogicalArray(value) if value.data.len() == 1 => Self::Bool(value.data[0] != 0),
            _ => Self::Unavailable,
        }
    }
    fn value(&self, c: &ScalarCooperation) -> BuiltinResult<Option<Value>> {
        Ok(match self {
            Self::Bool(value) => Some(Value::Bool(*value)),
            Self::Number(value) => Some(Value::Num(f64::from_bits(*value))),
            Self::Text(value) => Some(Value::string(crate::call::argument_text::copy_str(
                value,
                c.runtime(),
            )?)),
            Self::Empty => Some(super::super::public_values::empty(c)?),
            Self::Unavailable => None,
        })
    }
}

fn binding(value: &Value, field: Field) -> Option<&Value> {
    let Value::Struct(fields) = value else {
        return None;
    };
    fields.fields.iter().find_map(|(name, value)| {
        (Field::parse_for_solver(name, OptimSolver::Surrogateopt) == Some(field)).then_some(value)
    })
}

pub(super) struct Overlay {
    pub options: Options,
    pub saved: SavedOptions,
    pub output_changed: bool,
    pub plot_changed: bool,
    pub initial_supplied: Option<Value>,
}
impl SavedOptions {
    pub(super) async fn read(
        reader: &mut super::json::Reader,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<Self> {
        reader.token(b'{')?;
        reader.key("controls", true)?;
        let controls = reader.metadata(2048, c).await?;
        reader.key("display", false)?;
        let display = reader.string(c).await?;
        reader.key("checkpoint", false)?;
        let checkpoint = reader.string(c).await?;
        reader.key("vectorized", false)?;
        let vectorized = DormantSwitch::read(reader, c).await?;
        reader.token(b'}')?;
        Ok(Self {
            controls,
            display,
            checkpoint,
            vectorized,
        })
    }
    pub(super) async fn write(
        &self,
        writer: &mut super::json::Writer,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<()> {
        writer.token(b"{")?;
        writer.key("controls", true)?;
        writer.metadata(&self.controls, c).await?;
        writer.key("display", false)?;
        writer.string(&self.display, c).await?;
        writer.key("checkpoint", false)?;
        writer.string(&self.checkpoint, c).await?;
        writer.key("vectorized", false)?;
        self.vectorized.write(writer, c).await?;
        writer.token(b"}")
    }
    pub(super) fn checkpoint(&self) -> &str {
        &self.checkpoint
    }
    pub(super) fn capture(options: &Options) -> BuiltinResult<Self> {
        let vectorized = binding(options.declared.value(), Field::UseVectorized)
            .map_or(DormantSwitch::Bool(false), DormantSwitch::capture);
        Ok(Self {
            controls: options.controls,
            display: options.display.name().to_owned(),
            checkpoint: options.checkpoint.clone().unwrap_or_default(),
            vectorized,
        })
    }

    pub(super) async fn overlay(
        self,
        value: Value,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Overlay> {
        let value = if super::super::super::common::surrogate_options::empty(&value) {
            Value::Struct(runmat_value::UnboundedStructValueBuilder::new().finish())
        } else {
            value
        };
        let Value::Struct(fields) = &value else {
            return Err(error("checkpoint options must be a scalar structure or []"));
        };
        let mut supplied = Vec::new();
        for name in fields.fields.keys() {
            if name.eq_ignore_ascii_case("Solver") {
                continue;
            }
            let field = Field::parse_for_solver(name, OptimSolver::Surrogateopt)
                .ok_or_else(|| error(format!("unsupported checkpoint option {name}")))?;
            supplied.push(field);
        }
        let has = |field| supplied.contains(&field);
        let output_changed = has(Field::OutputFcn);
        let plot_changed = has(Field::PlotFcn);
        let mut merged = runmat_value::UnboundedStructValueBuilder::new();
        for (name, value) in &fields.fields {
            merged.insert(name, value.clone());
        }
        if !has(Field::UseParallel) {
            merged.insert(
                "UseParallel",
                Value::from(match self.controls.parallel {
                    crate::execution::CallbackParallelMode::Off => "off",
                    crate::execution::CallbackParallelMode::Auto => "auto",
                    crate::execution::CallbackParallelMode::On => "on",
                }),
            );
        }
        let vectorized = if has(Field::UseVectorized) {
            None
        } else {
            self.vectorized.value(c)?
        };
        if let Some(value) = vectorized {
            merged.insert("UseVectorized", value);
        }
        let mut options = Options::parse(Value::Struct(merged.finish()), c).await?;
        if !has(Field::UseVectorized)
            && matches!(self.vectorized, DormantSwitch::Unavailable)
            && options.controls.parallel == crate::execution::CallbackParallelMode::Off
        {
            return Err(error(
                "activating saved UseVectorized requires an explicit portable replacement",
            ));
        }
        let parsed = options.controls;
        let mut controls = self.controls;
        macro_rules! replace {
            ($field:ident, $member:ident) => {
                if has(Field::$field) {
                    controls.$member = parsed.$member;
                }
            };
        }
        replace!(MaxFunEvals, evaluations);
        replace!(MinSurrogatePoints, minimum_surrogate);
        replace!(BatchUpdateInterval, batch);
        replace!(ObjectiveLimit, objective_limit);
        replace!(MaxTime, time);
        // Parallel was merged before shared preparation, so vectorized
        // validation follows the actual requested effective-mode rule.
        controls.parallel = parsed.parallel;
        controls.vectorized = parsed.vectorized;
        if has(Field::TolCon)
            && parsed.nonlinear_tolerance.to_bits() != controls.nonlinear_tolerance.to_bits()
            || has(Field::MinSampleDistance)
                && parsed.sample_distance.to_bits() != controls.sample_distance.to_bits()
        {
            return Err(error(
                "checkpoint nonlinear tolerance and minimum sample distance cannot change",
            ));
        }
        if !has(Field::Display) {
            options.display = OptimDisplay::parse(&self.display)
                .filter(|value| OptimSolver::Surrogateopt.accepts_display(*value))
                .ok_or_else(|| error("saved Display is invalid"))?;
        }
        if !has(Field::CheckpointFile) {
            options.checkpoint = (!self.checkpoint.is_empty()).then_some(self.checkpoint);
        }
        options.controls = controls;
        let initial_supplied = has(Field::InitialPoints)
            .then(|| binding(options.public.value(), Field::InitialPoints).cloned())
            .flatten();
        let vectorized = if has(Field::UseVectorized) {
            binding(options.declared.value(), Field::UseVectorized)
                .map_or(DormantSwitch::Bool(false), DormantSwitch::capture)
        } else {
            self.vectorized
        };
        let saved = SavedOptions {
            controls,
            display: options.display.name().to_owned(),
            checkpoint: options.checkpoint.clone().unwrap_or_default(),
            vectorized,
        };
        Ok(Overlay {
            options,
            saved,
            output_changed,
            plot_changed,
            initial_supplied,
        })
    }
}
