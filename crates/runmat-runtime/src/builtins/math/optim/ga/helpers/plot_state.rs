//! Closed consumed-state projection; user fields/captures stay opaque.
use super::super::{common, invalid, numeric, options, score_value, ScalarCooperation};
use crate::BuiltinResult;
use runmat_builtins::{
    OptimGeneticPlotKind as Kind, OptimGeneticStateField as Field, OptimOptionField,
};
use runmat_value::{NumericDType, StructValue, Tensor, Value};
pub(super) struct State<'a> {
    value: &'a StructValue,
}
impl<'a> State<'a> {
    pub fn new(value: &'a Value) -> BuiltinResult<Self> {
        match value {
            Value::Struct(value) => Ok(Self { value }),
            _ => Err(invalid("genetic plot state must be a scalar structure")),
        }
    }
    pub fn field(&self, field: Field) -> BuiltinResult<&'a Value> {
        self.value
            .fields
            .get(field.name())
            .ok_or_else(|| invalid(format!("{} is required by this genetic plot", field.name())))
    }
    pub fn optional(&self, field: Field) -> Option<&'a Value> {
        self.value.fields.get(field.name())
    }
    pub fn preflight(&self, kind: Kind, cooperation: &ScalarCooperation) -> BuiltinResult<()> {
        use common::constrained_numeric::Geometry;
        options::admission::numeric_metadata(
            options::Owner::Solver,
            options::admission::Role::Count,
            self.field(Field::Generation)?,
            cooperation.runtime(),
        )?;
        if matches!(
            kind,
            Kind::BestFitness
                | Kind::BestIndividual
                | Kind::Range
                | Kind::ScoreDiversity
                | Kind::Scores
        ) {
            score_value::metadata(self.field(Field::Score)?, None, cooperation)?;
        }
        let role = match kind {
            Kind::BestIndividual | Kind::Distance => Some((Field::Population, Geometry::MatrixAny)),
            Kind::Expectation => Some((Field::Expectation, Geometry::Vector)),
            Kind::Selection => Some((Field::Selection, Geometry::Vector)),
            Kind::Genealogy => Some((Field::GeneticParents, Geometry::MatrixAny)),
            _ => None,
        };
        if let Some((field, geometry)) = role {
            numeric::preflight(
                self.field(field)?,
                field.name(),
                geometry,
                cooperation.runtime(),
            )?;
        }
        if kind == Kind::MaxConstraint {
            for field in [Field::NonlinearInequality, Field::NonlinearEquality] {
                numeric::preflight(
                    self.field(field)?,
                    field.name(),
                    Geometry::MatrixAny,
                    cooperation.runtime(),
                )?;
            }
        }
        if kind == Kind::Genealogy {
            if let Some(value) = self.optional(Field::TotalInnerGenerations) {
                options::admission::numeric_metadata(
                    options::Owner::Solver,
                    options::admission::Role::Count,
                    value,
                    cooperation.runtime(),
                )?;
            }
        }
        if kind == Kind::Stopping {
            numeric::preflight(
                self.field(Field::Best)?,
                "state.Best",
                Geometry::Vector,
                cooperation.runtime(),
            )?;
            options::admission::numeric_metadata(
                options::Owner::Solver,
                options::admission::Role::Count,
                self.field(Field::LastImprovement)?,
                cooperation.runtime(),
            )?;
        }
        Ok(())
    }
    pub async fn vector(
        &self,
        field: Field,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        let common::constrained_numeric::Numeric::Real { values, .. } =
            common::constrained_numeric::read(
                self.field(field)?.clone(),
                field.name(),
                common::constrained_numeric::Geometry::Vector,
                &numeric::INPUT_POLICY,
                cooperation,
            )
            .await?
        else {
            return Err(invalid("genetic plot requires real vector data"));
        };
        Ok(values)
    }
    pub async fn matrix(
        &self,
        field: Field,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Tensor> {
        let value = common::constrained_numeric::prepare(
            self.field(field)?.clone(),
            field.name(),
            common::constrained_numeric::Geometry::MatrixAny,
            &numeric::INPUT_POLICY,
            cooperation,
        )
        .await?
        .ok_or_else(|| invalid("genetic plot requires real matrix data"))?;
        if let Value::Tensor(value) = &value {
            if value.numeric_dtype() == NumericDType::F64 {
                return Ok(value.clone());
            }
        }
        let common::constrained_numeric::Numeric::Real { shape, values } =
            common::constrained_numeric::materialize_raw(
                value,
                field.name(),
                &numeric::INPUT_POLICY,
                cooperation,
            )
            .await?
        else {
            return Err(invalid("genetic plot requires real matrix data"));
        };
        Tensor::new(values, shape).map_err(invalid)
    }
    pub async fn count(
        &self,
        field: Field,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<usize> {
        super::admission::count(self.field(field)?, field.name(), cooperation).await
    }
    pub async fn scores(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<common::population::Fitness> {
        let source = self.field(Field::Score)?;
        let count = crate::value_fact::value_fact(source)
            .shape
            .element_count()
            .ok_or_else(|| invalid("genetic scores require known vector geometry"))?;
        score_value::decode(
            source.clone(),
            score_value::Role::Score,
            count,
            false,
            cooperation,
        )
        .await
    }
}
pub(super) fn stopping_options(
    value: &Value,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    super::admission::options(value)?;
    let fields = common::ScalarOptionFields::from_struct_for_solver(
        if let Value::Struct(value) = value {
            Some(value)
        } else {
            None
        },
        runmat_builtins::OptimSolver::Ga,
    );
    for field in [
        OptimOptionField::MaxIter,
        OptimOptionField::MaxStallIterations,
        OptimOptionField::MaxTime,
        OptimOptionField::FunctionTolerance,
    ] {
        if let Some(value) = fields.exact(field) {
            options::admission::metadata(
                options::Owner::Solver,
                field,
                value,
                cooperation.runtime(),
            )?;
        }
    }
    Ok(())
}
