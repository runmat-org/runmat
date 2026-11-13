//! Pure effective-algorithm selection and only the selected live CG controls.
use super::*;
use crate::builtins::math::optim::common::{
    derivative_scaling::DerivativeScaling, ScalarOptionFields,
};
use runmat_builtins::OptimSubproblemAlgorithm;
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim::lsqlin) enum Method {
    Cg { bandwidth: usize },
    Factorization,
}
pub(in crate::builtins::math::optim::lsqlin) struct Reflective {
    pub method: Method,
    pub typical: DerivativeScaling,
    pub maximum_cg_iterations: Option<usize>,
    pub cg_tolerance: f64,
    pub function_tolerance: f64,
}
pub(super) fn selected_input(
    value: Option<&runmat_value::StructValue>,
    linear_constraints: bool,
) -> BuiltinResult<OptimAlgorithm> {
    let fields = ScalarOptionFields::from_struct(value);
    let algorithm = OptimAlgorithm::parse(&fields.text(Field::Algorithm, ALGORITHM)?)
        .filter(|&v| OptimSolver::Lsqlin.accepts_algorithm(v))
        .ok_or_else(|| invalid("unsupported lsqlin algorithm"))?;
    Ok(
        if linear_constraints && algorithm == OptimAlgorithm::TrustRegionReflective {
            OptimAlgorithm::InteriorPoint
        } else {
            algorithm
        },
    )
}
pub(super) fn active(field: Field, algorithm: OptimAlgorithm) -> bool {
    let trust = algorithm == OptimAlgorithm::TrustRegionReflective;
    match field {
        Field::TypicalX
        | Field::FunctionTolerance
        | Field::MaxPCGIter
        | Field::TolPCG
        | Field::PrecondBandWidth
        | Field::SubproblemAlgorithm => trust,
        Field::TolCon => !trust,
        Field::TolX => {
            algorithm == OptimAlgorithm::ActiveSet || algorithm == OptimAlgorithm::InteriorPoint
        }
        Field::ObjectiveLimit => algorithm == OptimAlgorithm::ActiveSet,
        _ => true,
    }
}
pub(super) fn selected_active(
    field: Field,
    algorithm: OptimAlgorithm,
    method: Option<Method>,
) -> bool {
    active(field, algorithm)
        && !(algorithm == OptimAlgorithm::TrustRegionReflective && field == Field::PrecondBandWidth)
        && !(matches!(method, Some(Method::Factorization))
            && matches!(
                field,
                Field::MaxPCGIter | Field::TolPCG | Field::PrecondBandWidth
            ))
}
pub(super) async fn select(
    value: Option<&runmat_value::StructValue>,
    algorithm: OptimAlgorithm,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Method>> {
    if algorithm != OptimAlgorithm::TrustRegionReflective {
        return Ok(None);
    }
    let fields = ScalarOptionFields::from_struct(value);
    let method = OptimSubproblemAlgorithm::parse(&fields.text(Field::SubproblemAlgorithm, "cg")?)
        .ok_or_else(|| invalid("SubproblemAlgorithm must be cg or factorization"))?;
    if method == OptimSubproblemAlgorithm::Factorization {
        return Ok(Some(Method::Factorization));
    }
    let source = fields.exact(Field::PrecondBandWidth).map(|value| {
        runmat_value::StructValue::from_fields_unbounded([(
            "PrecondBandWidth".to_owned(),
            value.clone(),
        )])
    });
    let runtime = cooperation.runtime().cloned();
    let prepared = crate::builtins::math::optim::common::prepare_solver_options(
        OptimSolver::Lsqlin,
        source,
        Some(&runmat_builtins::LSQLIN_INTEGER_OPTION_EXTENSION),
        None,
        runtime.as_ref(),
        cooperation,
    )
    .await?;
    let fields = ScalarOptionFields::from_struct(prepared.as_ref().map(|v| v.structure()));
    let infinite = fields
        .exact(Field::PrecondBandWidth)
        .is_some_and(|value| match value {
            Value::Num(v) => *v == f64::INFINITY,
            Value::Tensor(v) if v.len() == 1 && v.integer_storage().is_none() => {
                crate::builtins::common::tensor::tensor_value_f64(v, 0) == f64::INFINITY
            }
            _ => false,
        });
    if infinite {
        Ok(Some(Method::Factorization))
    } else {
        Ok(Some(Method::Cg {
            bandwidth: fields.count(NAME, Field::PrecondBandWidth, 0)?,
        }))
    }
}
impl Reflective {
    pub(super) fn parse(
        fields: &ScalarOptionFields<'_>,
        legacy: f64,
        method: Method,
    ) -> BuiltinResult<Self> {
        let (maximum_cg_iterations, cg_tolerance) = if matches!(method, Method::Cg { .. }) {
            (
                fields
                    .exact(Field::MaxPCGIter)
                    .map(|_| fields.count(NAME, Field::MaxPCGIter, 1))
                    .transpose()?,
                fields.real(NAME, Field::TolPCG, 0.1)?,
            )
        } else {
            (None, 0.1)
        };
        let function_tolerance = fields.real(NAME, Field::FunctionTolerance, legacy)?;
        if maximum_cg_iterations == Some(0) || cg_tolerance <= 0. || function_tolerance < 0. {
            return Err(invalid(
                "MaxPCGIter and TolPCG must be positive; FunctionTolerance must be nonnegative",
            ));
        }
        let typical = DerivativeScaling::from_fields(NAME, fields, true)?;
        Ok(Self {
            method,
            typical,
            maximum_cg_iterations,
            cg_tolerance,
            function_tolerance,
        })
    }
}
