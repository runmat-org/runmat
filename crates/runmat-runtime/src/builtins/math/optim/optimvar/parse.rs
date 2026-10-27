//! Constructor boundary: typed dimensions, named axes and property roles.
use super::super::common::ScalarCooperation;
use super::*;
use runmat_builtins::OptimizationVariableProperty as Property;
use runmat_value::optimization_variable::OptimizationVariableDomain as Domain;

use runmat_value::optimization_variable::{
    OptimizationBoundStorage as Bound, OptimizationVariableCandidate,
};
use runmat_value::Utf8Text;
mod axes;
mod bounds;

pub(super) fn text(value: &Value, runtime: Option<&RuntimeContext>) -> BuiltinResult<Utf8Text> {
    if !matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        return Err(invalid("expected scalar text"));
    }
    if let Value::StringArray(array) = value {
        array.validate_layout().map_err(invalid)?;
    }
    crate::call::argument_text::convert(value, runtime)
}
fn constructor_text(value: &Value, cooperation: &ScalarCooperation) -> BuiltinResult<Utf8Text> {
    cooperation.check_allocation_epoch()?;
    let result = text(value, cooperation.runtime())?;
    cooperation.check_allocation_epoch()?;
    Ok(result)
}
pub(super) async fn constructor(
    args: crate::call::FunctionArgumentView<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<OptimizationVariableCandidate> {
    cooperation.check_allocation_epoch()?;
    let name = constructor_text(
        args.first()
            .ok_or_else(|| invalid("expected a variable name"))?,
        cooperation,
    )?;
    if name.len() > crate::builtins::common::identifiers::MATLAB_NAME_LENGTH_MAX
        || !crate::builtins::common::identifiers::is_valid_varname(&name)
    {
        return Err(invalid("Name must be a valid variable identifier"));
    }
    let mut split = 1;
    while split < args.len() {
        cooperation.check_allocation_epoch()?;
        if (matches!(&args[split], Value::String(_) | Value::CharArray(_))
            || matches!(&args[split],Value::StringArray(array) if array.data().len()==1))
            && Property::parse_name_value(&constructor_text(&args[split], cooperation)?).is_some()
        {
            break;
        }
        split += 1;
        cooperation.completed(1).await?;
    }
    if !(args.len() - split).is_multiple_of(2) {
        return Err(invalid("expected property name/value pairs"));
    }
    let (shape, index_names) = axes::constructor(args.range(1..split)?, cooperation).await?;
    let mut domain = Domain::Continuous;
    let mut lower = Bound::Uniform(f64::NEG_INFINITY);
    let mut upper = Bound::Uniform(f64::INFINITY);
    let mut seen = [false; 5];
    for pair in args.range(split..)?.chunks_exact(2) {
        let key = constructor_text(&pair[0], cooperation)?;
        let property = Property::parse_name_value(&key)
            .ok_or_else(|| invalid("unknown optimization-variable property"))?;
        let slot = Property::ALL
            .iter()
            .position(|known| *known == property)
            .expect("canonical property");
        if seen[slot] {
            return Err(invalid("duplicate optimization-variable property"));
        }
        seen[slot] = true;
        match property {
            Property::Name => {
                return Err(invalid(
                    "Name is read-only; supply it as the first argument",
                ))
            }
            Property::Type => {
                domain = Domain::parse(&constructor_text(&pair[1], cooperation)?)
                    .ok_or_else(|| invalid("unsupported optimization-variable Type"))?;
            }
            Property::IndexNames => {
                return Err(invalid(
                    "constructor index names belong in dimension arguments",
                ))
            }
            Property::LowerBound => lower = bounds::parse(&pair[1], &shape, cooperation).await?,
            Property::UpperBound => upper = bounds::parse(&pair[1], &shape, cooperation).await?,
        }
        cooperation.completed(1).await?;
    }
    let candidate = OptimizationVariableCandidate {
        name,
        domain,
        shape,
        index_names,
        lower,
        upper,
    };
    validate_domain(&candidate, cooperation).await?;
    cooperation.check_allocation_epoch()?;
    Ok(candidate)
}
pub(super) async fn validate_domain(
    candidate: &OptimizationVariableCandidate,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if candidate.domain.is_semi() {
        for bound in [&candidate.lower, &candidate.upper] {
            match bound {
                Bound::Uniform(value) => {
                    if !value.is_finite() || *value <= 0.0 || *value > 1e5 {
                        return Err(invalid("semi domains require bounds in (0,1e5]"));
                    }
                }
                Bound::PerElement(values) => {
                    for value in values
                        .as_f64_slice()
                        .ok_or_else(|| invalid("bounds require double storage"))?
                    {
                        if !value.is_finite() || *value <= 0.0 || *value > 1e5 {
                            return Err(invalid("semi domains require bounds in (0,1e5]"));
                        }
                        cooperation.completed(1).await?;
                    }
                }
            }
        }
    }
    Ok(())
}
pub(super) use bounds::parse as bound;

pub(super) use axes::property as index_names;
