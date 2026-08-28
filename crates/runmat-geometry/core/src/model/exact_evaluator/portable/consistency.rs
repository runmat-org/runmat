use super::super::incidence_consistency::validate_exact_incidence_with_parameters;
use super::super::{
    ExactCurveDefinition, ExactCurveImplementation, ExactPcurveDefinition,
    ExactPcurveImplementation, GeometryEvaluationControl, GeometryEvaluationError,
};
use super::PortableExactEvaluator;

impl PortableExactEvaluator<'_> {
    /// Checks every admitted portable incidence through the shared exact-geometry validator.
    pub fn validate_incidence_consistency(
        &self,
        tolerance_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<(), GeometryEvaluationError> {
        validate_exact_incidence_with_parameters(
            self.topology,
            self,
            tolerance_m,
            control,
            |edge, coedge, range| {
                let curve = &self.curve_record(&edge.curve_evaluator_id)?.implementation;
                let pcurve = &self
                    .pcurve_record(&coedge.pcurve_evaluator_id)?
                    .implementation;
                Ok(portable_breakpoints(range, curve, pcurve))
            },
        )
    }
}

fn portable_breakpoints(
    range: super::super::ParameterRange,
    curve: &ExactCurveImplementation,
    pcurve: &ExactPcurveImplementation,
) -> Vec<f64> {
    let mut parameters = Vec::new();
    if let ExactCurveImplementation::Portable {
        definition: ExactCurveDefinition::Nurbs { definition },
    } = curve
    {
        parameters.extend(
            definition
                .knots
                .iter()
                .copied()
                .filter(|value| *value > range.start && *value < range.end),
        );
    }
    if let ExactPcurveImplementation::Portable {
        definition: ExactPcurveDefinition::Nurbs { definition },
    } = pcurve
    {
        parameters.extend(
            definition
                .knots
                .iter()
                .copied()
                .filter(|value| *value > range.start && *value < range.end),
        );
    }
    parameters
}
