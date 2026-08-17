use super::super::{
    BodyMassProperties, ExactMassPropertiesEvaluator, ExactMassPropertiesImplementation,
    GeometryEvaluationControl, GeometryEvaluationError, MassPropertiesEvaluatorId,
};
use super::{kernel_owned, PortableExactEvaluator};

impl ExactMassPropertiesEvaluator for PortableExactEvaluator<'_> {
    fn mass_properties(
        &self,
        id: &MassPropertiesEvaluatorId,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<BodyMassProperties, GeometryEvaluationError> {
        let record = self.mass_properties_record(id)?;
        control.checkpoint()?;
        match &record.implementation {
            ExactMassPropertiesImplementation::Kernel { .. } => {
                Err(kernel_owned("mass-properties"))
            }
            ExactMassPropertiesImplementation::KernelValidated { properties, .. } => {
                control.consume_iterations(1)?;
                Ok(*properties)
            }
        }
    }
}
