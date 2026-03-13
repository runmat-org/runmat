use crate::FeaThermoMechanicalContext;

pub fn severity(context: Option<FeaThermoMechanicalContext>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    if !context.enabled {
        return 0.0;
    }
    let thermal_strain = (context.thermal_expansion_coefficient
        * context.applied_temperature_delta_k.abs())
    .clamp(0.0, 0.05);
    (thermal_strain / 0.05).clamp(0.0, 1.0)
}
