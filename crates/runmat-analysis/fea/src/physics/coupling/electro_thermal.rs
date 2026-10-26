use crate::assembly::ElectroThermalAssemblySummary;

pub fn severity(context: Option<&ElectroThermalAssemblySummary>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    (context.applied_voltage_v.powi(2)
        * context.base_electrical_conductivity_s_per_m.max(1.0e-9)
        * context
            .resistive_heating_coefficient
            .canonical_value()
            .max(0.0)
        / 1.0e7)
        .clamp(0.0, 1.0)
}
