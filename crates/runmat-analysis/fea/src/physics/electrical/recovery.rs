use super::{assembly::ElectricalAssembly, ElectricalStaticError};

pub(super) struct RecoveredElectrical {
    pub electric_field: Vec<[f64; 3]>,
    pub flux: Vec<[f64; 3]>,
    pub squared_field_integral: Vec<f64>,
    pub volumes: Vec<f64>,
}

pub(super) fn recover(
    assembly: &ElectricalAssembly,
    potential: &[f64],
) -> Result<RecoveredElectrical, ElectricalStaticError> {
    let elements = &assembly.projection.elements;
    // Independent per-element recovery on the admitted pool, collected in element order.
    let recovered = crate::algebra::element_assembly::ordered_map_range(elements.len(), |index| {
        let element = &elements[index];
        let mut average = [0.0; 3];
        let mut square = 0.0;
        let mut volume = 0.0;
        assembly
            .projection
            .visit_element_quadrature(element, |point| {
                let mut field = [0.0; 3];
                for (local, &node) in element.node_indices.iter().enumerate() {
                    for (axis, value) in field.iter_mut().enumerate() {
                        *value -= potential[node] * point.physical_shape_gradients()[local][axis];
                    }
                }
                volume += point.volume_weight_m3;
                square +=
                    field.iter().map(|value| value * value).sum::<f64>() * point.volume_weight_m3;
                for axis in 0..3 {
                    average[axis] += field[axis] * point.volume_weight_m3;
                }
            })
            .map_err(|error| ElectricalStaticError::Topology(error.to_string()))?;
        let average = average.map(|value| value / volume);
        if !volume.is_finite()
            || volume <= 0.0
            || !square.is_finite()
            || square < 0.0
            || average.iter().any(|value| !value.is_finite())
        {
            return Err(ElectricalStaticError::Operator(
                "invalid field recovery".into(),
            ));
        }
        let coefficient = assembly.coefficients.canonical_values()[index];
        if average
            .iter()
            .any(|value| !(value * coefficient).is_finite())
        {
            return Err(ElectricalStaticError::Operator(
                "nonfinite constitutive flux".into(),
            ));
        }
        Ok((
            average,
            average.map(|value| coefficient * value),
            square,
            volume,
        ))
    });
    let mut result = RecoveredElectrical {
        electric_field: Vec::with_capacity(elements.len()),
        flux: Vec::with_capacity(elements.len()),
        squared_field_integral: Vec::with_capacity(elements.len()),
        volumes: Vec::with_capacity(elements.len()),
    };
    for item in recovered {
        let (field, flux, square, volume) = item?;
        result.electric_field.push(field);
        result.flux.push(flux);
        result.squared_field_integral.push(square);
        result.volumes.push(volume);
    }
    Ok(result)
}
