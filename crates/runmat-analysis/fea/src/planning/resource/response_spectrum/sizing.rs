use crate::execution::{
    StructuralModalResponseSpectrumAssemblyRequest, StructuralModalResponseSpectrumDirectionRequest,
};

use super::{add, count, invalid, invalid_error, multiply};
use crate::planning::resource::{PreparedOutputResourceFacts, ResourcePlanningError};

const DIRECTION_FIXED_PAYLOAD_BYTES: u64 = 256 * 1024;
const ASSEMBLY_FIXED_PAYLOAD_BYTES: u64 = 256 * 1024;
const RECEIPT_PAYLOAD_BYTES: u64 = 16 * 1024;

pub(super) fn missing_mass_static_system_workspace_bytes(
    full_dimension: u64,
    reduced_dimension: u64,
    reduced_stiffness_nonzeros: u64,
    preconditioner_bytes: u64,
) -> Result<u64, ResourcePlanningError> {
    let reduced_stiffness_bytes = add(
        multiply(
            add(reduced_dimension, 1, "missing-mass reduced stiffness rows")?,
            8,
            "missing-mass reduced stiffness row offsets",
        )?,
        multiply(
            reduced_stiffness_nonzeros,
            16,
            "missing-mass reduced stiffness entries",
        )?,
        "missing-mass reduced stiffness storage",
    )?;
    let vector_values = add(
        multiply(
            full_dimension,
            6,
            "missing-mass full-order static workspace",
        )?,
        multiply(
            reduced_dimension,
            8,
            "missing-mass reduced static workspace",
        )?,
        "missing-mass static vector workspace",
    )?;
    add(
        add(
            reduced_stiffness_bytes,
            preconditioner_bytes,
            "missing-mass reduced system and preconditioner",
        )?,
        multiply(
            vector_values,
            8,
            "missing-mass static vector workspace bytes",
        )?,
        "missing-mass static system workspace",
    )
}

pub fn structural_modal_response_spectrum_direction_payload_bytes_upper_bound(
    request: &StructuralModalResponseSpectrumDirectionRequest,
) -> Result<u64, ResourcePlanningError> {
    request.validate().map_err(invalid_error)?;
    let request_bytes = count(
        request.canonical_bytes().map_err(invalid_error)?.len(),
        "request bytes",
    )?;
    let points = count(
        request.spectrum().abscissa.canonical_points().len(),
        "spectrum points",
    )?;
    let dofs = count(request.scope().physical_dof_count(), "physical DOFs")?;
    add(
        DIRECTION_FIXED_PAYLOAD_BYTES,
        add(
            multiply(request_bytes, 2, "embedded request copies")?,
            multiply(
                add(
                    multiply(dofs, 4, "direction response vectors")?,
                    multiply(points, 2, "spectrum axes")?,
                    "direction values",
                )?,
                8,
                "direction value bytes",
            )?,
            "direction variable payload",
        )?,
        "direction payload",
    )
}

pub fn structural_modal_response_spectrum_assembly_payload_bytes_upper_bound(
    request: &StructuralModalResponseSpectrumAssemblyRequest,
) -> Result<u64, ResourcePlanningError> {
    request.validate().map_err(invalid_error)?;
    let request_bytes = count(
        request.canonical_bytes().map_err(invalid_error)?.len(),
        "assembly request bytes",
    )?;
    let dofs = count(
        request.scope().physical_dof_count(),
        "assembly physical DOFs",
    )?;
    let directions = count(request.entries().len(), "assembly directions")?;
    add(
        ASSEMBLY_FIXED_PAYLOAD_BYTES,
        add(
            request_bytes,
            multiply(
                multiply(
                    dofs,
                    add(
                        multiply(directions, 4, "direction vectors")?,
                        1,
                        "final vector",
                    )?,
                    "assembly response values",
                )?,
                8,
                "assembly response bytes",
            )?,
            "assembly variable payload",
        )?,
        "assembly payload",
    )
}

pub const fn structural_modal_response_spectrum_receipt_payload_bytes_upper_bound() -> u64 {
    RECEIPT_PAYLOAD_BYTES
}

pub(super) fn validate_output(
    output: &PreparedOutputResourceFacts,
    expected: u64,
    label: &'static str,
) -> Result<(), ResourcePlanningError> {
    if output.encoded_payload_bytes_upper_bound.0 != expected
        || output.transport_closure_bytes_upper_bound.0 < expected
        || output.construction_resident_bytes_admission_budget.0 < expected
    {
        return Err(invalid(format!(
            "{label} output facts differ from its canonical payload ceiling"
        )));
    }
    Ok(())
}
