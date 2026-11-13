//! Human-facing documentation for FEA authoring names.
//!
//! Types and dimensions deliberately do not live here. The hover provider reads those from the
//! checked program, so this catalog cannot become a second semantic schema.

use runmat_analysis_language::{FeaType, PhysicalDimension};
use runmat_types::physical::{DynamicDimension, QuantityKind};

pub(super) struct AuthoringDoc {
    pub summary: &'static str,
    pub detail: &'static str,
}

pub(super) fn fallback_authoring_doc() -> AuthoringDoc {
    AuthoringDoc {
        summary: "Checked property of this simulation.",
        detail: "RunMat checks this property's value, shape, and physical units before running the simulation.",
    }
}

pub(super) fn authoring_doc(name: &str) -> Option<AuthoringDoc> {
    let (summary, detail) = match name {
        "assessment" => (
            "Engineering acceptance rule for a declared scalar study output.",
            "The comparison is checked for compatible physical dimensions. Its result is separate from solver convergence and publication quality.",
        ),
        "authored_samples" => (
            "Compare the scalar output at its declared sample coordinates.",
            "A sampled maximum does not certify the continuous peak between samples.",
        ),
        "certified_bound" => (
            "Require a certified absolute error bound to establish acceptance.",
            "Missing evidence or an uncertainty interval crossing the threshold yields an indeterminate assessment.",
        ),
        "mass_density" => (
            "Mass per unit volume used to assemble inertia, gravity, and other body-force terms.",
            "Required by dynamic structural procedures and any formulation whose coefficients depend on material density. The value must be finite and positive.",
        ),
        "youngs_modulus" => (
            "Uniaxial elastic stiffness for an isotropic linear material.",
            "Together with `poisson_ratio`, this defines the isotropic constitutive tensor. The value must be finite and positive.",
        ),
        "poisson_ratio" => (
            "Negative transverse-to-axial strain ratio for an isotropic linear material.",
            "This is dimensionless. Stable three-dimensional isotropic elasticity normally requires a value greater than -1 and less than 0.5.",
        ),
        "target_size" => (
            "Requested characteristic edge length for generated volume elements.",
            "This is a sizing target rather than a guarantee. Curvature, topology recovery, grading, and quality constraints may create smaller elements.",
        ),
        "maximum_deviation" => (
            "Largest permitted chordal distance between the boundary mesh and admitted CAD geometry.",
            "Lower values follow curved geometry more closely and usually increase boundary and volume element counts.",
        ),
        "maximum_grading_ratio" => (
            "Largest permitted size change between neighboring mesh regions.",
            "Values closer to 1 produce smoother size transitions and usually more elements. The value is dimensionless and must be greater than or equal to 1.",
        ),
        "deterministic_seed" => (
            "Seed used by deterministic tie-breaking in the meshing pipeline.",
            "Keeping the seed and all admitted inputs fixed preserves canonical mesh identity across repeat runs.",
        ),
        "order" => (
            "Polynomial order of generated tetrahedral elements.",
            "Use `linear` for Tet4 elements or `quadratic` for Tet10 elements with midside nodes.",
        ),
        "file" => (
            "Project-relative source file for this declaration.",
            "Paths resolve from the module containing the declaration and remain part of the checked source graph.",
        ),
        "units" => (
            "Length unit used to interpret unitless coordinates in imported geometry.",
            "Explicit study quantities keep their own units; this setting applies to the geometry source coordinates.",
        ),
        "maximum_nodes" => (
            "Hard upper bound on nodes produced by meshing.",
            "The run fails instead of silently exceeding this resource envelope.",
        ),
        "maximum_elements" => (
            "Hard upper bound on elements produced by meshing.",
            "The run fails instead of silently exceeding this resource envelope.",
        ),
        "thermal_conductivity" => (
            "Material heat-conduction coefficient.",
            "For an isotropic material this relates heat flux to the negative temperature gradient.",
        ),
        "specific_heat" => (
            "Heat capacity per unit mass.",
            "Transient thermal procedures combine this value with mass density to assemble thermal capacity.",
        ),
        "electrical_conductivity" => (
            "Material electrical conductivity.",
            "This relates electric field to conduction-current density in supported electromagnetic and electrothermal procedures.",
        ),
        "speed_of_sound" => (
            "Small-signal acoustic wave speed in the material.",
            "Together with mass density, this determines the bulk response used by acoustic formulations.",
        ),
        "damping_ratio" | "modal_damping_ratio" => (
            "Fraction of critical damping used by the selected dynamic procedure.",
            "This is dimensionless. A value of `0.02` represents two percent critical damping.",
        ),
        "residual_vectors" => (
            "Global directions whose inertia residual vectors a structural modal extraction adds to its basis.",
            "Each direction adds the static response to a unit acceleration in that direction, with the retained modes removed. Modal-reduced steps that consume the basis solve in modes plus residual vectors.",
        ),
        "static_correction" => (
            "Static correction of a modal-reduced frequency sweep: `applied_loads` or `none`.",
            "`applied_loads` adds the static response to the sweep's applied load to the reduced space, so the quasi-static part of the response is exact and the modal truncation measure falls sharply below the first retained mode.",
        ),
        "time_step" => (
            "Nominal integration increment for a transient procedure.",
            "Choose it to resolve the fastest retained dynamics and any sharp changes in applied signals.",
        ),
        "duration" | "end_time" => (
            "Physical time covered by a transient procedure or sampled signal.",
            "The value must be consistent with the signal domain and time-integration settings.",
        ),
        "frequency" => (
            "Excitation or evaluation frequency.",
            "Frequency quantities are checked dimensionally and converted to canonical SI frequency for execution.",
        ),
        "frequencies" => (
            "Ordered frequencies at which a frequency-domain procedure is evaluated.",
            "Each entry must carry a frequency unit unless its type supplies one through a checked expression.",
        ),
        "modes" => (
            "Number of eigenmodes requested from a modal extraction.",
            "Retaining more modes expands the represented dynamic bandwidth but increases solve and post-processing cost.",
        ),
        "minimum_cumulative_mass_ratio" => (
            "Required cumulative participating-mass fraction for retained modes.",
            "The modal procedure reports or fails the requirement rather than assuming the requested mode count is sufficient.",
        ),
        "direction" => (
            "Direction of the enclosing load, excitation, constraint, or operation.",
            "The accepted representation is determined by the enclosing typed resource and its coordinate frame.",
        ),
        "frame" => (
            "Coordinate frame in which vector components and directions are interpreted.",
            "Use an explicit local frame when components should follow part, material, or connector axes instead of global axes.",
        ),
        "value" => (
            "Typed value applied by the enclosing declaration.",
            "Its shape and physical dimension come from the load, constraint, signal, or initial-condition kind around it.",
        ),
        "target" => (
            "Typed geometry region or resource receiving this declaration.",
            "Topology role, cardinality, and physics compatibility are checked before execution.",
        ),
        "placement" => (
            "Execution placement policy for this study or sweep.",
            "Placement changes where work runs; it does not change the physical model or numerical meaning.",
        ),
        "workers" => (
            "Requested worker count for parallel or distributed execution.",
            "The scheduler validates this request against the selected execution backend and available resources.",
        ),
        "backend" => (
            "Runtime backend selected by an execution policy.",
            "Backend capability is checked separately from the physics contract so unsupported placement fails explicitly.",
        ),
        "checkpoint" => (
            "Persistence policy for restartable execution state.",
            "Checkpoints are execution artifacts; enabling them does not alter the governing model.",
        ),
        "field_id" => (
            "Stable identifier of a solved or derived result field.",
            "Field identity is used for result retrieval, partial access, plotting, and artifact manifests.",
        ),
        "artifact_id" => (
            "Stable identifier for a retained or published run artifact.",
            "Artifact identifiers remain distinct from display labels and filesystem paths.",
        ),
        _ => return None,
    };
    Some(AuthoringDoc { summary, detail })
}

pub(super) fn property_markdown(
    name: &str,
    value_type: Option<&FeaType>,
    documentation: AuthoringDoc,
) -> String {
    let type_line = value_type
        .map(|value_type| format!("\n\n```fea\n{name}: {}\n```", type_name(value_type)))
        .unwrap_or_default();
    let physical_contract = match value_type {
        Some(FeaType::Quantity(dimension)) => dimension_help(*dimension)
            .map(|(dimension, si_unit, examples)| {
                format!(
                    "\n\n| Physical contract | |\n|---|---|\n| Dimension | {dimension} |\n| Canonical SI | `{si_unit}` |\n| Examples | {examples} |"
                )
            })
            .unwrap_or_default(),
        _ => String::new(),
    };
    format!(
        "**`{name}`**{type_line}\n\n{}\n\n{}{physical_contract}",
        documentation.summary, documentation.detail
    )
}

pub(super) fn quantity_markdown(label: &str, source: &str, value_type: Option<&FeaType>) -> String {
    let physical_contract = match value_type {
        Some(FeaType::Quantity(dimension)) => dimension_help(*dimension)
            .map(|(dimension, si_unit, examples)| {
                format!(
                    "\n\n| Resolved contract | |\n|---|---|\n| Dimension | {dimension} |\n| Canonical SI | `{si_unit}` |\n| Other examples | {examples} |"
                )
            })
            .unwrap_or_default(),
        _ => String::new(),
    };
    format!(
        "**{label}**\n\n```fea\n{source}\n```\n\nThe unit fixes this value's physical dimension and scale. RunMat checks that contract before execution.{physical_contract}"
    )
}

fn type_name(value: &FeaType) -> String {
    match value {
        FeaType::Boolean => "boolean".into(),
        FeaType::Integer => "integer".into(),
        FeaType::Real => "real".into(),
        FeaType::Complex => "complex".into(),
        FeaType::String => "string".into(),
        FeaType::Enum => "enum".into(),
        FeaType::Quantity(dimension) => dimension_help(*dimension)
            .map(|(name, _, _)| name.to_string())
            .unwrap_or_else(|| "physical quantity".into()),
        FeaType::Vector { element, length } => length.map_or_else(
            || format!("vector<{}>", type_name(element)),
            |length| format!("vector<{}, {length}>", type_name(element)),
        ),
        FeaType::List(element) => format!("list<{}>", type_name(element)),
        FeaType::Record(_) => "record".into(),
        FeaType::Unknown => "inferred value".into(),
        FeaType::Error => "invalid value".into(),
        _ => "simulation value".into(),
    }
}

fn dimension_help(
    descriptor: DynamicDimension,
) -> Option<(&'static str, &'static str, &'static str)> {
    if descriptor.kind() == Some(QuantityKind::TemperatureDifference) {
        return Some(("temperature difference", "K", "`10 K`, `5 delta_degC`"));
    }
    if descriptor == DynamicDimension::quantity_type(QuantityKind::Angle)
        || descriptor.kind() == Some(QuantityKind::Angle)
    {
        return Some(("angle", "rad", "`45 deg`, `0.5 rad`"));
    }
    // A nominal angle power cannot be described using its SI exponents alone.
    if descriptor.angle_power() != 0 {
        return None;
    }
    let dimension = descriptor.dimension();
    Some(if dimension == PhysicalDimension::DIMENSIONLESS {
        ("dimensionless", "1", "`0.3`, `2 %`")
    } else if dimension == PhysicalDimension::LENGTH {
        ("length", "m", "`2 mm`, `0.75 m`, `4 in`")
    } else if dimension == PhysicalDimension::TIME {
        ("time", "s", "`250 us`, `0.1 s`, `2 min`")
    } else if dimension == PhysicalDimension::FREQUENCY {
        ("frequency", "Hz", "`60 Hz`, `2.5 kHz`")
    } else if dimension == PhysicalDimension::DENSITY {
        ("mass density", "kg/m^3", "`7850 kg/m^3`, `2.70 g/cm^3`")
    } else if dimension == PhysicalDimension::PRESSURE {
        ("pressure or stress", "Pa", "`200 GPa`, `12 MPa`, `5 psi`")
    } else if dimension == PhysicalDimension::TEMPERATURE {
        ("temperature", "K", "`293.15 K`, `20 degC`")
    } else if dimension == PhysicalDimension::VELOCITY {
        ("velocity", "m/s", "`3 m/s`, `25 mm/s`")
    } else if dimension == PhysicalDimension::ACCELERATION {
        ("acceleration", "m/s^2", "`9.80665 m/s^2`, `1 g`")
    } else if dimension == PhysicalDimension::FORCE {
        ("force", "N", "`120 N`, `4.5 kN`")
    } else if dimension == PhysicalDimension::MOMENT {
        ("moment", "N*m", "`40 N*m`, `2 kN*m`")
    } else if dimension == PhysicalDimension::POWER {
        ("power", "W", "`25 W`, `1.2 kW`")
    } else if dimension == PhysicalDimension::HEAT_FLUX {
        ("heat flux", "W/m^2", "`500 W/m^2`, `2 kW/m^2`")
    } else if dimension == PhysicalDimension::THERMAL_CONDUCTIVITY {
        (
            "thermal conductivity",
            "W/(m*K)",
            "`15 W/(m*K)`, `0.2 W/(m*K)`",
        )
    } else if dimension == PhysicalDimension::HEAT_TRANSFER_COEFFICIENT {
        (
            "heat transfer coefficient",
            "W/(m^2*K)",
            "`10 W/m^2/K`, `35 W/(m^2*K)`",
        )
    } else if dimension == PhysicalDimension::SPECIFIC_HEAT {
        ("specific heat", "J/(kg*K)", "`900 J/(kg*K)`")
    } else if dimension == PhysicalDimension::ELECTRICAL_CONDUCTIVITY {
        ("electrical conductivity", "S/m", "`5.8e7 S/m`")
    } else if dimension == PhysicalDimension::ELECTRIC_POTENTIAL {
        ("electric potential", "V", "`5 V`, `230 V`")
    } else if dimension == PhysicalDimension::ELECTRIC_CURRENT {
        ("electric current", "A", "`2 A`, `15 mA`")
    } else {
        return None;
    })
}
