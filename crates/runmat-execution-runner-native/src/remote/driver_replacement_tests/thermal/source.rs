use runmat_meshing_core::ElementOrder;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) enum Branch {
    Buckling,
    Modal,
    DirectHarmonic,
    ModalReducedHarmonic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) struct Case {
    pub branch: Branch,
    pub order: ElementOrder,
}

impl Case {
    pub(in super::super) fn matrix() -> impl Iterator<Item = Self> {
        [ElementOrder::Tet4, ElementOrder::Tet10]
            .into_iter()
            .flat_map(|order| {
                [
                    Branch::Buckling,
                    Branch::Modal,
                    Branch::DirectHarmonic,
                    Branch::ModalReducedHarmonic,
                ]
                .into_iter()
                .map(move |branch| Self { branch, order })
            })
    }

    pub(super) fn source(self) -> String {
        let mut source = SOURCE.to_owned();
        if self.order == ElementOrder::Tet10 {
            source = source.replace("order = linear", "order = quadratic");
        }
        // This fixture uses a Cartesian stop: general affine normal constraints are not
        // admitted by every consumer. Imported CAD normals must prove the face is x-normal
        // before a result is accepted as restrained thermal compression.
        source = source.replace("    constraint hot {", STOP);
        let consumer = match self.branch {
            Branch::Buckling => BUCKLING,
            Branch::Modal => MODAL,
            Branch::DirectHarmonic => DIRECT,
            Branch::ModalReducedHarmonic => REDUCED,
        };
        assert!(source.contains(ORIGINAL_CONSUMER));
        source = source.replace(ORIGINAL_CONSUMER, consumer);
        if matches!(
            self.branch,
            Branch::DirectHarmonic | Branch::ModalReducedHarmonic
        ) {
            source = source.replace(
                "    thermal {\n        reference_temperature",
                DAMPED_THERMAL,
            );
            // Probe only the response. The thermal and preload steps explicitly select no
            // mechanical load cases; the temperature state still drives preload compression.
            source = source.replace("    constraint hot {", PROBE);
            source = source.replace("thermal steady {}", "thermal steady { load_cases = [] }");
            source = source.replace(
                "temperature = analysis.heat.state.heated",
                "temperature = analysis.heat.state.heated\nload_cases = []",
            );
            source = source.replace("modes = 3", "modes = 3\nload_cases = []");
        }
        source
    }
}

const SOURCE: &str =
    include_str!("../../../../../../docs/simulation/examples/thermal-prestress.fea");
const STOP: &str = r#"    constraint expansion_stop {
        structural.prescribed {
            target = geometry.part.region.hot
            frame = global
            ux = 0 m
        }
    }
    constraint hot {"#;
const DAMPED_THERMAL: &str = r#"    rayleigh_damping {
        mass_coefficient = 0.2 Hz
        stiffness_coefficient = 200 us
    }
    thermal {
        reference_temperature"#;
const PROBE: &str = r#"    load transverse_probe {
        force on geometry.part.region.hot = [0, 1, 1] N
    }
    load_case response_probe {
        include load.transverse_probe
    }
    constraint hot {"#;
const ORIGINAL_CONSUMER: &str = r#"    analysis modes after analysis.preload {
        structural prestressed_modal {
            prestress = analysis.preload.state.geometric_stiffness
            modes = 1
        }
    }"#;
const MODAL: &str = ORIGINAL_CONSUMER;
const BUCKLING: &str = r#"    analysis buckling after analysis.preload {
        structural buckling {
            prestress = analysis.preload.state.geometric_stiffness
            modes = 1
        }
    }"#;
const DIRECT: &str = r#"    analysis response after analysis.preload {
        structural prestressed_harmonic {
            load_cases = [load_case.response_probe]
            prestress = analysis.preload.state.geometric_stiffness
            frequencies = [60 Hz]
        }
    }"#;
const REDUCED: &str = r#"    analysis modes after analysis.preload {
        structural prestressed_modal {
            prestress = analysis.preload.state.geometric_stiffness
            modes = 3
        }
    }
    analysis response after analysis.modes {
        structural prestressed_harmonic {
            load_cases = [load_case.response_probe]
            prestress = analysis.preload.state.geometric_stiffness
            basis = analysis.modes.state.modal_basis
            frequencies = [60 Hz]
        }
    }"#;
