//! The study output each view draws. A consumer opens an example on its named result by this
//! name, so it comes from the study's authored output declarations, never from the view id.
use super::profiles::ViewSpec;
use runmat_analysis_core::{
    AuthoredOutput, AuthoredOutputExpression, FieldLocation, ObservationQuantity,
};
use std::collections::BTreeMap;

/// One authored output that republishes a solved field unchanged
/// (`output temperature = analysis.heat.field.node.temperature`).
pub(in crate::commands::fea::snapshot) struct FieldOutput {
    step_id: String,
    quantity: ObservationQuantity,
    location: FieldLocation,
    name: String,
}

/// Names each view by the study output it draws.
pub(in crate::commands::fea::snapshot) enum OutputNames {
    /// A fresh capture: a view draws the output whose field has the view's quantity and
    /// location in the view run's step. The first such output in authored order names it.
    Authored {
        fields: Vec<FieldOutput>,
        /// Each captured run's analysis step, by run id.
        steps: BTreeMap<String, String>,
    },
    /// A redraw of a saved snapshot keeps the names its views were saved with, by view id.
    Saved(BTreeMap<String, String>),
}

impl OutputNames {
    /// The names a fresh capture gives: `outputs` are the study's authored outputs and `runs`
    /// each captured run's id and step.
    pub(in crate::commands::fea::snapshot) fn authored<'a>(
        outputs: &[AuthoredOutput],
        runs: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Self {
        Self::Authored {
            fields: outputs
                .iter()
                .filter_map(|output| match &output.expression {
                    AuthoredOutputExpression::AnalysisField {
                        step_id,
                        quantity,
                        location,
                    } => Some(FieldOutput {
                        step_id: step_id.clone(),
                        quantity: *quantity,
                        location: *location,
                        name: output.output_id.clone(),
                    }),
                    _ => None,
                })
                .collect(),
            steps: runs
                .into_iter()
                .map(|(run, step)| (run.to_string(), step.to_string()))
                .collect(),
        }
    }

    /// The study output `spec` draws; `None` for the solver mesh and for a field no authored
    /// output republishes.
    pub(super) fn name(&self, spec: &ViewSpec) -> Option<String> {
        match self {
            Self::Authored { fields, steps } => {
                let (quantity, location) = spec.source?;
                let step = steps.get(&spec.run_id)?;
                fields
                    .iter()
                    .find(|output| {
                        &output.step_id == step
                            && output.quantity == quantity
                            && output.location == location
                    })
                    .map(|output| output.name.clone())
            }
            Self::Saved(names) => names.get(&spec.id).cloned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::profiles::{tests as profile, CaptureProfile};
    use super::*;
    use runmat_analysis_core::{
        AnalysisFieldLayout, AuthoredOutputPolicy, FieldLocation as L, ObservationQuantity as Q,
    };

    fn field_output(name: &str, step: &str, quantity: Q, location: L) -> AuthoredOutput {
        AuthoredOutput {
            output_id: name.into(),
            expression: AuthoredOutputExpression::AnalysisField {
                step_id: step.into(),
                quantity,
                location,
            },
            policy: AuthoredOutputPolicy::default(),
        }
    }

    fn named(profile: &CaptureProfile, names: &OutputNames) -> Vec<(String, Option<String>)> {
        profile
            .views
            .iter()
            .map(|view| (view.id.clone(), names.name(view)))
            .collect()
    }

    #[test]
    fn field_views_and_their_variants_name_the_authored_output_they_draw() {
        let fields = [
            profile::published(
                "electrical.potential",
                AnalysisFieldLayout::scalar(Q::ElectricPotential, L::Node),
                vec![4],
            ),
            profile::published(
                "electrical.current_density",
                AnalysisFieldLayout::vector3(Q::ElectricCurrentDensity, L::Element),
                vec![5, 3],
            ),
        ];
        let captured =
            CaptureProfile::from_published(&[("run", &fields)], &profile::Values::default())
                .unwrap();
        // `current_density` is authored; the potential is published but no output names it.
        // A reduction over a region (`peak_current_density`) republishes no field.
        let names = OutputNames::authored(
            &[
                field_output(
                    "current_density",
                    "conduction",
                    Q::ElectricCurrentDensity,
                    L::Element,
                ),
                AuthoredOutput {
                    output_id: "peak_current_density".into(),
                    expression: AuthoredOutputExpression::OutputReference {
                        output_id: "current_density".into(),
                    },
                    policy: AuthoredOutputPolicy::default(),
                },
            ],
            [("run", "conduction")],
        );
        let named = named(&captured, &names);
        let output = |id: &str| {
            named
                .iter()
                .find(|(view, _)| view == id)
                .unwrap_or_else(|| panic!("no view {id}: {named:?}"))
                .1
                .as_deref()
        };
        assert_eq!(output("geometry"), None);
        assert_eq!(output("electric-potential"), None);
        assert_eq!(output("current-density"), Some("current_density"));
        assert_eq!(output("current-density-section"), Some("current_density"));
        // The plain field view comes before its variants, so a consumer that opens the first
        // view of an output opens the plain one.
        let first = named
            .iter()
            .find(|(_, output)| output.as_deref() == Some("current_density"))
            .unwrap();
        assert_eq!(first.0, "current-density");
        // Every view carries no mode.
        assert!(captured.views.iter().all(|view| view.mode.is_none()));
    }

    #[test]
    fn an_output_of_another_step_names_nothing() {
        let fields = [profile::published(
            "heat.temperature",
            AnalysisFieldLayout::scalar(Q::Temperature, L::Node),
            vec![4],
        )];
        let captured =
            CaptureProfile::from_published(&[("run", &fields)], &profile::Values::default())
                .unwrap();
        let names = OutputNames::authored(
            &[field_output(
                "temperature",
                "cooldown",
                Q::Temperature,
                L::Node,
            )],
            [("run", "heat")],
        );
        assert!(captured.views.iter().all(|view| names.name(view).is_none()));
        // The same declaration on the captured step names the view.
        let names = OutputNames::authored(
            &[field_output("temperature", "heat", Q::Temperature, L::Node)],
            [("run", "heat")],
        );
        assert_eq!(
            named(&captured, &names)[1],
            ("temperature".to_string(), Some("temperature".to_string()))
        );
    }

    #[test]
    fn buckling_mode_views_name_the_output_and_their_one_based_mode() {
        let mut fields: Vec<_> = (0..2)
            .map(|mode| {
                profile::published(
                    &format!("buckling.mode_shape.{}", mode + 1),
                    AnalysisFieldLayout::vector3(Q::Displacement, L::Node).mode(mode),
                    vec![4, 3],
                )
            })
            .collect();
        fields.push(profile::published(
            "buckling.load_factor",
            AnalysisFieldLayout::scalar(Q::LoadFactor, L::Mode),
            vec![2],
        ));
        let values = profile::Values {
            lists: [("buckling.load_factor".to_string(), vec![3.5, 9.1])].into(),
            ..Default::default()
        };
        let captured = CaptureProfile::from_published(&[("run", &fields)], &values).unwrap();
        let names = OutputNames::authored(
            &[field_output(
                "buckled_shapes",
                "buckling",
                Q::Displacement,
                L::Node,
            )],
            [("run", "buckling")],
        );
        let modes: Vec<_> = captured
            .views
            .iter()
            .map(|view| (view.id.as_str(), names.name(view), view.mode))
            .collect();
        assert_eq!(
            modes,
            [
                ("geometry", None, None),
                ("mode-1", Some("buckled_shapes".to_string()), Some(1)),
                ("mode-2", Some("buckled_shapes".to_string()), Some(2)),
            ]
        );
    }

    #[test]
    fn a_redraw_keeps_the_saved_names_by_view_id() {
        let fields = [profile::published(
            "heat.temperature",
            AnalysisFieldLayout::scalar(Q::Temperature, L::Node),
            vec![4],
        )];
        let captured =
            CaptureProfile::from_published(&[("run", &fields)], &profile::Values::default())
                .unwrap();
        let names = OutputNames::Saved([("temperature".to_string(), "t".to_string())].into());
        assert_eq!(
            named(&captured, &names),
            [
                ("geometry".to_string(), None),
                ("temperature".to_string(), Some("t".to_string())),
            ]
        );
    }
}
