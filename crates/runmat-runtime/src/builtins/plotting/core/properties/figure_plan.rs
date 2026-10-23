//! Parse Figure edits once; the same typed values price and commit the update.
use super::super::state::PropertyMutationTarget;
use super::*;

pub(super) struct PreparedFigureProperties {
    edits: Vec<FigurePropertyEdit>,
}

enum FigurePropertyEdit {
    Name(String),
    NumberTitle(bool),
    Visible(bool),
    Position([f64; 4]),
    Color(glam::Vec4),
    Tag(String),
    CurrentAxes {
        index: usize,
        receiver: Option<runmat_value::GraphicsReceiverRef>,
    },
    SuperTitle {
        text: Option<String>,
        style: Option<TextStyle>,
    },
}

impl PreparedFigureProperties {
    pub(super) fn new(
        handle: FigureHandle,
        args: crate::call::FunctionArgumentView<'_>,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        if args.is_empty() || !args.len().is_multiple_of(2) {
            return Err(plotting_error(
                builtin,
                format!("{builtin}: property/value arguments must come in pairs"),
            ));
        }
        let mut edits = Vec::with_capacity(args.len() / 2);
        for pair in args.chunks_exact(2) {
            edits.push(FigurePropertyEdit::parse(
                &pair[0],
                &pair[1],
                Some(handle),
                builtin,
            )?);
        }
        Ok(Self { edits })
    }

    pub(super) fn proposal(
        &self,
        selection: &super::super::state::GraphicsPropertySelection,
        builtin: &'static str,
    ) -> BuiltinResult<crate::context::RuntimeGraphicsSceneProposal> {
        let original = PropertyMutationTarget::selected(selection.clone())
            .figure_snapshot(selection.figure_handle())
            .map_err(|source| map_figure_error(builtin, source))?;
        let mut overlay = original.metadata_overlay();
        let proposed = original
            .retained_snapshot_bytes_with_metadata(self.edits.iter().map(|edit| {
                match edit {
                    FigurePropertyEdit::Name(name) => overlay.name = Some(name),
                    FigurePropertyEdit::CurrentAxes { index, .. } => {
                        overlay.active_axes = Some(*index)
                    }
                    FigurePropertyEdit::SuperTitle { text, style } => {
                        if let Some(text) = text {
                            overlay.sg_title = Some(text);
                        }
                        if let Some(style) = style {
                            overlay.sg_title_style = style;
                        }
                    }
                    FigurePropertyEdit::NumberTitle(_)
                    | FigurePropertyEdit::Visible(_)
                    | FigurePropertyEdit::Position(_)
                    | FigurePropertyEdit::Color(_)
                    | FigurePropertyEdit::Tag(_) => {}
                }
                overlay
            }))
            .ok_or_else(|| {
                crate::build_runtime_error("proposed Figure snapshot retention cannot be bounded")
                    .with_builtin(builtin)
                    .with_identifier("RunMat:graphics:ProposedSceneBudgetUnavailable")
                    .build()
            })?;
        crate::context::RuntimeGraphicsSceneProposal::from_property(selection, &original, proposed)
    }

    pub(super) fn apply(
        self,
        authority: &PropertyMutationTarget,
        handle: FigureHandle,
        builtin: &'static str,
    ) -> BuiltinResult<PropertySetOutcome> {
        // The original selection is validated again by each actual write guard.
        for edit in self.edits {
            edit.apply(authority, handle, builtin)?;
        }
        PropertySetOutcome::presentation(authority, handle, builtin)
    }
}

impl FigurePropertyEdit {
    fn parse(
        key: &Value,
        value: &Value,
        target: Option<FigureHandle>,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        let key = property_name(key, builtin)?;
        let opts = LineStyleParseOptions::generic(builtin);
        Ok(match key.as_str() {
            "name" => Self::Name(
                value_as_text_string(value)
                    .map_err(|source| {
                        crate::builtins::plotting::plotting_error_with_source(
                            builtin,
                            source.to_string(),
                            source,
                        )
                    })?
                    .ok_or_else(|| {
                        plotting_error(builtin, format!("{builtin}: Name must be text"))
                    })?,
            ),
            "tag" => Self::Tag(
                value_as_text_string(value)
                    .map_err(|source| {
                        crate::builtins::plotting::plotting_error_with_source(
                            builtin,
                            source.to_string(),
                            source,
                        )
                    })?
                    .ok_or_else(|| {
                        plotting_error(builtin, format!("{builtin}: Tag must be text"))
                    })?,
            ),
            "numbertitle" => Self::NumberTitle(value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: NumberTitle must be logical"))
            })?),
            "visible" => Self::Visible(value_as_bool(value).ok_or_else(|| {
                plotting_error(builtin, format!("{builtin}: Visible must be logical"))
            })?),
            "position" => Self::Position(parse_figure_position(value, builtin)?),
            "color" => Self::Color(parse_color_value(&opts, value)?),
            "currentaxes" => {
                let selected = receiver::select(value, builtin)?;
                let (figure, index, receiver) = match selected {
                    Some(selected) => match selected.receiver().kind() {
                        runmat_value::GraphicsReceiverKind::Axes { figure, index } => (
                            FigureHandle::from(figure),
                            index,
                            Some(selected.receiver().clone()),
                        ),
                        _ => {
                            return Err(plotting_error(
                                builtin,
                                format!("{builtin}: CurrentAxes must be an axes handle"),
                            ))
                        }
                    },
                    None => {
                        let PlotHandle::Axes(figure, index) = resolve_plot_handle(value, builtin)?
                        else {
                            return Err(plotting_error(
                                builtin,
                                format!("{builtin}: CurrentAxes must be an axes handle"),
                            ));
                        };
                        (figure, index, None)
                    }
                };
                let Some(target) = target else {
                    return Err(plotting_error(
                        builtin,
                        format!("{builtin}: CurrentAxes requires an existing target figure"),
                    ));
                };
                if figure != target {
                    return Err(plotting_error(
                        builtin,
                        format!("{builtin}: CurrentAxes must belong to the target figure"),
                    ));
                }
                Self::CurrentAxes { index, receiver }
            }
            "sgtitle" => {
                if let Some(text) = value_as_text_string(value).map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        builtin,
                        source.to_string(),
                        source,
                    )
                })? {
                    Self::SuperTitle {
                        text: Some(text),
                        style: None,
                    }
                } else {
                    let scalar = handle_scalar(value, builtin)?;
                    let (source, _, kind) = decode_plot_object_handle(scalar)
                        .map_err(|source| map_figure_error(builtin, source))?;
                    if kind != PlotObjectKind::SuperTitle {
                        return Err(plotting_error(
                            builtin,
                            format!(
                                "{builtin}: expected a matching text handle for `{}`",
                                key_name(PlotObjectKind::SuperTitle)
                            ),
                        ));
                    }
                    let figure = PropertyMutationTarget::alias()
                        .figure_snapshot(source)
                        .map_err(|source| {
                            plotting_error_with_source(
                                builtin,
                                format!("{builtin}: invalid figure handle"),
                                source,
                            )
                        })?;
                    if Some(source) == target {
                        // Copying this Figure's own title is an identity at this
                        // point in the ordered plan, including earlier edits.
                        Self::SuperTitle {
                            text: None,
                            style: None,
                        }
                    } else {
                        Self::SuperTitle {
                            text: figure.sg_title.clone(),
                            style: Some(figure.sg_title_style.clone()),
                        }
                    }
                }
            }
            other => {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: unsupported figure property `{other}`"),
                ))
            }
        })
    }

    fn apply(
        self,
        authority: &PropertyMutationTarget,
        handle: FigureHandle,
        builtin: &'static str,
    ) -> BuiltinResult<()> {
        use super::super::state;
        match self {
            Self::Name(name) => state::set_figure_name_for_property(authority, handle, name),
            Self::Tag(tag) => state::set_figure_tag_for_property(authority, handle, tag),
            Self::NumberTitle(enabled) => {
                state::set_figure_number_title_for_property(authority, handle, enabled)
            }
            Self::Visible(visible) => {
                state::set_figure_visible_for_property(authority, handle, visible).map(|_| ())
            }
            Self::Position(position) => {
                state::set_figure_position_for_property(authority, handle, position)
            }
            Self::Color(color) => {
                state::set_figure_background_color_for_property(authority, handle, color)
            }
            Self::CurrentAxes { index, receiver } => match receiver {
                Some(receiver) => state::select_axes_receiver_for_property(authority, &receiver),
                None => state::select_axes_for_figure_for_property(authority, handle, index),
            },
            Self::SuperTitle { text, style } => {
                state::set_sg_title_properties_for_figure_for_property(
                    authority, handle, text, style,
                )
                .map(|_| ())
            }
        }
        .map_err(|source| map_figure_error(builtin, source))
    }
}

pub(super) fn validate(
    key: &Value,
    value: &Value,
    target: Option<FigureHandle>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    FigurePropertyEdit::parse(key, value, target, builtin).map(|_| ())
}
