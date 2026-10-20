//! Figure property rendering from one selected metadata snapshot.

use super::super::state::{FigurePropertySnapshot, GraphicsPropertySelection};
use super::*;

pub(super) fn get_compatibility(
    handle: FigureHandle,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let snapshot = super::super::state::figure_property_snapshot_alias(handle)
        .map_err(|error| map_figure_error(builtin, error))?;
    render(snapshot, property, builtin)
}

pub(super) fn get_selected(
    selection: &GraphicsPropertySelection,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let snapshot = selection
        .figure_snapshot()
        .map_err(|error| map_figure_error(builtin, error))?;
    render(snapshot, property, builtin)
}

fn render(
    snapshot: FigurePropertySnapshot,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let FigurePropertySnapshot {
        selection,
        handle,
        figure,
        axes,
        current_axes,
        has_sg_title,
        tag,
        waitbar,
    } = snapshot;
    let sg_title_handle =
        super::super::state::encode_plot_object_handle(handle, 0, PlotObjectKind::SuperTitle);
    let result = match property.map(canonical_property_name).as_deref() {
        None => {
            let mut st = runmat_value::UnboundedStructValueBuilder::new();
            st.insert("Handle", Value::Num(handle.as_u32() as f64));
            st.insert("Number", Value::Num(handle.as_u32() as f64));
            st.insert("Type", Value::String("figure".into()));
            st.insert("CurrentAxes", Value::Num(current_axes));
            let mut children = axes;
            if has_sg_title {
                children.push(sg_title_handle);
            }
            st.insert("Children", handles_value(children));
            st.insert("Parent", Value::Num(f64::NAN));
            st.insert(
                "Name",
                Value::string(figure.name.clone().unwrap_or_default()),
            );
            st.insert("NumberTitle", Value::Bool(figure.number_title));
            st.insert("Visible", Value::Bool(figure.visible));
            st.insert("Position", figure_position_value(figure.position));
            st.insert("Color", rgb_triplet_value(figure.background_color));
            st.insert("Tag", Value::string(tag));
            if let Some(waitbar) = waitbar {
                st.insert("WaitbarProgress", Value::Num(waitbar.progress));
                st.insert("WaitbarMessage", Value::string(waitbar.message));
            }
            st.insert("SGTitle", Value::Num(sg_title_handle));
            Ok(Value::Struct(st.finish()))
        }
        Some("number") => Ok(Value::Num(handle.as_u32() as f64)),
        Some("type") => Ok(Value::String("figure".into())),
        Some("currentaxes") => Ok(Value::Num(current_axes)),
        Some("children") => Ok(handles_value({
            let mut children = axes;
            if has_sg_title {
                children.push(sg_title_handle);
            }
            children
        })),
        Some("parent") => Ok(Value::Num(f64::NAN)),
        Some("name") => Ok(Value::string(figure.name.clone().unwrap_or_default())),
        Some("numbertitle") => Ok(Value::Bool(figure.number_title)),
        Some("visible") => Ok(Value::Bool(figure.visible)),
        Some("position") => Ok(figure_position_value(figure.position)),
        Some("color") => Ok(rgb_triplet_value(figure.background_color)),
        Some("waitbarprogress") => Ok(Value::Num(
            waitbar.map(|state| state.progress).unwrap_or(f64::NAN),
        )),
        Some("waitbarmessage") => Ok(Value::string(
            waitbar.map(|state| state.message).unwrap_or_default(),
        )),
        Some("tag") => Ok(Value::string(tag)),
        Some("sgtitle") => Ok(Value::Num(sg_title_handle)),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported figure property `{other}`"),
        )),
    };
    let value = result?;
    if let Some(selection) = selection {
        selection
            .ensure_current()
            .map_err(|error| map_figure_error(builtin, error))?;
    }
    Ok(value)
}
