//! Palette mutation preserves selection and updates direct-image color limits before publication.
use super::*;

#[cfg(test)]
pub fn set_colormap_with_length(colormap: ColorMap, length: usize) {
    let (handle, figure) = {
        let mut reg = registry();
        let handle = reg.current;
        let direct_images = direct_image_limits_updates(&reg.plot_children, handle, None);
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state
            .figure
            .set_axes_colormap_with_length(axes, colormap, length);
        refresh_direct_image_limits(&direct_images, axes, length, &mut state.figure);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    notify_with_figure(handle, &figure, FigureEventKind::Updated);
}

pub(crate) fn set_colormap_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    colormap: ColorMap,
) -> Result<(), FigureError> {
    set_colormap_for_axes_with_length_for_property(
        property,
        handle,
        axes_index,
        colormap,
        DEFAULT_COLORMAP_LENGTH,
    )
}

pub fn set_colormap_for_axes_with_length(
    handle: FigureHandle,
    axes_index: usize,
    colormap: ColorMap,
    length: usize,
) -> Result<(), FigureError> {
    set_colormap_for_axes_with_length_for_property(
        &PropertyMutationTarget::alias(),
        handle,
        axes_index,
        colormap,
        length,
    )
}

pub(crate) fn set_colormap_for_axes_with_length_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    colormap: ColorMap,
    length: usize,
) -> Result<(), FigureError> {
    let figure = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let direct_images =
            direct_image_limits_updates(&reg.plot_children, handle, Some(axes_index));
        let state = existing_axes_state_mut(&mut reg, handle, axes_index)?;
        state
            .figure
            .set_axes_colormap_with_length(axes_index, colormap, length);
        refresh_direct_image_limits(&direct_images, axes_index, length, &mut state.figure);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure, FigureEventKind::Updated)?;
    Ok(())
}

pub fn set_colormap_for_figure_with_length(
    handle: FigureHandle,
    colormap: ColorMap,
    length: usize,
) -> Result<(), FigureError> {
    let figure = {
        let mut reg = registry();
        let direct_images = direct_image_limits_updates(&reg.plot_children, handle, None);
        let state = existing_state_mut(&mut reg, handle)?;
        state.figure.set_figure_colormap(colormap, length);
        for axes in 0..state.figure.axes_metadata.len() {
            refresh_direct_image_limits(&direct_images, axes, length, &mut state.figure);
        }
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure, FigureEventKind::Updated);
    Ok(())
}

pub fn figure_colormap_snapshot(handle: FigureHandle) -> Result<(ColorMap, usize), FigureError> {
    let reg = registry();
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    Ok((
        state.figure.default_colormap.clone(),
        state.figure.default_colormap_length,
    ))
}

fn direct_image_limits_updates(
    children: &PlotChildRegistry,
    figure: FigureHandle,
    axes_index: Option<usize>,
) -> Vec<(usize, usize, bool)> {
    children
        .values()
        .filter_map(|child| match child {
            PlotChildHandleState::Image(image)
                if image.figure == figure
                    && axes_index.is_none_or(|axes| image.axes_index == axes)
                    && image.c_data_mapping == "direct" =>
            {
                Some((
                    image.axes_index,
                    image.plot_index,
                    image
                        .c_data
                        .as_ref()
                        .is_some_and(|tensor| tensor.integer_storage().is_some()),
                ))
            }
            _ => None,
        })
        .collect()
}

fn refresh_direct_image_limits(
    direct_images: &[(usize, usize, bool)],
    axes_index: usize,
    length: usize,
    plot_figure: &mut Figure,
) {
    for &(_, plot_index, integer) in direct_images
        .iter()
        .filter(|(axes, _, _)| *axes == axes_index)
    {
        if let Some(runmat_plot::plots::figure::PlotElement::Surface(surface)) =
            plot_figure.get_plot_mut(plot_index)
        {
            surface.set_color_limits(Some(
                crate::builtins::plotting::image::direct_colormap_limits(integer, length),
            ));
        }
    }
}

pub fn colormap_length_for_axes(handle: FigureHandle, axes_index: usize) -> usize {
    let reg = registry();
    reg.figures
        .get(&handle)
        .and_then(|state| state.figure.axes_metadata(axes_index))
        .map(|meta| meta.colormap_length)
        .unwrap_or(DEFAULT_COLORMAP_LENGTH)
}

pub fn current_colormap_length() -> usize {
    let reg = registry();
    reg.figures
        .get(&reg.current)
        .and_then(|state| state.figure.axes_metadata(state.active_axes))
        .map(|meta| meta.colormap_length)
        .unwrap_or(DEFAULT_COLORMAP_LENGTH)
}

#[cfg(test)]
#[path = "colormap/tests.rs"]
mod tests;
