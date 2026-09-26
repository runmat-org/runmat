use super::*;
#[test]
fn command_descriptors_cover_core_forms() {
    let grid_labels: Vec<&str> = GRID_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(grid_labels.contains(&"enabled = grid()"));
    assert!(grid_labels.contains(&"enabled = grid(mode)"));

    let box_labels: Vec<&str> = BOX_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(box_labels.contains(&"enabled = box()"));
    assert!(box_labels.contains(&"enabled = box(mode)"));

    let axis_labels: Vec<&str> = AXIS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(axis_labels.contains(&"lim = axis()"));
    assert!(axis_labels.contains(&"lim = axis(ax)"));
    assert!(axis_labels.contains(&"axis([xmin xmax ymin ymax | ... cmin cmax])"));
    assert!(axis_labels.contains(&"axis(mode)"));
    assert!(axis_labels.contains(&"axis(visibility)"));

    let cla_labels: Vec<&str> = CLA_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(cla_labels.contains(&"ok = cla()"));

    let colormap_labels: Vec<&str> = COLORMAP_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(colormap_labels.contains(&"cmap = colormap()"));
    assert!(colormap_labels.contains(&"cmap = colormap(target)"));
    assert!(colormap_labels.contains(&"colormap(name)"));
    assert!(colormap_labels.contains(&"colormap(map)"));
    assert!(colormap_labels.contains(&"colormap(target, map)"));

    let shading_labels: Vec<&str> = SHADING_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(shading_labels.contains(&"ok = shading(mode)"));

    let hidden_labels: Vec<&str> = HIDDEN_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(hidden_labels.contains(&"enabled = hidden()"));
    assert!(hidden_labels.contains(&"enabled = hidden(mode)"));
    assert!(hidden_labels.contains(&"enabled = hidden(ax)"));
    assert!(hidden_labels.contains(&"enabled = hidden(ax, mode)"));

    let colorbar_labels: Vec<&str> = COLORBAR_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(colorbar_labels.contains(&"enabled = colorbar()"));
    assert!(colorbar_labels.contains(&"enabled = colorbar(mode)"));
}
