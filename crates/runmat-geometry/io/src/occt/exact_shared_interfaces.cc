#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_shared_interfaces.hxx"

#include <BOPAlgo_Builder.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepGProp.hxx>
#include <BRep_Builder.hxx>
#include <GProp_GProps.hxx>
#include <Message_ProgressIndicator.hxx>
#include <Message_Report.hxx>
#include <TDF_LabelSequence.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

void check_cancelled(const OcctImportOptions& options) {
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
}

class CancelProgress final : public Message_ProgressIndicator {
public:
  explicit CancelProgress(const OcctImportOptions& options) : options_(options) {}

protected:
  Standard_Boolean UserBreak() override {
    return options_.cancel_token_id != 0 && occt_import_cancelled(options_.cancel_token_id);
  }
  void Show(const Message_ProgressScope&, const Standard_Boolean) override {}

private:
  OcctImportOptions options_;
};

struct PlacedBody {
  TopoDS_Shape shape;
  ExactNameTable names;
  std::string display_name;
  // The occurrence path by instance names below the top-level shape (`left` or `wing/aileron`),
  // empty for a top-level part or when an instance on the path is unnamed.
  std::string occurrence_path;
};

// The path of `component` below `parent`: nullptr when `parent` has none or the instance is
// unnamed (or its name holds the path separator).
std::unique_ptr<std::string> component_path(const std::string* parent,
                                            const TDF_Label& component) {
  const std::string name = exact_label_name(component);
  if (parent == nullptr || name.empty() || name.find('/') != std::string::npos) {
    return nullptr;
  }
  return std::make_unique<std::string>(parent->empty() ? name : *parent + "/" + name);
}

void collect_bodies(const TDF_Label& occurrence,
                    const TopLoc_Location& parent_location,
                    const std::string* path,
                    std::uint32_t depth,
                    std::vector<PlacedBody>& bodies,
                    const OcctImportOptions& options) {
  check_cancelled(options);
  if (occurrence.IsNull() || depth > 256) {
    throw std::runtime_error("OCCT exact assembly exceeds its occurrence depth budget");
  }
  TDF_Label definition = occurrence;
  TDF_Label referred;
  if (XCAFDoc_ShapeTool::IsReference(occurrence) &&
      XCAFDoc_ShapeTool::GetReferredShape(occurrence, referred) && !referred.IsNull()) {
    definition = referred;
  }
  if (XCAFDoc_ShapeTool::IsAssembly(definition)) {
    const TopLoc_Location location = parent_location * XCAFDoc_ShapeTool::GetLocation(occurrence);
    TDF_LabelSequence components;
    XCAFDoc_ShapeTool::GetComponents(definition, components, Standard_False);
    for (Standard_Integer index = 1; index <= components.Length(); ++index) {
      const std::unique_ptr<std::string> child = component_path(path, components.Value(index));
      collect_bodies(
          components.Value(index), location, child.get(), depth + 1, bodies, options);
    }
    return;
  }
  TopoDS_Shape shape = XCAFDoc_ShapeTool::GetShape(definition);
  if (shape.IsNull()) {
    throw std::runtime_error("OCCT exact assembly occurrence has no shape");
  }
  const TopLoc_Location placement =
      parent_location * XCAFDoc_ShapeTool::GetLocation(occurrence) * shape.Location();
  shape.Location(TopLoc_Location());
  const ExactNameTable names = ExactNameTable::from_definition(definition, occurrence);
  PlacedBody body;
  body.display_name = names.body_name();
  body.occurrence_path = path == nullptr ? std::string() : *path;
  // The joined shape is written without placements: each part's placement is applied to its
  // geometry, so every shape in the result is in one frame and no kernel location survives.
  if (placement.IsIdentity()) {
    body.shape = shape;
    body.names = names;
  } else {
    BRepBuilderAPI_Transform placed(shape, placement.Transformation(), Standard_True);
    if (!placed.IsDone()) {
      throw std::runtime_error("OCCT could not place " +
                               (body.display_name.empty() ? std::string("a part")
                                                          : "'" + body.display_name + "'") +
                               " for shared interfaces");
    }
    body.shape = placed.Shape();
    body.names = names.transferred(
        shape, [&placed](const TopoDS_Shape& source) { return placed.ModifiedShape(source); });
  }
  bodies.push_back(std::move(body));
}

double solid_volume(const TopoDS_Shape& solid) {
  GProp_GProps properties;
  BRepGProp::VolumeProperties(solid, properties);
  return std::abs(properties.Mass());
}

std::string quoted(const std::string& name, std::size_t ordinal) {
  return name.empty() ? "body " + std::to_string(ordinal + 1) : "'" + name + "'";
}

std::vector<TopoDS_Shape> images(BOPAlgo_Builder& builder,
                                 const TopoDS_Shape& shape,
                                 TopAbs_ShapeEnum kind) {
  std::vector<TopoDS_Shape> result;
  const TopTools_ListOfShape& modified = builder.Modified(shape);
  if (modified.IsEmpty()) {
    if (!builder.IsDeleted(shape)) {
      result.push_back(shape);
    }
    return result;
  }
  for (TopTools_ListOfShape::Iterator it(modified); it.More(); it.Next()) {
    if (it.Value().ShapeType() == kind) {
      result.push_back(it.Value());
    }
  }
  return result;
}

} // namespace

ExactSharedInterfaces share_exact_interfaces(const TopoDS_Shape& root,
                                             const Handle(XCAFDoc_ShapeTool)& shape_tool,
                                             bool has_xcaf,
                                             const OcctImportOptions& options) {
  std::vector<PlacedBody> bodies;
  if (has_xcaf && !shape_tool.IsNull()) {
    TDF_LabelSequence free_shapes;
    shape_tool->GetFreeShapes(free_shapes);
    for (Standard_Integer index = 1; index <= free_shapes.Length(); ++index) {
      const std::string top;
      collect_bodies(free_shapes.Value(index), TopLoc_Location(), &top, 0, bodies, options);
    }
  } else {
    PlacedBody body;
    body.shape = root;
    if (!root.Location().IsIdentity()) {
      TopoDS_Shape unplaced = root;
      unplaced.Location(TopLoc_Location());
      body.shape = BRepBuilderAPI_Transform(unplaced, root.Location().Transformation(),
                                            Standard_True)
                       .Shape();
    } else {
      body.shape.Location(TopLoc_Location());
    }
    bodies.push_back(std::move(body));
  }

  struct Input {
    TopoDS_Shape solid;
    std::size_t body = 0;
    double volume = 0.0;
  };
  std::vector<Input> inputs;
  for (std::size_t body = 0; body < bodies.size(); ++body) {
    std::size_t solids = 0;
    for (TopExp_Explorer explorer(bodies[body].shape, TopAbs_SOLID); explorer.More();
         explorer.Next()) {
      inputs.push_back({explorer.Current(), body, solid_volume(explorer.Current())});
      ++solids;
    }
    if (solids == 0) {
      throw std::runtime_error("shared interfaces join solid bodies, but " +
                               quoted(bodies[body].display_name, body) + " has no solid");
    }
    if (TopExp_Explorer(bodies[body].shape, TopAbs_FACE, TopAbs_SHELL).More() ||
        TopExp_Explorer(bodies[body].shape, TopAbs_SHELL, TopAbs_SOLID).More()) {
      throw std::runtime_error("shared interfaces join solid bodies, but " +
                               quoted(bodies[body].display_name, body) +
                               " also has faces outside its solids");
    }
  }

  // A General Fuse needs at least two arguments. A single solid has no other body to touch,
  // so it is its own image and keeps every name.
  const bool fuse = inputs.size() > 1;
  BOPAlgo_Builder builder;
  for (const Input& input : inputs) {
    builder.AddArgument(input.solid);
  }
  builder.SetRunParallel(Standard_True);
  builder.SetNonDestructive(Standard_True);
  if (fuse) {
    CancelProgress progress(options);
    builder.Perform(progress.Start());
  }
  check_cancelled(options);
  const auto image_of = [&](const TopoDS_Shape& shape, TopAbs_ShapeEnum kind) {
    return fuse ? images(builder, shape, kind) : std::vector<TopoDS_Shape>{shape};
  };
  if (fuse && builder.HasErrors()) {
    std::ostringstream report;
    builder.DumpErrors(report);
    throw std::runtime_error("OCCT could not join touching bodies into shared interfaces: " +
                             report.str());
  }

  ExactSharedInterfaces result;
  BRep_Builder compound_builder;
  TopoDS_Compound compound;
  compound_builder.MakeCompound(compound);
  std::unordered_map<const TopoDS_TShape*, std::size_t> owners;
  // (rank, volume, name) per shared face, edge or vertex, then sorted into the documented
  // precedence: authored names (rank 0) before their occurrence-qualified forms (rank 1).
  std::unordered_map<const TopoDS_TShape*,
                     std::pair<TopoDS_Shape, std::vector<std::tuple<int, double, std::string>>>>
      subshape_names;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    check_cancelled(options);
    const Input& input = inputs[index];
    const PlacedBody& body = bodies[input.body];
    const std::vector<TopoDS_Shape> solid_images = image_of(input.solid, TopAbs_SOLID);
    if (solid_images.size() != 1) {
      throw std::runtime_error(quoted(body.display_name, input.body) +
                               " interpenetrates another body; shared interfaces need bodies "
                               "that touch without overlapping");
    }
    const TopoDS_Shape& image = solid_images.front();
    const auto [owner, inserted] = owners.emplace(image.TShape().get(), index);
    if (!inserted) {
      throw std::runtime_error(quoted(body.display_name, input.body) + " and " +
                               quoted(bodies[inputs[owner->second].body].display_name,
                                      inputs[owner->second].body) +
                               " occupy the same volume; shared interfaces need bodies that "
                               "touch without overlapping");
    }
    compound_builder.Add(compound, image);
    const std::vector<std::string> solid_names = body.names.solid_names(input.solid);
    for (const std::string& name : solid_names) {
      result.names.add_solid_name(image, name);
    }
    if (!body.occurrence_path.empty()) {
      for (const std::string& name : solid_names) {
        result.names.add_solid_name(image, body.occurrence_path + "/" + name);
      }
    }
    for (const std::string& tag : body.names.solid_tags(input.solid)) {
      result.names.add_solid_tag(image, tag);
    }
    for (const TopAbs_ShapeEnum kind : {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX}) {
      TopTools_IndexedMapOfShape subshapes;
      TopExp::MapShapes(input.solid, kind, subshapes);
      for (Standard_Integer sub = 1; sub <= subshapes.Extent(); ++sub) {
        const TopoDS_Shape& subshape = subshapes(sub);
        const std::vector<std::string>& tags = body.names.subshape_tags(subshape);
        const std::vector<std::string>& names = body.names.subshape_names(subshape);
        if (tags.empty() && names.empty()) {
          continue;
        }
        for (const TopoDS_Shape& subshape_image : image_of(subshape, kind)) {
          for (const std::string& tag : tags) {
            result.names.add_subshape_tag(subshape_image, tag);
          }
          if (names.empty()) {
            continue;
          }
          auto& entry = subshape_names[subshape_image.TShape().get()];
          entry.first = subshape_image;
          for (const std::string& name : names) {
            entry.second.emplace_back(0, input.volume, name);
            if (!body.occurrence_path.empty()) {
              entry.second.emplace_back(1, input.volume, body.occurrence_path + "/" + name);
            }
          }
        }
      }
    }
  }
  for (auto& [key, entry] : subshape_names) {
    (void)key;
    std::sort(entry.second.begin(), entry.second.end());
    for (const auto& [rank, volume, name] : entry.second) {
      (void)rank;
      (void)volume;
      result.names.add_subshape_name(entry.first, name);
    }
  }
  result.shape = compound;
  return result;
}

} // namespace occt_backend
} // namespace runmat_geometry_io
