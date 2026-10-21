#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_names.hxx"

#include <TCollection_AsciiString.hxx>
#include <TCollection_ExtendedString.hxx>
#include <TDF_LabelSequence.hxx>
#include <TColStd_HSequenceOfExtendedString.hxx>
#include <TDataStd_Name.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_LayerTool.hxx>
#include <TopExp_Explorer.hxx>
#include <XCAFDoc_ShapeTool.hxx>

#include <algorithm>
#include <cctype>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

const std::vector<std::string>& no_names() {
  static const std::vector<std::string> empty;
  return empty;
}

std::string trimmed(std::string value) {
  const auto space = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), space));
  value.erase(std::find_if_not(value.rbegin(), value.rend(), space).base(), value.end());
  return value;
}

std::string utf8(const TCollection_ExtendedString& value) {
  const TCollection_AsciiString text(value, '\0');
  return trimmed(std::string(text.ToCString()));
}

std::vector<std::string> label_layers(const Handle(XCAFDoc_LayerTool)& layers,
                                      const TDF_Label& label) {
  std::vector<std::string> result;
  if (layers.IsNull() || label.IsNull()) {
    return result;
  }
  Handle(TColStd_HSequenceOfExtendedString) names = layers->GetLayers(label);
  if (names.IsNull()) {
    return result;
  }
  for (Standard_Integer index = 1; index <= names->Length(); ++index) {
    const std::string name = utf8(names->Value(index));
    if (!name.empty() && std::find(result.begin(), result.end(), name) == result.end()) {
      result.push_back(name);
    }
  }
  return result;
}

} // namespace

std::string exact_label_name(const TDF_Label& label) {
  if (label.IsNull()) {
    return {};
  }
  Handle(TDataStd_Name) name;
  if (!label.FindAttribute(TDataStd_Name::GetID(), name) || name.IsNull()) {
    return {};
  }
  // A zero replacement character requests UTF-8 for non-ASCII names.
  return utf8(name->Get());
}

void ExactNameTable::add(std::unordered_map<const TopoDS_TShape*, std::vector<std::string>>& map,
                         const TopoDS_Shape& shape,
                         const std::string& name) {
  if (shape.IsNull() || name.empty()) {
    return;
  }
  std::vector<std::string>& names = map[shape.TShape().get()];
  if (std::find(names.begin(), names.end(), name) == names.end()) {
    names.push_back(name);
  }
}

void ExactNameTable::add_face_name(const TopoDS_Shape& face, const std::string& name) {
  retained_.push_back(face);
  add(faces_, face, name);
}

void ExactNameTable::add_solid_name(const TopoDS_Shape& solid, const std::string& name) {
  retained_.push_back(solid);
  add(solids_, solid, name);
}

void ExactNameTable::add_face_tag(const TopoDS_Shape& face, const std::string& tag) {
  retained_.push_back(face);
  add(face_tags_, face, tag);
}

void ExactNameTable::add_solid_tag(const TopoDS_Shape& solid, const std::string& tag) {
  retained_.push_back(solid);
  add(solid_tags_, solid, tag);
}

const std::vector<std::string>& ExactNameTable::face_tags(const TopoDS_Shape& face) const {
  const auto found = face_tags_.find(face.TShape().get());
  return found == face_tags_.end() ? no_names() : found->second;
}

std::vector<std::string> ExactNameTable::solid_tags(const TopoDS_Shape& solid) const {
  std::vector<std::string> result = body_tags_;
  const auto found = solid_tags_.find(solid.TShape().get());
  if (found != solid_tags_.end()) {
    for (const std::string& tag : found->second) {
      if (std::find(result.begin(), result.end(), tag) == result.end()) {
        result.push_back(tag);
      }
    }
  }
  return result;
}

ExactNameTable ExactNameTable::transferred(
    const TopoDS_Shape& source,
    const std::function<TopoDS_Shape(const TopoDS_Shape&)>& image) const {
  ExactNameTable result;
  result.body_name_ = body_name_;
  result.body_tags_ = body_tags_;
  for (TopExp_Explorer faces(source, TopAbs_FACE); faces.More(); faces.Next()) {
    const TopoDS_Shape target = image(faces.Current());
    for (const std::string& name : face_names(faces.Current())) {
      result.add_face_name(target, name);
    }
    for (const std::string& tag : face_tags(faces.Current())) {
      result.add_face_tag(target, tag);
    }
  }
  for (TopExp_Explorer solids(source, TopAbs_SOLID); solids.More(); solids.Next()) {
    const TopoDS_Shape target = image(solids.Current());
    const auto names = solids_.find(solids.Current().TShape().get());
    if (names != solids_.end()) {
      for (const std::string& name : names->second) {
        result.add_solid_name(target, name);
      }
    }
    const auto tags = solid_tags_.find(solids.Current().TShape().get());
    if (tags != solid_tags_.end()) {
      for (const std::string& tag : tags->second) {
        result.add_solid_tag(target, tag);
      }
    }
  }
  return result;
}

void ExactNameTable::set_body_name(const std::string& name) {
  body_name_ = name;
}

const std::vector<std::string>& ExactNameTable::face_names(const TopoDS_Shape& face) const {
  const auto found = faces_.find(face.TShape().get());
  return found == faces_.end() ? no_names() : found->second;
}

std::vector<std::string> ExactNameTable::solid_names(const TopoDS_Shape& solid) const {
  const auto found = solids_.find(solid.TShape().get());
  if (found != solids_.end() && !found->second.empty()) {
    return found->second;
  }
  if (!body_name_.empty()) {
    return {body_name_};
  }
  return {};
}

ExactNameTable ExactNameTable::from_definition(const TDF_Label& definition,
                                               const TDF_Label& occurrence) {
  ExactNameTable table;
  std::string body = exact_label_name(definition);
  if (body.empty() && !occurrence.IsNull() && !(occurrence == definition)) {
    body = exact_label_name(occurrence);
  }
  table.set_body_name(body);
  if (definition.IsNull()) {
    return table;
  }
  const Handle(XCAFDoc_LayerTool) layers = XCAFDoc_DocumentTool::LayerTool(definition);
  table.body_tags_ = label_layers(layers, definition);
  if (!occurrence.IsNull() && !(occurrence == definition)) {
    for (const std::string& tag : label_layers(layers, occurrence)) {
      if (std::find(table.body_tags_.begin(), table.body_tags_.end(), tag) ==
          table.body_tags_.end()) {
        table.body_tags_.push_back(tag);
      }
    }
  }
  TDF_LabelSequence labels;
  if (!XCAFDoc_ShapeTool::GetSubShapes(definition, labels)) {
    return table;
  }
  for (Standard_Integer index = 1; index <= labels.Length(); ++index) {
    const TDF_Label& label = labels.Value(index);
    const std::string name = exact_label_name(label);
    const std::vector<std::string> tags = label_layers(layers, label);
    if (name.empty() && tags.empty()) {
      continue;
    }
    const TopoDS_Shape shape = XCAFDoc_ShapeTool::GetShape(label);
    if (shape.IsNull()) {
      continue;
    }
    // A named group of solids names each solid; a named shell or group of faces names each face.
    if (shape.ShapeType() > TopAbs_FACE) {
      continue;
    }
    if (TopExp_Explorer(shape, TopAbs_SOLID).More()) {
      for (TopExp_Explorer solids(shape, TopAbs_SOLID); solids.More(); solids.Next()) {
        table.add_solid_name(solids.Current(), name);
        for (const std::string& tag : tags) {
          table.add_solid_tag(solids.Current(), tag);
        }
      }
    } else {
      for (TopExp_Explorer faces(shape, TopAbs_FACE); faces.More(); faces.Next()) {
        table.add_face_name(faces.Current(), name);
        for (const std::string& tag : tags) {
          table.add_face_tag(faces.Current(), tag);
        }
      }
    }
  }
  return table;
}

void append_names(rust::Vec<rust::String>& destination, const std::vector<std::string>& names) {
  for (const std::string& name : names) {
    destination.push_back(rust::String(name));
  }
}

} // namespace occt_backend
} // namespace runmat_geometry_io
