#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "occt_bridge.hxx"

#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <Message_ProgressRange.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <ShapeFix_Shape.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Shape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Mat.hxx>
#include <gp_Pnt.hxx>
#include <gp_XYZ.hxx>

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

void require_count(rust::Slice<const double> values,
                   std::size_t expected,
                   const char* field) {
  if (values.size() != expected) {
    throw std::runtime_error(std::string(field) + " has an invalid value count");
  }
  for (double value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(std::string(field) + " contains a non-finite value");
    }
  }
}

void require_operands(const rust::Vec<OcctFeatureOperand>& operands,
                      std::size_t minimum,
                      const char* operation) {
  if (operands.size() < minimum) {
    throw std::runtime_error(std::string(operation) + " has too few operands");
  }
}

TopoDS_Shape read_operand(const OcctFeatureOperand& operand) {
  if (operand.representation.empty()) {
    throw std::runtime_error("OCCT feature operand is empty");
  }
  std::string bytes(reinterpret_cast<const char*>(operand.representation.data()),
                    operand.representation.size());
  std::istringstream stream(bytes);
  TopoDS_Shape shape;
  BRep_Builder builder;
  BRepTools::Read(shape, stream, builder, Message_ProgressRange());
  if (shape.IsNull()) {
    throw std::runtime_error("OCCT feature operand is not a valid BREP shape");
  }
  return shape;
}

gp_Dir direction(rust::Slice<const double> values, std::size_t offset) {
  const double x = values[offset];
  const double y = values[offset + 1];
  const double z = values[offset + 2];
  if (x * x + y * y + z * z <= 0.0) {
    throw std::runtime_error("OCCT feature direction must be nonzero");
  }
  return gp_Dir(x, y, z);
}

gp_Pnt point(rust::Slice<const double> values, std::size_t offset) {
  return gp_Pnt(values[offset], values[offset + 1], values[offset + 2]);
}

TopoDS_Shape transformed(const TopoDS_Shape& shape,
                         rust::Slice<const double> matrices,
                         std::size_t offset) {
  const double tolerance = 1.0e-12;
  if (std::abs(matrices[offset + 12]) > tolerance ||
      std::abs(matrices[offset + 13]) > tolerance ||
      std::abs(matrices[offset + 14]) > tolerance ||
      std::abs(matrices[offset + 15] - 1.0) > tolerance) {
    throw std::runtime_error("OCCT feature transform must be affine");
  }
  gp_GTrsf transform;
  transform.SetVectorialPart(gp_Mat(matrices[offset], matrices[offset + 1],
                                    matrices[offset + 2], matrices[offset + 4],
                                    matrices[offset + 5], matrices[offset + 6],
                                    matrices[offset + 8], matrices[offset + 9],
                                    matrices[offset + 10]));
  transform.SetTranslationPart(gp_XYZ(matrices[offset + 3],
                                      matrices[offset + 7],
                                      matrices[offset + 11]));
  BRepBuilderAPI_GTransform builder(shape, transform, Standard_True);
  if (!builder.IsDone()) {
    throw std::runtime_error("OCCT general transform did not complete");
  }
  return builder.Shape();
}

template <typename Algorithm>
TopoDS_Shape checked_boolean(Algorithm& algorithm, const char* operation) {
  algorithm.SetNonDestructive(Standard_True);
  algorithm.SetCheckInverted(Standard_True);
  algorithm.Build();
  if (!algorithm.IsDone() || algorithm.Shape().IsNull()) {
    throw std::runtime_error(std::string("OCCT ") + operation + " did not complete");
  }
  algorithm.SimplifyResult();
  ShapeUpgrade_UnifySameDomain unify(algorithm.Shape(), Standard_True, Standard_True,
                                     Standard_True);
  unify.Build();
  if (unify.Shape().IsNull()) {
    throw std::runtime_error(std::string("OCCT ") + operation +
                             " same-domain unification failed");
  }
  ShapeFix_Shape repair(unify.Shape());
  repair.Perform();
  if (repair.Shape().IsNull()) {
    throw std::runtime_error(std::string("OCCT ") + operation +
                             " topology repair failed");
  }
  return repair.Shape();
}

template <typename Algorithm>
TopoDS_Shape execute_boolean_algorithm(Algorithm& algorithm,
                                       const TopoDS_Shape& object,
                                       const std::vector<TopoDS_Shape>& tools,
                                       const char* operation) {
  TopTools_ListOfShape arguments;
  arguments.Append(object);
  TopTools_ListOfShape tool_list;
  for (const TopoDS_Shape& tool : tools) {
    tool_list.Append(tool);
  }
  algorithm.SetArguments(arguments);
  algorithm.SetTools(tool_list);
  return checked_boolean(algorithm, operation);
}

TopoDS_Shape execute_boolean(OcctFeatureKind kind,
                             const rust::Vec<OcctFeatureOperand>& operands) {
  require_operands(operands, 2, "boolean feature");
  const TopoDS_Shape object = read_operand(operands[0]);
  std::vector<TopoDS_Shape> tools;
  tools.reserve(operands.size() - 1);
  for (std::size_t index = 1; index < operands.size(); ++index) {
    tools.push_back(read_operand(operands[index]));
  }
  if (kind == OcctFeatureKind::Union) {
    BRepAlgoAPI_Fuse algorithm;
    return execute_boolean_algorithm(algorithm, object, tools, "union");
  }
  if (kind == OcctFeatureKind::Subtract) {
    BRepAlgoAPI_Cut algorithm;
    return execute_boolean_algorithm(algorithm, object, tools, "subtraction");
  }
  BRepAlgoAPI_Common algorithm;
  return execute_boolean_algorithm(algorithm, object, tools, "intersection");
}

TopoDS_Shape execute_feature(OcctFeatureKind kind,
                             rust::Slice<const double> scalars,
                             rust::Slice<const double> matrices,
                             const rust::Vec<OcctFeatureOperand>& operands) {
  switch (kind) {
    case OcctFeatureKind::Box:
      require_count(scalars, 3, "box");
      return BRepPrimAPI_MakeBox(scalars[0], scalars[1], scalars[2]).Shape();
    case OcctFeatureKind::Cylinder:
      require_count(scalars, 8, "cylinder");
      return BRepPrimAPI_MakeCylinder(
                 gp_Ax2(point(scalars, 5), direction(scalars, 2)),
                 scalars[0], scalars[1])
          .Shape();
    case OcctFeatureKind::Sphere:
      require_count(scalars, 4, "sphere");
      return BRepPrimAPI_MakeSphere(point(scalars, 1), scalars[0]).Shape();
    case OcctFeatureKind::Transform:
      require_operands(operands, 1, "transform");
      require_count(matrices, 16, "transform matrix");
      return transformed(read_operand(operands[0]), matrices, 0);
    case OcctFeatureKind::Instance: {
      require_operands(operands, 1, "instance");
      if (matrices.empty() || matrices.size() % 16 != 0) {
        throw std::runtime_error("OCCT instance transforms must contain 16 values each");
      }
      TopoDS_Compound compound;
      BRep_Builder builder;
      builder.MakeCompound(compound);
      const TopoDS_Shape source = read_operand(operands[0]);
      for (std::size_t offset = 0; offset < matrices.size(); offset += 16) {
        builder.Add(compound, transformed(source, matrices, offset));
      }
      return compound;
    }
    case OcctFeatureKind::Union:
    case OcctFeatureKind::Subtract:
    case OcctFeatureKind::Intersect:
      return execute_boolean(kind, operands);
  }
  throw std::runtime_error("unsupported OCCT exact feature kind");
}

}  // namespace

rust::Vec<std::uint8_t> execute_exact_feature(
    OcctFeatureKind kind,
    rust::Slice<const double> scalars,
    rust::Slice<const double> matrices,
    rust::Vec<OcctFeatureOperand> operands,
    std::uint64_t max_representation_bytes) {
  if (max_representation_bytes == 0) {
    throw std::runtime_error("OCCT feature representation byte bound must be positive");
  }
  TopoDS_Shape shape = execute_feature(kind, scalars, matrices, operands);
  if (shape.IsNull()) {
    throw std::runtime_error("OCCT feature produced a null shape");
  }
  BRepTools::Clean(shape);
  if (!BRepCheck_Analyzer(shape, Standard_True).IsValid()) {
    throw std::runtime_error("OCCT feature produced an invalid BREP shape");
  }
  std::ostringstream stream;
  BRepTools::Write(shape, stream, Message_ProgressRange());
  const std::string bytes = stream.str();
  if (bytes.empty() || bytes.size() > max_representation_bytes) {
    throw std::runtime_error("OCCT feature representation exceeds its admitted byte bound");
  }
  rust::Vec<std::uint8_t> result;
  result.reserve(bytes.size());
  for (unsigned char byte : bytes) {
    result.push_back(static_cast<std::uint8_t>(byte));
  }
  return result;
}

}  // namespace occt_backend
}  // namespace runmat_geometry_io
