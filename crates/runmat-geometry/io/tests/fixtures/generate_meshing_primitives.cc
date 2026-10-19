#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeSolid.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepLib.hxx>
#include <BRepPrimAPI_MakeCone.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <BRepTools.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Ax3.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <gp_Sphere.hxx>
#include <gp_Trsf.hxx>
#include <gp_Torus.hxx>
#include <gp_Vec.hxx>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kPi = 3.14159265358979323846;

bool write_shape(const std::filesystem::path& directory, const std::string& name,
                 const TopoDS_Shape& shape) {
  if (shape.IsNull() || !BRepCheck_Analyzer(shape).IsValid()) {
    std::cerr << name << " is not a valid exact solid\n";
    return false;
  }
  const auto path = directory / name;
  if (!BRepTools::Write(shape, path.string().c_str())) {
    std::cerr << "failed to write " << path << '\n';
    return false;
  }
  return true;
}

TopoDS_Shape translated(const TopoDS_Shape& shape, const gp_Vec& offset) {
  gp_Trsf transform;
  transform.SetTranslation(offset);
  return BRepBuilderAPI_Transform(shape, transform, true).Shape();
}

TopoDS_Shape rotated_about_y(const TopoDS_Shape& shape) {
  gp_Trsf transform;
  transform.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 1.0, 0.0)),
                        1.57079632679489661923);
  return BRepBuilderAPI_Transform(shape, transform, true).Shape();
}

template <typename Surface>
TopoDS_Shape solid_from_parameter_patches(const Surface& surface, double u_min,
                                          double u_max, double v_min,
                                          double v_max, int u_parts,
                                          int v_parts) {
  BRepBuilderAPI_Sewing sewing(1.0e-9);
  for (int u_index = 0; u_index < u_parts; ++u_index) {
    const double first_u =
        u_min + (u_max - u_min) * u_index / u_parts;
    const double last_u =
        u_min + (u_max - u_min) * (u_index + 1) / u_parts;
    for (int v_index = 0; v_index < v_parts; ++v_index) {
      const double first_v =
          v_min + (v_max - v_min) * v_index / v_parts;
      const double last_v =
          v_min + (v_max - v_min) * (v_index + 1) / v_parts;
      BRepBuilderAPI_MakeFace face(surface, first_u, last_u, first_v, last_v);
      if (!face.IsDone()) {
        throw std::runtime_error("failed to construct an analytic face patch");
      }
      sewing.Add(face.Face());
    }
  }
  sewing.Perform();
  const TopoDS_Shape sewn = sewing.SewedShape();
  TopExp_Explorer shells(sewn, TopAbs_SHELL);
  if (!shells.More()) {
    throw std::runtime_error("analytic face patches did not form a shell");
  }
  TopoDS_Solid solid = BRepBuilderAPI_MakeSolid(TopoDS::Shell(shells.Current())).Solid();
  if (shells.Next(); shells.More()) {
    throw std::runtime_error("analytic face patches formed multiple shells");
  }
  if (!BRepLib::OrientClosedSolid(solid)) {
    throw std::runtime_error("analytic face patches did not form a closed solid");
  }
  return solid;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: generate_meshing_primitives <fixture-directory>\n";
    return 2;
  }
  const std::filesystem::path directory(argv[1]);
  const gp_Ax2 z_axis(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0));
  const gp_Ax3 z_frame(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0));
  const TopoDS_Shape cylinder = BRepPrimAPI_MakeCylinder(z_axis, 1.0, 2.0).Shape();
  const TopoDS_Shape millimeter_cylinder =
      BRepPrimAPI_MakeCylinder(z_axis, 1000.0, 2000.0).Shape();

  const bool written =
      write_shape(directory, "meshing_cylinder.brep", cylinder) &&
      write_shape(directory, "meshing_cone.brep",
                  BRepPrimAPI_MakeCone(z_axis, 1.0, 0.35, 2.0).Shape()) &&
      write_shape(directory, "meshing_sphere.brep",
                  solid_from_parameter_patches(gp_Sphere(z_frame, 1.0), 0.0,
                                               2.0 * kPi, -0.5 * kPi,
                                               0.5 * kPi, 4, 2)) &&
      write_shape(directory, "meshing_torus.brep",
                  solid_from_parameter_patches(gp_Torus(z_frame, 2.0, 0.5), 0.0,
                                               2.0 * kPi, -kPi, kPi, 4,
                                               4)) &&
      write_shape(directory, "meshing_cylinder_mm.brep", millimeter_cylinder) &&
      write_shape(directory, "meshing_cylinder_translated.brep",
                  translated(cylinder, gp_Vec(7.0, -3.0, 4.0))) &&
      write_shape(directory, "meshing_cylinder_rotated.brep",
                  rotated_about_y(cylinder));
  return written ? 0 : 1;
}
