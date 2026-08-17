#pragma once

#include "rust/cxx.h"

#include <cstdint>

namespace runmat_geometry_io {
namespace occt_backend {

enum class OcctCadFormat : std::uint8_t;
struct OcctImportOptions;
struct OcctImportPayload;
struct OcctExactShapePayload;
struct OcctCurveRangePayload;
struct OcctCurveDerivativesPayload;
struct OcctCurveProjectionPayload;
struct OcctPcurveDerivativesPayload;
struct OcctSurfacePropertiesPayload;
struct OcctSurfaceDerivativesPayload;
struct OcctSurfaceProjectionPayload;
struct OcctMassPropertiesPayload;
struct OcctPreviewSessionChunkOptions;
struct OcctPreviewSessionChunkPayload;
struct OcctPreviewSessionStartPayload;

OcctImportPayload import_cad_bytes(rust::Str path,
                                   rust::Slice<const std::uint8_t> bytes,
                                   OcctCadFormat format,
                                   OcctImportOptions options);
OcctExactShapePayload import_exact_cad_bytes(
    rust::Str path,
    rust::Slice<const std::uint8_t> bytes,
    OcctCadFormat format,
    OcctImportOptions options);
OcctPreviewSessionStartPayload start_cad_preview_session(
    rust::Str path,
    rust::Slice<const std::uint8_t> bytes,
    OcctCadFormat format,
    OcctImportOptions options);
OcctPreviewSessionChunkPayload read_cad_preview_session_chunk(
    std::uint64_t session_id,
    OcctPreviewSessionChunkOptions options);
void close_cad_preview_session(std::uint64_t session_id);
std::uint64_t start_exact_evaluator_session(
    rust::Slice<const std::uint8_t> representation,
    double meters_per_source_unit);
OcctCurveRangePayload exact_curve_range(std::uint64_t session_id,
                                        std::uint64_t shape_key);
OcctCurveDerivativesPayload exact_curve_derivatives(std::uint64_t session_id,
                                                    std::uint64_t shape_key,
                                                    double parameter);
double exact_curve_arc_length(std::uint64_t session_id,
                              std::uint64_t shape_key,
                              double start,
                              double end,
                              double absolute_error_m);
OcctCurveProjectionPayload exact_curve_inverse_project(
    std::uint64_t session_id,
    std::uint64_t shape_key,
    rust::Slice<const double> point_m,
    double absolute_error_m);
OcctPcurveDerivativesPayload exact_pcurve_derivatives(
    std::uint64_t session_id,
    std::uint64_t face_key,
    std::uint64_t wire_key,
    std::uint64_t coedge_position,
    std::int8_t seam_image,
    double parameter);
OcctCurveRangePayload exact_pcurve_range(std::uint64_t session_id,
                                         std::uint64_t face_key,
                                         std::uint64_t wire_key,
                                         std::uint64_t coedge_position,
                                         std::int8_t seam_image);
std::int8_t exact_trim_classify(std::uint64_t session_id,
                                std::uint64_t face_key,
                                double u,
                                double v,
                                double boundary_tolerance_uv);
OcctSurfacePropertiesPayload exact_surface_properties(std::uint64_t session_id,
                                                      std::uint64_t face_key);
OcctSurfaceDerivativesPayload exact_surface_derivatives(std::uint64_t session_id,
                                                        std::uint64_t face_key,
                                                        double u,
                                                        double v);
OcctSurfaceProjectionPayload exact_surface_closest_point(
    std::uint64_t session_id,
    std::uint64_t face_key,
    rust::Slice<const double> point_m,
    double absolute_error_m);
OcctMassPropertiesPayload exact_mass_properties(std::uint64_t session_id,
                                                std::uint64_t shape_key);
void close_exact_evaluator_session(std::uint64_t session_id);

} // namespace occt_backend
} // namespace runmat_geometry_io
