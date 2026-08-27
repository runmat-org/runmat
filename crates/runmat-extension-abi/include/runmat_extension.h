#ifndef RUNMAT_EXTENSION_H
#define RUNMAT_EXTENSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RUNMAT_EXTENSION_ABI_MAJOR 1
#define RUNMAT_EXTENSION_ABI_MINOR 1
#define RUNMAT_EXTENSION_QUERY_SYMBOL runmat_extension_query_v1

typedef struct { uint16_t major; uint16_t minor; } RunMatAbiVersion;
typedef struct { uint64_t bits; } RunMatExtensionCapabilities;
#define RUNMAT_EXTENSION_CAP_INVOKE (UINT64_C(1) << 0)
#define RUNMAT_EXTENSION_CAP_READ_VALUE (UINT64_C(1) << 1)
#define RUNMAT_EXTENSION_CAP_WRITE_VALUE (UINT64_C(1) << 2)
#define RUNMAT_EXTENSION_CAP_CALLBACK (UINT64_C(1) << 3)
#define RUNMAT_EXTENSION_CAP_TRANSFER (UINT64_C(1) << 4)
#define RUNMAT_EXTENSION_CAP_SERIALIZE (UINT64_C(1) << 5)
#define RUNMAT_EXTENSION_CAP_ZERO_COPY (UINT64_C(1) << 6)
#define RUNMAT_EXTENSION_CAP_BUFFER_LEASES (UINT64_C(1) << 7)
typedef struct { uint64_t host; uint64_t resource; uint64_t generation; } RunMatForeignHandle;
typedef struct { uint64_t resource; uint64_t generation; } RunMatValueHandle;
typedef struct { uint64_t resource; uint64_t generation; } RunMatBufferLeaseHandle;
typedef struct { const uint8_t *data; size_t length; } RunMatUtf8View;
typedef struct { RunMatUtf8View identifier; RunMatUtf8View message; } RunMatErrorView;

typedef enum {
  RUNMAT_STATUS_OK = 0,
  RUNMAT_STATUS_INVALID_ARGUMENT = 1,
  RUNMAT_STATUS_UNSUPPORTED = 2,
  RUNMAT_STATUS_CANCELLED = 3,
  RUNMAT_STATUS_FAILED = 4,
  RUNMAT_STATUS_PANIC = 5,
  RUNMAT_STATUS_ABI_MISMATCH = 6,
  RUNMAT_STATUS_STALE_HANDLE = 7,
  RUNMAT_STATUS_AFFINITY_VIOLATION = 8
} RunMatStatusCode;

typedef enum {
  RUNMAT_VALUE_UNKNOWN = 0,
  RUNMAT_VALUE_SCALAR = 1,
  RUNMAT_VALUE_DENSE = 2,
  RUNMAT_VALUE_SPARSE = 3,
  RUNMAT_VALUE_LOGICAL = 4,
  RUNMAT_VALUE_CHARACTER = 5,
  RUNMAT_VALUE_STRING = 6,
  RUNMAT_VALUE_CELL = 7,
  RUNMAT_VALUE_STRUCTURE = 8,
  RUNMAT_VALUE_OBJECT = 9,
  RUNMAT_VALUE_CALLABLE = 10,
  RUNMAT_VALUE_FOREIGN = 11
} RunMatValueKind;

typedef enum {
  RUNMAT_ELEMENT_UNKNOWN = 0,
  RUNMAT_ELEMENT_F64 = 1,
  RUNMAT_ELEMENT_F32 = 2,
  RUNMAT_ELEMENT_I8 = 3,
  RUNMAT_ELEMENT_I16 = 4,
  RUNMAT_ELEMENT_I32 = 5,
  RUNMAT_ELEMENT_I64 = 6,
  RUNMAT_ELEMENT_U8 = 7,
  RUNMAT_ELEMENT_U16 = 8,
  RUNMAT_ELEMENT_U32 = 9,
  RUNMAT_ELEMENT_U64 = 10,
  RUNMAT_ELEMENT_LOGICAL = 11,
  RUNMAT_ELEMENT_CHARACTER_U32 = 12,
  RUNMAT_ELEMENT_COMPLEX_F64 = 13,
  RUNMAT_ELEMENT_COMPLEX_F32 = 14
} RunMatElementType;

typedef struct {
  const uint8_t *data;
  size_t byte_length;
  const size_t *shape;
  size_t rank;
  RunMatElementType element_type;
  uint32_t flags;
} RunMatBufferView;

typedef struct {
  RunMatBufferView view;
  RunMatBufferLeaseHandle handle;
} RunMatBufferLease;

typedef struct {
  void *context;
  const RunMatValueHandle *arguments;
  size_t argument_count;
  size_t requested_outputs;
  const void *cancellation;
} RunMatExtensionCall;

typedef struct {
  RunMatStatusCode status;
  const RunMatValueHandle *outputs;
  size_t output_count;
  RunMatErrorView error;
} RunMatExtensionResult;

typedef struct RunMatHostVTable RunMatHostVTable;
struct RunMatHostVTable {
  RunMatAbiVersion abi_version;
  size_t struct_size;
  RunMatExtensionCapabilities capabilities;
  void *context;
  RunMatStatusCode (*retain_value)(void *, RunMatValueHandle);
  RunMatStatusCode (*release_value)(void *, RunMatValueHandle);
  RunMatStatusCode (*value_kind)(void *, RunMatValueHandle, RunMatValueKind *);
  RunMatStatusCode (*borrow_buffer)(void *, RunMatValueHandle, RunMatBufferView *);
  RunMatStatusCode (*invoke_callback)(void *, RunMatUtf8View, const RunMatExtensionCall *, RunMatExtensionResult *);
  bool (*is_cancelled)(void *, const void *);
  RunMatStatusCode (*borrow_buffer_lease)(void *, RunMatValueHandle, RunMatBufferLease *);
  RunMatStatusCode (*release_buffer)(void *, RunMatBufferLeaseHandle);
};

typedef struct RunMatExtensionVTable RunMatExtensionVTable;
struct RunMatExtensionVTable {
  RunMatAbiVersion abi_version;
  size_t struct_size;
  RunMatExtensionCapabilities required_host_capabilities;
  RunMatExtensionCapabilities provided_capabilities;
  RunMatStatusCode (*initialize)(const RunMatHostVTable *, void **);
  RunMatStatusCode (*invoke)(void *, RunMatUtf8View, const RunMatExtensionCall *, RunMatExtensionResult *);
  void (*shutdown)(void *);
};

typedef RunMatStatusCode (*RunMatExtensionQueryFn)(RunMatAbiVersion, RunMatExtensionVTable *);

#ifdef __cplusplus
}
#endif

#endif
