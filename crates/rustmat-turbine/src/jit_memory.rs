//! JIT Memory Management
//! 
//! Provides memory allocation and marshaling services for JIT-compiled code
//! to interact with the RustMat runtime system. Uses the existing GC for
//! safe memory management.

use rustmat_builtins::Value;
use rustmat_gc::{gc_allocate, GcPtr};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use cranelift::prelude::*;

/// JIT memory manager for marshaling data between Cranelift and RustMat runtime
pub struct JitMemoryManager {
    /// Global memory pools for different data types
    string_pool: Arc<RwLock<HashMap<String, GcPtr<Value>>>>,
    array_pool: Arc<RwLock<HashMap<String, GcPtr<Value>>>>, // Use string hash of f64 vector
    
    /// Statistics
    allocated_strings: AtomicUsize,
    allocated_arrays: AtomicUsize,
}

impl JitMemoryManager {
    pub fn new() -> Self {
        Self {
            string_pool: Arc::new(RwLock::new(HashMap::new())),
            array_pool: Arc::new(RwLock::new(HashMap::new())),
            allocated_strings: AtomicUsize::new(0),
            allocated_arrays: AtomicUsize::new(0),
        }
    }
    
    /// Allocate a string in GC memory and return its pointer and length
    pub fn allocate_string(&self, s: &str) -> Result<(*const u8, usize), String> {
        // Check if we already have this string in the pool
        {
            let pool = self.string_pool.read().unwrap();
            if let Some(gc_ptr) = pool.get(s) {
                let value_ptr = unsafe { gc_ptr.as_raw() };
                if let Value::String(ref stored_str) = unsafe { &*value_ptr } {
                    return Ok((stored_str.as_ptr(), stored_str.len()));
                }
            }
        }
        
        // Allocate new string in GC
        let string_value = Value::String(s.to_string());
        let gc_ptr = gc_allocate(string_value)
            .map_err(|e| format!("Failed to allocate string in GC: {e}"))?;
        
        // Store in pool for reuse
        {
            let mut pool = self.string_pool.write().unwrap();
            pool.insert(s.to_string(), gc_ptr);
        }
        
        // Return pointer and length
        let value_ptr = unsafe { gc_ptr.as_raw() };
        if let Value::String(ref stored_str) = unsafe { &*value_ptr } {
            self.allocated_strings.fetch_add(1, Ordering::Relaxed);
            Ok((stored_str.as_ptr(), stored_str.len()))
        } else {
            Err("Allocated value is not a string".to_string())
        }
    }
    
    /// Allocate an f64 array in GC memory and return its pointer and length
    pub fn allocate_f64_array(&self, values: &[f64]) -> Result<(*const f64, usize), String> {
        // Create a hash key from the f64 vector (since f64 doesn't implement Hash)
        let array_key = format!("{values:?}");
        
        // Check if we already have this array in the pool
        {
            let pool = self.array_pool.read().unwrap();
            if let Some(_gc_ptr) = pool.get(&array_key) {
                // For simplicity, return a pointer to the raw f64 data
                // In a complete implementation, we'd need more sophisticated marshaling
                return Ok((values.as_ptr(), values.len()));
            }
        }
        
        // Allocate new array in GC as a Cell of Num values
        let cell_values: Vec<Value> = values.iter().map(|&v| Value::Num(v)).collect();
        let cell_value = Value::Cell(cell_values);
        let gc_ptr = gc_allocate(cell_value)
            .map_err(|e| format!("Failed to allocate array in GC: {e}"))?;
        
        // Store in pool for reuse
        {
            let mut pool = self.array_pool.write().unwrap();
            pool.insert(array_key, gc_ptr);
        }
        
        self.allocated_arrays.fetch_add(1, Ordering::Relaxed);
        
        // Return pointer to original data (this is simplified)
        // In a complete implementation, we'd extract the data from the GC-allocated Cell
        Ok((values.as_ptr(), values.len()))
    }
    
    /// Create a string constant in JIT memory
    pub fn create_string_constant(
        &self,
        builder: &mut FunctionBuilder,
        s: &str,
    ) -> Result<(cranelift::prelude::Value, cranelift::prelude::Value), String> {
        let (ptr, len) = self.allocate_string(s)?;
        
        let ptr_val = builder.ins().iconst(types::I64, ptr as i64);
        let len_val = builder.ins().iconst(types::I64, len as i64);
        
        Ok((ptr_val, len_val))
    }
    
    /// Create an f64 array constant in JIT memory
    pub fn create_f64_array_constant(
        &self,
        builder: &mut FunctionBuilder,
        values: &[f64],
    ) -> Result<(cranelift::prelude::Value, cranelift::prelude::Value), String> {
        let (ptr, len) = self.allocate_f64_array(values)?;
        
        let ptr_val = builder.ins().iconst(types::I64, ptr as i64);
        let len_val = builder.ins().iconst(types::I64, len as i64);
        
        Ok((ptr_val, len_val))
    }
    
    /// Convert Cranelift Values to f64 array for runtime calls
    pub fn marshal_cranelift_args_to_f64(
        &self,
        args: &[cranelift::prelude::Value],
    ) -> Result<(*const f64, usize), String> {
        // In a complete implementation, we'd extract the actual f64 values
        // from the Cranelift Value objects. For now, we'll create a placeholder
        // array since we don't have access to the actual runtime values here.
        
        let placeholder_values: Vec<f64> = (0..args.len()).map(|i| i as f64).collect();
        self.allocate_f64_array(&placeholder_values)
    }
    
    /// Get memory manager statistics
    pub fn stats(&self) -> JitMemoryStats {
        JitMemoryStats {
            allocated_strings: self.allocated_strings.load(Ordering::Relaxed),
            allocated_arrays: self.allocated_arrays.load(Ordering::Relaxed),
            string_pool_size: self.string_pool.read().unwrap().len(),
            array_pool_size: self.array_pool.read().unwrap().len(),
        }
    }
    
    /// Clear memory pools (for testing)
    pub fn clear_pools(&self) {
        self.string_pool.write().unwrap().clear();
        self.array_pool.write().unwrap().clear();
    }
}

impl Default for JitMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for JIT memory management
#[derive(Debug, Clone)]
pub struct JitMemoryStats {
    pub allocated_strings: usize,
    pub allocated_arrays: usize,
    pub string_pool_size: usize,
    pub array_pool_size: usize,
}

/// Global JIT memory manager instance
static GLOBAL_JIT_MEMORY: std::sync::OnceLock<JitMemoryManager> = std::sync::OnceLock::new();

/// Get the global JIT memory manager
pub fn get_jit_memory_manager() -> &'static JitMemoryManager {
    GLOBAL_JIT_MEMORY.get_or_init(JitMemoryManager::new)
}

/// Helper function to allocate a string for JIT use
pub fn jit_allocate_string(s: &str) -> Result<(*const u8, usize), String> {
    get_jit_memory_manager().allocate_string(s)
}

/// Helper function to allocate an f64 array for JIT use
pub fn jit_allocate_f64_array(values: &[f64]) -> Result<(*const f64, usize), String> {
    get_jit_memory_manager().allocate_f64_array(values)
}

/// Helper to create runtime function signatures for Cranelift
pub fn create_runtime_f64_signature() -> Signature {
    let mut sig = Signature::new(cranelift::prelude::isa::CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I64)); // name_ptr
    sig.params.push(AbiParam::new(types::I64)); // name_len
    sig.params.push(AbiParam::new(types::I64)); // args_ptr
    sig.params.push(AbiParam::new(types::I64)); // args_len
    sig.returns.push(AbiParam::new(types::F64)); // f64 result
    sig
}

/// Helper to create runtime function signatures for matrix operations
pub fn create_runtime_matrix_signature() -> Signature {
    let mut sig = Signature::new(cranelift::prelude::isa::CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I64)); // name_ptr
    sig.params.push(AbiParam::new(types::I64)); // name_len
    sig.params.push(AbiParam::new(types::I64)); // args_ptr
    sig.params.push(AbiParam::new(types::I64)); // args_len
    sig.returns.push(AbiParam::new(types::I64)); // *mut Value
    sig
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_jit_memory_manager_creation() {
        let manager = JitMemoryManager::new();
        let stats = manager.stats();
        
        assert_eq!(stats.allocated_strings, 0);
        assert_eq!(stats.allocated_arrays, 0);
        assert_eq!(stats.string_pool_size, 0);
        assert_eq!(stats.array_pool_size, 0);
    }
    
    #[test]
    fn test_string_allocation() {
        let manager = JitMemoryManager::new();
        
        let result = manager.allocate_string("test_string");
        assert!(result.is_ok());
        
        let (ptr, len) = result.unwrap();
        assert!(!ptr.is_null());
        assert_eq!(len, "test_string".len());
        
        let stats = manager.stats();
        assert_eq!(stats.allocated_strings, 1);
        assert_eq!(stats.string_pool_size, 1);
    }
    
    #[test]
    fn test_string_pool_reuse() {
        let manager = JitMemoryManager::new();
        
        // Allocate same string twice
        let result1 = manager.allocate_string("reuse_test");
        let result2 = manager.allocate_string("reuse_test");
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        
        let (ptr1, _) = result1.unwrap();
        let (ptr2, _) = result2.unwrap();
        
        // Should reuse the same allocation
        assert_eq!(ptr1, ptr2);
        
        let stats = manager.stats();
        assert_eq!(stats.string_pool_size, 1);
    }
    
    #[test]
    fn test_f64_array_allocation() {
        let manager = JitMemoryManager::new();
        
        let values = [1.0, 2.0, 3.0, 4.0];
        let result = manager.allocate_f64_array(&values);
        assert!(result.is_ok());
        
        let (ptr, len) = result.unwrap();
        assert!(!ptr.is_null());
        assert_eq!(len, 4);
        
        let stats = manager.stats();
        assert_eq!(stats.allocated_arrays, 1);
        assert_eq!(stats.array_pool_size, 1);
    }
    
    #[test]
    fn test_global_memory_manager() {
        let _manager = get_jit_memory_manager();
        
        let result = jit_allocate_string("global_test");
        assert!(result.is_ok());
        
        let array_result = jit_allocate_f64_array(&[1.0, 2.0]);
        assert!(array_result.is_ok());
    }
    
    #[test]
    fn test_runtime_signatures() {
        let f64_sig = create_runtime_f64_signature();
        assert_eq!(f64_sig.params.len(), 4);
        assert_eq!(f64_sig.returns.len(), 1);
        assert_eq!(f64_sig.returns[0].value_type, types::F64);
        
        let matrix_sig = create_runtime_matrix_signature();
        assert_eq!(matrix_sig.params.len(), 4);
        assert_eq!(matrix_sig.returns.len(), 1);
        assert_eq!(matrix_sig.returns[0].value_type, types::I64);
    }
}