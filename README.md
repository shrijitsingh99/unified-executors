# unified-executors

## TODO

### Executor Design
* **Design**
    1. Add allocator to executors (v0.1)
    2. Improve type trait to check executor instance type (v0.1)
    3. best_fit executor, depends on PCL integration (v0.1)
    4. Decay policies
    5. Fallback
    6. Additional type traits (Continuous)
    7. Add support for additional type traits to check and match various executor instance types
    8. Combined executors
    9. Shapes (Low)
    10. Common module for common and test related macros/functions
    11. cuda_executor (Discussion after v0.1 integration)
    
 * **Test**
    1. require, prefer (both as a class method and a separate function) (v0.1)
    2. Property methods (v0.1)
    3. Executor type traits (v0.1)
    4. Add executor tests (Continuous)
    
* **Additional**
    1. Setup CI (Low)

### PCL Integration
* **Integration**
    1. Modify for C++11 compatibility
    2. Integrate with filter module
    3. Integrate with segmentation module
    4. Analyze breakdown of functions for reusability between executors   
    5. best_fit executor analysis
    6. GPU module restructure (Discussion after v0.1 integration)
    
### Strech Goals
* Additional SIMD implementations for functions
* Additional CUDA implementation for functions

