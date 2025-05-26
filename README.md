# Mobius Strip Geometric Modeling

A comprehensive Python implementation for modeling and analyzing Mobius strips using parametric equations. This project computes key geometric properties including surface area and edge length, with rich 3D visualization capabilities.

## 🌀 Features

- **Parametric Modeling**: Complete implementation of Mobius strip parametric equations
- **Geometric Analysis**: Accurate computation of surface area and edge length
- **Dual Calculation Methods**: Both numerical integration and mesh approximation
- **3D Visualization**: Interactive plotting with customizable rendering options
- **Parameter Flexibility**: Adjustable radius, width, and mesh resolution
- **Convergence Analysis**: Built-in validation and accuracy testing

## 📋 Requirements

```
numpy >= 1.19.0
matplotlib >= 3.3.0
scipy >= 1.5.0
```

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/Charanyedida/mobius-strip.git
cd mobius-strip
```

2. Install required dependencies:
```bash
pip install numpy matplotlib scipy
```

## 📖 Usage

### Basic Usage

```python
from mobius_strip import MobiusStrip

# Create a Mobius strip with default parameters
strip = MobiusStrip(radius=2.0, width=1.0, resolution=50)

# Display geometric properties
strip.print_properties()

# Create 3D visualization
fig, ax = strip.plot_3d()
plt.show()
```

### Advanced Usage

```python
# Create custom Mobius strip
strip = MobiusStrip(radius=3.0, width=0.8, resolution=80)

# Compute surface area using different methods
area_numerical = strip.compute_surface_area(method='numerical')
area_mesh = strip.compute_surface_area(method='mesh')

# Compute edge length
edge_length = strip.compute_edge_length()

# Get specific points on the surface
x, y, z = strip.get_point(u=np.pi/2, v=0.2)

# Custom visualization
fig, ax = strip.plot_3d(
    show_wireframe=True, 
    show_surface=True, 
    alpha=0.7
)
```

### Running the Complete Demo

```bash
python mobius_strip.py
```

This will:
- Create multiple Mobius strips with different parameters
- Display geometric properties for each
- Generate 3D visualizations
- Perform convergence analysis

## 🔬 Mathematical Background

### Parametric Equations

The Mobius strip is defined by the parametric equations:

```
x(u,v) = (R + v⋅cos(u/2))⋅cos(u)
y(u,v) = (R + v⋅cos(u/2))⋅sin(u)
z(u,v) = v⋅sin(u/2)
```

Where:
- `u ∈ [0, 2π]` - angle parameter
- `v ∈ [-w/2, w/2]` - width parameter
- `R` - radius (distance from center to strip)
- `w` - strip width

### Surface Area Calculation

Two methods are implemented:

1. **Numerical Integration**:
   ```
   Surface Area = ∫∫ |∂r/∂u × ∂r/∂v| du dv
   ```

2. **Mesh Approximation**:
   - Divides surface into triangular elements
   - Computes area of each triangle
   - Sums total area

### Edge Length

The Mobius strip has a unique topological property: it has only **one edge** that forms a continuous loop. The edge length is computed by parametrizing this boundary curve and integrating along its length.

## 🏗️ Class Structure

### MobiusStrip Class

```python
class MobiusStrip:
    def __init__(self, radius=2.0, width=1.0, resolution=50)
    def get_point(self, u, v)                    # Get point at parameters (u,v)
    def compute_surface_area(self, method)       # Calculate surface area
    def compute_edge_length(self)                # Calculate edge length
    def plot_3d(self, options)                  # 3D visualization
    def print_properties(self)                   # Display all properties
```

### Key Methods

- **`get_point(u, v)`**: Returns 3D coordinates for given parameters
- **`compute_surface_area(method)`**: Calculates surface area using 'numerical' or 'mesh' method
- **`compute_edge_length()`**: Computes the length of the single edge boundary
- **`plot_3d()`**: Creates interactive 3D visualization with customizable options

## 📊 Example Output

```
Mobius Strip Properties:
========================================
Radius (R): 2.0
Width (w): 1.0
Resolution: 50
========================================
Surface Area (numerical): 21.9911
Surface Area (mesh approx): 21.8764
Difference: 0.1147
Edge Length: 14.1372
Theoretical estimate: 12.5664
Relative error: 75.01%
```

## 🎨 Visualization Features

- **Surface Rendering**: Smooth surface with customizable colormap
- **Wireframe Overlay**: Shows mesh structure
- **Edge Highlighting**: Emphasizes the single boundary edge
- **Transparency Control**: Adjustable alpha for better visualization
- **Aspect Ratio**: Proper 3D scaling and perspective

## 🔧 Configuration Options

### Strip Parameters
- `radius`: Distance from center to strip (default: 2.0)
- `width`: Width of the strip (default: 1.0)
- `resolution`: Mesh density (default: 50)

### Visualization Options
- `show_wireframe`: Display mesh lines (default: True)
- `show_surface`: Display surface (default: True)
- `alpha`: Surface transparency (default: 0.7)

## 📈 Performance Notes

- **Memory Usage**: O(n²) where n is resolution
- **Computation Time**: Numerical integration is O(1), mesh approximation is O(n²)
- **Recommended Resolution**: 50-100 for interactive use, 200+ for high-quality output

## 🧪 Validation and Testing

The implementation includes several validation methods:

1. **Convergence Testing**: Results improve with increasing resolution
2. **Method Comparison**: Numerical vs. mesh approximation agreement
3. **Parameter Sensitivity**: Appropriate scaling with parameter changes
4. **Topological Verification**: Single edge property validation

### Running Tests

```python
# Convergence analysis
for resolution in [20, 40, 60, 80, 100]:
    strip = MobiusStrip(resolution=resolution)
    area = strip.compute_surface_area('numerical')
    print(f"Resolution {resolution}: Area = {area:.4f}")
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional geometric properties (Gaussian curvature, mean curvature)
- Alternative parametrizations
- Animation capabilities
- Performance optimizations
- Additional visualization options

## 📚 Educational Applications

This implementation is ideal for:

- **Differential Geometry**: Understanding non-orientable surfaces
- **Topology**: Visualizing single-sided surfaces
- **Numerical Methods**: Comparing integration techniques
- **Computer Graphics**: Parametric surface modeling
- **Mathematics Education**: Interactive geometric exploration

## 🐛 Known Issues

- High resolution (>200) may cause memory issues on limited systems
- Numerical integration accuracy depends on scipy version
- 3D rendering performance varies with graphics capabilities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- Differential Geometry of Curves and Surfaces by Manfredo P. do Carmo
- Elementary Differential Geometry by Barrett O'Neill
- Numerical Methods for Scientists and Engineers by Richard Hamming

## 📞 Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact [yedidacharan931@gmail.com].

---

**Happy Modeling! 🌀**
