import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Mobius Strip Explorer",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class MobiusStrip:
    """
    A class to model a Mobius strip using parametric equations and compute
    key geometric properties including surface area and edge length.
    """
    
    def __init__(self, radius=2.0, width=1.0, resolution=50):
        """Initialize the Mobius strip with given parameters."""
        self.R = radius
        self.w = width
        self.n = resolution
        
        # Parameter ranges
        self.u_range = np.linspace(0, 2*np.pi, self.n)
        self.v_range = np.linspace(-self.w/2, self.w/2, self.n)
        
        # Generate mesh
        self.U, self.V = np.meshgrid(self.u_range, self.v_range)
        self._compute_surface_points()
        
    def _compute_surface_points(self):
        """Compute the 3D coordinates of all points on the Mobius strip surface."""
        # Parametric equations
        self.X = (self.R + self.V * np.cos(self.U/2)) * np.cos(self.U)
        self.Y = (self.R + self.V * np.cos(self.U/2)) * np.sin(self.U)
        self.Z = self.V * np.sin(self.U/2)
        
    def get_point(self, u, v):
        """Get a single point on the Mobius strip surface."""
        x = (self.R + v * np.cos(u/2)) * np.cos(u)
        y = (self.R + v * np.cos(u/2)) * np.sin(u)
        z = v * np.sin(u/2)
        return (x, y, z)
    
    def _partial_derivatives(self, u, v):
        """Compute partial derivatives of the parametric surface."""
        # ∂r/∂u
        r_u_x = -(self.R + v * np.cos(u/2)) * np.sin(u) - (v/2) * np.sin(u/2) * np.cos(u)
        r_u_y = (self.R + v * np.cos(u/2)) * np.cos(u) - (v/2) * np.sin(u/2) * np.sin(u)
        r_u_z = (v/2) * np.cos(u/2)
        
        # ∂r/∂v
        r_v_x = np.cos(u/2) * np.cos(u)
        r_v_y = np.cos(u/2) * np.sin(u)
        r_v_z = np.sin(u/2)
        
        r_u = np.array([r_u_x, r_u_y, r_u_z])
        r_v = np.array([r_v_x, r_v_y, r_v_z])
        
        return r_u, r_v
    
    def _surface_element_magnitude(self, u, v):
        """Compute the magnitude of the surface element |∂r/∂u × ∂r/∂v|."""
        r_u, r_v = self._partial_derivatives(u, v)
        cross_product = np.cross(r_u, r_v, axis=0)
        magnitude = np.sqrt(np.sum(cross_product**2, axis=0))
        return magnitude
    
    def compute_surface_area(self, method='numerical'):
        """Compute the surface area of the Mobius strip."""
        if method == 'numerical':
            def integrand(v, u):
                return self._surface_element_magnitude(u, v)
            
            area, _ = dblquad(integrand, 0, 2*np.pi, -self.w/2, self.w/2)
            return area
            
        elif method == 'mesh':
            total_area = 0
            for i in range(self.n - 1):
                for j in range(self.n - 1):
                    u1, v1 = self.u_range[i], self.v_range[j]
                    u2, v2 = self.u_range[i+1], self.v_range[j+1]
                    
                    # Four corners of the mesh element
                    p1 = np.array(self.get_point(u1, v1))
                    p2 = np.array(self.get_point(u2, v1))
                    p3 = np.array(self.get_point(u2, v2))
                    p4 = np.array(self.get_point(u1, v2))
                    
                    # Split into two triangles and compute their areas
                    area1 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p4 - p1))
                    area2 = 0.5 * np.linalg.norm(np.cross(p3 - p2, p4 - p2))
                    
                    total_area += area1 + area2
                    
            return total_area
    
    def compute_edge_length(self):
        """Compute the length of the edge boundary of the Mobius strip."""
        edge_points = []
        u_edge = np.linspace(0, 2*np.pi, self.n*2)
        
        # Trace the edge at v = w/2
        for u in u_edge:
            point = self.get_point(u, self.w/2)
            edge_points.append(point)
        
        # Compute length by summing distances between consecutive points
        edge_points = np.array(edge_points)
        differences = np.diff(edge_points, axis=0)
        distances = np.sqrt(np.sum(differences**2, axis=1))
        edge_length = np.sum(distances)
        
        return edge_length
    
    def plot_3d(self, show_wireframe=True, show_surface=True, alpha=0.7, colormap='viridis'):
        """Create a 3D visualization of the Mobius strip."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if show_surface:
            surf = ax.plot_surface(self.X, self.Y, self.Z, 
                          alpha=alpha, cmap=colormap, 
                          linewidth=0, antialiased=True)
        
        if show_wireframe:
            ax.plot_wireframe(self.X, self.Y, self.Z, 
                            color='black', alpha=0.3, linewidth=0.5)
        
        # Add edge highlighting
        edge_u = np.linspace(0, 2*np.pi, 100)
        edge_top = np.array([self.get_point(u, self.w/2) for u in edge_u])
        edge_bottom = np.array([self.get_point(u, -self.w/2) for u in edge_u])
        
        ax.plot(edge_top[:, 0], edge_top[:, 1], edge_top[:, 2], 
                'r-', linewidth=3, label='Edge')
        ax.plot(edge_bottom[:, 0], edge_bottom[:, 1], edge_bottom[:, 2], 
                'r-', linewidth=3)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'Mobius Strip (R={self.R}, w={self.w})', fontsize=14)
        
        # Set equal aspect ratio
        max_range = max(np.ptp(self.X), np.ptp(self.Y), np.ptp(self.Z)) / 2
        mid_x = (np.max(self.X) + np.min(self.X)) / 2
        mid_y = (np.max(self.Y) + np.min(self.Y)) / 2
        mid_z = (np.max(self.Z) + np.min(self.Z)) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Improve viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return fig, ax

# Main Streamlit App
def main():
    # Header
    st.markdown('<div class="main-header"><h1>🌀 Mobius Strip Geometric Explorer</h1><p>Interactive visualization and analysis of Mobius strip properties</p></div>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Control Panel")
    st.sidebar.markdown("Adjust the parameters below to explore different Mobius strips:")
    
    # Parameter controls
    radius = st.sidebar.slider(
        "📏 Radius (R)", 
        min_value=0.5, 
        max_value=5.0, 
        value=2.0, 
        step=0.1,
        help="Distance from the center to the strip"
    )
    
    width = st.sidebar.slider(
        "📐 Width (w)", 
        min_value=0.2, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="Width of the strip"
    )
    
    resolution = st.sidebar.slider(
        "🔍 Resolution", 
        min_value=20, 
        max_value=100, 
        value=50, 
        step=10,
        help="Number of points in the mesh (higher = more detailed)"
    )
    
    # Visualization options
    st.sidebar.markdown("## 🎨 Visualization Options")
    
    show_surface = st.sidebar.checkbox("Show Surface", value=True)
    show_wireframe = st.sidebar.checkbox("Show Wireframe", value=True)
    
    alpha = st.sidebar.slider(
        "Transparency", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.1
    )
    
    colormap = st.sidebar.selectbox(
        "Color Scheme",
        ["viridis", "plasma", "inferno", "magma", "cool", "hot", "spring", "summer"]
    )
    
    # Create Mobius strip
    with st.spinner('Generating Mobius strip...'):
        strip = MobiusStrip(radius=radius, width=width, resolution=resolution)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("## 📊 Geometric Properties")
        
        # Calculate properties
        with st.spinner('Computing surface area...'):
            area_numerical = strip.compute_surface_area('numerical')
        
        with st.spinner('Computing mesh approximation...'):
            area_mesh = strip.compute_surface_area('mesh')
            
        with st.spinner('Computing edge length...'):
            edge_length = strip.compute_edge_length()
        
        # Display metrics
        st.metric(
            "Surface Area (Numerical)", 
            f"{area_numerical:.4f}",
            help="Calculated using numerical integration"
        )
        
        st.metric(
            "Surface Area (Mesh)", 
            f"{area_mesh:.4f}",
            delta=f"{area_mesh - area_numerical:.4f}",
            help="Calculated using mesh triangulation"
        )
        
        st.metric(
            "Edge Length", 
            f"{edge_length:.4f}",
            help="Length of the single boundary edge"
        )
        
        difference = abs(area_numerical - area_mesh)
        st.metric(
            "Method Difference", 
            f"{difference:.4f}",
            help="Difference between numerical and mesh methods"
        )
        
        # Theoretical comparison
        theoretical_area = 2 * np.pi * radius * width
        relative_error = abs(area_numerical - theoretical_area) / theoretical_area * 100
        
        st.markdown("### 🧮 Theoretical Comparison")
        st.info(f"""
        **Theoretical Estimate:** {theoretical_area:.4f}  
        **Relative Error:** {relative_error:.2f}%  
        
        *Note: Theoretical estimate is approximate*
        """)
    
    with col2:
        st.markdown("## 🌀 3D Visualization")
        
        # Generate 3D plot
        with st.spinner('Rendering 3D visualization...'):
            fig, ax = strip.plot_3d(
                show_wireframe=show_wireframe,
                show_surface=show_surface,
                alpha=alpha,
                colormap=colormap
            )
            st.pyplot(fig)
            plt.close(fig)  # Prevent memory leaks
    
    # Additional analysis section
    st.markdown("---")
    st.markdown("## 📈 Advanced Analysis")
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["🔄 Convergence Analysis", "📐 Parameter Study", "🧮 Mathematical Details"])
    
    with tab1:
        st.markdown("### Convergence Analysis")
        st.markdown("See how the calculations converge as resolution increases:")
        
        if st.button("🚀 Run Convergence Analysis", type="primary"):
            resolutions = [20, 30, 40, 50, 60, 80, 100]
            areas_numerical = []
            areas_mesh = []
            edge_lengths = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, res in enumerate(resolutions):
                status_text.text(f'Computing for resolution {res}...')
                test_strip = MobiusStrip(radius=radius, width=width, resolution=res)
                
                area_num = test_strip.compute_surface_area('numerical')
                area_mesh = test_strip.compute_surface_area('mesh')
                edge_len = test_strip.compute_edge_length()
                
                areas_numerical.append(area_num)
                areas_mesh.append(area_mesh)
                edge_lengths.append(edge_len)
                
                progress_bar.progress((i + 1) / len(resolutions))
            
            status_text.text('Analysis complete!')
            
            # Create convergence plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Surface area convergence
            ax1.plot(resolutions, areas_numerical, 'bo-', linewidth=2, markersize=8, label='Numerical')
            ax1.plot(resolutions, areas_mesh, 'ro-', linewidth=2, markersize=8, label='Mesh')
            ax1.set_xlabel('Resolution')
            ax1.set_ylabel('Surface Area')
            ax1.set_title('Surface Area Convergence')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Edge length convergence
            ax2.plot(resolutions, edge_lengths, 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('Resolution')
            ax2.set_ylabel('Edge Length')
            ax2.set_title('Edge Length Convergence')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Show convergence data table
            convergence_df = pd.DataFrame({
                'Resolution': resolutions,
                'Surface Area (Numerical)': [f"{a:.4f}" for a in areas_numerical],
                'Surface Area (Mesh)': [f"{a:.4f}" for a in areas_mesh],
                'Edge Length': [f"{e:.4f}" for e in edge_lengths]
            })
            
            st.markdown("### 📋 Convergence Data")
            st.dataframe(convergence_df, use_container_width=True)
    
    with tab2:
        st.markdown("### Parameter Sensitivity Study")
        st.markdown("Explore how geometric properties change with parameters:")
        
        study_param = st.selectbox("Parameter to study:", ["Radius", "Width"])
        
        if st.button("📊 Run Parameter Study", type="primary"):
            if study_param == "Radius":
                param_values = np.linspace(0.5, 4.0, 10)
                areas = []
                edges = []
                
                progress_bar = st.progress(0)
                for i, r in enumerate(param_values):
                    test_strip = MobiusStrip(radius=r, width=width, resolution=50)
                    areas.append(test_strip.compute_surface_area('numerical'))
                    edges.append(test_strip.compute_edge_length())
                    progress_bar.progress((i + 1) / len(param_values))
                    
                x_label = "Radius"
                param_name = "radius"
                
            else:  # Width
                param_values = np.linspace(0.2, 2.0, 10)
                areas = []
                edges = []
                
                progress_bar = st.progress(0)
                for i, w in enumerate(param_values):
                    test_strip = MobiusStrip(radius=radius, width=w, resolution=50)
                    areas.append(test_strip.compute_surface_area('numerical'))
                    edges.append(test_strip.compute_edge_length())
                    progress_bar.progress((i + 1) / len(param_values))
                    
                x_label = "Width"
                param_name = "width"
            
            # Create parameter study plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.plot(param_values, areas, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel(x_label)
            ax1.set_ylabel('Surface Area')
            ax1.set_title(f'Surface Area vs {x_label}')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(param_values, edges, 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel(x_label)
            ax2.set_ylabel('Edge Length')
            ax2.set_title(f'Edge Length vs {x_label}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    with tab3:
        st.markdown("### 🧮 Mathematical Background")
        
        st.markdown("""
        #### Parametric Equations
        The Mobius strip is defined by:
        
        ```
        x(u,v) = (R + v⋅cos(u/2))⋅cos(u)
        y(u,v) = (R + v⋅cos(u/2))⋅sin(u)  
        z(u,v) = v⋅sin(u/2)
        ```
        
        Where:
        - `u ∈ [0, 2π]` - angle parameter around the strip
        - `v ∈ [-w/2, w/2]` - width parameter across the strip
        - `R` - radius (distance from center to strip)
        - `w` - strip width
        
        #### Surface Area Calculation
        Surface area is computed using the formula:
        
        ```
        Surface Area = ∫∫ |∂r/∂u × ∂r/∂v| du dv
        ```
        
        Where `∂r/∂u` and `∂r/∂v` are the partial derivatives of the position vector.
        
        #### Topological Properties
        - **Non-orientable**: The Mobius strip has no distinct "inside" or "outside"
        - **Single edge**: What appears as two edges is actually one continuous boundary
        - **Single surface**: The strip has only one side due to the half-twist
        """)
        
        # Current parameters summary
        st.markdown("#### Current Strip Parameters")
        st.code(f"""
        Radius (R): {radius}
        Width (w): {width}
        Resolution: {resolution}
        
        Parameter ranges:
        u ∈ [0, 2π]
        v ∈ [{-width/2:.2f}, {width/2:.2f}]
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌀 Mobius Strip Explorer | Built with Streamlit | 
        <a href='https://github.com/yourusername/mobius-strip-modeling' target='_blank'>View Source Code</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
