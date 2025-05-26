import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad
import warnings
warnings.filterwarnings('ignore')

class MobiusStrip:
    """
    A class to model a Mobius strip using parametric equations and compute
    key geometric properties including surface area and edge length.
    
    Parametric equations:
    x(u,v) = (R + v*cos(u/2)) * cos(u)
    y(u,v) = (R + v*cos(u/2)) * sin(u)  
    z(u,v) = v * sin(u/2)
    
    Where u ∈ [0, 2π] and v ∈ [-w/2, w/2]
    """
    
    def __init__(self, radius=2.0, width=1.0, resolution=50):
        """
        Initialize the Mobius strip with given parameters.
        
        Parameters:
        -----------
        radius : float
            Distance from center to the strip (R)
        width : float
            Width of the strip (w)
        resolution : int
            Number of points in each parameter direction for mesh generation
        """
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
        """
        Get a single point on the Mobius strip surface.
        
        Parameters:
        -----------
        u : float
            Parameter u ∈ [0, 2π]
        v : float  
            Parameter v ∈ [-w/2, w/2]
            
        Returns:
        --------
        tuple: (x, y, z) coordinates
        """
        x = (self.R + v * np.cos(u/2)) * np.cos(u)
        y = (self.R + v * np.cos(u/2)) * np.sin(u)
        z = v * np.sin(u/2)
        return (x, y, z)
    
    def _partial_derivatives(self, u, v):
        """
        Compute partial derivatives of the parametric surface.
        
        Returns:
        --------
        tuple: (r_u, r_v) where each is a 3D vector
        """
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
        """
        Compute the magnitude of the surface element |∂r/∂u × ∂r/∂v|.
        This is used for surface area integration.
        """
        r_u, r_v = self._partial_derivatives(u, v)
        cross_product = np.cross(r_u, r_v, axis=0)
        magnitude = np.sqrt(np.sum(cross_product**2, axis=0))
        return magnitude
    
    def compute_surface_area(self, method='numerical'):
        """
        Compute the surface area of the Mobius strip.
        
        Parameters:
        -----------
        method : str
            'numerical' for numerical integration, 'mesh' for mesh approximation
            
        Returns:
        --------
        float: Surface area
        """
        if method == 'numerical':
            # Use scipy's double integration
            def integrand(v, u):
                return self._surface_element_magnitude(u, v)
            
            area, _ = dblquad(integrand, 0, 2*np.pi, -self.w/2, self.w/2)
            return area
            
        elif method == 'mesh':
            # Approximate using mesh triangulation
            du = 2*np.pi / (self.n - 1)
            dv = self.w / (self.n - 1)
            
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
        """
        Compute the length of the edge boundary of the Mobius strip.
        The Mobius strip has a single edge that forms a closed loop.
        
        Returns:
        --------
        float: Edge length
        """
        # The edge is at v = ±w/2, but due to the twist, we only trace one edge
        # as it forms a continuous loop
        edge_points = []
        u_edge = np.linspace(0, 2*np.pi, self.n*2)  # Higher resolution for edge
        
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
    
    def plot_3d(self, show_wireframe=True, show_surface=True, alpha=0.7):
        """
        Create a 3D visualization of the Mobius strip.
        
        Parameters:
        -----------
        show_wireframe : bool
            Whether to show wireframe
        show_surface : bool
            Whether to show surface
        alpha : float
            Transparency level for surface
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if show_surface:
            ax.plot_surface(self.X, self.Y, self.Z, 
                          alpha=alpha, cmap='viridis', 
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
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Mobius Strip (R={self.R}, w={self.w})')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = max(np.ptp(self.X), np.ptp(self.Y), np.ptp(self.Z)) / 2
        mid_x = (np.max(self.X) + np.min(self.X)) / 2
        mid_y = (np.max(self.Y) + np.min(self.Y)) / 2
        mid_z = (np.max(self.Z) + np.min(self.Z)) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        return fig, ax
    
    def print_properties(self):
        """Print computed geometric properties."""
        print(f"Mobius Strip Properties:")
        print(f"{'='*40}")
        print(f"Radius (R): {self.R}")
        print(f"Width (w): {self.w}")
        print(f"Resolution: {self.n}")
        print(f"{'='*40}")
        
        # Surface area
        area_numerical = self.compute_surface_area('numerical')
        area_mesh = self.compute_surface_area('mesh')
        print(f"Surface Area (numerical): {area_numerical:.4f}")
        print(f"Surface Area (mesh approx): {area_mesh:.4f}")
        print(f"Difference: {abs(area_numerical - area_mesh):.4f}")
        
        # Edge length
        edge_length = self.compute_edge_length()
        print(f"Edge Length: {edge_length:.4f}")
        
        # Theoretical comparison (for validation)
        # For a Mobius strip, approximate surface area ≈ 2πR*w (rough estimate)
        theoretical_area = 2 * np.pi * self.R * self.w
        print(f"Theoretical estimate: {theoretical_area:.4f}")
        print(f"Relative error: {abs(area_numerical - theoretical_area)/theoretical_area * 100:.2f}%")

def main():
    """
    Main function to demonstrate the Mobius strip modeling.
    """
    print("Mobius Strip Geometric Modeling")
    print("=" * 50)
    
    # Create Mobius strip instances with different parameters
    strips = [
        MobiusStrip(radius=2.0, width=1.0, resolution=50),
        MobiusStrip(radius=3.0, width=0.5, resolution=60),
        MobiusStrip(radius=1.5, width=1.5, resolution=40)
    ]
    
    # Analyze each strip
    for i, strip in enumerate(strips):
        print(f"\nStrip {i+1}:")
        strip.print_properties()
        
        # Create visualization
        fig, ax = strip.plot_3d()
        plt.show()
    
    # Demonstrate convergence with resolution
    print(f"\nConvergence Analysis:")
    print(f"{'Resolution':<12} {'Surface Area':<15} {'Edge Length':<12}")
    print("-" * 40)
    
    test_strip = MobiusStrip(radius=2.0, width=1.0, resolution=20)
    for res in [20, 40, 60, 80, 100]:
        test_strip.n = res
        test_strip.u_range = np.linspace(0, 2*np.pi, res)
        test_strip.v_range = np.linspace(-test_strip.w/2, test_strip.w/2, res)
        test_strip.U, test_strip.V = np.meshgrid(test_strip.u_range, test_strip.v_range)
        test_strip._compute_surface_points()
        
        area = test_strip.compute_surface_area('numerical')
        edge = test_strip.compute_edge_length()
        print(f"{res:<12} {area:<15.4f} {edge:<12.4f}")

if __name__ == "__main__":
    main()