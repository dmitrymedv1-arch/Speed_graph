import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, FancyArrowPatch, Circle
import numpy as np
from enum import Enum
from typing import Dict, Tuple, List, Optional
import io
from io import BytesIO
import zipfile

# Page configuration
st.set_page_config(
    page_title="Speed Graph Generator",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GaugeStyle(Enum):
    """Available gauge styles for scientific visualization."""
    CLASSIC = "Classic"
    TECH = "Tech"
    GRADIENT = "Gradient"
    GRADIENT_PULSE = "Gradient Pulse"
    NEON_FUTURE = "Neon Future"
    MATERIAL_SCIENCE = "Material Science"
    BIOMEDICAL = "Biomedical"
    QUANTUM = "Quantum"
    CELESTIAL = "Celestial"
    GEOMETRIC = "Geometric"
    MINIMALIST = "Minimalist"


class CircularGauge:
    """
    Class for creating circular gauges with arrow indicators.
    Supports 11 modern styles for scientific visualization.
    """
    
    def __init__(self, 
                 title: str = "Measurement",
                 max_value: float = 100.0,
                 font_size: int = 12,
                 style: GaugeStyle = GaugeStyle.CLASSIC,
                 decimals: int = 0):
        """
        Initialize gauge with scientific visualization parameters.
        
        Args:
            title: Gauge title
            max_value: Maximum value for the scale
            font_size: Base font size
            style: Visual style of the gauge
            decimals: Number of decimal places to display
        """
        self.title = title
        self.max_value = max_value
        self.font_size = font_size
        self.style = style
        self.decimals = decimals
        
        # Get color scheme based on style
        self.colors = self._get_default_colors(style)
        
        # Display geometry parameters
        self.gauge_radius = 0.8
        self.center = (0, 0)
        self.arrow_length = 0.7
        self.start_angle = 180
        self.end_angle = 0
        
    def _get_default_colors(self, style: GaugeStyle) -> Dict:
        """Get default color scheme for the selected style."""
        color_schemes = {
            GaugeStyle.CLASSIC: {
                'face': '#f0f0f0',
                'edge': '#333333',
                'arrow': '#2E86AB',
                'text': '#000000',
                'ticks': '#666666',
                'value_text': '#D90429',
                'title': '#1B3B6F',
                'gauge_low': '#4CAF50',
                'gauge_mid': '#FFC107',
                'gauge_high': '#F44336',
                'inner_circle': '#ffffff',
                'highlight': '#ffffff'
            },
            GaugeStyle.TECH: {
                'face': '#0a192f',
                'edge': '#64ffda',
                'arrow': '#64ffda',
                'text': '#ccd6f6',
                'ticks': '#8892b0',
                'value_text': '#ff6b6b',
                'title': '#64ffda',
                'gauge_low': '#00ff88',
                'gauge_mid': '#ffd166',
                'gauge_high': '#ff6b6b',
                'inner_circle': '#112240',
                'highlight': '#64ffda'
            },
            GaugeStyle.GRADIENT: {
                'face': '#f8f9fa',
                'edge': '#495057',
                'arrow': '#228be6',
                'text': '#343a40',
                'ticks': '#868e96',
                'value_text': '#fa5252',
                'title': '#4263eb',
                'gauge_low': '#51cf66',
                'gauge_mid': '#ffd43b',
                'gauge_high': '#ff6b6b',
                'inner_circle': '#ffffff',
                'highlight': '#dee2e6'
            },
            GaugeStyle.GRADIENT_PULSE: {
                'face': '#0a0a0a',
                'edge': '#ffffff',
                'arrow': '#ffffff',
                'text': '#ffffff',
                'ticks': 'rgba(255, 255, 255, 0.7)',
                'value_text': '#ffffff',
                'title': '#ffffff',
                'gauge_low': '#ff3366',
                'gauge_mid': '#00ff9d',
                'gauge_high': '#3366ff',
                'inner_circle': '#000000',
                'highlight': '#ffffff'
            },
            GaugeStyle.NEON_FUTURE: {
                'face': '#000814',
                'edge': '#ff00ff',
                'arrow': '#00ffff',
                'text': '#ffffff',
                'ticks': '#ff00ff',
                'value_text': '#ffff00',
                'title': '#ff00ff',
                'gauge_low': '#00ff00',
                'gauge_mid': '#ffff00',
                'gauge_high': '#ff0000',
                'inner_circle': '#001233',
                'highlight': '#00ffff'
            },
            GaugeStyle.MATERIAL_SCIENCE: {
                'face': '#f5f5f5',
                'edge': '#607d8b',
                'arrow': '#3f51b5',
                'text': '#37474f',
                'ticks': '#78909c',
                'value_text': '#d32f2f',
                'title': '#1a237e',
                'gauge_low': '#388e3c',
                'gauge_mid': '#ffb300',
                'gauge_high': '#f44336',
                'inner_circle': '#ffffff',
                'highlight': '#e0e0e0'
            },
            GaugeStyle.BIOMEDICAL: {
                'face': '#f0f4f8',
                'edge': '#0052cc',
                'arrow': '#00a896',
                'text': '#2d3748',
                'ticks': '#718096',
                'value_text': '#d64545',
                'title': '#0052cc',
                'gauge_low': '#38b000',
                'gauge_mid': '#ffd000',
                'gauge_high': '#ff0054',
                'inner_circle': '#ffffff',
                'highlight': '#ebf8ff'
            },
            GaugeStyle.QUANTUM: {
                'face': '#000033',
                'edge': '#66ffcc',
                'arrow': '#ff66cc',
                'text': '#ccffff',
                'ticks': '#66ffcc',
                'value_text': '#ff9966',
                'title': '#66ffcc',
                'gauge_low': '#33ccff',
                'gauge_mid': '#ffcc00',
                'gauge_high': '#ff3366',
                'inner_circle': '#000066',
                'highlight': '#66ffcc'
            },
            GaugeStyle.CELESTIAL: {
                'face': '#0f0c29',
                'edge': '#ff9a00',
                'arrow': '#ff5e62',
                'text': '#ffffff',
                'ticks': '#a463f5',
                'value_text': '#ff9a00',
                'title': '#ff5e62',
                'gauge_low': '#38ef7d',
                'gauge_mid': '#ff9a00',
                'gauge_high': '#ff5e62',
                'inner_circle': '#1a1a40',
                'highlight': '#ff9a00'
            },
            GaugeStyle.GEOMETRIC: {
                'face': '#ffffff',
                'edge': '#222222',
                'arrow': '#222222',
                'text': '#222222',
                'ticks': '#666666',
                'value_text': '#ff3b30',
                'title': '#222222',
                'gauge_low': '#34c759',
                'gauge_mid': '#ffcc00',
                'gauge_high': '#ff3b30',
                'inner_circle': '#f2f2f7',
                'highlight': '#8e8e93'
            },
            GaugeStyle.MINIMALIST: {
                'face': '#ffffff',
                'edge': '#e0e0e0',
                'arrow': '#4a90e2',
                'text': '#333333',
                'ticks': '#bdbdbd',
                'value_text': '#4a90e2',
                'title': '#333333',
                'gauge_low': '#7ed321',
                'gauge_mid': '#f5a623',
                'gauge_high': '#d0021b',
                'inner_circle': '#fafafa',
                'highlight': '#e0e0e0'
            }
        }
        return color_schemes.get(style, color_schemes[GaugeStyle.CLASSIC])
    
    def set_color_scheme(self, 
                        face_color: str = None,
                        edge_color: str = None,
                        arrow_color: str = None,
                        text_color: str = None,
                        gauge_low: str = None,
                        gauge_mid: str = None,
                        gauge_high: str = None,
                        inner_circle_color: str = None,
                        highlight_color: str = None):
        """
        Customize color scheme.
        """
        if face_color:
            self.colors['face'] = face_color
        if edge_color:
            self.colors['edge'] = edge_color
        if arrow_color:
            self.colors['arrow'] = arrow_color
        if text_color:
            self.colors['text'] = text_color
        if gauge_low:
            self.colors['gauge_low'] = gauge_low
        if gauge_mid:
            self.colors['gauge_mid'] = gauge_mid
        if gauge_high:
            self.colors['gauge_high'] = gauge_high
        if inner_circle_color:
            self.colors['inner_circle'] = inner_circle_color
        if highlight_color:
            self.colors['highlight'] = highlight_color
    
    def _calculate_angle(self, value: float, normalized: bool = False) -> float:
        """
        Calculate arrow angle based on current value.
        """
        if normalized:
            normalized_value = max(0, min(1, value))
        else:
            normalized_value = max(0, min(1, value / self.max_value))
        
        angle = self.start_angle - normalized_value * (self.start_angle - self.end_angle)
        return angle
    
    def _get_arrow_color(self, value: float, normalized: bool = False) -> str:
        """
        Get arrow color based on value position.
        """
        if normalized:
            normalized_value = max(0, min(1, value))
        else:
            normalized_value = max(0, min(1, value / self.max_value))
        
        if self.style == GaugeStyle.GRADIENT_PULSE:
            # Dynamic rainbow colors for pulse style
            hue = normalized_value
            r = int(127.5 * (1 + np.sin(2 * np.pi * hue)))
            g = int(127.5 * (1 + np.sin(2 * np.pi * hue + 2 * np.pi / 3)))
            b = int(127.5 * (1 + np.sin(2 * np.pi * hue + 4 * np.pi / 3)))
            return f'#{r:02x}{g:02x}{b:02x}'
        elif self.style == GaugeStyle.QUANTUM:
            # Quantum interference pattern
            r = int(255 * (0.5 + 0.5 * np.sin(4 * np.pi * normalized_value)))
            g = int(255 * (0.5 + 0.5 * np.sin(4 * np.pi * normalized_value + np.pi/2)))
            b = int(255 * (0.5 + 0.5 * np.sin(4 * np.pi * normalized_value + np.pi)))
            return f'#{r:02x}{g:02x}{b:02x}'
        else:
            if normalized_value < 0.5:
                t = normalized_value * 2
                return self._interpolate_color(
                    self.colors['gauge_low'], 
                    self.colors['gauge_mid'], 
                    t
                )
            else:
                t = (normalized_value - 0.5) * 2
                return self._interpolate_color(
                    self.colors['gauge_mid'], 
                    self.colors['gauge_high'], 
                    t
                )
    
    def _interpolate_color(self, color1: str, color2: str, t: float) -> str:
        """
        Interpolate between two hex colors.
        """
        try:
            r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        except:
            r1, g1, b1 = 255, 255, 255
            
        try:
            r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        except:
            r2, g2, b2 = 255, 255, 255
        
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _format_value(self, value: float, normalized: bool = False) -> str:
        """
        Format value based on decimal settings.
        """
        if self.decimals == 0:
            return f"{value:.0f}"
        else:
            return f"{value:.{self.decimals}f}"
    
    def create_gauge(self, 
                     current_value: float, 
                     normalized: bool = False,
                     show_grid: bool = True,
                     figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create and return gauge figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Setup axes
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Set background color
        ax.set_facecolor(self.colors['face'])
        
        # Select drawing method based on style
        if self.style == GaugeStyle.CLASSIC:
            self._draw_classic(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.TECH:
            self._draw_tech(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.GRADIENT:
            self._draw_gradient(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.GRADIENT_PULSE:
            self._draw_gradient_pulse(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.NEON_FUTURE:
            self._draw_neon_future(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.MATERIAL_SCIENCE:
            self._draw_material_science(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.BIOMEDICAL:
            self._draw_biomedical(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.QUANTUM:
            self._draw_quantum(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.CELESTIAL:
            self._draw_celestial(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.GEOMETRIC:
            self._draw_geometric(ax, current_value, normalized, show_grid)
        elif self.style == GaugeStyle.MINIMALIST:
            self._draw_minimalist(ax, current_value, normalized, show_grid)
        
        plt.tight_layout()
        return fig
    
    def _draw_classic(self, ax, current_value, normalized, show_grid):
        """Classic scientific gauge style."""
        # Outer circle with rim
        outer_circle = patches.Circle(
            self.center, 
            radius=self.gauge_radius + 0.05,
            facecolor='white',
            edgecolor=self.colors['edge'],
            linewidth=3,
            zorder=1
        )
        ax.add_patch(outer_circle)
        
        # Main gauge arc
        arc = Arc(
            self.center,
            width=2 * self.gauge_radius,
            height=2 * self.gauge_radius,
            angle=0,
            theta1=self.start_angle,
            theta2=self.end_angle,
            linewidth=25,
            color='#f0f0f0',
            zorder=2
        )
        ax.add_patch(arc)
        
        # Colored value arc with gradient segments
        norm_value = current_value / self.max_value if not normalized else current_value
        num_segments = 20
        for i in range(num_segments):
            seg_start = i / num_segments
            seg_end = (i + 1) / num_segments
            
            if seg_end <= norm_value:
                seg_color = self._get_arrow_color(seg_end if normalized else seg_end * self.max_value, normalized)
                
                seg_arc = Arc(
                    self.center,
                    width=2 * self.gauge_radius,
                    height=2 * self.gauge_radius,
                    angle=0,
                    theta1=self.start_angle - seg_end * (self.start_angle - self.end_angle),
                    theta2=self.start_angle - seg_start * (self.start_angle - self.end_angle),
                    linewidth=20,
                    color=seg_color,
                    zorder=3
                )
                ax.add_patch(seg_arc)
        
        # Add scale marks
        if show_grid:
            self._add_scale_marks(ax, normalized)
        
        # Draw arrow
        angle = self._calculate_angle(current_value, normalized)
        arrow_color = self._get_arrow_color(current_value, normalized)
        
        arrow = FancyArrowPatch(
            self.center,
            (self.arrow_length * np.cos(np.radians(angle)), 
             self.arrow_length * np.sin(np.radians(angle))),
            arrowstyle='->',
            mutation_scale=40,
            linewidth=8,
            color=arrow_color,
            zorder=5
        )
        ax.add_patch(arrow)
        
        # Center pivot
        center_circle = patches.Circle(
            self.center, 
            radius=0.06,
            facecolor='white',
            edgecolor=self.colors['edge'],
            linewidth=3,
            zorder=6
        )
        ax.add_patch(center_circle)
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _add_scale_marks(self, ax, normalized):
        """Add scale marks with proper labeling."""
        if normalized:
            major_ticks_normalized = np.linspace(0, 1, 6)
            tick_values = major_ticks_normalized
            tick_labels = [self._format_value(tick) for tick in tick_values]
            use_ticks_for_angle = major_ticks_normalized
        else:
            num_ticks = 6
            major_ticks_normalized = np.linspace(0, 1, num_ticks)
            tick_values = major_ticks_normalized * self.max_value
            
            tick_labels = []
            for value in tick_values:
                if self.max_value >= 1000:
                    tick_labels.append(f"{value/1000:.1f}k")
                elif self.max_value >= 100:
                    tick_labels.append(self._format_value(value))
                else:
                    tick_labels.append(self._format_value(value))
            
            use_ticks_for_angle = major_ticks_normalized
        
        # Draw ticks and labels
        for i, tick in enumerate(use_ticks_for_angle):
            angle = self._calculate_angle(tick, normalized=True)
            
            # Tick line
            inner_radius = self.gauge_radius - 0.05
            outer_radius = self.gauge_radius + 0.05
            
            x_inner = inner_radius * np.cos(np.radians(angle))
            y_inner = inner_radius * np.sin(np.radians(angle))
            x_outer = outer_radius * np.cos(np.radians(angle))
            y_outer = outer_radius * np.sin(np.radians(angle))
            
            ax.plot(
                [x_inner, x_outer],
                [y_inner, y_outer],
                color=self.colors['ticks'],
                linewidth=2.5,
                solid_capstyle='round',
                zorder=4
            )
            
            # Label
            label_radius = outer_radius + 0.15
            x_label = label_radius * np.cos(np.radians(angle))
            y_label = label_radius * np.sin(np.radians(angle))
            
            # Auto-align text
            ha = 'center'
            if angle > 100:
                ha = 'right'
            elif angle < 80:
                ha = 'left'
            
            va = 'center'
            if angle > 160 or angle < 20:
                va = 'center'
            
            ax.text(
                x_label, y_label,
                tick_labels[i],
                ha=ha,
                va=va,
                fontsize=self.font_size,
                fontweight='bold',
                color=self.colors['text'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'),
                zorder=4
            )
    
    def _draw_tech(self, ax, current_value, normalized, show_grid):
        """Tech style with glitch effects."""
        # Dark background with grid
        ax.add_patch(patches.Rectangle(
            (-1.2, -0.3), 2.4, 1.5, facecolor=self.colors['face'], edgecolor='none', zorder=0
        ))
        
        # Grid lines
        for x in np.linspace(-1, 1, 11):
            ax.plot([x, x], [-0.2, 1.1], color='#112240', linewidth=0.5, alpha=0.3, zorder=0)
        for y in np.linspace(-0.2, 1.1, 11):
            ax.plot([-1, 1], [y, y], color='#112240', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Outer neon ring
        outer_ring = patches.Circle(
            self.center, 
            radius=self.gauge_radius + 0.1,
            facecolor='none',
            edgecolor=self.colors['edge'],
            linewidth=4,
            alpha=0.8,
            zorder=1
        )
        ax.add_patch(outer_ring)
        
        # Inner ring
        inner_ring = patches.Circle(
            self.center, 
            radius=self.gauge_radius - 0.1,
            facecolor=self.colors['inner_circle'],
            edgecolor='none',
            alpha=0.7,
            zorder=2
        )
        ax.add_patch(inner_ring)
        
        # Value segments
        norm_value = current_value / self.max_value if not normalized else current_value
        num_segments = 50
        for i in range(num_segments):
            seg_start = i / num_segments
            seg_end = (i + 1) / num_segments
            
            if seg_end <= norm_value:
                seg_angle_start = self.start_angle - seg_start * (self.start_angle - self.end_angle)
                seg_angle_end = self.start_angle - seg_end * (self.start_angle - self.end_angle)
                
                # Create segment
                theta = np.linspace(np.radians(seg_angle_start), np.radians(seg_angle_end), 50)
                x = self.gauge_radius * np.cos(theta)
                y = self.gauge_radius * np.sin(theta)
                
                ax.fill_between(x, y, y*0.95, color=self._get_arrow_color(
                    seg_end if normalized else seg_end * self.max_value, normalized
                ), alpha=0.7, zorder=3)
        
        # Scale marks
        if show_grid:
            self._add_scale_marks(ax, normalized)
        
        # Arrow with glow effect
        angle = self._calculate_angle(current_value, normalized)
        arrow_color = self._get_arrow_color(current_value, normalized)
        
        # Main arrow
        arrow = FancyArrowPatch(
            self.center,
            (self.arrow_length * np.cos(np.radians(angle)), 
             self.arrow_length * np.sin(np.radians(angle))),
            arrowstyle='->',
            mutation_scale=35,
            linewidth=6,
            color=arrow_color,
            zorder=5
        )
        ax.add_patch(arrow)
        
        # Arrow glow layers
        for i in range(3):
            glow_arrow = FancyArrowPatch(
                self.center,
                (self.arrow_length * np.cos(np.radians(angle)), 
                 self.arrow_length * np.sin(np.radians(angle))),
                arrowstyle='->',
                mutation_scale=35 + i*5,
                linewidth=2,
                color=arrow_color,
                alpha=0.2/(i+1),
                zorder=4
            )
            ax.add_patch(glow_arrow)
        
        # Center point
        ax.add_patch(patches.Circle(
            self.center, radius=0.05, facecolor=arrow_color, edgecolor='none', zorder=6
        ))
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _draw_gradient(self, ax, current_value, normalized, show_grid):
        """Smooth gradient style."""
        # Gradient background
        gradient = plt.Circle(self.center, 1.1, transform=ax.transData)
        gradient.set_facecolor(self.colors['face'])
        ax.add_patch(gradient)
        
        # Main arc outline
        theta = np.linspace(np.radians(self.start_angle), np.radians(self.end_angle), 100)
        x = self.gauge_radius * np.cos(theta)
        y = self.gauge_radius * np.sin(theta)
        
        # Create gradient fill
        norm_value = current_value / self.max_value if not normalized else current_value
        active_theta = np.linspace(np.radians(self.start_angle), 
                                 np.radians(self.start_angle - norm_value * (self.start_angle - self.end_angle)), 100)
        active_x = self.gauge_radius * np.cos(active_theta)
        active_y = self.gauge_radius * np.sin(active_theta)
        
        # Gradient fill segments
        for i in range(len(active_x)-1):
            color = self._get_arrow_color(
                i/(len(active_x)-1) if normalized else i/(len(active_x)-1) * self.max_value, 
                normalized
            )
            ax.fill_between([active_x[i], active_x[i+1]], [active_y[i], active_y[i+1]], 
                           [active_y[i]*0.7, active_y[i+1]*0.7], color=color, alpha=0.7, zorder=2)
        
        # Outline
        ax.plot(x, y, color=self.colors['edge'], linewidth=2, zorder=3)
        
        # Arrow
        angle = self._calculate_angle(current_value, normalized)
        arrow_color = self._get_arrow_color(current_value, normalized)
        
        arrow = FancyArrowPatch(
            self.center,
            (self.arrow_length * np.cos(np.radians(angle)), 
             self.arrow_length * np.sin(np.radians(angle))),
            arrowstyle='->',
            mutation_scale=35,
            linewidth=6,
            color=arrow_color,
            zorder=5
        )
        ax.add_patch(arrow)
        
        # Scale marks
        if show_grid:
            self._add_scale_marks(ax, normalized)
        
        # Center
        ax.add_patch(patches.Circle(
            self.center, radius=0.05, facecolor=arrow_color, edgecolor='none', zorder=6
        ))
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _draw_gradient_pulse(self, ax, current_value, normalized, show_grid):
        """Gradient Pulse style with dynamic colors."""
        # Dark background
        ax.set_facecolor(self.colors['face'])
        
        # Pulsing outer ring
        outer_ring = patches.Circle(
            self.center, 
            radius=self.gauge_radius + 0.1,
            facecolor='none',
            edgecolor=self.colors['edge'],
            linewidth=3,
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(outer_ring)
        
        # Multi-color gradient arc
        norm_value = current_value / self.max_value if not normalized else current_value
        
        # Create smooth gradient with many segments
        num_segments = 60
        for i in range(num_segments):
            seg_start = i / num_segments
            seg_end = (i + 1) / num_segments
            
            if seg_end <= norm_value:
                # Create rainbow gradient
                hue = i / num_segments
                r = int(127.5 * (1 + np.sin(2 * np.pi * hue)))
                g = int(127.5 * (1 + np.sin(2 * np.pi * hue + 2 * np.pi / 3)))
                b = int(127.5 * (1 + np.sin(2 * np.pi * hue + 4 * np.pi / 3)))
                seg_color = f'#{r:02x}{g:02x}{b:02x}'
                
                seg_arc = Arc(
                    self.center,
                    width=2 * self.gauge_radius,
                    height=2 * self.gauge_radius,
                    angle=0,
                    theta1=self.start_angle - seg_end * (self.start_angle - self.end_angle),
                    theta2=self.start_angle - seg_start * (self.start_angle - self.end_angle),
                    linewidth=12,
                    color=seg_color,
                    alpha=0.9,
                    zorder=3
                )
                ax.add_patch(seg_arc)
        
        # Pulsing arrow with multiple layers
        angle = self._calculate_angle(current_value, normalized)
        
        # Dynamic arrow color
        hue = norm_value
        r = int(127.5 * (1 + np.sin(2 * np.pi * hue)))
        g = int(127.5 * (1 + np.sin(2 * np.pi * hue + 2 * np.pi / 3)))
        b = int(127.5 * (1 + np.sin(2 * np.pi * hue + 4 * np.pi / 3)))
        arrow_color = f'#{r:02x}{g:02x}{b:02x}'
        
        # Multiple glow layers for pulsing effect
        for i in range(5):
            pulse_arrow = FancyArrowPatch(
                self.center,
                (self.arrow_length * np.cos(np.radians(angle)), 
                 self.arrow_length * np.sin(np.radians(angle))),
                arrowstyle='->',
                mutation_scale=45 + i*10,
                linewidth=8 - i*1.5,
                color=arrow_color,
                alpha=0.2/(i+1),
                zorder=4 + i
            )
            ax.add_patch(pulse_arrow)
        
        # Main arrow
        arrow = FancyArrowPatch(
            self.center,
            (self.arrow_length * np.cos(np.radians(angle)), 
             self.arrow_length * np.sin(np.radians(angle))),
            arrowstyle='->',
            mutation_scale=45,
            linewidth=8,
            color=arrow_color,
            zorder=9
        )
        ax.add_patch(arrow)
        
        # Pulsing center point
        ax.add_patch(patches.Circle(
            self.center, radius=0.06, facecolor=arrow_color, edgecolor='none', zorder=10
        ))
        
        # Center glow layers
        for i in range(3):
            pulse_center = patches.Circle(
                self.center, 
                radius=0.06 + i*0.03,
                facecolor=arrow_color,
                edgecolor='none',
                alpha=0.2/(i+1),
                zorder=9 - i
            )
            ax.add_patch(pulse_center)
        
        # Minimal scale marks
        if show_grid:
            if normalized:
                major_ticks_normalized = np.linspace(0, 1, 6)
                tick_labels = [self._format_value(tick) for tick in major_ticks_normalized]
            else:
                major_ticks_normalized = np.linspace(0, 1, 6)
                tick_values = major_ticks_normalized * self.max_value
                tick_labels = []
                for value in tick_values:
                    if self.max_value >= 1000:
                        tick_labels.append(f"{value/1000:.1f}k")
                    elif self.max_value >= 100:
                        tick_labels.append(self._format_value(value))
                    else:
                        tick_labels.append(self._format_value(value))
            
            for i, tick in enumerate(major_ticks_normalized):
                angle = self._calculate_angle(tick, normalized=True)
                
                # Glowing dots for scale marks
                point_radius = self.gauge_radius + 0.03
                x_point = point_radius * np.cos(np.radians(angle))
                y_point = point_radius * np.sin(np.radians(angle))
                
                # Glow effect
                ax.scatter(
                    x_point, y_point,
                    s=150,
                    color='white',
                    alpha=0.3,
                    zorder=4
                )
                
                # Main point
                ax.scatter(
                    x_point, y_point,
                    s=50,
                    color='white',
                    edgecolor='none',
                    alpha=0.8,
                    zorder=5
                )
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _draw_neon_future(self, ax, current_value, normalized, show_grid):
        """Neon Future style with cyberpunk aesthetics."""
        # Deep space background
        ax.set_facecolor(self.colors['face'])
        
        # Add starfield effect
        for _ in range(50):
            x = np.random.uniform(-1.1, 1.1)
            y = np.random.uniform(-0.2, 1.1)
            size = np.random.uniform(0.5, 2)
            ax.scatter(x, y, s=size, color='white', alpha=0.3, zorder=0)
        
        # Glowing outer ring
        for i in range(3):
            ring = patches.Circle(
                self.center,
                radius=self.gauge_radius + 0.15 - i*0.02,
                facecolor='none',
                edgecolor=self.colors['edge'],
                linewidth=2,
                alpha=0.3/(i+1),
                zorder=1
            )
            ax.add_patch(ring)
        
        # Holographic gauge surface
        inner_circle = patches.Circle(
            self.center,
            radius=self.gauge_radius - 0.1,
            facecolor=self.colors['inner_circle'],
            edgecolor='none',
            alpha=0.6,
            zorder=2
        )
        ax.add_patch(inner_circle)
        
        # Value indication with neon segments
        norm_value = current_value / self.max_value if not normalized else current_value
        num_segments = 40
        
        for i in range(num_segments):
            seg_start = i / num_segments
            seg_end = (i + 1) / num_segments
            
            if seg_end <= norm_value:
                # Neon gradient
                hue = i / num_segments
                r = int(255 * (0.5 + 0.5 * np.sin(6 * np.pi * hue)))
                g = int(255 * (0.5 + 0.5 * np.sin(6 * np.pi * hue + np.pi/3)))
                b = int(255 * (0.5 + 0.5 * np.sin(6 * np.pi * hue + 2*np.pi/3)))
                seg_color = f'#{r:02x}{g:02x}{b:02x}'
                
                seg_arc = Arc(
                    self.center,
                    width=2 * self.gauge_radius,
                    height=2 * self.gauge_radius,
                    angle=0,
                    theta1=self.start_angle - seg_end * (self.start_angle - self.end_angle),
                    theta2=self.start_angle - seg_start * (self.start_angle - self.end_angle),
                    linewidth=15,
                    color=seg_color,
                    alpha=0.8,
                    zorder=3
                )
                ax.add_patch(seg_arc)
        
        # Neon arrow with data stream effect
        angle = self._calculate_angle(current_value, normalized)
        arrow_color = self._get_arrow_color(current_value, normalized)
        
        # Data stream lines
        for i in range(5):
            stream_angle = angle + np.random.uniform(-2, 2)
            length = self.arrow_length * np.random.uniform(0.7, 1.0)
            ax.plot(
                [0, length * np.cos(np.radians(stream_angle))],
                [0, length * np.sin(np.radians(stream_angle))],
                color=arrow_color,
                linewidth=1,
                alpha=0.3,
                zorder=4
            )
        
        # Main arrow
        arrow = FancyArrowPatch(
            self.center,
            (self.arrow_length * np.cos(np.radians(angle)), 
             self.arrow_length * np.sin(np.radians(angle))),
            arrowstyle='->',
            mutation_scale=40,
            linewidth=7,
            color=arrow_color,
            zorder=5
        )
        ax.add_patch(arrow)
        
        # Glowing center
        center_glow = patches.Circle(
            self.center,
            radius=0.08,
            facecolor=arrow_color,
            edgecolor='none',
            alpha=0.8,
            zorder=6
        )
        ax.add_patch(center_glow)
        
        # Scale marks with neon glow
        if show_grid:
            self._add_neon_scale_marks(ax, normalized)
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _draw_material_science(self, ax, current_value, normalized, show_grid):
        """Material Science style with clean, professional aesthetics."""
        # Clean background
        ax.set_facecolor(self.colors['face'])
        
        # Subtle grid pattern
        for r in np.linspace(0.2, 0.9, 4):
            circle = patches.Circle(
                self.center,
                radius=r,
                facecolor='none',
                edgecolor=self.colors['highlight'],
                linewidth=0.5,
                alpha=0.1,
                linestyle='--',
                zorder=0
            )
            ax.add_patch(circle)
        
        # Outer ring with material texture effect
        outer_ring = patches.Circle(
            self.center,
            radius=self.gauge_radius + 0.08,
            facecolor='none',
            edgecolor=self.colors['edge'],
            linewidth=4,
            zorder=1
        )
        ax.add_patch(outer_ring)
        
        # Inner disc
        inner_disc = patches.Circle(
            self.center,
            radius=self.gauge_radius - 0.12,
            facecolor=self.colors['inner_circle'],
            edgecolor='none',
            zorder=2
        )
        ax.add_patch(inner_disc)
        
        # Value indicator with material gradient
        norm_value = current_value / self.max_value if not normalized else current_value
        
        # Create smooth material gradient
        theta_active = np.linspace(
            np.radians(self.start_angle),
            np.radians(self.start_angle - norm_value * (self.start_angle - self.end_angle)),
            100
        )
        
        for i in range(len(theta_active)-1):
            t = i / (len(theta_active)-1)
            color = self._interpolate_color(
                self.colors['gauge_low'],
                self.colors['gauge_high'],
                t
            )
            
            # Draw segment with material appearance
            theta_seg = theta_active[i:i+2]
            x_outer = self.gauge_radius * np.cos(theta_seg)
            y_outer = self.gauge_radius * np.sin(theta_seg)
            x_inner = (self.gauge_radius - 0.15) * np.cos(theta_seg)
            y_inner = (self.gauge_radius - 0.15) * np.sin(theta_seg)
            
            ax.fill_between(
                np.concatenate([x_outer, x_inner[::-1]]),
                np.concatenate([y_outer, y_inner[::-1]]),
                color=color,
                alpha=0.9,
                zorder=3
            )
        
        # Precision arrow
        angle = self._calculate_angle(current_value, normalized)
        arrow_color = self._get_arrow_color(current_value, normalized)
        
        arrow = FancyArrowPatch(
            self.center,
            (self.arrow_length * np.cos(np.radians(angle)), 
             self.arrow_length * np.sin(np.radians(angle))),
            arrowstyle='->',
            mutation_scale=30,
            linewidth=5,
            color=arrow_color,
            zorder=5
        )
        ax.add_patch(arrow)
        
        # Center hub with precision marking
        center_hub = patches.Circle(
            self.center,
            radius=0.04,
            facecolor='white',
            edgecolor=arrow_color,
            linewidth=2,
            zorder=6
        )
        ax.add_patch(center_hub)
        
        # Scale marks
        if show_grid:
            self._add_precision_scale_marks(ax, normalized)
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _draw_biomedical(self, ax, current_value, normalized, show_grid):
        """Biomedical style with health monitoring aesthetics."""
        # Medical chart background
        ax.set_facecolor(self.colors['face'])
        
        # Calculate normalized value
        norm_value = current_value / self.max_value if not normalized else current_value
        
        # ECG-style grid
        for y in np.linspace(-0.1, 0.9, 11):
            ax.plot([-1, 1], [y, y], color=self.colors['highlight'], linewidth=0.5, alpha=0.2, zorder=0)
        
        # Pulse waveform effect on gauge
        theta = np.linspace(np.radians(self.start_angle), np.radians(self.end_angle), 200)
        pulse_amplitude = 0.03
        pulse_freq = 5
        
        r_pulse = self.gauge_radius + pulse_amplitude * np.sin(pulse_freq * theta)
        x_pulse = r_pulse * np.cos(theta)
        y_pulse = r_pulse * np.sin(theta)
        
        ax.plot(x_pulse, y_pulse, color=self.colors['edge'], linewidth=1, alpha=0.3, zorder=1)
        
        # Main gauge body
        gauge_body = patches.Circle(
            self.center,
            radius=self.gauge_radius,
            facecolor='none',
            edgecolor=self.colors['edge'],
            linewidth=3,
            zorder=2
        )
        ax.add_patch(gauge_body)
        
        # Calculate normalized value
        norm_value = current_value / self.max_value if not normalized else current_value
        
        # Health indicator zones
        zone_colors = [self.colors['gauge_low'], self.colors['gauge_mid'], self.colors['gauge_high']]
        zone_limits = [0.33, 0.66, 1.0]
        
        for i, zone_end in enumerate(zone_limits):
            zone_start = 0 if i == 0 else zone_limits[i-1]
            
            if norm_value > zone_start:
                zone_value = min(norm_value, zone_end) - zone_start
                if zone_value > 0:
                    zone_arc = Arc(
                        self.center,
                        width=2 * self.gauge_radius,
                        height=2 * self.gauge_radius,
                        angle=0,
                        theta1=self.start_angle - zone_end * (self.start_angle - self.end_angle),
                        theta2=self.start_angle - zone_start * (self.start_angle - self.end_angle),
                        linewidth=20,
                        color=zone_colors[i],
                        alpha=0.7,
                        zorder=3
                    )
                    ax.add_patch(zone_arc)
        
        # Heartbeat-style arrow
        angle = self._calculate_angle(current_value, normalized)
        arrow_color = self._get_arrow_color(current_value, normalized)
        
        # Pulsing arrow effect
        for i in range(3):
            pulse_arrow = FancyArrowPatch(
                self.center,
                (self.arrow_length * np.cos(np.radians(angle)), 
                 self.arrow_length * np.sin(np.radians(angle))),
                arrowstyle='->',
                mutation_scale=35 + i*3,
                linewidth=6 - i*1,
                color=arrow_color,
                alpha=0.8/(i+1),
                zorder=4 + i
            )
            ax.add_patch(pulse_arrow)
        
        # Vital sign center
        center_vital = patches.Circle(
            self.center,
            radius=0.05,
            facecolor=arrow_color,
            edgecolor='white',
            linewidth=2,
            zorder=7
        )
        ax.add_patch(center_vital)
        
        # Medical scale marks
        if show_grid:
            self._add_medical_scale_marks(ax, normalized)
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _draw_quantum(self, ax, current_value, normalized, show_grid):
        """Quantum physics inspired style with wave functions."""
        # Quantum field background
        ax.set_facecolor(self.colors['face'])
        
        # Quantum wave interference pattern
        x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-0.2, 1, 30))
        wave = np.sin(8*x_grid) * np.cos(8*y_grid)
        ax.contourf(x_grid, y_grid, wave, levels=20, alpha=0.05, cmap='coolwarm', zorder=0)
        
        # Quantum tunneling rings
        for i in range(5):
            quantum_ring = patches.Circle(
                self.center,
                radius=self.gauge_radius + 0.1 - i*0.03,
                facecolor='none',
                edgecolor=self.colors['edge'],
                linewidth=1 + i*0.5,
                alpha=0.2,
                linestyle=':',
                zorder=1
            )
            ax.add_patch(quantum_ring)
        
        # Quantum state indicator
        norm_value = current_value / self.max_value if not normalized else current_value
        
        # Wave function probability distribution
        num_points = 200
        theta = np.linspace(np.radians(self.start_angle), np.radians(self.end_angle), num_points)
        probability = np.exp(-50*(np.linspace(0, 1, num_points) - norm_value)**2)
        
        for i in range(num_points-1):
            if probability[i] > 0.01:
                arc_segment = Arc(
                    self.center,
                    width=2 * (self.gauge_radius + 0.05 * probability[i]),
                    height=2 * (self.gauge_radius + 0.05 * probability[i]),
                    angle=0,
                    theta1=self.start_angle - (i+1)/num_points * (self.start_angle - self.end_angle),
                    theta2=self.start_angle - i/num_points * (self.start_angle - self.end_angle),
                    linewidth=15 * probability[i],
                    color=self._get_arrow_color(i/num_points if normalized else i/num_points * self.max_value, normalized),
                    alpha=0.8,
                    zorder=3
                )
                ax.add_patch(arc_segment)
        
        # Quantum superposition arrow
        angle = self._calculate_angle(current_value, normalized)
        arrow_color = self._get_arrow_color(current_value, normalized)
        
        # Multiple quantum states (superposition)
        for i in range(3):
            quantum_angle = angle + np.random.uniform(-5, 5)
            quantum_arrow = FancyArrowPatch(
                self.center,
                (self.arrow_length * np.cos(np.radians(quantum_angle)), 
                 self.arrow_length * np.sin(np.radians(quantum_angle))),
                arrowstyle='->',
                mutation_scale=30 + i*5,
                linewidth=4 - i,
                color=arrow_color,
                alpha=0.6/(i+1),
                zorder=4 + i
            )
            ax.add_patch(quantum_arrow)
        
        # Quantum singularity center
        ax.add_patch(patches.Circle(
            self.center, radius=0.04, facecolor=arrow_color, edgecolor='none', zorder=7
        ))
        
        # Quantum scale marks
        if show_grid:
            self._add_quantum_scale_marks(ax, normalized)
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _draw_celestial(self, ax, current_value, normalized, show_grid):
        """Celestial style with astronomical themes."""
        # Starfield background
        ax.set_facecolor(self.colors['face'])
        
        # Create starfield
        for _ in range(100):
            x = np.random.uniform(-1.1, 1.1)
            y = np.random.uniform(-0.2, 1.1)
            size = np.random.uniform(0.5, 3)
            brightness = np.random.uniform(0.3, 1.0)
            ax.scatter(x, y, s=size, color='white', alpha=brightness, zorder=0)
        
        # Planetary rings
        for i in range(3):
            ring = patches.Circle(
                self.center,
                radius=self.gauge_radius + 0.15 - i*0.05,
                facecolor='none',
                edgecolor=self.colors['edge'],
                linewidth=1 + i,
                alpha=0.3,
                linestyle='-' if i == 0 else '--',
                zorder=1
            )
            ax.add_patch(ring)
        
        # Nebula effect in gauge
        norm_value = current_value / self.max_value if not normalized else current_value
        num_segments = 80
        
        for i in range(num_segments):
            seg_start = i / num_segments
            seg_end = (i + 1) / num_segments
            
            if seg_end <= norm_value:
                # Nebula colors
                r = int(255 * (0.2 + 0.8 * np.sin(4 * np.pi * seg_start)))
                g = int(255 * (0.2 + 0.8 * np.sin(4 * np.pi * seg_start + np.pi/2)))
                b = int(255 * (0.2 + 0.8 * np.sin(4 * np.pi * seg_start + np.pi)))
                # Ensure values are within 0-255 range
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                seg_color = f'#{r:02x}{g:02x}{b:02x}'
                
                seg_arc = Arc(
                    self.center,
                    width=2 * self.gauge_radius,
                    height=2 * self.gauge_radius,
                    angle=0,
                    theta1=self.start_angle - seg_end * (self.start_angle - self.end_angle),
                    theta2=self.start_angle - seg_start * (self.start_angle - self.end_angle),
                    linewidth=10,
                    color=seg_color,
                    alpha=0.7,
                    zorder=3
                )
                ax.add_patch(seg_arc)
        
        # Comet-like arrow
        angle = self._calculate_angle(current_value, normalized)
        arrow_color = self._get_arrow_color(current_value, normalized)
        
        # Comet tail
        tail_length = 0.3
        for i in range(10):
            tail_pos = i / 10
            tail_alpha = 0.1 * (1 - tail_pos)
            tail_point = (
                tail_pos * self.arrow_length * np.cos(np.radians(angle)),
                tail_pos * self.arrow_length * np.sin(np.radians(angle))
            )
            ax.scatter(
                tail_point[0], tail_point[1],
                s=100 * (1 - tail_pos),
                color=arrow_color,
                alpha=tail_alpha,
                zorder=4
            )
        
        # Main arrow (comet head)
        arrow = FancyArrowPatch(
            self.center,
            (self.arrow_length * np.cos(np.radians(angle)), 
             self.arrow_length * np.sin(np.radians(angle))),
            arrowstyle='->',
            mutation_scale=40,
            linewidth=6,
            color=arrow_color,
            zorder=5
        )
        ax.add_patch(arrow)
        
        # Star center
        center_star = patches.Circle(
            self.center,
            radius=0.06,
            facecolor=arrow_color,
            edgecolor='white',
            linewidth=2,
            zorder=6
        )
        ax.add_patch(center_star)
        
        # Constellation scale marks
        if show_grid:
            self._add_celestial_scale_marks(ax, normalized)
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _draw_geometric(self, ax, current_value, normalized, show_grid):
        """Geometric style with clean lines and shapes."""
        # Pure white background
        ax.set_facecolor(self.colors['face'])
        
        # Geometric pattern background
        for angle in np.linspace(0, 360, 12, endpoint=False):
            x_line = 1.1 * np.cos(np.radians(angle))
            y_line = 1.1 * np.sin(np.radians(angle))
            ax.plot([0, x_line], [0, y_line], color=self.colors['highlight'], linewidth=0.5, alpha=0.2, zorder=0)
        
        # Outer geometric ring
        outer_ring = patches.Circle(
            self.center,
            radius=self.gauge_radius + 0.05,
            facecolor='none',
            edgecolor=self.colors['edge'],
            linewidth=2,
            zorder=1
        )
        ax.add_patch(outer_ring)
        
        # Inner geometric ring
        inner_ring = patches.Circle(
            self.center,
            radius=self.gauge_radius - 0.15,
            facecolor='none',
            edgecolor=self.colors['edge'],
            linewidth=1,
            linestyle='--',
            alpha=0.5,
            zorder=1
        )
        ax.add_patch(inner_ring)
        
        # Geometric segments (triangular)
        norm_value = current_value / self.max_value if not normalized else current_value
        num_segments = 24
        
        for i in range(num_segments):
            seg_start = i / num_segments
            seg_end = (i + 1) / num_segments
            
            if seg_end <= norm_value:
                # Geometric color based on position
                if seg_start < 0.5:
                    seg_color = self.colors['gauge_low']
                else:
                    seg_color = self.colors['gauge_high']
                
                # Create triangular segment
                angle_start = self.start_angle - seg_start * (self.start_angle - self.end_angle)
                angle_mid = self.start_angle - (seg_start + seg_end)/2 * (self.start_angle - self.end_angle)
                angle_end = self.start_angle - seg_end * (self.start_angle - self.end_angle)
                
                # Points for triangle
                inner_radius = self.gauge_radius - 0.1
                outer_radius = self.gauge_radius
                
                x = [
                    inner_radius * np.cos(np.radians(angle_start)),
                    outer_radius * np.cos(np.radians(angle_mid)),
                    inner_radius * np.cos(np.radians(angle_end))
                ]
                y = [
                    inner_radius * np.sin(np.radians(angle_start)),
                    outer_radius * np.sin(np.radians(angle_mid)),
                    inner_radius * np.sin(np.radians(angle_end))
                ]
                
                ax.fill(x, y, color=seg_color, alpha=0.8, zorder=2)
        
        # Geometric arrow (clean lines)
        angle = self._calculate_angle(current_value, normalized)
        arrow_color = self.colors['arrow']
        
        # Arrow shaft
        ax.plot(
            [0, self.arrow_length * np.cos(np.radians(angle))],
            [0, self.arrow_length * np.sin(np.radians(angle))],
            color=arrow_color,
            linewidth=4,
            solid_capstyle='round',
            zorder=3
        )
        
        # Arrow head (triangle)
        head_length = 0.1
        head_angle = 30
        
        # Calculate arrowhead points
        tip_x = self.arrow_length * np.cos(np.radians(angle))
        tip_y = self.arrow_length * np.sin(np.radians(angle))
        
        left_x = tip_x - head_length * np.cos(np.radians(angle - head_angle))
        left_y = tip_y - head_length * np.sin(np.radians(angle - head_angle))
        
        right_x = tip_x - head_length * np.cos(np.radians(angle + head_angle))
        right_y = tip_y - head_length * np.sin(np.radians(angle + head_angle))
        
        arrow_head = patches.Polygon(
            [[tip_x, tip_y], [left_x, left_y], [right_x, right_y]],
            facecolor=arrow_color,
            edgecolor='none',
            zorder=3
        )
        ax.add_patch(arrow_head)
        
        # Geometric center
        center_square = patches.Rectangle(
            [-0.03, -0.03], 0.06, 0.06,
            facecolor=arrow_color,
            edgecolor=self.colors['edge'],
            linewidth=1,
            zorder=4
        )
        ax.add_patch(center_square)
        
        # Geometric scale marks
        if show_grid:
            self._add_geometric_scale_marks(ax, normalized)
        
        # Display value
        self._add_value_display_simple(ax, current_value, normalized)
    
    def _draw_minimalist(self, ax, current_value, normalized, show_grid):
        """Minimalist style with essential elements only."""
        # Clean background
        ax.set_facecolor(self.colors['face'])
        
        # Very subtle outer circle
        outer_circle = patches.Circle(
            self.center,
            radius=self.gauge_radius + 0.02,
            facecolor='none',
            edgecolor=self.colors['edge'],
            linewidth=1,
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(outer_circle)
        
        # Value indicator (thin line)
        norm_value = current_value / self.max_value if not normalized else current_value
        
        indicator_arc = Arc(
            self.center,
            width=2 * self.gauge_radius,
            height=2 * self.gauge_radius,
            angle=0,
            theta1=self.start_angle,
            theta2=self.start_angle - norm_value * (self.start_angle - self.end_angle),
            linewidth=2,
            color=self.colors['arrow'],
            zorder=2
        )
        ax.add_patch(indicator_arc)
        
        # Minimal arrow
        angle = self._calculate_angle(current_value, normalized)
        
        # Thin arrow line
        ax.plot(
            [0, self.arrow_length * np.cos(np.radians(angle))],
            [0, self.arrow_length * np.sin(np.radians(angle))],
            color=self.colors['arrow'],
            linewidth=2,
            solid_capstyle='round',
            zorder=3
        )
        
        # Small dot at end
        ax.scatter(
            self.arrow_length * np.cos(np.radians(angle)),
            self.arrow_length * np.sin(np.radians(angle)),
            s=20,
            color=self.colors['arrow'],
            zorder=3
        )
        
        # Center dot
        ax.scatter(0, 0, s=30, color=self.colors['arrow'], zorder=4)
        
        # Minimal scale marks (only key positions)
        if show_grid:
            key_positions = [0, 0.25, 0.5, 0.75, 1.0] if normalized else [0, self.max_value/4, self.max_value/2, 3*self.max_value/4, self.max_value]
            
            for pos in key_positions:
                if normalized:
                    angle = self._calculate_angle(pos, normalized=True)
                    label = self._format_value(pos)
                else:
                    angle = self._calculate_angle(pos, normalized=False)
                    label = self._format_value(pos)
                
                # Tiny tick
                x_tick = (self.gauge_radius + 0.02) * np.cos(np.radians(angle))
                y_tick = (self.gauge_radius + 0.02) * np.sin(np.radians(angle))
                
                ax.scatter(x_tick, y_tick, s=10, color=self.colors['text'], alpha=0.5, zorder=2)
        
        # Display value (minimal)
        self._add_value_display_minimal(ax, current_value, normalized)
    
    def _add_neon_scale_marks(self, ax, normalized):
        """Add neon-style scale marks."""
        if normalized:
            ticks = np.linspace(0, 1, 6)
            labels = [self._format_value(tick) for tick in ticks]
        else:
            ticks = np.linspace(0, self.max_value, 6)
            labels = [self._format_value(tick) for tick in ticks]
            ticks = ticks / self.max_value
        
        for tick, label in zip(ticks, labels):
            angle = self._calculate_angle(tick, normalized=True)
            
            # Neon glow effect
            for i in range(3):
                glow_radius = self.gauge_radius + 0.05 + i*0.02
                x_glow = glow_radius * np.cos(np.radians(angle))
                y_glow = glow_radius * np.sin(np.radians(angle))
                ax.scatter(x_glow, y_glow, s=200/(i+1), color=self.colors['edge'], alpha=0.1, zorder=4)
            
            # Main tick
            x_tick = (self.gauge_radius + 0.05) * np.cos(np.radians(angle))
            y_tick = (self.gauge_radius + 0.05) * np.sin(np.radians(angle))
            ax.scatter(x_tick, y_tick, s=50, color=self.colors['edge'], zorder=5)
    
    def _add_precision_scale_marks(self, ax, normalized):
        """Add precision scale marks for material science style."""
        if normalized:
            major_ticks = np.linspace(0, 1, 6)
            minor_ticks = np.linspace(0, 1, 21)
        else:
            major_ticks = np.linspace(0, self.max_value, 6)
            minor_ticks = np.linspace(0, self.max_value, 21)
            major_ticks_norm = major_ticks / self.max_value
            minor_ticks_norm = minor_ticks / self.max_value
        
        # Minor ticks
        for tick in (minor_ticks_norm if not normalized else minor_ticks):
            angle = self._calculate_angle(tick, normalized=True)
            inner_r = self.gauge_radius - 0.05
            outer_r = self.gauge_radius - 0.08
            
            x_inner = inner_r * np.cos(np.radians(angle))
            y_inner = inner_r * np.sin(np.radians(angle))
            x_outer = outer_r * np.cos(np.radians(angle))
            y_outer = outer_r * np.sin(np.radians(angle))
            
            ax.plot([x_inner, x_outer], [y_inner, y_outer], 
                   color=self.colors['ticks'], linewidth=0.5, alpha=0.5, zorder=4)
        
        # Major ticks
        for i, tick in enumerate((major_ticks_norm if not normalized else major_ticks)):
            angle = self._calculate_angle(tick, normalized=True)
            
            # Tick line
            inner_r = self.gauge_radius - 0.05
            outer_r = self.gauge_radius - 0.11
            
            x_inner = inner_r * np.cos(np.radians(angle))
            y_inner = inner_r * np.sin(np.radians(angle))
            x_outer = outer_r * np.cos(np.radians(angle))
            y_outer = outer_r * np.sin(np.radians(angle))
            
            ax.plot([x_inner, x_outer], [y_inner, y_outer], 
                   color=self.colors['ticks'], linewidth=1.5, zorder=4)
            
            # Label
            label_r = self.gauge_radius - 0.16
            x_label = label_r * np.cos(np.radians(angle))
            y_label = label_r * np.sin(np.radians(angle))
            
            value = major_ticks[i] if not normalized else tick
            ax.text(x_label, y_label, self._format_value(value),
                   ha='center', va='center', fontsize=self.font_size-2,
                   color=self.colors['text'], zorder=4)
    
    def _add_medical_scale_marks(self, ax, normalized):
        """Add medical-style scale marks."""
        if normalized:
            ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = [self._format_value(t) for t in ticks]
        else:
            ticks = np.linspace(0, self.max_value, 6)
            labels = [self._format_value(t) for t in ticks]
            ticks = ticks / self.max_value
        
        for tick, label in zip(ticks, labels):
            angle = self._calculate_angle(tick, normalized=True)
            
            # Cross-shaped tick
            size = 0.02
            x_center = self.gauge_radius * np.cos(np.radians(angle))
            y_center = self.gauge_radius * np.sin(np.radians(angle))
            
            # Horizontal line
            ax.plot([x_center-size, x_center+size], [y_center, y_center],
                   color=self.colors['ticks'], linewidth=1.5, zorder=4)
            # Vertical line
            ax.plot([x_center, x_center], [y_center-size, y_center+size],
                   color=self.colors['ticks'], linewidth=1.5, zorder=4)
    
    def _add_quantum_scale_marks(self, ax, normalized):
        """Add quantum-style scale marks."""
        if normalized:
            ticks = np.linspace(0, 1, 7)
            labels = [self._format_value(t) for t in ticks]
        else:
            ticks = np.linspace(0, self.max_value, 7)
            labels = [self._format_value(t) for t in ticks]
            ticks = ticks / self.max_value
        
        for tick, label in zip(ticks, labels):
            angle = self._calculate_angle(tick, normalized=True)
            
            # Quantum probability cloud
            cloud_radius = 0.03
            for _ in range(10):
                cloud_angle = angle + np.random.uniform(-5, 5)
                cloud_dist = np.random.uniform(self.gauge_radius - 0.05, self.gauge_radius + 0.05)
                x_cloud = cloud_dist * np.cos(np.radians(cloud_angle))
                y_cloud = cloud_dist * np.sin(np.radians(cloud_angle))
                ax.scatter(x_cloud, y_cloud, s=10, color=self.colors['ticks'], alpha=0.3, zorder=4)
            
            # Main quantum state indicator
            x_main = self.gauge_radius * np.cos(np.radians(angle))
            y_main = self.gauge_radius * np.sin(np.radians(angle))
            ax.scatter(x_main, y_main, s=30, color=self.colors['ticks'], alpha=0.8, zorder=5)
    
    def _add_celestial_scale_marks(self, ax, normalized):
        """Add celestial-style scale marks."""
        if normalized:
            ticks = np.linspace(0, 1, 8)
            labels = [self._format_value(t) for t in ticks]
        else:
            ticks = np.linspace(0, self.max_value, 8)
            labels = [self._format_value(t) for t in ticks]
            ticks = ticks / self.max_value
        
        for i, (tick, label) in enumerate(zip(ticks, labels)):
            angle = self._calculate_angle(tick, normalized=True)
            
            # Star-shaped mark
            star_radius = 0.03
            x_star = (self.gauge_radius + 0.05) * np.cos(np.radians(angle))
            y_star = (self.gauge_radius + 0.05) * np.sin(np.radians(angle))
            
            # Create simple star
            star_angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
            star_x = x_star + star_radius * np.cos(star_angles) * (np.abs(np.sin(2*star_angles)) + 0.5)
            star_y = y_star + star_radius * np.sin(star_angles) * (np.abs(np.sin(2*star_angles)) + 0.5)
            
            ax.fill(star_x, star_y, color='gold' if i % 2 == 0 else 'silver', alpha=0.8, zorder=4)
    
    def _add_geometric_scale_marks(self, ax, normalized):
        """Add geometric-style scale marks."""
        if normalized:
            ticks = np.linspace(0, 1, 5)
            labels = [self._format_value(t) for t in ticks]
        else:
            ticks = np.linspace(0, self.max_value, 5)
            labels = [self._format_value(t) for t in ticks]
            ticks = ticks / self.max_value
        
        for tick, label in zip(ticks, labels):
            angle = self._calculate_angle(tick, normalized=True)
            
            # Square tick marks
            size = 0.02
            x_center = self.gauge_radius * np.cos(np.radians(angle))
            y_center = self.gauge_radius * np.sin(np.radians(angle))
            
            square = patches.Rectangle(
                [x_center-size/2, y_center-size/2], size, size,
                facecolor=self.colors['ticks'],
                edgecolor='none',
                zorder=4
            )
            ax.add_patch(square)
    
    def _add_value_display_simple(self, ax, current_value, normalized):
        """
        Display only the value in the center (no percentage or max).
        """
        # Title
        ax.text(
            0, 1.15,
            self.title,
            ha='center',
            va='center',
            fontsize=self.font_size + 8,
            fontweight='bold',
            color=self.colors['title'],
            zorder=10
        )
        
        # Current value (large, centered)
        formatted_value = self._format_value(current_value)
        
        ax.text(
            0, -0.15,
            formatted_value,
            ha='center',
            va='center',
            fontsize=self.font_size + 16,
            fontweight='bold',
            color=self.colors['value_text'],
            zorder=10
        )
    
    def _add_value_display_minimal(self, ax, current_value, normalized):
        """
        Minimal value display for minimalist style.
        """
        # Title (small and subtle)
        ax.text(
            0, 1.05,
            self.title,
            ha='center',
            va='center',
            fontsize=self.font_size + 4,
            color=self.colors['title'],
            alpha=0.8,
            zorder=10
        )
        
        # Value (clean and simple)
        formatted_value = self._format_value(current_value)
        
        ax.text(
            0, -0.05,
            formatted_value,
            ha='center',
            va='center',
            fontsize=self.font_size + 12,
            color=self.colors['value_text'],
            zorder=10
        )


def main():
    """Main Streamlit application."""
    
    # Application header
    st.title("🎛️ Speed Graph Generator")
    st.markdown("""
    Generate circular gauges for scientific visualization and data presentation. 
    Supports single gauges or multiple sample comparison with 11 different visualization styles.
    """)
    
    # Sidebar with configuration options
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display mode selection
        mode = st.radio(
            "Display Mode",
            ["Single Gauge", "Multiple Samples"],
            index=0
        )
        
        # Main parameters
        title = st.text_input("Title", "Measurement Value")
        max_value = st.number_input("Maximum Value", 0.1, 10000.0, 100.0, 0.1)
        font_size = st.slider("Font Size", 8, 24, 12)
        
        # Decimal places configuration
        decimals = st.selectbox(
            "Decimal Places",
            [0, 1, 2, 3],
            index=0,
            help="Number of decimal places to display for values"
        )
        
        # Style selection from all available styles
        style = st.selectbox(
            "Gauge Style",
            [style.value for style in GaugeStyle],
            index=0
        )
        
        # Get style enum object
        style_obj = GaugeStyle(style)
        
        # Mode-specific parameters
        if mode == "Single Gauge":
            current_value = st.slider(
                "Current Value", 
                0.0, float(max_value), 65.0, 0.1
            )
            show_normalized = st.checkbox("Show Normalized Gauge", True)
        else:
            num_samples = st.slider("Number of Samples", 2, 8, 3)
            show_normalized = st.checkbox("Show Normalized Comparison", True)
            
            # Sample configuration
            st.subheader("🧪 Sample Configuration")
            samples = []
            sample_colors = [
                '#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0',
                '#118AB2', '#EF476F', '#7209B7', '#3A86FF'
            ]
            
            for i in range(num_samples):
                with st.expander(f"Sample {i+1}", expanded=i < 3):
                    col1, col2 = st.columns(2)
                    with col1:
                        sample_name = st.text_input(f"Name {i+1}", f"Sample {i+1}", key=f"name_{i}")
                    with col2:
                        sample_value = st.number_input(
                            f"Value {i+1}", 
                            0.0, float(max_value), 
                            (i+1) * 15.0, 0.1,
                            key=f"value_{i}"
                        )
                    sample_color = st.color_picker(
                        f"Color {i+1}", 
                        sample_colors[i % len(sample_colors)],
                        key=f"color_{i}"
                    )
                    
                    samples.append({
                        'name': sample_name,
                        'value': sample_value,
                        'color': sample_color
                    })
        
        # Additional parameters
        show_grid = st.checkbox("Show Scale Marks", True)
        dpi = st.slider("DPI for Export", 300, 600, 300, 50)
        
        # Color scheme customization
        st.subheader("🎨 Color Scheme")
        
        # Define color schemes for each style
        color_schemes = {
            "Classic": {
                'face_color': '#f0f0f0',
                'edge_color': '#333333',
                'arrow_color': '#2E86AB',
                'text_color': '#000000',
                'gauge_low': '#4CAF50',
                'gauge_mid': '#FFC107',
                'gauge_high': '#F44336',
                'inner_circle_color': '#ffffff',
                'highlight_color': '#ffffff'
            },
            "Tech": {
                'face_color': '#0a192f',
                'edge_color': '#64ffda',
                'arrow_color': '#64ffda',
                'text_color': '#ccd6f6',
                'gauge_low': '#00ff88',
                'gauge_mid': '#ffd166',
                'gauge_high': '#ff6b6b',
                'inner_circle_color': '#112240',
                'highlight_color': '#64ffda'
            },
            "Gradient": {
                'face_color': '#f8f9fa',
                'edge_color': '#495057',
                'arrow_color': '#228be6',
                'text_color': '#343a40',
                'gauge_low': '#51cf66',
                'gauge_mid': '#ffd43b',
                'gauge_high': '#ff6b6b',
                'inner_circle_color': '#ffffff',
                'highlight_color': '#dee2e6'
            },
            "Gradient Pulse": {
                'face_color': '#0a0a0a',
                'edge_color': '#ffffff',
                'arrow_color': '#ffffff',
                'text_color': '#ffffff',
                'gauge_low': '#ff3366',
                'gauge_mid': '#00ff9d',
                'gauge_high': '#3366ff',
                'inner_circle_color': '#000000',
                'highlight_color': '#ffffff'
            },
            "Neon Future": {
                'face_color': '#000814',
                'edge_color': '#ff00ff',
                'arrow_color': '#00ffff',
                'text_color': '#ffffff',
                'gauge_low': '#00ff00',
                'gauge_mid': '#ffff00',
                'gauge_high': '#ff0000',
                'inner_circle_color': '#001233',
                'highlight_color': '#00ffff'
            },
            "Material Science": {
                'face_color': '#f5f5f5',
                'edge_color': '#607d8b',
                'arrow_color': '#3f51b5',
                'text_color': '#37474f',
                'gauge_low': '#388e3c',
                'gauge_mid': '#ffb300',
                'gauge_high': '#f44336',
                'inner_circle_color': '#ffffff',
                'highlight_color': '#e0e0e0'
            },
            "Biomedical": {
                'face_color': '#f0f4f8',
                'edge_color': '#0052cc',
                'arrow_color': '#00a896',
                'text_color': '#2d3748',
                'gauge_low': '#38b000',
                'gauge_mid': '#ffd000',
                'gauge_high': '#ff0054',
                'inner_circle_color': '#ffffff',
                'highlight_color': '#ebf8ff'
            },
            "Quantum": {
                'face_color': '#000033',
                'edge_color': '#66ffcc',
                'arrow_color': '#ff66cc',
                'text_color': '#ccffff',
                'gauge_low': '#33ccff',
                'gauge_mid': '#ffcc00',
                'gauge_high': '#ff3366',
                'inner_circle_color': '#000066',
                'highlight_color': '#66ffcc'
            },
            "Celestial": {
                'face_color': '#0f0c29',
                'edge_color': '#ff9a00',
                'arrow_color': '#ff5e62',
                'text_color': '#ffffff',
                'gauge_low': '#38ef7d',
                'gauge_mid': '#ff9a00',
                'gauge_high': '#ff5e62',
                'inner_circle_color': '#1a1a40',
                'highlight_color': '#ff9a00'
            },
            "Geometric": {
                'face_color': '#ffffff',
                'edge_color': '#222222',
                'arrow_color': '#222222',
                'text_color': '#222222',
                'gauge_low': '#34c759',
                'gauge_mid': '#ffcc00',
                'gauge_high': '#ff3b30',
                'inner_circle_color': '#f2f2f7',
                'highlight_color': '#8e8e93'
            },
            "Minimalist": {
                'face_color': '#ffffff',
                'edge_color': '#e0e0e0',
                'arrow_color': '#4a90e2',
                'text_color': '#333333',
                'gauge_low': '#7ed321',
                'gauge_mid': '#f5a623',
                'gauge_high': '#d0021b',
                'inner_circle_color': '#fafafa',
                'highlight_color': '#e0e0e0'
            }
        }
        
        if style in color_schemes:
            colors = color_schemes[style]
            
            col1, col2 = st.columns(2)
            with col1:
                face_color = st.color_picker("Background", colors['face_color'])
                edge_color = st.color_picker("Edge", colors['edge_color'])
                arrow_color = st.color_picker("Arrow", colors['arrow_color'])
                text_color = st.color_picker("Text", colors['text_color'])
            
            with col2:
                gauge_low = st.color_picker("Low Zone", colors['gauge_low'])
                gauge_mid = st.color_picker("Medium Zone", colors['gauge_mid'])
                gauge_high = st.color_picker("High Zone", colors['gauge_high'])
                inner_circle_color = st.color_picker("Inner Circle", colors['inner_circle_color'])
                highlight_color = st.color_picker("Highlight", colors['highlight_color'])
    
    # Main content area
    st.header("📊 Gauge Visualization")
    
    if mode == "Single Gauge":
        # Create single gauge display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Standard Gauge")
            gauge = CircularGauge(
                title=title,
                max_value=max_value,
                font_size=font_size,
                style=style_obj,
                decimals=decimals
            )
            
            gauge.set_color_scheme(
                face_color=face_color,
                edge_color=edge_color,
                arrow_color=arrow_color,
                text_color=text_color,
                gauge_low=gauge_low,
                gauge_mid=gauge_mid,
                gauge_high=gauge_high,
                inner_circle_color=inner_circle_color,
                highlight_color=highlight_color
            )
            
            fig = gauge.create_gauge(current_value, normalized=False, show_grid=show_grid)
            st.pyplot(fig)
            
            # Download button for standard gauge
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            st.download_button(
                label="📥 Download Standard Gauge",
                data=buf.getvalue(),
                file_name=f"gauge_{title.replace(' ', '_')}_{style.lower()}.png",
                mime="image/png"
            )
            plt.close(fig)
        
        if show_normalized:
            with col2:
                st.subheader("Normalized Gauge")
                normalized_gauge = CircularGauge(
                    title=f"{title} (Normalized)",
                    max_value=1.0,
                    font_size=font_size,
                    style=style_obj,
                    decimals=decimals
                )
                
                normalized_gauge.set_color_scheme(
                    face_color=face_color,
                    edge_color=edge_color,
                    arrow_color=arrow_color,
                    text_color=text_color,
                    gauge_low=gauge_low,
                    gauge_mid=gauge_mid,
                    gauge_high=gauge_high,
                    inner_circle_color=inner_circle_color,
                    highlight_color=highlight_color
                )
                
                normalized_value = current_value / max_value
                norm_fig = normalized_gauge.create_gauge(
                    normalized_value, 
                    normalized=True, 
                    show_grid=show_grid
                )
                st.pyplot(norm_fig)
                
                # Download button for normalized gauge
                buf_norm = io.BytesIO()
                norm_fig.savefig(buf_norm, format='png', dpi=dpi, bbox_inches='tight')
                st.download_button(
                    label="📥 Download Normalized Gauge",
                    data=buf_norm.getvalue(),
                    file_name=f"gauge_{title.replace(' ', '_')}_normalized_{style.lower()}.png",
                    mime="image/png"
                )
                plt.close(norm_fig)
        
        # Value information
        formatted_value = f"{current_value:.{decimals}f}" if decimals > 0 else f"{current_value:.0f}"
        st.info(f"**Value:** {formatted_value} / {max_value:.{decimals}f}" if decimals > 0 else f"{formatted_value} / {max_value:.0f}")
        
    else:  # Multiple Samples mode
        if show_normalized:
            # Display both standard and normalized gauges
            st.subheader("Standard Gauges")
            cols = st.columns(num_samples)
            
            # Standard gauges
            for i, (sample, col) in enumerate(zip(samples, cols)):
                with col:
                    gauge = CircularGauge(
                        title=sample['name'],
                        max_value=max_value,
                        font_size=font_size - 2,
                        style=style_obj,
                        decimals=decimals
                    )
                    
                    # Use sample color for the gauge
                    gauge.set_color_scheme(
                        face_color=face_color,
                        edge_color=edge_color,
                        arrow_color=sample['color'],
                        text_color=text_color,
                        gauge_low=sample['color'],
                        gauge_mid=sample['color'],
                        gauge_high=sample['color'],
                        inner_circle_color=inner_circle_color,
                        highlight_color=highlight_color
                    )
                    
                    fig = gauge.create_gauge(
                        sample['value'], 
                        normalized=False, 
                        show_grid=show_grid,
                        figsize=(5, 4)
                    )
                    st.pyplot(fig)
                    plt.close(fig)
            
            st.subheader("Normalized Gauges")
            cols_norm = st.columns(num_samples)
            
            # Normalized gauges
            for i, (sample, col) in enumerate(zip(samples, cols_norm)):
                with col:
                    normalized_gauge = CircularGauge(
                        title=f"{sample['name']} (Norm)",
                        max_value=1.0,
                        font_size=font_size - 2,
                        style=style_obj,
                        decimals=decimals
                    )
                    
                    normalized_gauge.set_color_scheme(
                        face_color=face_color,
                        edge_color=edge_color,
                        arrow_color=sample['color'],
                        text_color=text_color,
                        gauge_low=sample['color'],
                        gauge_mid=sample['color'],
                        gauge_high=sample['color'],
                        inner_circle_color=inner_circle_color,
                        highlight_color=highlight_color
                    )
                    
                    normalized_value = sample['value'] / max_value
                    norm_fig = normalized_gauge.create_gauge(
                        normalized_value, 
                        normalized=True, 
                        show_grid=show_grid,
                        figsize=(5, 4)
                    )
                    st.pyplot(norm_fig)
                    plt.close(norm_fig)
            
            # Create ZIP download button for all gauges
            if st.button("📥 Download All Gauges as ZIP"):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Standard gauges
                    for i, sample in enumerate(samples):
                        gauge = CircularGauge(
                            title=sample['name'],
                            max_value=max_value,
                            font_size=font_size,
                            style=style_obj,
                            decimals=decimals
                        )
                        
                        gauge.set_color_scheme(
                            face_color=face_color,
                            edge_color=edge_color,
                            arrow_color=sample['color'],
                            text_color=text_color,
                            gauge_low=sample['color'],
                            gauge_mid=sample['color'],
                            gauge_high=sample['color'],
                            inner_circle_color=inner_circle_color,
                            highlight_color=highlight_color
                        )
                        
                        fig_buffer = BytesIO()
                        fig = gauge.create_gauge(
                            sample['value'], 
                            normalized=False, 
                            show_grid=show_grid
                        )
                        fig.savefig(fig_buffer, format='png', dpi=dpi, bbox_inches='tight')
                        zip_file.writestr(
                            f"gauge_{sample['name'].replace(' ', '_')}_{style.lower()}.png",
                            fig_buffer.getvalue()
                        )
                        plt.close(fig)
                    
                    # Normalized gauges
                    for i, sample in enumerate(samples):
                        normalized_gauge = CircularGauge(
                            title=f"{sample['name']}_normalized",
                            max_value=1.0,
                            font_size=font_size,
                            style=style_obj,
                            decimals=decimals
                        )
                        
                        normalized_gauge.set_color_scheme(
                            face_color=face_color,
                            edge_color=edge_color,
                            arrow_color=sample['color'],
                            text_color=text_color,
                            gauge_low=sample['color'],
                            gauge_mid=sample['color'],
                            gauge_high=sample['color'],
                            inner_circle_color=inner_circle_color,
                            highlight_color=highlight_color
                        )
                        
                        normalized_value = sample['value'] / max_value
                        norm_buffer = BytesIO()
                        norm_fig = normalized_gauge.create_gauge(
                            normalized_value, 
                            normalized=True, 
                            show_grid=show_grid
                        )
                        norm_fig.savefig(norm_buffer, format='png', dpi=dpi, bbox_inches='tight')
                        zip_file.writestr(
                            f"gauge_{sample['name'].replace(' ', '_')}_normalized_{style.lower()}.png",
                            norm_buffer.getvalue()
                        )
                        plt.close(norm_fig)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="📥 Download ZIP with all gauges",
                    data=zip_buffer.getvalue(),
                    file_name=f"gauges_{title.replace(' ', '_')}_{style.lower()}.zip",
                    mime="application/zip"
                )
        
        else:
            # Display only standard gauges
            cols = st.columns(num_samples)
            
            for i, (sample, col) in enumerate(zip(samples, cols)):
                with col:
                    gauge = CircularGauge(
                        title=sample['name'],
                        max_value=max_value,
                        font_size=font_size,
                        style=style_obj,
                        decimals=decimals
                    )
                    
                    gauge.set_color_scheme(
                        face_color=face_color,
                        edge_color=edge_color,
                        arrow_color=sample['color'],
                        text_color=text_color,
                        gauge_low=sample['color'],
                        gauge_mid=sample['color'],
                        gauge_high=sample['color'],
                        inner_circle_color=inner_circle_color,
                        highlight_color=highlight_color
                    )
                    
                    fig = gauge.create_gauge(
                        sample['value'], 
                        normalized=False, 
                        show_grid=show_grid
                    )
                    st.pyplot(fig)
                    
                    # Individual download button for each gauge
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
                    st.download_button(
                        label=f"📥 Download {sample['name']}",
                        data=buf.getvalue(),
                        file_name=f"gauge_{sample['name'].replace(' ', '_')}_{style.lower()}.png",
                        mime="image/png",
                        key=f"download_{i}"
                    )
                    plt.close(fig)
        
        # Sample statistics
        values = [s['value'] for s in samples]
        avg_value = np.mean(values)
        min_value = np.min(values)
        max_value_val = np.max(values)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{avg_value:.{decimals}f}" if decimals > 0 else f"{avg_value:.0f}")
        with col2:
            st.metric("Minimum", f"{min_value:.{decimals}f}" if decimals > 0 else f"{min_value:.0f}")
        with col3:
            st.metric("Maximum", f"{max_value_val:.{decimals}f}" if decimals > 0 else f"{max_value_val:.0f}")
        with col4:
            range_val = max_value_val - min_value
            st.metric("Range", f"{range_val:.{decimals}f}" if decimals > 0 else f"{range_val:.0f}")
    
    # Footer with features and instructions
    st.markdown("---")
    st.markdown("""
    ### 🎯 Features
    - **11 Gauge Styles:** Classic, Tech, Gradient, Gradient Pulse, Neon Future, Material Science, Biomedical, Quantum, Celestial, Geometric, Minimalist
    - **Two Display Modes:** Single gauge or multiple sample comparison
    - **Decimal Control:** Configure number of decimal places (0-3)
    - **Normalization:** Display gauges with 0-1 normalized scale
    - **Full Color Customization:** Control over all visual elements
    - **High-Quality Export:** Adjustable DPI (300-600) for publication quality
    - **Scientific Ready:** Designed for research presentations and academic papers
    
    ### 📝 Instructions
    1. Configure all settings in the sidebar
    2. For multiple samples: expand each sample section to configure individually
    3. Use download buttons to save individual gauges or entire sets as ZIP
    4. Use higher DPI settings for publication-quality images
    5. Experiment with different styles for various presentation contexts
    """)
    
    # About section
    with st.expander("ℹ️ About this Application"):
        st.markdown("""
        **Speed Graph Generator**  
        Advanced tool for scientific data visualization and presentation.
        
        This application generates circular gauge indicators optimized for scientific 
        communication. Each gauge displays quantitative values with clear visual 
        indicators using various modern and scientific design styles.
        
        **Primary use cases:**
        - Experimental results visualization
        - Performance metrics display
        - Comparative analysis of multiple samples
        - Scientific presentations and publications
        - Data dashboards and monitoring displays
        
        **Key features:**
        - 11 distinct visualization styles
        - Precise control over numerical formatting
        - High-quality export for publications
        - Customizable color schemes for brand consistency
        - Support for both absolute and normalized values
        
        **developed by @daM, @CTA, https://chimicatechnoacta.ru **
        
        *Version 2.0 - Enhanced with 7 new scientific visualization styles*
        """)


if __name__ == "__main__":
    main()





