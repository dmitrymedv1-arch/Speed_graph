import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, FancyArrowPatch, Circle
import numpy as np
from enum import Enum
from typing import Dict, Tuple
import io

# Настройки страницы
st.set_page_config(
    page_title="Speed map Generator",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GaugeStyle(Enum):
    """Gauge styles for infographics."""
    CLASSIC = "Classic"
    TECH = "Tech"
    GRADIENT = "Gradient"
    GRADIENT_PULSE = "Gradient Pulse"


class CircularGauge:
    """
    Class for creating circular gauges with arrow.
    Supports 4 modern styles.
    """
    
    def __init__(self, 
                 title: str = "Value",
                 max_value: float = 100.0,
                 font_size: int = 12,
                 style: GaugeStyle = GaugeStyle.CLASSIC):
        """
        Initialize gauge.
        """
        self.title = title
        self.max_value = max_value
        self.font_size = font_size
        self.style = style
        
        # Default color scheme based on style
        self.colors = self._get_default_colors(style)
        
        # Display parameters
        self.gauge_radius = 0.8
        self.center = (0, 0)
        self.arrow_length = 0.7
        self.start_angle = 180
        self.end_angle = 0
        
    def _get_default_colors(self, style: GaugeStyle) -> Dict:
        """Get default color scheme for style."""
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
                'gauge_high': '#F44336'
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
                'gauge_high': '#ff6b6b'
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
                'gauge_high': '#ff6b6b'
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
                'gauge_high': '#3366ff'
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
                        gauge_high: str = None):
        """
        Set color scheme.
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
    
    def _calculate_angle(self, value: float, normalized: bool = False) -> float:
        """
        Calculate arrow angle based on value.
        """
        if normalized:
            normalized_value = max(0, min(1, value))
        else:
            normalized_value = max(0, min(1, value / self.max_value))
        
        angle = self.start_angle - normalized_value * (self.start_angle - self.end_angle)
        return angle
    
    def _get_arrow_color(self, value: float, normalized: bool = False) -> str:
        """
        Get arrow color based on value.
        """
        if normalized:
            normalized_value = max(0, min(1, value))
        else:
            normalized_value = max(0, min(1, value / self.max_value))
        
        if self.style == GaugeStyle.GRADIENT_PULSE:
            # For Gradient Pulse, use dynamic rainbow colors
            hue = normalized_value
            r = int(127.5 * (1 + np.sin(2 * np.pi * hue)))
            g = int(127.5 * (1 + np.sin(2 * np.pi * hue + 2 * np.pi / 3)))
            b = int(127.5 * (1 + np.sin(2 * np.pi * hue + 4 * np.pi / 3)))
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
        Interpolate between two colors.
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
    
    def create_gauge(self, 
                     current_value: float, 
                     normalized: bool = False,
                     show_grid: bool = True,
                     figsize: Tuple[int, int] = (8, 6)):
        """
        Create gauge figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Setup axes
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Set background
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
        
        plt.tight_layout()
        return fig
    
    def _draw_classic(self, ax, current_value, normalized, show_grid):
        """Classic style."""
        # Outer circle
        outer_circle = patches.Circle(
            self.center, 
            radius=self.gauge_radius + 0.05,
            facecolor='white',
            edgecolor=self.colors['edge'],
            linewidth=3,
            zorder=1
        )
        ax.add_patch(outer_circle)
        
        # Background arc
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
        
        # Colored arc (gradient)
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
        
        # Scale marks
        if show_grid:
            self._add_scale_marks(ax, normalized)
        
        # Arrow
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
        
        # Center circle
        center_circle = patches.Circle(
            self.center, 
            radius=0.06,
            facecolor='white',
            edgecolor=self.colors['edge'],
            linewidth=3,
            zorder=6
        )
        ax.add_patch(center_circle)
        
        # Display values
        self._add_value_display(ax, current_value, normalized)
    
    def _add_scale_marks(self, ax, normalized):
        """Add scale marks."""
        # Use normalized ticks for angle calculation, but actual values for labels
        if normalized:
            # For normalized gauge: 0, 0.2, 0.4, 0.6, 0.8, 1.0
            major_ticks_normalized = np.linspace(0, 1, 6)
            tick_values = major_ticks_normalized
            tick_labels = [f"{tick:.1f}" for tick in tick_values]
            use_ticks_for_angle = major_ticks_normalized
        else:
            # For regular gauge: distribute normalized values evenly, but show actual values as labels
            num_ticks = 6
            major_ticks_normalized = np.linspace(0, 1, num_ticks)
            tick_values = major_ticks_normalized * self.max_value
            
            # Format labels based on value magnitude
            tick_labels = []
            for value in tick_values:
                if self.max_value >= 1000:
                    tick_labels.append(f"{value/1000:.1f}k")
                elif self.max_value >= 100:
                    tick_labels.append(f"{value:.0f}")
                else:
                    tick_labels.append(f"{value:.1f}")
            
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
        
        # Grid
        for x in np.linspace(-1, 1, 11):
            ax.plot([x, x], [-0.2, 1.1], color='#112240', linewidth=0.5, alpha=0.3, zorder=0)
        for y in np.linspace(-0.2, 1.1, 11):
            ax.plot([-1, 1], [y, y], color='#112240', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Outer ring with neon glow
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
            facecolor='#112240',
            edgecolor='none',
            alpha=0.7,
            zorder=2
        )
        ax.add_patch(inner_ring)
        
        # Scale segments
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
        
        # Arrow
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
        
        # Arrow glow
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
        
        # Center
        ax.add_patch(patches.Circle(
            self.center, radius=0.05, facecolor=arrow_color, edgecolor='none', zorder=6
        ))
        
        # Display values
        self._add_value_display(ax, current_value, normalized, style='tech')
    
    def _draw_gradient(self, ax, current_value, normalized, show_grid):
        """Gradient style."""
        # Gradient background
        gradient = plt.Circle(self.center, 1.1, transform=ax.transData)
        gradient.set_facecolor(self.colors['face'])
        ax.add_patch(gradient)
        
        # Main circle with gradient
        theta = np.linspace(np.radians(self.start_angle), np.radians(self.end_angle), 100)
        x = self.gauge_radius * np.cos(theta)
        y = self.gauge_radius * np.sin(theta)
        
        # Create gradient
        norm_value = current_value / self.max_value if not normalized else current_value
        active_theta = np.linspace(np.radians(self.start_angle), 
                                 np.radians(self.start_angle - norm_value * (self.start_angle - self.end_angle)), 100)
        active_x = self.gauge_radius * np.cos(active_theta)
        active_y = self.gauge_radius * np.sin(active_theta)
        
        # Gradient fill
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
        
        # Display values
        self._add_value_display(ax, current_value, normalized)
    
    def _draw_gradient_pulse(self, ax, current_value, normalized, show_grid):
        """Gradient Pulse style with pulsing effects."""
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
        
        # Pulsing arrow
        angle = self._calculate_angle(current_value, normalized)
        
        # Get color from gradient for arrow
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
        
        # Pulsing center
        ax.add_patch(patches.Circle(
            self.center, radius=0.06, facecolor=arrow_color, edgecolor='none', zorder=10
        ))
        
        # Multiple center pulses
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
        
        # Scale marks (minimal for pulse style)
        if show_grid:
            if normalized:
                major_ticks_normalized = np.linspace(0, 1, 6)
                tick_labels = [f"{tick:.1f}" for tick in major_ticks_normalized]
            else:
                major_ticks_normalized = np.linspace(0, 1, 6)
                tick_values = major_ticks_normalized * self.max_value
                tick_labels = []
                for value in tick_values:
                    if self.max_value >= 1000:
                        tick_labels.append(f"{value/1000:.1f}k")
                    elif self.max_value >= 100:
                        tick_labels.append(f"{value:.0f}")
                    else:
                        tick_labels.append(f"{value:.1f}")
            
            for i, tick in enumerate(major_ticks_normalized):
                angle = self._calculate_angle(tick, normalized=True)
                
                # Glowing dots instead of lines
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
        
        # Display values with gradient
        self._add_value_display(ax, current_value, normalized, style='pulse')
    
    def _add_value_display(self, ax, current_value, normalized, **kwargs):
        """Display value and title."""
        style_specific = kwargs.get('style', '')
        
        # Title
        title_y = 1.05
        if style_specific == 'tech':
            title_color = self.colors['title']
            title_weight = 'bold'
        elif style_specific == 'pulse':
            # Get gradient color for pulse style
            hue = (current_value / self.max_value) if not normalized else current_value
            r = int(127.5 * (1 + np.sin(2 * np.pi * hue)))
            g = int(127.5 * (1 + np.sin(2 * np.pi * hue + 2 * np.pi / 3)))
            b = int(127.5 * (1 + np.sin(2 * np.pi * hue + 4 * np.pi / 3)))
            title_color = f'#{r:02x}{g:02x}{b:02x}'
            title_weight = 'bold'
        else:
            title_color = self.colors['title']
            title_weight = 'bold'
        
        ax.text(
            0, title_y,
            self.title,
            ha='center',
            va='center',
            fontsize=self.font_size + 8,
            fontweight=title_weight,
            color=title_color,
            zorder=10
        )
        
        # Current value
        if not normalized:
            value_text = f"{current_value:.2f}"
            percent_text = f"({current_value/self.max_value:.1%})"
            max_text = f"of {self.max_value:.2f}"
        else:
            value_text = f"{current_value:.3f}"
            percent_text = f"({current_value:.1%})"
            max_text = "normalized [0, 1]"
        
        # Main value
        if style_specific == 'pulse':
            hue = (current_value / self.max_value) if not normalized else current_value
            r = int(127.5 * (1 + np.sin(2 * np.pi * hue)))
            g = int(127.5 * (1 + np.sin(2 * np.pi * hue + 2 * np.pi / 3)))
            b = int(127.5 * (1 + np.sin(2 * np.pi * hue + 4 * np.pi / 3)))
            value_color = f'#{r:02x}{g:02x}{b:02x}'
        else:
            value_color = self.colors['value_text']
        
        ax.text(
            0, -0.1,
            value_text,
            ha='center',
            va='center',
            fontsize=self.font_size + 10,
            fontweight='bold',
            color=value_color,
            zorder=10
        )
        
        # Percentage
        text_color = self.colors['text']
        if style_specific == 'pulse':
            text_color = 'white'
        
        ax.text(
            0, -0.18,
            percent_text,
            ha='center',
            va='center',
            fontsize=self.font_size + 4,
            color=text_color,
            zorder=10
        )
        
        # Maximum value or normalization info
        ax.text(
            0, -0.25,
            max_text,
            ha='center',
            va='center',
            fontsize=self.font_size,
            color=text_color,
            alpha=0.7,
            zorder=10
        )


def main():
    """Main Streamlit app."""
    
    # Заголовок приложения
    st.title("🎛️ Speed map Generator")
    st.markdown("""
    Generate circular gauges with arrows for scientific visualization. 
    Supports single gauges or multiple sample comparison with 4 different styles.
    """)
    
    # Сайдбар с настройками
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Режим отображения
        mode = st.radio(
            "Display Mode",
            ["Single Gauge", "Multiple Samples"],
            index=0
        )
        
        # Основные параметры
        title = st.text_input("Title", "Reaction Speed")
        max_value = st.number_input("Maximum Value", 0.1, 10000.0, 100.0, 0.1)
        font_size = st.slider("Font Size", 8, 24, 12)
        style = st.selectbox(
            "Gauge Style",
            [style.value for style in GaugeStyle],
            index=0
        )
        
        # Получаем объект стиля
        style_obj = GaugeStyle(style)
        
        # Параметры в зависимости от режима
        if mode == "Single Gauge":
            current_value = st.slider(
                "Current Value", 
                0.0, float(max_value), 65.0, 0.1
            )
            show_normalized = st.checkbox("Show Normalized Gauge", True)
        else:
            num_samples = st.slider("Number of Samples", 2, 8, 3)
            show_normalized = st.checkbox("Show Normalized Comparison", True)
            
            # Настройки для каждого образца
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
        
        # Дополнительные параметры
        show_grid = st.checkbox("Show Scale Marks", True)
        dpi = st.slider("DPI for Export", 300, 600, 300, 50)
        
        # Настройки цветов для стиля
        st.subheader("🎨 Color Scheme")
        
        color_schemes = {
            "Classic": {
                'face_color': '#f0f0f0',
                'edge_color': '#333333',
                'arrow_color': '#2E86AB',
                'text_color': '#000000',
                'gauge_low': '#4CAF50',
                'gauge_mid': '#FFC107',
                'gauge_high': '#F44336'
            },
            "Tech": {
                'face_color': '#0a192f',
                'edge_color': '#64ffda',
                'arrow_color': '#64ffda',
                'text_color': '#ccd6f6',
                'gauge_low': '#00ff88',
                'gauge_mid': '#ffd166',
                'gauge_high': '#ff6b6b'
            },
            "Gradient": {
                'face_color': '#f8f9fa',
                'edge_color': '#495057',
                'arrow_color': '#228be6',
                'text_color': '#343a40',
                'gauge_low': '#51cf66',
                'gauge_mid': '#ffd43b',
                'gauge_high': '#ff6b6b'
            },
            "Gradient Pulse": {
                'face_color': '#0a0a0a',
                'edge_color': '#ffffff',
                'arrow_color': '#ffffff',
                'text_color': '#ffffff',
                'gauge_low': '#ff3366',
                'gauge_mid': '#00ff9d',
                'gauge_high': '#3366ff'
            }
        }
        
        if style in color_schemes:
            colors = color_schemes[style]
            
            col1, col2 = st.columns(2)
            with col1:
                face_color = st.color_picker("Background", colors['face_color'])
                edge_color = st.color_picker("Edge", colors['edge_color'])
                arrow_color = st.color_picker("Arrow", colors['arrow_color'])
            
            with col2:
                text_color = st.color_picker("Text", colors['text_color'])
                gauge_low = st.color_picker("Low Zone", colors['gauge_low'])
                gauge_mid = st.color_picker("Medium Zone", colors['gauge_mid'])
                gauge_high = st.color_picker("High Zone", colors['gauge_high'])
    
    # Основная область контента
    st.header("📊 Gauge Visualization")
    
    if mode == "Single Gauge":
        # Создаем один датчик
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Standard Gauge")
            gauge = CircularGauge(
                title=title,
                max_value=max_value,
                font_size=font_size,
                style=style_obj
            )
            
            gauge.set_color_scheme(
                face_color=face_color,
                edge_color=edge_color,
                arrow_color=arrow_color,
                text_color=text_color,
                gauge_low=gauge_low,
                gauge_mid=gauge_mid,
                gauge_high=gauge_high
            )
            
            fig = gauge.create_gauge(current_value, normalized=False, show_grid=show_grid)
            st.pyplot(fig)
            
            # Кнопка загрузки
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
                    style=style_obj
                )
                
                normalized_gauge.set_color_scheme(
                    face_color=face_color,
                    edge_color=edge_color,
                    arrow_color=arrow_color,
                    text_color=text_color,
                    gauge_low=gauge_low,
                    gauge_mid=gauge_mid,
                    gauge_high=gauge_high
                )
                
                normalized_value = current_value / max_value
                norm_fig = normalized_gauge.create_gauge(
                    normalized_value, 
                    normalized=True, 
                    show_grid=show_grid
                )
                st.pyplot(norm_fig)
                
                # Кнопка загрузки
                buf_norm = io.BytesIO()
                norm_fig.savefig(buf_norm, format='png', dpi=dpi, bbox_inches='tight')
                st.download_button(
                    label="📥 Download Normalized Gauge",
                    data=buf_norm.getvalue(),
                    file_name=f"gauge_{title.replace(' ', '_')}_normalized_{style.lower()}.png",
                    mime="image/png"
                )
                plt.close(norm_fig)
        
        # Информация о значениях
        st.info(f"**Value:** {current_value:.2f} / {max_value} ({current_value/max_value:.1%})")
        
    else:  # Multiple Samples
        # Определяем макет для образцов
        if show_normalized:
            # Два ряда: стандартные и нормализованные
            st.subheader("Standard Gauges")
            cols = st.columns(num_samples)
            
            # Стандартные датчики
            for i, (sample, col) in enumerate(zip(samples, cols)):
                with col:
                    gauge = CircularGauge(
                        title=sample['name'],
                        max_value=max_value,
                        font_size=font_size - 2,
                        style=style_obj
                    )
                    
                    # Используем цвет образца для стрелки
                    gauge.set_color_scheme(
                        face_color=face_color,
                        edge_color=edge_color,
                        arrow_color=sample['color'],
                        text_color=text_color,
                        gauge_low=sample['color'],
                        gauge_mid=sample['color'],
                        gauge_high=sample['color']
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
            
            # Нормализованные датчики
            for i, (sample, col) in enumerate(zip(samples, cols_norm)):
                with col:
                    normalized_gauge = CircularGauge(
                        title=f"{sample['name']} (Norm)",
                        max_value=1.0,
                        font_size=font_size - 2,
                        style=style_obj
                    )
                    
                    normalized_gauge.set_color_scheme(
                        face_color=face_color,
                        edge_color=edge_color,
                        arrow_color=sample['color'],
                        text_color=text_color,
                        gauge_low=sample['color'],
                        gauge_mid=sample['color'],
                        gauge_high=sample['color']
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
            
            # Создаем общую фигуру для загрузки
            if st.button("📥 Download All Gauges as ZIP"):
                import zipfile
                from io import BytesIO
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Стандартные датчики
                    for i, sample in enumerate(samples):
                        gauge = CircularGauge(
                            title=sample['name'],
                            max_value=max_value,
                            font_size=font_size,
                            style=style_obj
                        )
                        
                        gauge.set_color_scheme(
                            face_color=face_color,
                            edge_color=edge_color,
                            arrow_color=sample['color'],
                            text_color=text_color,
                            gauge_low=sample['color'],
                            gauge_mid=sample['color'],
                            gauge_high=sample['color']
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
                    
                    # Нормализованные датчики
                    for i, sample in enumerate(samples):
                        normalized_gauge = CircularGauge(
                            title=f"{sample['name']}_normalized",
                            max_value=1.0,
                            font_size=font_size,
                            style=style_obj
                        )
                        
                        normalized_gauge.set_color_scheme(
                            face_color=face_color,
                            edge_color=edge_color,
                            arrow_color=sample['color'],
                            text_color=text_color,
                            gauge_low=sample['color'],
                            gauge_mid=sample['color'],
                            gauge_high=sample['color']
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
            # Только стандартные датчики
            cols = st.columns(num_samples)
            
            for i, (sample, col) in enumerate(zip(samples, cols)):
                with col:
                    gauge = CircularGauge(
                        title=sample['name'],
                        max_value=max_value,
                        font_size=font_size,
                        style=style_obj
                    )
                    
                    gauge.set_color_scheme(
                        face_color=face_color,
                        edge_color=edge_color,
                        arrow_color=sample['color'],
                        text_color=text_color,
                        gauge_low=sample['color'],
                        gauge_mid=sample['color'],
                        gauge_high=sample['color']
                    )
                    
                    fig = gauge.create_gauge(
                        sample['value'], 
                        normalized=False, 
                        show_grid=show_grid
                    )
                    st.pyplot(fig)
                    
                    # Кнопка загрузки для каждого датчика
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
        
        # Статистика
        values = [s['value'] for s in samples]
        avg_value = np.mean(values)
        min_value = np.min(values)
        max_value_val = np.max(values)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{avg_value:.2f}")
        with col2:
            st.metric("Minimum", f"{min_value:.2f}")
        with col3:
            st.metric("Maximum", f"{max_value_val:.2f}")
        with col4:
            st.metric("Range", f"{max_value_val - min_value:.2f}")
    
    # Информация в нижней части
    st.markdown("---")
    st.markdown("""
    ### 🎯 Features
    - **4 Gauge Styles:** Classic, Tech, Gradient, Gradient Pulse
    - **Two Modes:** Single gauge or multiple sample comparison
    - **Normalization:** Display gauges with 0-1 scale
    - **Custom Colors:** Full control over all color elements
    - **High-Quality Export:** Adjustable DPI (300-600)
    - **Scientific Ready:** Perfect for research presentations and papers
    
    ### 📝 Instructions
    1. Configure settings in the sidebar
    2. For multiple samples: expand each sample to configure
    3. Click download buttons to save gauges
    4. Use high DPI for publication-quality images
    """)
    
    # Информация о приложении
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        **Speed map generator**  
        Created for scientific visualization of quantitative data.
        
        This tool generates circular gauge indicators similar to speedometers
        but adapted for scientific purposes. Each gauge shows a value relative
        to a maximum, with clear visual indication via an arrow pointer.
        
        **Use cases:**
        - Experimental results visualization
        - Performance metrics display
        - Comparative analysis of samples
        - Scientific presentations and publications
        
        **developed by @daM, @CTA, https://chimicatechnoacta.ru **.
        """)

if __name__ == "__main__":

    main()
