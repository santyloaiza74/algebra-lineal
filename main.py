import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import math

# Configuración inicial
np.set_printoptions(precision=3, suppress=True)

## 1. Funciones básicas de transformación homogénea

def rotx(theta):
    """Matriz de rotación alrededor del eje X"""
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(theta), -math.sin(theta), 0],
        [0, math.sin(theta), math.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def roty(theta):
    """Matriz de rotación alrededor del eje Y"""
    return np.array([
        [math.cos(theta), 0, math.sin(theta), 0],
        [0, 1, 0, 0],
        [-math.sin(theta), 0, math.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def rotz(theta):
    """Matriz de rotación alrededor del eje Z"""
    return np.array([
        [math.cos(theta), -math.sin(theta), 0, 0],
        [math.sin(theta), math.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def transl(x, y, z):
    """Matriz de traslación"""
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

## 2. Clase para el robot (brazo articulado de 3 GDL)

class Robot3DOF:
    def __init__(self, lengths=[1.0, 1.0, 0.5]):
        """
        Inicializa el robot con las longitudes de los eslabones
        :param lengths: [l1, l2, l3] longitudes de los eslabones
        """
        self.lengths = lengths
        self.joints = [np.eye(4) for _ in range(4)]  # 4 frames: base + 3 articulaciones
        self.fig = None
        self.ax = None
        self.line = None
        self.points = None
        
    def forward_kinematics(self, theta1, theta2, theta3):
        """
        Calcula la cinemática directa del robot
        :param theta1: Ángulo de la articulación 1 (radianes)
        :param theta2: Ángulo de la articulación 2 (radianes)
        :param theta3: Ángulo de la articulación 3 (radianes)
        :return: Matriz de transformación del efector final
        """
        l1, l2, l3 = self.lengths
        
        # Transformaciones sucesivas
        T01 = rotz(theta1) @ transl(0, 0, l1/2)  # Base a articulación 1
        T12 = rotx(theta2) @ transl(0, 0, l1/2)  # Articulación 1 a 2
        T23 = rotx(theta3) @ transl(0, l2, 0)    # Articulación 2 a 3
        T3E = transl(0, l3, 0)                   # Articulación 3 a efector
        
        # Actualizar las posiciones de las articulaciones
        self.joints[0] = np.eye(4)  # Base
        self.joints[1] = T01        # Articulación 1
        self.joints[2] = T01 @ T12   # Articulación 2
        self.joints[3] = T01 @ T12 @ T23 @ T3E  # Efector final
        
        return self.joints[3]
    
    def plot_setup(self):
        """Configura la figura para visualización"""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Configuración de los ejes
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 2.5])
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        self.ax.set_title('Robot de 3 GDL - Cinemática Directa')
        
        # Dibujar el robot inicialmente
        positions = [T[:3, 3] for T in self.joints]
        x, y, z = zip(*positions)
        self.line, = self.ax.plot(x, y, z, 'o-', linewidth=3, markersize=8, 
                                color='blue', markerfacecolor='red')
        
        # Añadir cuadrícula y estilo
        self.ax.grid(True)
        
    def update_plot(self, theta1, theta2, theta3):
        """Actualiza el gráfico con nuevos ángulos"""
        self.forward_kinematics(theta1, theta2, theta3)
        positions = [T[:3, 3] for T in self.joints]
        x, y, z = zip(*positions)
        
        self.line.set_data(x, y)
        self.line.set_3d_properties(z)
        
        # Actualizar los límites de los ejes si es necesario
        max_range = max(max(x)-min(x), max(y)-min(y), max(z)-min(z)) * 0.7
        mid_x = (max(x)+min(x)) * 0.5
        mid_y = (max(y)+min(y)) * 0.5
        mid_z = (max(z)+min(z)) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.fig.canvas.draw_idle()

## 3. Visualización interactiva con sliders

def create_interactive_robot():
    """Crea una visualización interactiva del robot con controles deslizantes"""
    robot = Robot3DOF()
    robot.plot_setup()
    
    # Ajustar el layout para hacer espacio para los sliders
    plt.subplots_adjust(bottom=0.3)
    
    # Crear ejes para los sliders
    ax_slider1 = plt.axes([0.2, 0.2, 0.6, 0.03])
    ax_slider2 = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_slider3 = plt.axes([0.2, 0.1, 0.6, 0.03])
    
    # Crear los sliders
    slider1 = Slider(ax_slider1, 'X', -np.pi, np.pi, valinit=0)
    slider2 = Slider(ax_slider2, 'Y', -np.pi/2, np.pi/2, valinit=0)
    slider3 = Slider(ax_slider3, 'Z', -np.pi/2, np.pi/2, valinit=0)
    
    # Función de actualización
    def update(val):
        theta1 = slider1.val
        theta2 = slider2.val
        theta3 = slider3.val
        robot.update_plot(theta1, theta2, theta3)
        
        # Mostrar posición del efector final
        T = robot.joints[3]
        print(f"Posición efector: X={T[0,3]:.2f}, Y={T[1,3]:.2f}, Z={T[2,3]:.2f}")
    
    # Conectar los sliders a la función de actualización
    slider1.on_changed(update)
    slider2.on_changed(update)
    slider3.on_changed(update)
    
    # Configuración inicial
    robot.update_plot(0, 0, 0)
    plt.show()

## 4. Ejemplo de trayectoria del efector final

def plot_trajectory():
    """Muestra una trayectoria circular del efector final"""
    robot = Robot3DOF()
    robot.plot_setup()
    
    # Generar puntos de una trayectoria circular
    theta1_vals = np.linspace(0, 2*np.pi, 50)
    x_vals, y_vals, z_vals = [], [], []
    
    for theta1 in theta1_vals:
        # Cinemática directa para ángulos que forman un círculo
        robot.forward_kinematics(theta1, np.pi/4, np.pi/6)
        T = robot.joints[3]
        x_vals.append(T[0,3])
        y_vals.append(T[1,3])
        z_vals.append(T[2,3])
    
    # Dibujar la trayectoria
    robot.ax.plot(x_vals, y_vals, z_vals, 'r-', linewidth=2, alpha=0.5)
    
    # Dibujar la posición final del robot
    robot.update_plot(theta1_vals[-1], np.pi/4, np.pi/6)
    
    plt.title('Trayectoria del Efector Final')
    plt.show()

## 5. Menú principal para seleccionar la visualización

def main():
    print("Proyecto de Cinemática de Robots")
    print("1. Visualización interactiva con controles deslizantes")
    print("2. Ejemplo de trayectoria circular del efector")
    print("3. Salir")
    
    choice = input("Selecciona una opción (1-3): ")
    
    if choice == '1':
        create_interactive_robot()
    elif choice == '2':
        plot_trajectory()
    else:
        print("Saliendo del programa...")

if __name__ == "__main__":
    main()