# Importación de librerías generales del sistema
import rclpy                # Inicializador principal de la red de ROS 2 en Python.
import cv2                  # OpenCV para crear la ventana del menú interactivo (Dashboard).
import numpy as np          # NumPy para crear la matriz vacía (el lienzo negro del menú).
import os                   # Para crear carpetas dinámicamente en el sistema operativo.
import shutil               # Para copiar y pegar archivos (las fotos de evidencia) a las carpetas.

# --- MODIFICACIÓN: Importaciones de tus propios módulos ---
# Llama a todas las funciones que analizamos anteriormente desde la carpeta del paquete 'maze_navigator'.
from maze_navigator.robot_ros import MazeRobotNode
from maze_navigator.vision_laberinto import paso1_capturar_camara, paso2_reconstruccion, paso3_dibujar_puntos
from maze_navigator.algoritmos_ruta import detectar_aberturas_360, paso4_resolver_laberinto
from maze_navigator.navegacion_ar import iniciar_realidad_aumentada


# =====================================================================
# PASO 5: DASHBOARD DE CONTROL UNIFICADO (Interfaz de Usuario)
# =====================================================================
def paso5_dashboard_interactivo(resultados, p_ent, p_sal, ros_node, trial_path):
    # LÓGICA DE DECISIÓN DEL "MEJOR ALGORITMO":
    # Analiza el diccionario de resultados. Busca cuál tiene menos "pasos" físicos.
    # Si hay un empate en pasos, el desempate lo gana el que revisó menos nodos ('n').
    ganador_key = min(resultados.keys(), key=lambda k: (resultados[k]['pasos'], resultados[k]['n']))
    
    # Crea una ventana de OpenCV para el menú. WINDOW_AUTOSIZE evita que la ventana se distorsione o encoja.
    cv2.namedWindow("Dashboard Analitico", cv2.WINDOW_AUTOSIZE)
    estado_actual = None # Variable que guarda qué botón presionó el usuario (1, 2, 3, 4 o 5).

    while True: # Bucle infinito para mantener el menú abierto y receptivo.
        # Crea un lienzo base de 1050 píxeles de ancho por 600 de alto (3 canales RGB).
        lienzo = np.zeros((600, 1050, 3), dtype=np.uint8)
        
        # Dibuja un rectángulo gris oscuro en la parte derecha (desde el px 600 al 1050) para el panel de texto.
        cv2.rectangle(lienzo, (600, 0), (1050, 600), (25, 25, 25), -1) 
        
        # --- RENDERIZADO DE TEXTOS E INSTRUCCIONES ---
        cv2.putText(lienzo, "--- PANEL DE NAVEGACION ---", (620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(lienzo, "[1] A* [2] Dijkstra", (620, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(lienzo, "[3] BFS    [4] DFS", (620, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(lienzo, "[5] Mejor Algoritmo", (620, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.putText(lienzo, "[v] INICIAR PILOTO AUTOMATICO", (620, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(lienzo, "[q] Salir", (620, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.line(lienzo, (620, 215), (1030, 215), (100, 100, 100), 2) # Línea divisoria decorativa.
        
        # --- LÓGICA DE VISUALIZACIÓN DINÁMICA ---
        if estado_actual:
            # Si eligió el '5', cargamos la información del ganador. Si no, cargamos el que haya elegido.
            k_show = ganador_key if estado_actual == '5' else estado_actual
            res = resultados.get(k_show, resultados['1']) # Extrae los datos del diccionario.
            
            # Redimensiona la imagen del laberinto dibujado a 600x600 para que encaje en el lado izquierdo.
            img_res = cv2.resize(res['img'], (600, 600))
            
            # Si eligió el "Mejor Algoritmo", le pone un marco amarillo de 10px para resaltarlo visualmente.
            if estado_actual == '5': 
                img_res = cv2.copyMakeBorder(cv2.resize(res['img'], (580,580)), 10,10,10,10, cv2.BORDER_CONSTANT, value=[0, 215, 255])
                
            # Pega la imagen en la mitad izquierda del lienzo.
            lienzo[:, :600] = img_res
            
            # Imprime las estadísticas matemáticas del algoritmo seleccionado en la mitad derecha.
            cv2.putText(lienzo, res['nombre'], (620, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(lienzo, f"Pasos Fisicos: {res['pasos']}", (620, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(lienzo, f"Tiempo CPU:    {res['t']:.2f} ms", (620, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(lienzo, f"Nodos Revisados: {res['n']}", (620, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                
        else:
            # Texto por defecto si no ha tocado nada todavía.
            cv2.putText(lienzo, "Escoge un algoritmo del menu", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        cv2.imshow("Dashboard Analitico", lienzo)
        
        # --- LECTURA DE TECLADO ---
        k = cv2.waitKey(0) & 0xFF            # Espera a que el usuario presione una tecla.
        char_k = chr(k) if k < 256 else ''   # Convierte el código de la tecla a un caracter (string).
        
        if char_k in ['1', '2', '3', '4', '5']:
            estado_actual = char_k # Cambia la vista al algoritmo correspondiente.
            
        elif char_k == 'v':
            # Si presiona 'v', lanza la Realidad Aumentada (el archivo navegacion_ar.py).
            if estado_actual:
                # Obtiene la ruta del algoritmo elegido para pasársela al auto.
                res = resultados.get(estado_actual if estado_actual != '5' else ganador_key, resultados['1'])
                # Llama a la función principal pasándole TODA la información (Ruta, colores, nodo de ROS2, etc).
                iniciar_realidad_aumentada(res['camino'], p_ent, p_sal, res['color_ar'], 5, 5, ros_node, res, trial_path)
            else:
                print("⚠️ Selecciona un algoritmo antes de iniciar el Piloto Automatico.")
                
        elif char_k == 'q':
            break # Rompe el bucle para cerrar el programa.

    cv2.destroyAllWindows()


# =====================================================================
# FUNCIÓN MAIN (Punto de Ejecución Principal)
# =====================================================================
def main(args=None):
    # 1. Inicializa la comunicación de red ROS 2
    rclpy.init(args=args)
    ros_node = MazeRobotNode() # Crea el nodo transmisor/receptor WiFi.

    # 2. Define los nombres temporales de las fotos generadas por los pasos
    img_cap, img_opt, img_grf = "12.png", "laberinto_optimizado.png", "laberinto_nodos.png"
    
    # 3. EJECUCIÓN DEL PIPELINE SECUENCIAL
    # Si la captura (Paso 1) fue exitosa...
    if paso1_capturar_camara(img_cap):
        # Si la reconstrucción de paredes (Paso 2) fue exitosa...
        if paso2_reconstruccion(img_cap, img_opt):
            # Busca entradas y salidas en el laberinto limpio.
            p_ent, p_sal = detectar_aberturas_360(img_opt, cells_x=5, cells_y=5)
            # Dibuja los puntitos amarillos, el rojo (Start) y el verde (End).
            paso3_dibujar_puntos(img_opt, p_ent, p_sal, img_grf, cells_x=5, cells_y=5)
            # El cerebro matemático resuelve la ruta por los 4 algoritmos posibles.
            datos_resultados = paso4_resolver_laberinto(img_grf, p_ent, p_sal, cells_x=5, cells_y=5)
            
            # 4. GESTIÓN DE DATOS EXPERIMENTALES (Sistema de Archivos)
            # Crea una carpeta maestra para no perder los datos de tesis.
            base_dir = "Datos Experimentales"
            os.makedirs(base_dir, exist_ok=True)
            
            # Lógica para crear carpetas numeradas (Prueba 1, Prueba 2...) sin sobrescribir las anteriores.
            trial_num = 1
            while os.path.exists(os.path.join(base_dir, f"Prueba {trial_num}")):
                trial_num += 1
            trial_path = os.path.join(base_dir, f"Prueba {trial_num}")
            os.makedirs(trial_path)

            # 5. GUARDADO DE EVIDENCIA
            # Mueve copias de las fotos base (temporales) a la carpeta de la Prueba definitiva para guardarlas para siempre.
            shutil.copy(img_cap, os.path.join(trial_path, "captura_original.png"))
            shutil.copy(img_opt, os.path.join(trial_path, "laberinto_optimizado.png"))
            shutil.copy(img_grf, os.path.join(trial_path, "laberinto_nodos.png"))
            
            # 6. LANZAR EL MENÚ
            # Una vez que todo está procesado y guardado, abre la interfaz interactiva para el usuario.
            paso5_dashboard_interactivo(datos_resultados, p_ent, p_sal, ros_node, trial_path)
            
    # Al cerrar todo, destruye el nodo de comunicaciones y apaga el sistema de ROS 2 limpiamente.
    ros_node.destroy_node()
    rclpy.shutdown()

# Punto de entrada de Python. Si corres este archivo directamente, ejecuta la función main().
if __name__ == "__main__":
    main()
