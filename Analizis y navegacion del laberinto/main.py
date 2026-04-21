import rclpy
import cv2
import numpy as np
import os
import shutil

# --- MODIFICACIÓN: Agregar "maze_navigator." a los imports locales ---
from maze_navigator.robot_ros import MazeRobotNode
from maze_navigator.vision_laberinto import paso1_capturar_camara, paso2_reconstruccion, paso3_dibujar_puntos
from maze_navigator.algoritmos_ruta import detectar_aberturas_360, paso4_resolver_laberinto
from maze_navigator.navegacion_ar import iniciar_realidad_aumentada



# =====================================================================
# PASO 5: DASHBOARD DE CONTROL UNIFICADO
# =====================================================================
def paso5_dashboard_interactivo(resultados, p_ent, p_sal, ros_node, trial_path):
    ganador_key = min(resultados.keys(), key=lambda k: (resultados[k]['pasos'], resultados[k]['n']))
    
    # --- MODIFICACION: WINDOW_AUTOSIZE fuerza el tamano real sin encogerse ---
    cv2.namedWindow("Dashboard Analitico", cv2.WINDOW_AUTOSIZE)
    estado_actual = None 

    while True:
        # Lienzo ajustado a 600 de alto por 1050 de ancho
        lienzo = np.zeros((600, 1050, 3), dtype=np.uint8)
        cv2.rectangle(lienzo, (600, 0), (1050, 600), (25, 25, 25), -1) 
        
        cv2.putText(lienzo, "--- PANEL DE NAVEGACION ---", (620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(lienzo, "[1] A* [2] Dijkstra", (620, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(lienzo, "[3] BFS    [4] DFS", (620, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(lienzo, "[5] Mejor Algoritmo", (620, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.putText(lienzo, "[v] INICIAR PILOTO AUTOMATICO", (620, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(lienzo, "[q] Salir", (620, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.line(lienzo, (620, 215), (1030, 215), (100, 100, 100), 2)
        
        if estado_actual:
            k_show = ganador_key if estado_actual == '5' else estado_actual
            res = resultados.get(k_show, resultados['1']) 
            
            # Imagen ajustada a 600x600
            img_res = cv2.resize(res['img'], (600, 600))
            if estado_actual == '5': 
                img_res = cv2.copyMakeBorder(cv2.resize(res['img'], (580,580)), 10,10,10,10, cv2.BORDER_CONSTANT, value=[0, 215, 255])
            lienzo[:, :600] = img_res
            
            cv2.putText(lienzo, res['nombre'], (620, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(lienzo, f"Pasos Fisicos: {res['pasos']}", (620, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(lienzo, f"Tiempo CPU:    {res['t']:.2f} ms", (620, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(lienzo, f"Nodos Revisados: {res['n']}", (620, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                
        else:
            cv2.putText(lienzo, "Escoge un algoritmo del menu", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        cv2.imshow("Dashboard Analitico", lienzo)
        
        k = cv2.waitKey(0) & 0xFF
        char_k = chr(k) if k < 256 else ''
        
        if char_k in ['1', '2', '3', '4', '5']:
            estado_actual = char_k
        elif char_k == 'v':
            if estado_actual:
                res = resultados.get(estado_actual if estado_actual != '5' else ganador_key, resultados['1'])
                # Se envía el diccionario de resultados y la ruta de la prueba para guardar datos experimentales
                iniciar_realidad_aumentada(res['camino'], p_ent, p_sal, res['color_ar'], 5, 5, ros_node, res, trial_path)
            else:
                print("⚠️ Selecciona un algoritmo antes de iniciar el Piloto Automatico.")
        elif char_k == 'q':
            break

    cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    ros_node = MazeRobotNode()

    img_cap, img_opt, img_grf = "12.png", "laberinto_optimizado.png", "laberinto_nodos.png"
    
    if paso1_capturar_camara(img_cap):
        if paso2_reconstruccion(img_cap, img_opt):
            p_ent, p_sal = detectar_aberturas_360(img_opt, cells_x=5, cells_y=5)
            paso3_dibujar_puntos(img_opt, p_ent, p_sal, img_grf, cells_x=5, cells_y=5)
            datos_resultados = paso4_resolver_laberinto(img_grf, p_ent, p_sal, cells_x=5, cells_y=5)
            
            # --- MODIFICACION: GESTION DE DATOS EXPERIMENTALES ---
            base_dir = "Datos Experimentales"
            os.makedirs(base_dir, exist_ok=True)
            
            trial_num = 1
            while os.path.exists(os.path.join(base_dir, f"Prueba {trial_num}")):
                trial_num += 1
            trial_path = os.path.join(base_dir, f"Prueba {trial_num}")
            os.makedirs(trial_path)

            # Guardar imagenes base renombradas en la carpeta de la prueba
            shutil.copy(img_cap, os.path.join(trial_path, "captura_original.png"))
            shutil.copy(img_opt, os.path.join(trial_path, "laberinto_optimizado.png"))
            shutil.copy(img_grf, os.path.join(trial_path, "laberinto_nodos.png"))
            
            paso5_dashboard_interactivo(datos_resultados, p_ent, p_sal, ros_node, trial_path)
            
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
