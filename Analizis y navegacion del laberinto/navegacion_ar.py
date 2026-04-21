import cv2
import numpy as np
import time
import rclpy
import os

def iniciar_realidad_aumentada(camino_logico, puerta_ent, puerta_sal, color_bgr_ignorado, cells_x, cells_y, ros_node, datos_resultado, trial_path):
    tamano_warp = 800
    margen_corte = 7 
    
    ancho_real = tamano_warp - (2 * margen_corte)
    alto_real = tamano_warp - (2 * margen_corte)
    paso_x = ancho_real / cells_x
    paso_y = alto_real / cells_y
    
    def coord(n):
        if n == 'START': p = puerta_ent
        elif n == 'END': p = puerta_sal
        else: return (int((n[0] + 0.5) * paso_x) + margen_corte, int((n[1] + 0.5) * paso_y) + margen_corte)
        if p[0] == 'N': return (int((p[1] + 0.5) * paso_x) + margen_corte, 0)
        if p[0] == 'S': return (int((p[1] + 0.5) * paso_x) + margen_corte, tamano_warp - 1)
        if p[0] == 'O': return (0, int((p[1] + 0.5) * paso_y) + margen_corte)
        if p[0] == 'E': return (tamano_warp - 1, int((p[1] + 0.5) * paso_y) + margen_corte)

    puntos = [coord(n) for n in camino_logico]
    
    nodos_clave = []
    
    v_in = (puntos[1][0] - puntos[0][0], puntos[1][1] - puntos[0][1])
    v_out = (puntos[2][0] - puntos[1][0], puntos[2][1] - puntos[1][1])
    cp_inicio = v_in[0] * v_out[1] - v_in[1] * v_out[0]
    
    if cp_inicio != 0: nodos_clave.append({'pt': puntos[1], 'tipo': 'GIRO_INICIO', 'cmd': "DERECHA" if cp_inicio > 0 else "IZQUIERDA"})
    else: nodos_clave.append({'pt': puntos[1], 'tipo': 'INICIO_RECTO', 'cmd': 'ADELANTE'})

    for i in range(2, len(puntos) - 2):
        v_in = (puntos[i][0] - puntos[i-1][0], puntos[i][1] - puntos[i-1][1])
        v_out = (puntos[i+1][0] - puntos[i][0], puntos[i+1][1] - puntos[i][1])
        cp = v_in[0] * v_out[1] - v_in[1] * v_out[0]
        if cp != 0: nodos_clave.append({'pt': puntos[i], 'tipo': 'GIRO', 'cmd': "DERECHA" if cp > 0 else "IZQUIERDA"})
            
    nodos_clave.append({'pt': puntos[-2], 'tipo': 'FIN', 'cmd': 'STOP'})

    indice_objetivo = 0
    if nodos_clave[0]['tipo'] == 'INICIO_RECTO':
        for i in range(1, len(nodos_clave)):
            if nodos_clave[i]['tipo'] in ['GIRO', 'FIN']:
                indice_objetivo = i
                break

    longitud_guia = min(paso_x, paso_y) * 0.5
    
    # --- MODIFICACIÓN: Vectores teóricos puros aplicados al nodo ---
    v_ent_x = puntos[1][0] - puntos[0][0]
    v_ent_y = puntos[1][1] - puntos[0][1]
    norm_ent = max(1e-5, np.hypot(v_ent_x, v_ent_y))
    pt_guia_entrada = (int(nodos_clave[0]['pt'][0] - (v_ent_x / norm_ent) * longitud_guia), int(nodos_clave[0]['pt'][1] - (v_ent_y / norm_ent) * longitud_guia))
    
    v_sal_x = puntos[-1][0] - puntos[-2][0]
    v_sal_y = puntos[-1][1] - puntos[-2][1]
    norm_sal = max(1e-5, np.hypot(v_sal_x, v_sal_y))
    pt_guia_salida = (int(nodos_clave[-1]['pt'][0] + (v_sal_x / norm_sal) * longitud_guia), int(nodos_clave[-1]['pt'][1] + (v_sal_y / norm_sal) * longitud_guia))

    # --- MODIFICACION: ALMACENAMIENTO DE DATOS EXPERIMENTALES DEL ALGORITMO ---
    nombre_algoritmo = datos_resultado['nombre']
    algo_folder_name = nombre_algoritmo.replace("*", "_star") 
    algo_path = os.path.join(trial_path, algo_folder_name)
    os.makedirs(algo_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(algo_path, f"Resolucion_{algo_folder_name}.png"), datos_resultado['img'])
    
    with open(os.path.join(algo_path, f"Datos_{algo_folder_name}.txt"), "w") as f:
        f.write(f"Algoritmo: {nombre_algoritmo}\n")
        f.write(f"Nodos para resolver (Pasos fisicos): {datos_resultado['pasos']}\n")
        f.write(f"Tiempo CPU: {datos_resultado['t']:.2f} ms\n")
        f.write(f"Nodos explorados (Revisados): {datos_resultado['n']}\n")

    cap = cv2.VideoCapture(2)
    print("\n🟢 PRUEBA DE MOTOR: SEGUIDOR CONTINUO SIN DETENCIONES 🟢")
    print("Manten la camara quieta. Observa el acoplamiento logico-fisico.")
    
    tiempo_inicio = time.time()
    mapeo_bloqueado = False
    navegacion_iniciada = False
    nodo_alcanzado = False  
    
    tiempo_fase_asoc = 0
    asociacion_calculada = False
    asociacion_hecha = False
    reemplazo_hecho = False
    
    tablero_cnt_est = None
    M_inv_est = None
    M_est = None
    arucos_maze = {}      

    # NUEVAS VARIABLES PARA EL CRONOMETRO Y GUARDADO
    tiempo_inicio_recorrido = 0.0
    tiempo_transcurrido = 0.0
    cronometro_corriendo = False
    tiempo_guardado = False

    # VARIABLES PARA GRABACIÓN DE VIDEO
    video_writer = None
    video_count = 0

    # NUEVAS VARIABLES PARA DETECCIÓN DE VEHÍCULO ESTÁTICO (ATASCO)
    posicion_anterior_auto = None
    tiempo_estatico = time.time()
    estado_retroceso = False
    tiempo_inicio_retroceso = 0.0
    
    # --- VARIABLE DE ESTADO PARA ENVIAR ÓRDENES SOLO UNA VEZ ---
    ultimo_comando_enviado = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        h_frame, w_frame = frame.shape[:2]
        frame_limpio = frame.copy()
        tiempo_actual = time.time()
        
        comando_ros2 = "STOP"
        bloquear_envio = False 

        # LÓGICA DE ACTUALIZACIÓN DEL CRONÓMETRO
        if cronometro_corriendo:
            tiempo_transcurrido = time.time() - tiempo_inicio_recorrido

        if not mapeo_bloqueado:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blur, 50, 150)
            edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            tablero_temp = None
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and cv2.contourArea(c) > 10000:
                    tablero_temp = approx
                    break
            if tablero_temp is not None:
                tablero_cnt_est = tablero_temp
                pts = tablero_cnt_est.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1); rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
                diff = np.diff(pts, axis=1); rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
                dst = np.array([[0, 0], [tamano_warp - 1, 0], [tamano_warp - 1, tamano_warp - 1], [0, tamano_warp - 1]], dtype="float32")
                M_inv_est = cv2.getPerspectiveTransform(dst, rect)
                M_est = cv2.getPerspectiveTransform(rect, dst)
            if tiempo_actual - tiempo_inicio >= 2.0 and tablero_cnt_est is not None:
                mapeo_bloqueado = True

        cx_auto, cy_auto = None, None
        
        # --- ÚNICO SENSOR CENTRAL ---
        pt_sensor_central = None

        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
            corners, ids, rejected = detector.detectMarkers(frame_limpio)
        except AttributeError:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
            corners, ids, rejected = cv2.aruco.detectMarkers(frame_limpio, aruco_dict, parameters=cv2.aruco.DetectorParameters_create())

        if ids is not None:
            for i, aruco_id in enumerate(ids):
                id_val = aruco_id[0]
                cx = int(np.mean(corners[i][0][:, 0]))
                cy = int(np.mean(corners[i][0][:, 1]))
                
                if id_val == 100:
                    cx_auto, cy_auto = cx, cy
                    
                    longitud_extension_t = 12 
                    radio_puntos = 4          

                    esquinas = corners[i][0]
                    front_x = (esquinas[0][0] + esquinas[1][0]) / 2.0
                    front_y = (esquinas[0][1] + esquinas[1][1]) / 2.0
                    back_x = (esquinas[2][0] + esquinas[3][0]) / 2.0
                    back_y = (esquinas[2][1] + esquinas[3][1]) / 2.0
                    
                    vec_fwd = np.array([front_x - back_x, front_y - back_y])
                    norm_fwd = max(1e-5, np.linalg.norm(vec_fwd))
                    vec_fwd_dir = vec_fwd / norm_fwd
                    
                    # --- EL SENSOR CENTRAL ES LA PUNTA DE LA LÍNEA ---
                    pt_sensor_central = (int(front_x + vec_fwd_dir[0] * longitud_extension_t), 
                                         int(front_y + vec_fwd_dir[1] * longitud_extension_t))
                    
                    cv2.line(frame, (int(back_x), int(back_y)), pt_sensor_central, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, pt_sensor_central, radio_puntos, (0, 0, 255), -1) 
                    
                elif 1 <= id_val <= 25 and not asociacion_calculada:
                    if M_est is not None:
                        pt_w_array = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), M_est)[0][0]
                        arucos_maze[id_val] = (int(pt_w_array[0]), int(pt_w_array[1]))

        # --- VARIABLES PARA ALMACENAR LOS TEXTOS DEL PANEL INFERIOR ---
        txt_fase = ""
        txt_tiempo = ""
        txt_instrucciones = ""
        txt_sensor_central = ""
        txt_sensor_lado = ""
        txt_completado = ""
        txt_movimiento = ""

        if tablero_cnt_est is not None:
            lienzo_ar_dinamico = np.zeros((tamano_warp, tamano_warp, 4), dtype=np.uint8)
            
            if mapeo_bloqueado:
                if tiempo_fase_asoc == 0: tiempo_fase_asoc = time.time()
                dt_asoc = time.time() - tiempo_fase_asoc
                
                if dt_asoc < 1.0:
                    txt_fase = "FASE 1: IDENTIFICANDO PUNTOS"
                    for nodo in nodos_clave: cv2.circle(lienzo_ar_dinamico, nodo['pt'], 6, (0, 0, 255, 255), -1) 
                    for pt_w in arucos_maze.values(): cv2.circle(lienzo_ar_dinamico, pt_w, 6, (255, 255, 0, 255), -1) 

                elif dt_asoc < 2.0:
                    txt_fase = "FASE 2: CALCULANDO VECTORES"
                    if not asociacion_calculada:
                        for nodo in nodos_clave:
                            mejor_dist, mejor_pt_w = float('inf'), None
                            for pt_w in arucos_maze.values():
                                dist = np.hypot(nodo['pt'][0] - pt_w[0], nodo['pt'][1] - pt_w[1])
                                if dist < mejor_dist: mejor_dist, mejor_pt_w = dist, pt_w
                            if mejor_pt_w is not None: nodo['pt_aruco_w'] = mejor_pt_w
                        asociacion_calculada = True
                    
                    for nodo in nodos_clave:
                        cv2.circle(lienzo_ar_dinamico, nodo['pt'], 6, (0, 0, 255, 255), -1)
                        if 'pt_aruco_w' in nodo:
                            cv2.line(lienzo_ar_dinamico, nodo['pt'], nodo['pt_aruco_w'], (0, 255, 255, 255), 2)
                    for pt_w in arucos_maze.values():
                        cv2.circle(lienzo_ar_dinamico, pt_w, 6, (255, 255, 0, 255), -1)

                elif dt_asoc < 3.0:
                    txt_fase = "FASE 3: SNAP COMPLETADO"
                    if not asociacion_hecha:
                        for nodo in nodos_clave:
                            if 'pt_aruco_w' in nodo: nodo['pt'] = nodo['pt_aruco_w']
                        
                        # --- MODIFICACIÓN: Vectores teóricos puros aplicados al nodo en Fase 3 ---
                        v_ent_x = puntos[1][0] - puntos[0][0]
                        v_ent_y = puntos[1][1] - puntos[0][1]
                        norm_ent = max(1e-5, np.hypot(v_ent_x, v_ent_y))
                        pt_guia_entrada = (int(nodos_clave[0]['pt'][0] - (v_ent_x / norm_ent) * longitud_guia), int(nodos_clave[0]['pt'][1] - (v_ent_y / norm_ent) * longitud_guia))
                        
                        v_sal_x = puntos[-1][0] - puntos[-2][0]
                        v_sal_y = puntos[-1][1] - puntos[-2][1]
                        norm_sal = max(1e-5, np.hypot(v_sal_x, v_sal_y))
                        pt_guia_salida = (int(nodos_clave[-1]['pt'][0] + (v_sal_x / norm_sal) * longitud_guia), int(nodos_clave[-1]['pt'][1] + (v_sal_y / norm_sal) * longitud_guia))
                        asociacion_hecha = True

                else:
                    if not reemplazo_hecho:
                        reemplazo_hecho = True
                        
                        puntos_ruta = [pt_guia_entrada] + [n['pt'] for n in nodos_clave] + [pt_guia_salida]
                        nodos_suavizados = []
                        
                        d_chaflan = 35  
                        
                        for i in range(len(nodos_clave)):
                            A = puntos_ruta[i]     
                            B = puntos_ruta[i+1]   
                            C = puntos_ruta[i+2]   
                            
                            if nodos_clave[i]['tipo'] in ['GIRO', 'GIRO_INICIO', 'FIN', 'INICIO_RECTO']:
                                v_in = np.array([A[0] - B[0], A[1] - B[1]], dtype=float)
                                v_out = np.array([C[0] - B[0], C[1] - B[1]], dtype=float)
                                n_in = np.linalg.norm(v_in)
                                n_out = np.linalg.norm(v_out)
                                
                                if n_in > 0 and n_out > 0:
                                    v_in /= n_in
                                    v_out /= n_out
                                    
                                    d_in = min(d_chaflan, n_in / 2.1)
                                    d_out = min(d_chaflan, n_out / 2.1)
                                    
                                    pt_in = (int(B[0] + v_in[0] * d_in), int(B[1] + v_in[1] * d_in))
                                    pt_out = (int(B[0] + v_out[0] * d_out), int(B[1] + v_out[1] * d_out))
                                    
                                    nodos_suavizados.append({'pt': pt_in, 'tipo': 'PRE_GIRO', 'cmd': 'ADELANTE'})
                                    nodos_suavizados.append({'pt': pt_out, 'tipo': 'POST_GIRO', 'cmd': 'ADELANTE'})
                                else:
                                    nodos_suavizados.append(nodos_clave[i])
                            else:
                                nodos_suavizados.append(nodos_clave[i])
                        
                        nodos_clave = nodos_suavizados

            if reemplazo_hecho:
                grosor_linea_negra = 10
                
                txt_tiempo = f"Tiempo: {tiempo_transcurrido:.2f} s"

                dibujar_entrada = False
                if indice_objetivo == 0:
                    dibujar_entrada = True
                elif indice_objetivo == 1 and nodos_clave[0]['tipo'] == 'INICIO_RECTO':
                    dibujar_entrada = True
                    
                if dibujar_entrada:
                    cv2.line(lienzo_ar_dinamico, pt_guia_entrada, nodos_clave[0]['pt'], (0, 0, 0, 255), grosor_linea_negra, cv2.LINE_AA)
                
                cv2.line(lienzo_ar_dinamico, nodos_clave[-1]['pt'], pt_guia_salida, (0, 0, 0, 255), grosor_linea_negra, cv2.LINE_AA)

                inicio_lineas = max(1, indice_objetivo)
                for i in range(inicio_lineas, len(nodos_clave)):
                    cv2.line(lienzo_ar_dinamico, nodos_clave[i-1]['pt'], nodos_clave[i]['pt'], (0, 0, 0, 255), grosor_linea_negra)

                v_meta = np.array([pt_guia_salida[0] - nodos_clave[-1]['pt'][0], pt_guia_salida[1] - nodos_clave[-1]['pt'][1]], dtype=float)
                norm_meta = np.linalg.norm(v_meta)
                if norm_meta > 0:
                    v_meta /= norm_meta
                    v_perp_meta = np.array([-v_meta[1], v_meta[0]])
                    largo_meta = 60 
                    pt_meta_1 = (int(pt_guia_salida[0] + v_perp_meta[0] * largo_meta), int(pt_guia_salida[1] + v_perp_meta[1] * largo_meta))
                    pt_meta_2 = (int(pt_guia_salida[0] - v_perp_meta[0] * largo_meta), int(pt_guia_salida[1] - v_perp_meta[1] * largo_meta))
                    cv2.line(lienzo_ar_dinamico, pt_meta_1, pt_meta_2, (0, 255, 0, 255), 8, cv2.LINE_AA) 

            if M_inv_est is not None:
                holograma_est = cv2.warpPerspective(lienzo_ar_dinamico, M_inv_est, (w_frame, h_frame))
                alpha = holograma_est[:, :, 3] / 255.0
                alpha_3d = np.dstack((alpha, alpha, alpha))
                frame = ((alpha_3d * holograma_est[:, :, :3].astype(float)) + ((1.0 - alpha_3d) * frame.astype(float))).astype(np.uint8)

            # -----------------------------------------------------------------
            # LÓGICA DE CONTROL Y UI FINAL
            # -----------------------------------------------------------------
            if reemplazo_hecho:
                if not navegacion_iniciada:
                    txt_instrucciones = "PRESIONA 'i' INICIAR | 'r' RESET"
                    comando_ros2 = "STOP"
                else:
                    if cx_auto is not None and cy_auto is not None and vec_fwd_dir is not None:
                        mid_x = cx_auto
                        mid_y = cy_auto

                        if pt_sensor_central is not None:
                            pt_central_w = cv2.perspectiveTransform(np.array([[[pt_sensor_central[0], pt_sensor_central[1]]]], dtype=np.float32), M_est)[0][0]
                            pt_auto_w = cv2.perspectiveTransform(np.array([[[cx_auto, cy_auto]]], dtype=np.float32), M_est)[0][0]
                            
                            puntos_ruta = [pt_guia_entrada] + [n['pt'] for n in nodos_clave] + [pt_guia_salida]

                            def dist_punto_segmento(p, a, b):
                                p_arr, a_arr, b_arr = np.array(p), np.array(a), np.array(b)
                                ab = b_arr - a_arr
                                ap = p_arr - a_arr
                                norm_ab = np.dot(ab, ab)
                                if norm_ab == 0: return np.linalg.norm(ap)
                                t = max(0.0, min(1.0, np.dot(ap, ab) / norm_ab))
                                proy = a_arr + t * ab
                                return np.linalg.norm(p_arr - proy), proy

                            dist_minima = float('inf')
                            punto_mas_cercano = None

                            for j in range(len(puntos_ruta)-1):
                                d, proy = dist_punto_segmento(pt_central_w, puntos_ruta[j], puntos_ruta[j+1])
                                if d < dist_minima:
                                    dist_minima = d
                                    punto_mas_cercano = proy

                            umbral_sensor = 10 
                            sensor_en_linea = dist_minima < umbral_sensor
                            
                            vec_sensor = np.array([pt_central_w[0] - pt_auto_w[0], pt_central_w[1] - pt_auto_w[1]])
                            
                            vec_ruta = np.array([punto_mas_cercano[0] - pt_auto_w[0], punto_mas_cercano[1] - pt_auto_w[1]])
                            
                            cross_product = (vec_sensor[0] * vec_ruta[1] - vec_sensor[1] * vec_ruta[0])

                            txt_sensor_central = f"Sensor: {'EN LINEA' if sensor_en_linea else 'FUERA'}"
                            txt_sensor_lado = f"Lado: {'-' if sensor_en_linea else ('IZQ' if cross_product < 0 else 'DER')}"

                        if not nodo_alcanzado:
                            if cx_auto is not None and cy_auto is not None:
                                if posicion_anterior_auto is None:
                                    posicion_anterior_auto = (cx_auto, cy_auto)
                                    tiempo_estatico = time.time()
                                else:
                                    dist_movimiento = np.hypot(cx_auto - posicion_anterior_auto[0], cy_auto - posicion_anterior_auto[1])
                                    if dist_movimiento > 10:  
                                        posicion_anterior_auto = (cx_auto, cy_auto)
                                        tiempo_estatico = time.time()
                            
                            if estado_retroceso:
                                comando_ros2 = "RETROCEDER"
                                if time.time() - tiempo_inicio_retroceso >= 0.5:
                                    estado_retroceso = False
                                    tiempo_estatico = time.time()  
                                else:
                                    bloquear_envio = True 

                            elif tiempo_transcurrido > 2.0 and time.time() - tiempo_estatico >= 2.0 and ros_node.comando_confirmado():
                                estado_retroceso = True
                                tiempo_inicio_retroceso = time.time()
                                comando_ros2 = "RETROCEDER"
                            else:
                                if sensor_en_linea:
                                    comando_ros2 = "ADELANTE"
                                else:
                                    dist_a_meta, _ = dist_punto_segmento(pt_auto_w, puntos_ruta[-2], puntos_ruta[-1])
                                    
                                    if dist_a_meta < umbral_sensor:
                                         comando_ros2 = "ADELANTE"
                                    else:
                                        if cross_product < 0:
                                            comando_ros2 = "CORREGIR_DERECHA"
                                        else:
                                            comando_ros2 = "CORREGIR_IZQUIERDA"
                                    
                            v_exit = np.array([puntos_ruta[-1][0] - puntos_ruta[-2][0], puntos_ruta[-1][1] - puntos_ruta[-2][1]], dtype=float)
                            norm_exit = np.linalg.norm(v_exit)
                            if norm_exit > 0:
                                v_exit_dir = v_exit / norm_exit
                                vec_auto_meta = np.array([pt_auto_w[0] - puntos_ruta[-1][0], pt_auto_w[1] - puntos_ruta[-1][1]], dtype=float)
                                
                                if np.dot(vec_auto_meta, v_exit_dir) > 0:
                                    comando_ros2 = "STOP"
                                    nodo_alcanzado = True
                                    cronometro_corriendo = False
                                    
                                    if not tiempo_guardado:
                                        ros_node.detener_recoleccion()
                                        
                                        if video_writer:
                                            video_writer.release()
                                            video_writer = None
                                            
                                        raw_path = os.path.join(algo_path, f"Metricas_Raw_{algo_folder_name}.txt")
                                        with open(raw_path, "w") as f_raw:
                                            f_raw.write("Latencia(s)\tJitter(s)\tCPU(%)\tMemoria(MB)\tEficiencia(%)\n")
                                            for l, j, c, m, e in zip(ros_node.latencias, ros_node.jitters, ros_node.cpus, ros_node.memorias, ros_node.energias):
                                                f_raw.write(f"{l:.4f}\t{j:.4f}\t{c:.2f}\t{m:.2f}\t{e:.2f}\n")
                                        
                                        avg_lat = np.mean(ros_node.latencias) if ros_node.latencias else 0
                                        avg_jit = np.mean(ros_node.jitters) if ros_node.jitters else 0
                                        avg_cpu = np.mean(ros_node.cpus) if ros_node.cpus else 0
                                        avg_mem = np.mean(ros_node.memorias) if ros_node.memorias else 0
                                        avg_ene = np.mean(ros_node.energias) if ros_node.energias else 0
                                        
                                        main_txt_path = os.path.join(algo_path, f"Datos_{algo_folder_name}.txt")
                                        with open(main_txt_path, "a") as f:
                                            f.write(f"\n--- RENDIMIENTO DE ROS 2 ---\n")
                                            f.write(f"Latencia Promedio: {avg_lat:.4f} s\n")
                                            f.write(f"Jitter Promedio: {avg_jit:.4f} s\n")
                                            f.write(f"CPU Promedio: {avg_cpu:.2f} %\n")
                                            f.write(f"Memoria Promedio: {avg_mem:.2f} MB\n")
                                            f.write(f"Eficiencia Energetica Estimada: {avg_ene:.2f} %\n")
                                            f.write(f"\nTiempo Total de Resolucion: {tiempo_transcurrido:.2f} s\n")
                                            
                                        tiempo_guardado = True
                            
                        else:
                            comando_ros2 = "STOP"
                            txt_completado = "LABERINTO COMPLETADO"
                    
                    else:
                        comando_ros2 = "IGNORAR"
                            
        if comando_ros2 == "IGNORAR":
            color_cmd = (0, 165, 255) 
            txt_movimiento = f"MOVIMIENTO AUTO: ARUCO PERDIDO"
        else:
            color_cmd = (0, 255, 0) if comando_ros2 == "ADELANTE" else ((0, 255, 255) if "CORREGIR" in comando_ros2 else (0, 0, 255))
            txt_movimiento = f"MOVIMIENTO AUTO: {comando_ros2}"

        if comando_ros2 != "IGNORAR":
            if not bloquear_envio:
                if comando_ros2 != ultimo_comando_enviado:
                    ros_node.publicar_comando(comando_ros2)
                    ultimo_comando_enviado = comando_ros2

        rclpy.spin_once(ros_node, timeout_sec=0.001)
        
        frame_ar_resized = cv2.resize(frame, (700, 540))
        vista_doble = np.zeros((540, 1120, 3), dtype=np.uint8)
        vista_doble[0:540, 0:700] = frame_ar_resized
        cv2.line(vista_doble, (700, 0), (700, 540), (100, 100, 100), 2)
        
        x_base = 720
        cv2.putText(vista_doble, "--- INFO ---", (x_base, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        if txt_fase: cv2.putText(vista_doble, txt_fase, (x_base, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if txt_instrucciones: cv2.putText(vista_doble, txt_instrucciones, (x_base, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif txt_tiempo: cv2.putText(vista_doble, txt_tiempo, (x_base, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if txt_sensor_central: cv2.putText(vista_doble, txt_sensor_central, (x_base, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if txt_sensor_lado: cv2.putText(vista_doble, txt_sensor_lado, (x_base, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
        cv2.putText(vista_doble, txt_movimiento, (x_base, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color_cmd, 2)
        if txt_completado: cv2.putText(vista_doble, txt_completado, (x_base, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if video_writer:
            video_writer.write(vista_doble)

        cv2.imshow("VALIDACION VECTOR (q: Salir)", vista_doble)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'): 
            ros_node.publicar_comando("STOP")
            if video_writer: video_writer.release()
            break
        elif tecla == ord('i') and reemplazo_hecho: 
            if not navegacion_iniciada:
                navegacion_iniciada, tiempo_inicio_recorrido, cronometro_corriendo, nodo_alcanzado, tiempo_guardado = True, time.time(), True, False, False
                posicion_anterior_auto, tiempo_estatico, estado_retroceso = None, time.time(), False
                ros_node.iniciar_recoleccion()
                ultimo_comando_enviado = None 
                video_count += 1
                video_filename = os.path.join(algo_path, f"Video_{algo_folder_name}_{video_count}.mp4")
                video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 7.6, (1120, 540))
        elif tecla == ord('r'):
            navegacion_iniciada, cronometro_corriendo, tiempo_transcurrido, nodo_alcanzado, tiempo_guardado = False, False, 0.0, False, False
            ros_node.publicar_comando("STOP")
            ros_node.iniciar_recoleccion()
            posicion_anterior_auto, tiempo_estatico, estado_retroceso = None, time.time(), False
            ultimo_comando_enviado = None 
            if video_writer: video_writer.release(); video_writer = None
            with open(os.path.join(algo_path, f"Datos_{algo_folder_name}.txt"), "w") as f:
                f.write(f"Algoritmo: {nombre_algoritmo}\n")
                f.write(f"Nodos para resolver (Pasos fisicos): {datos_resultado['pasos']}\n")
                f.write(f"Tiempo CPU: {datos_resultado['t']:.2f} ms\n")
                f.write(f"Nodos explorados (Revisados): {datos_resultado['n']}\n")
            
            raw_path = os.path.join(algo_path, f"Metricas_Raw_{algo_folder_name}.txt")
            if os.path.exists(raw_path): os.remove(raw_path)
            
    cap.release()
    cv2.destroyWindow("VALIDACION VECTOR (q: Salir)")
