# Importación de librerías para visión por computadora y manipulación de matrices
import cv2                  # OpenCV: Librería principal para leer la cámara, detectar contornos y aplicar filtros.
import numpy as np          # NumPy: Para crear matrices vacías y hacer cálculos matemáticos de las coordenadas.
from PIL import Image, ImageDraw # Pillow: Para dibujar los círculos de colores de los nodos al final.

# =====================================================================
# FUNCIÓN: RECORTE INTELIGENTE (Eliminar bordes blancos inútiles)
# =====================================================================
def recorte_inteligente_por_contornos(captura_limpia):
    # Invierte los colores (blanco a negro y viceversa). OpenCV detecta mejor contornos blancos sobre fondo negro.
    img_inv = cv2.bitwise_not(captura_limpia)
    
    # Busca todos los contornos (bordes de las paredes del laberinto) en la imagen invertida.
    cnts, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Inicializa las variables para buscar la "caja delimitadora" (bounding box) más grande que encierre todo el laberinto.
    min_x, min_y, max_x, max_y, hay_contornos = float('inf'), float('inf'), 0, 0, False

    for c in cnts: # Revisa cada contorno encontrado...
        if cv2.contourArea(c) > 100: # Si el contorno es más grande que 100 píxeles (ignora el ruido/basurita):
            hay_contornos = True
            # Obtiene las coordenadas (x, y) y el tamaño (ancho, alto) del cuadrado que encierra este contorno.
            x, y, w, h = cv2.boundingRect(c)
            # Actualiza los límites extremos para ir armando una caja gigante que encierre TODOS los contornos.
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if x + w > max_x: max_x = x + w
            if y + h > max_y: max_y = y + h

    # Si la hoja estaba en blanco y no hay nada, devuelve la imagen tal cual.
    if not hay_contornos: return captura_limpia
    
    margen = 5 # Deja un pequeño margen de 5 píxeles de aire alrededor del recorte.
    h_img, w_img = captura_limpia.shape # Obtiene las medidas originales de la imagen.
    
    # Calcula desde dónde hasta dónde recortar, asegurándose de no salirse de los bordes reales de la imagen (usando min y max).
    y_start, y_end = max(0, min_y - margen), min(h_img, max_y + margen)
    x_start, x_end = max(0, min_x - margen), min(w_img, max_x + margen)
    
    # Retorna la matriz de la imagen recortada exactamente al tamaño del laberinto.
    return captura_limpia[y_start:y_end, x_start:x_end]

# =====================================================================
# PASO 1: CAPTURAR, ENDEREZAR Y LIMPIAR LA FOTO
# =====================================================================
def paso1_capturar_camara(ruta_salida="12.png"):
    MARGEN_CORTE = 7 
    cap = cv2.VideoCapture(2) # Abre la cámara externa (índice 2).
    if not cap.isOpened(): return False # Si no detecta cámara, aborta.

    print("\n" + "="*50)
    print(" PASO 1: CAPTURA DE CÁMARA")
    print("Apunta al laberinto. Presiona 's' para capturar o 'q' para salir.")
    tamano_warp = 800 # Tamaño estándar final al que forzaremos el laberinto.
    exito = False

    while True: # Bucle de video en tiempo real.
        ret, frame = cap.read() # Lee un fotograma de la cámara.
        if not ret: break
        output = frame.copy() # Copia para mostrar en pantalla sin rayar la original.
        
        # Preprocesamiento para encontrar el papel/tablero:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Pasa a blanco y negro.
        blur = cv2.GaussianBlur(gray, (7, 7), 0)       # Difumina para eliminar ruido pequeño.
        edges = cv2.Canny(blur, 50, 150)               # Detecta los bordes afilados (líneas).
        edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1) # Engrosa los bordes para que no se rompan.
        
        # Busca los contornos de la imagen procesada y se queda con los 5 más grandes.
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        tablero_cnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True) # Calcula el perímetro del contorno.
            approx = cv2.approxPolyDP(c, 0.02 * peri, True) # Intenta simplificar el contorno a líneas rectas.
            # Si tiene exactamente 4 lados (es un rectángulo/papel) y es muy grande:
            if len(approx) == 4 and cv2.contourArea(c) > 10000:
                tablero_cnt = approx # Encontramos el tablero del laberinto.
                cv2.drawContours(output, [approx], -1, (0, 255, 0), 3) # Lo dibuja en verde en la pantalla.
                break

        cv2.imshow("Camara - 's' capturar", output)
        key = cv2.waitKey(1) & 0xFF
        
        # Si presionas 's' y sí encontró un rectángulo verde:
        if key == ord('s') and tablero_cnt is not None:
            # Reorganiza los 4 puntos del tablero para saber cuál es Arriba-Izq, Arriba-Der, Abajo-Izq, Abajo-Der.
            pts = tablero_cnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
            rect[0], rect[2], rect[1], rect[3] = pts[np.argmin(s)], pts[np.argmax(s)], pts[np.argmin(diff)], pts[np.argmax(diff)]
            
            # Define las coordenadas de un cuadrado perfecto de 800x800 píxeles.
            dst = np.array([[0, 0], [tamano_warp - 1, 0], [tamano_warp - 1, tamano_warp - 1], [0, tamano_warp - 1]], dtype="float32")
            
            # MAGIA: Calcula la matriz matemática para deformar la foto sesgada en un cuadrado plano y perfecto (Vista de Pájaro).
            M = cv2.getPerspectiveTransform(rect, dst)
            captura_enderezada = cv2.warpPerspective(frame, M, (tamano_warp, tamano_warp))
            
            # Recorta un poco los bordes para eliminar la cinta adhesiva o bordes del papel.
            captura_recortada = captura_enderezada[MARGEN_CORTE:tamano_warp-MARGEN_CORTE, MARGEN_CORTE:tamano_warp-MARGEN_CORTE]
            
            # Binarización Robusta: Convierte la imagen en blanco y negro PURO sin grises, eliminando sombras.
            blur_recorte = cv2.GaussianBlur(cv2.cvtColor(captura_recortada, cv2.COLOR_BGR2GRAY), (9, 9), 0)
            captura_limpia = cv2.adaptiveThreshold(blur_recorte, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)
            
            # Llama a la primera función para cortarle el exceso de blanco alrededor de las paredes reales.
            captura_final_ajustada = recorte_inteligente_por_contornos(captura_limpia)
            
            # Guarda y muestra el resultado de esta etapa.
            cv2.imwrite(ruta_salida, captura_final_ajustada)
            cv2.destroyWindow("Camara - 's' capturar")
            cv2.imshow("Paso 1: Recorte", captura_final_ajustada)
            cv2.waitKey(0); cv2.destroyAllWindows()
            exito = True; break
        elif key == ord('q'): break # Si presionas 'q', aborta.
    cap.release(); cv2.destroyAllWindows()
    return exito

# =====================================================================
# PASO 2: RECONSTRUCCIÓN (Reparar paredes rotas de la foto)
# =====================================================================
def paso2_reconstruccion(ruta_ent, ruta_sal="laberinto_perfecto.png"):
    print("\n" + "="*50)
    print(" PASO 2: RECONSTRUCCIÓN GEOMÉTRICA")
    print("="*50)
    
    img_original = cv2.imread(ruta_ent)
    if img_original is None: return False

    # Preparación de la imagen a blanco y negro invertido (paredes blancas, fondo negro).
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    muros = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    
    h_img, w_img = muros.shape
    lienzo = np.zeros_like(muros) # Crea un "lienzo" totalmente negro del mismo tamaño.

    tolerancia_borde = 50 # Píxeles de distancia para estirar una pared hasta chocar con el borde de la imagen.
    kernel_len = 50       # Tamaño de la regla para buscar líneas rectas (mínimo 50 píxeles de largo).

    # Este bucle ejecuta morfología matemática dos veces: primero busca líneas horizontales puras, luego líneas verticales puras.
    for mask in [cv2.morphologyEx(muros, cv2.MORPH_OPEN, np.ones((1, kernel_len), np.uint8)), cv2.morphologyEx(muros, cv2.MORPH_OPEN, np.ones((kernel_len, 1), np.uint8))]:
        # Extrae los contornos de las líneas rectas perfectas encontradas.
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, w, h = cv2.boundingRect(c) # Obtiene las coordenadas y tamaño de esa pared.
            
            # ESTIRAMIENTO DE BORDES: Si una pared está muy cerca del borde (menor a tolerancia_borde), 
            # la "estira" matemáticamente hasta tocar el borde. Así evita que el laberinto tenga "fugas" en las esquinas.
            if x < tolerancia_borde: w += x; x = 0
            if y < tolerancia_borde: h += y; y = 0
            if x + w > w_img - tolerancia_borde: w = w_img - x
            if y + h > h_img - tolerancia_borde: h = h_img - y
            
            # Dibuja esa pared perfecta y recta de color blanco sobre nuestro lienzo negro vacío.
            cv2.rectangle(lienzo, (x, y), (x + w, y + h), 255, -1)
            
    # Guarda la imagen reconstruida (invirtiendo el color para que el fondo sea blanco y las paredes negras).
    cv2.imwrite(ruta_sal, cv2.bitwise_not(lienzo))
    cv2.imshow("Paso 2: Geometria", cv2.resize(cv2.bitwise_not(lienzo), (500, 500)))
    cv2.waitKey(0); cv2.destroyAllWindows()
    return True

# =====================================================================
# PASO 3: DIBUJAR PUNTOS DE REFERENCIA DEL GRAFO
# =====================================================================
def paso3_dibujar_puntos(ruta, p_ent, p_sal, r_sal="laberinto_puntos_completo.png", cells_x=5, cells_y=5):
    print("\n" + "="*50)
    print(" PASO 3: EXTRACCIÓN DE GRAFOS")
    print("="*50)
    
    img = Image.open(ruta).convert('RGB') # Carga la imagen reconstruida matemáticamente.
    d, px, py = ImageDraw.Draw(img), img.size[0]/cells_x, img.size[1]/cells_y
    rp = max(2, int(min(px, py)*0.05)) # Calcula el radio del punto basado en el tamaño de la celda.
    
    # Recorre las 25 celdas y dibuja un punto AMARILLO en el centro exacto de cada una.
    for r in range(cells_y):
        for c in range(cells_x): d.ellipse([(int((c+0.5)*px)-rp, int((r+0.5)*py)-rp), (int((c+0.5)*px)+rp, int((r+0.5)*py)+rp)], fill=(255,255,0))
        
    # Subfunción que traduce la puerta (ej. Norte, columna 2) a su posición X,Y exacta en los bordes.
    def cp(p):
        if p[0]=='N': return (int((p[1]+0.5)*px), 0)
        if p[0]=='S': return (int((p[1]+0.5)*px), img.size[1]-1)
        if p[0]=='O': return (0, int((p[1]+0.5)*py))
        if p[0]=='E': return (img.size[0]-1, int((p[1]+0.5)*py))
        
    # Dibuja la Entrada en ROJO y la Salida en VERDE (más grandes que los amarillos, rp*2).
    xe, ye = cp(p_ent); d.ellipse([(xe-rp*2, ye-rp*2), (xe+rp*2, ye+rp*2)], fill=(255,0,0))
    xs, ys = cp(p_sal); d.ellipse([(xs-rp*2, ys-rp*2), (xs+rp*2, ys+rp*2)], fill=(0,255,0))
    
    # Guarda y muestra el resultado.
    img.save(r_sal)
    cv2.imshow("Paso 3: Grafos", cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (500,500)))
    cv2.waitKey(0); cv2.destroyAllWindows()
