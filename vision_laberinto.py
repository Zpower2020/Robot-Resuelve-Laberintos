import cv2
import numpy as np
from PIL import Image, ImageDraw

def recorte_inteligente_por_contornos(captura_limpia):
    img_inv = cv2.bitwise_not(captura_limpia)
    cnts, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_x, min_y, max_x, max_y, hay_contornos = float('inf'), float('inf'), 0, 0, False

    for c in cnts:
        if cv2.contourArea(c) > 100:
            hay_contornos = True
            x, y, w, h = cv2.boundingRect(c)
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if x + w > max_x: max_x = x + w
            if y + h > max_y: max_y = y + h

    if not hay_contornos: return captura_limpia
    margen = 5 
    h_img, w_img = captura_limpia.shape
    y_start, y_end = max(0, min_y - margen), min(h_img, max_y + margen)
    x_start, x_end = max(0, min_x - margen), min(w_img, max_x + margen)
    return captura_limpia[y_start:y_end, x_start:x_end]

def paso1_capturar_camara(ruta_salida="12.png"):
    MARGEN_CORTE = 7 
    cap = cv2.VideoCapture(2) 
    if not cap.isOpened(): return False

    print("\n" + "="*50)
    print(" PASO 1: CAPTURA DE CÁMARA")
    print("Apunta al laberinto. Presiona 's' para capturar o 'q' para salir.")
    tamano_warp = 800
    exito = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        tablero_cnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 10000:
                tablero_cnt = approx
                cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
                break

        cv2.imshow("Camara - 's' capturar", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and tablero_cnt is not None:
            pts = tablero_cnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
            rect[0], rect[2], rect[1], rect[3] = pts[np.argmin(s)], pts[np.argmax(s)], pts[np.argmin(diff)], pts[np.argmax(diff)]
            dst = np.array([[0, 0], [tamano_warp - 1, 0], [tamano_warp - 1, tamano_warp - 1], [0, tamano_warp - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            captura_enderezada = cv2.warpPerspective(frame, M, (tamano_warp, tamano_warp))
            captura_recortada = captura_enderezada[MARGEN_CORTE:tamano_warp-MARGEN_CORTE, MARGEN_CORTE:tamano_warp-MARGEN_CORTE]
            blur_recorte = cv2.GaussianBlur(cv2.cvtColor(captura_recortada, cv2.COLOR_BGR2GRAY), (9, 9), 0)
            captura_limpia = cv2.adaptiveThreshold(blur_recorte, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)
            captura_final_ajustada = recorte_inteligente_por_contornos(captura_limpia)
            cv2.imwrite(ruta_salida, captura_final_ajustada)
            cv2.destroyWindow("Camara - 's' capturar")
            cv2.imshow("Paso 1: Recorte", captura_final_ajustada)
            cv2.waitKey(0); cv2.destroyAllWindows()
            exito = True; break
        elif key == ord('q'): break
    cap.release(); cv2.destroyAllWindows()
    return exito

def paso2_reconstruccion(ruta_ent, ruta_sal="laberinto_perfecto.png"):
    print("\n" + "="*50)
    print(" PASO 2: RECONSTRUCCIÓN GEOMÉTRICA")
    print("="*50)
    
    img_original = cv2.imread(ruta_ent)
    if img_original is None: return False

    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    muros = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    
    h_img, w_img = muros.shape
    lienzo = np.zeros_like(muros)

    tolerancia_borde = 50
    kernel_len = 50

    for mask in [cv2.morphologyEx(muros, cv2.MORPH_OPEN, np.ones((1, kernel_len), np.uint8)), cv2.morphologyEx(muros, cv2.MORPH_OPEN, np.ones((kernel_len, 1), np.uint8))]:
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, w, h = cv2.boundingRect(c)
            if x < tolerancia_borde: w += x; x = 0
            if y < tolerancia_borde: h += y; y = 0
            if x + w > w_img - tolerancia_borde: w = w_img - x
            if y + h > h_img - tolerancia_borde: h = h_img - y
            cv2.rectangle(lienzo, (x, y), (x + w, y + h), 255, -1)
    cv2.imwrite(ruta_sal, cv2.bitwise_not(lienzo))
    cv2.imshow("Paso 2: Geometria", cv2.resize(cv2.bitwise_not(lienzo), (500, 500)))
    cv2.waitKey(0); cv2.destroyAllWindows()
    return True

def paso3_dibujar_puntos(ruta, p_ent, p_sal, r_sal="laberinto_puntos_completo.png", cells_x=5, cells_y=5):
    print("\n" + "="*50)
    print(" PASO 3: EXTRACCIÓN DE GRAFOS")
    print("="*50)
    
    img = Image.open(ruta).convert('RGB')
    d, px, py = ImageDraw.Draw(img), img.size[0]/cells_x, img.size[1]/cells_y
    rp = max(2, int(min(px, py)*0.05))
    
    for r in range(cells_y):
        for c in range(cells_x): d.ellipse([(int((c+0.5)*px)-rp, int((r+0.5)*py)-rp), (int((c+0.5)*px)+rp, int((r+0.5)*py)+rp)], fill=(255,255,0))
        
    def cp(p):
        if p[0]=='N': return (int((p[1]+0.5)*px), 0)
        if p[0]=='S': return (int((p[1]+0.5)*px), img.size[1]-1)
        if p[0]=='O': return (0, int((p[1]+0.5)*py))
        if p[0]=='E': return (img.size[0]-1, int((p[1]+0.5)*py))
        
    xe, ye = cp(p_ent); d.ellipse([(xe-rp*2, ye-rp*2), (xe+rp*2, ye+rp*2)], fill=(255,0,0))
    xs, ys = cp(p_sal); d.ellipse([(xs-rp*2, ys-rp*2), (xs+rp*2, ys+rp*2)], fill=(0,255,0))
    
    img.save(r_sal)
    cv2.imshow("Paso 3: Grafos", cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (500,500)))
    cv2.waitKey(0); cv2.destroyAllWindows()