# Importación de librerías necesarias para el procesamiento de la imagen y los algoritmos
import cv2                  # Librería OpenCV para procesamiento de imágenes matricial.
import numpy as np          # NumPy para operaciones matemáticas y manejo de matrices (arrays).
import time                 # Librería para medir el tiempo de ejecución (rendimiento).
import heapq                # Módulo para crear "Colas de Prioridad" (esencial para A* y Dijkstra).
import itertools            # Módulo para crear contadores únicos (desempate en las colas de prioridad).
from collections import deque # Módulo para crear colas doblemente terminadas (esencial para BFS).
from PIL import Image, ImageDraw # Pillow para dibujar gráficos (rutas y nodos) con transparencia sobre la imagen.

# =====================================================================
# FUNCIÓN 1: DETECCIÓN DE ENTRADA Y SALIDA
# =====================================================================
def detectar_aberturas_360(ruta, cells_x=5, cells_y=5):
    # Abre la imagen en blanco y negro y la convierte a formato RGB para poder leer sus píxeles.
    img = Image.open(ruta).convert('RGB')
    
    # Calcula cuántos píxeles mide de ancho (px) y alto (py) cada "celda" del laberinto virtual.
    # 'off' (offset) es un margen de 5 píxeles hacia adentro para no leer justo el borde cortado de la imagen.
    px, py, off = img.size[0] / cells_x, img.size[1] / cells_y, 5
    
    ab = [] # Lista vacía donde guardaremos las aberturas (puertas) que encontremos.
    
    # Escaneo de los bordes Norte (Arriba) y Sur (Abajo)
    for c in range(cells_x): # Itera sobre cada columna
        # Obtiene el color del píxel en el centro de la celda 'c', en el borde superior (off).
        # Si el valor rojo [0] es mayor a 128 (es decir, es un píxel blanco/claro, no hay pared negra):
        if img.getpixel((int((c+0.5)*px), off))[0] > 128: 
            ab.append(('N', c, int((c+0.5)*px), 0)) # Guarda la abertura indicando que está en el Norte ('N').
            
        # Hace lo mismo pero en el borde inferior (img.size[1]-1-off).
        if img.getpixel((int((c+0.5)*px), img.size[1]-1-off))[0] > 128: 
            ab.append(('S', c, int((c+0.5)*px), img.size[1]-1)) # Guarda la abertura indicando que está en el Sur ('S').
            
    # Escaneo de los bordes Oeste (Izquierda) y Este (Derecha)
    for r in range(cells_y): # Itera sobre cada fila
        # Escanea el borde izquierdo.
        if img.getpixel((off, int((r+0.5)*py)))[0] > 128: 
            ab.append(('O', r, 0, int((r+0.5)*py))) # Guarda la abertura indicando que está en el Oeste ('O').
            
        # Escanea el borde derecho.
        if img.getpixel((img.size[0]-1-off, int((r+0.5)*py)))[0] > 128: 
            ab.append(('E', r, img.size[0]-1, int((r+0.5)*py))) # Guarda la abertura indicando que está en el Este ('E').
    
    # Si encontró al menos 2 aberturas en el perímetro:
    if len(ab) >= 2:
        # Las ordena basándose en sus coordenadas 'x' y 'y' para mantener un orden lógico (arriba-abajo, izq-der).
        ab.sort(key=lambda i: (i[2], i[3]))
        # Retorna la primera abertura como Entrada y la segunda como Salida.
        return ab[0], ab[1]
    
    # Si el laberinto está cerrado y no encuentra puertas, retorna un error en texto.
    return "Error", "Error"

# =====================================================================
# FUNCIÓN 2: TRADUCTOR DE RUTA A INSTRUCCIONES DE CONDUCCIÓN
# =====================================================================
def generar_comandos_auto(nodos, puerta_ent):
    # Esta función recibe la ruta ganadora (ej. [START, (0,0), (0,1), END]) y la traduce a comandos (Adelante, Derecha).
    comandos = []
    lado = puerta_ent[0] # Obtiene por qué punto cardinal entró el robot (N, S, E, O).
    
    # Define la dirección "vectorial" inicial hacia donde mira la trompa del auto al entrar.
    if lado == 'N': current_dir = (0, 1)   # Si entra por arriba, mira hacia abajo (positivo en Y).
    elif lado == 'S': current_dir = (0, -1) # Si entra por abajo, mira hacia arriba (negativo en Y).
    elif lado == 'O': current_dir = (1, 0)  # Si entra por la izquierda, mira hacia la derecha (positivo en X).
    else: current_dir = (-1, 0)             # Si entra por la derecha, mira hacia la izquierda (negativo en X).

    # Bucle que analiza cada paso de la ruta con el paso inmediatamente siguiente.
    for i in range(len(nodos) - 1):
        curr, nxt = nodos[i], nodos[i+1] # 'curr' es el nodo actual, 'nxt' es el siguiente paso.
        
        # Ignoramos matemáticamente los nodos virtuales START y END, ya que no son coordenadas matemáticas.
        if curr == 'START' or nxt == 'START' or curr == 'END' or nxt == 'END': continue
        
        # Calcula el vector de movimiento matemático restando el nodo siguiente menos el actual.
        move = (nxt[0] - curr[0], nxt[1] - curr[1])
        
        # Lógica Vectorial de Giro:
        if move == current_dir:
            # Si el vector de movimiento es igual a hacia donde mira el auto, solo avanza.
            comandos.append("Adelante")
        elif (current_dir[1], -current_dir[0]) == move:
            # Si aplicamos una rotación matemática de 90° a la derecha y coincide con el movimiento, debe girar y luego avanzar.
            comandos.extend(["Derecha", "Adelante"])
            current_dir = move # Actualizamos hacia dónde mira el auto ahora.
        elif (-current_dir[1], current_dir[0]) == move:
            # Si aplicamos una rotación matemática de 90° a la izquierda y coincide, gira y avanza.
            comandos.extend(["Izquierda", "Adelante"])
            current_dir = move # Actualizamos hacia dónde mira el auto ahora.

    # Compresión de comandos consecutivos:
    # Transforma una lista como ["Adelante", "Adelante", "Derecha"] en algo legible como "Adelante x2, Derecha".
    comp, adv = [], 0 # 'comp' es la lista comprimida, 'adv' cuenta cuántos "Adelante" seguidos van.
    for cmd in comandos:
        if cmd == "Adelante": 
            adv += 1 # Si la orden es avanzar, suma 1 al contador.
        else:
            # Si hay una orden de giro, primero anota cuántos adelantes se hicieron antes del giro.
            if adv > 0: comp.append(f"Adelante x{adv}"); adv = 0
            # Anota la orden de giro.
            comp.append(cmd)
            
    # Si terminamos el laberinto con movimientos de avanzar, los anota al final.
    if adv > 0: comp.append(f"Adelante x{adv}")
    
    # Convierte la lista comprimida en un solo texto separado por " -> " para la interfaz.
    return " -> ".join(comp), comandos

# =====================================================================
# FUNCIÓN 3: EL CEREBRO DE RESOLUCIÓN (GRAFOS Y BÚSQUEDA)
# =====================================================================
def paso4_resolver_laberinto(ruta_img, puerta_ent, puerta_sal, cells_x=5, cells_y=5):
    # Carga la imagen optimizada (blanco y negro puro) en modo escala de grises ('L').
    img = Image.open(ruta_img).convert('L')
    px, py = img.size[0] / cells_x, img.size[1] / cells_y # Calcula tamaño de celda en píxeles.
    
    # Subfunción que convierte una coordenada de celda abstracta (ej. c=1, r=1) a su píxel exacto en el lienzo.
    def obtener_coordenada(c, r): return int((c + 0.5) * px), int((r + 0.5) * py)
    
    # Subfunción vital: Verifica si hay un muro negro bloqueando el paso entre el punto 1 y el punto 2.
    def hay_pared(x1, y1, x2, y2):
        for i in range(16): # Traza una línea imaginaria entre los dos puntos y la divide en 16 "muestras".
            # Si alguna de esas 16 muestras cae sobre un píxel muy oscuro (<128), asume que chocó con una pared.
            if img.getpixel((int(x1 + (x2 - x1) * i / 16.0), int(y1 + (y2 - y1) * i / 16.0))) < 128: 
                return True
        # Si revisó las 16 muestras y todas son blancas, no hay pared bloqueando.
        return False

    grafo = {} # Diccionario que guardará el mapa navegable (El Grafo Matemático).
    
    # 1. GENERACIÓN DEL GRAFO: Mapeo de caminos válidos.
    for r in range(cells_y):
        for c in range(cells_x):
            nodo = (c, r) # Identificador del nodo actual.
            grafo[nodo] = [] # Inicializa la lista de caminos válidos para este nodo.
            x1, y1 = obtener_coordenada(c, r) # Coordenada en píxeles de la celda actual.
            
            # Revisa a sus vecinos (Arriba, Abajo, Izquierda, Derecha).
            for dc, dr in [(1,0), (0,1), (-1,0), (0,-1)]:
                nc, nr = c + dc, r + dr # Calcula la celda vecina.
                # Si el vecino no se sale de los límites del mapa (ej. no es -1 o > 5)...
                if 0 <= nc < cells_x and 0 <= nr < cells_y:
                    x2, y2 = obtener_coordenada(nc, nr) # Calcula la coordenada del vecino.
                    # Si NO hay una pared bloqueando el paso entre mi celda y el vecino:
                    if not hay_pared(x1, y1, x2, y2):
                        # Lo agrego a mi lista de caminos permitidos.
                        grafo[nodo].append((nc, nr))

    # 2. CONEXIÓN DE LOS NODOS DE INICIO Y FIN (START/END)
    # Crea un nodo "START" virtual. Se conecta a la celda del laberinto que corresponda a la puerta de entrada.
    grafo['START'] = [(puerta_ent[1], 0) if puerta_ent[0] == 'N' else (puerta_ent[1], cells_y-1) if puerta_ent[0] == 'S' else (0, puerta_ent[1]) if puerta_ent[0] == 'O' else (cells_x-1, puerta_ent[1])]
    
    # La meta ("END") no tiene salidas, por lo que su lista de caminos válidos está vacía.
    grafo['END'] = []
    
    # Agrega el nodo "END" virtual como vecino de la celda del laberinto por donde debemos salir.
    nodo_fin_real = (puerta_sal[1], 0) if puerta_sal[0] == 'N' else (puerta_sal[1], cells_y-1) if puerta_sal[0] == 'S' else (0, puerta_sal[1]) if puerta_sal[0] == 'O' else (cells_x-1, puerta_sal[1])
    if nodo_fin_real in grafo: grafo[nodo_fin_real].append('END')

    # 3. DIBUJADO DE LAS EXPLORACIONES (Visualización de Nodos)
    def trazar_ruta_con_exploracion(ruta, visitados, color_explorado):
        # Convierte la imagen a formato RGBA (con transparencia/canal Alpha).
        img_color = Image.open(ruta_img).convert("RGBA")
        capa_overlay = Image.new("RGBA", img_color.size, (0,0,0,0)) # Crea una capa transparente del mismo tamaño.
        d = ImageDraw.Draw(capa_overlay) # Herramienta para dibujar en la capa.
        rp = max(2, int(min(px, py) * 0.15)) # Radio de los círculos (nodos) a dibujar.
        
        # Dibuja un círculo semitransparente por CADA nodo que el algoritmo revisó (pensando caminos).
        for nodo in visitados:
            if nodo not in ['START', 'END']:
                cx, cy = obtener_coordenada(*nodo)
                d.ellipse([(cx-rp, cy-rp), (cx+rp, cy+rp)], fill=color_explorado + (150,))
                
        # Dibuja la línea gruesa de la ruta ganadora final.
        puntos = []
        for n in ruta:
            if n == 'START': puntos.append((puerta_ent[2], puerta_ent[3]))
            elif n == 'END': puntos.append((puerta_sal[2], puerta_sal[3]))
            else: puntos.append(obtener_coordenada(*n))
        d.line(puntos, fill=color_explorado + (255,), width=max(3, int(min(px, py)*0.2)), joint="curve")
        
        # Mezcla la imagen original con la capa de dibujos y la convierte a matriz OpenCV (numpy).
        return cv2.cvtColor(np.array(Image.alpha_composite(img_color, capa_overlay).convert("RGB")), cv2.COLOR_RGB2BGR)

    res = {} # Diccionario donde guardaremos los resultados de los 4 algoritmos para compararlos.

    # -----------------------------------------------------------------
    # ALGORITMO 1: A* (A-Star) - Heurístico e Inteligente
    # -----------------------------------------------------------------
    ts = time.perf_counter() # Inicia el cronómetro de alto rendimiento.
    # Función Heurística (h): Adivina la distancia matemática en línea recta (Distancia Manhattan) hacia el nodo final.
    h = lambda n: 0 if n in ['START', 'END'] else abs(n[0] - nodo_fin_real[0]) + abs(n[1] - nodo_fin_real[1])
    
    op = [] # Open Set: La lista priorizada de nodos pendientes por evaluar.
    cnt = itertools.count() # Contador para evitar empates de prioridad si dos nodos cuestan lo mismo.
    
    # Inserta el primer nodo ('START'). Formato: (costo f, orden de llegada, nodo).
    heapq.heappush(op, (0, next(cnt), 'START')) 
    
    # 'cf' (came from) guarda de qué nodo vine. 'g' es el costo acumulado desde el inicio.
    cf, g = {}, {n: float('inf') for n in grafo}; g['START'] = 0
    # 'f' es el costo estimado total (lo que ya caminé 'g' + lo que me falta 'h').
    f = {n: float('inf') for n in grafo}; f['START'] = h('START')
    
    vis_a = set() # Set para almacenar qué nodos ya exploramos (sin repetir).
    
    while op: # Mientras haya caminos que explorar...
        curr = heapq.heappop(op)[2] # Saca el nodo más prometedor (el de menor 'f') de la lista.
        vis_a.add(curr) # Lo marcamos como "revisado".
        
        if curr == 'END': # ¡Llegamos a la meta!
            # Reconstruimos la ruta ganadora leyendo de atrás hacia adelante en el diccionario 'cf'.
            cam_a = ['END']; 
            while cam_a[-1] != 'START': 
                cam_a.append(cf[cam_a[-1]])
            cam_a.reverse() # Invertimos la lista para que quede de INICIO a FIN.
            break # Salimos del bucle.
            
        # Revisa a todos los vecinos válidos del nodo actual.
        for vec in grafo.get(curr, []):
            tg = g[curr] + 1 # Costo tentativo: lo que ya caminé + 1 paso hacia el vecino.
            # Si este camino tentativo es MEJOR (más corto) que el que el vecino conocía:
            if tg < g.get(vec, float('inf')):
                cf[vec] = curr # Anota que la mejor forma de llegar a este vecino es desde mi nodo actual.
                g[vec] = tg # Actualiza el costo 'g'.
                f[vec] = tg + h(vec) # Actualiza el costo 'f' (g + Heurística).
                # Mete este vecino a la lista priorizada para evaluarlo pronto.
                heapq.heappush(op, (f[vec], next(cnt), vec)) 

    if cam_a: 
        ui_str, _ = generar_comandos_auto(cam_a, puerta_ent) # Traduce la ruta a texto (Adelante, Derecha).
        # Guarda todos los datos experimentales de A* en el diccionario de resultados.
        res['1'] = {'nombre': "A*", 'img': trazar_ruta_con_exploracion(cam_a, vis_a, (0, 255, 0)), 'color_ar': (0, 255, 0), 'camino': cam_a, 'pasos': len(cam_a)-1, 't': (time.perf_counter()-ts)*1000, 'n': len(vis_a), 'ui': ui_str}

    # -----------------------------------------------------------------
    # ALGORITMO 2: Dijkstra - Seguro pero Exhaustivo (Sin Heurística)
    # -----------------------------------------------------------------
    ts = time.perf_counter() # Inicia cronómetro.
    op_d = [] # Open Set de Dijkstra.
    cnt_d = itertools.count()
    heapq.heappush(op_d, (0, next(cnt_d), 'START'))
    
    # En Dijkstra, 'f' no existe. Solo usa 'g' (distancia caminada real).
    cf_d, g_d = {}, {n: float('inf') for n in grafo}; g_d['START'] = 0
    vis_d = set()
    cam_d = None
    
    while op_d:
        curr_dist, _, curr = heapq.heappop(op_d) # Saca el nodo más "cercano" según 'g'.
        vis_d.add(curr)
        
        if curr == 'END': 
            cam_d = ['END']
            while cam_d[-1] != 'START': 
                cam_d.append(cf_d[cam_d[-1]])
            cam_d.reverse()
            break
            
        # Al igual que A*, revisa vecinos, pero no calcula ninguna Heurística 'h()'.
        for vec in grafo.get(curr, []):
            td = g_d[curr] + 1
            if td < g_d.get(vec, float('inf')):
                cf_d[vec] = curr
                g_d[vec] = td
                heapq.heappush(op_d, (td, next(cnt_d), vec)) # Prioriza puramente por 'td' (Costo G real).
                
    if cam_d: 
        ui_str, _ = generar_comandos_auto(cam_d, puerta_ent)
        res['2'] = {'nombre': "Dijkstra", 'img': trazar_ruta_con_exploracion(cam_d, vis_d, (255, 0, 0)), 'color_ar': (255, 0, 0), 'camino': cam_d, 'pasos': len(cam_d)-1, 't': (time.perf_counter()-ts)*1000, 'n': len(vis_d), 'ui': ui_str}

    # -----------------------------------------------------------------
    # ALGORITMO 3: BFS (Búsqueda en Anchura) - Fuerza Bruta en Ondas
    # -----------------------------------------------------------------
    ts = time.perf_counter() # Inicia cronómetro.
    
    # BFS usa una Cola estándar (FIFO: Primero en entrar, primero en salir).
    cola, vis_b, cam_b = deque([('START', ['START'])]), set(), None
    
    while cola:
        # Saca siempre el elemento más ANTIGUO (El de la izquierda de la cola). 
        # Esto hace que explore el laberinto como una onda expansiva, garantizando encontrar el camino más corto.
        curr, path = cola.popleft(); vis_b.add(curr) 
        
        if curr == 'END': 
            cam_b = path; break # Encontró la meta, se queda con la ruta armada.
            
        for vec in grafo.get(curr, []):
            # Si el vecino no ha sido visitado ni está esperando ya en la cola:
            if vec not in vis_b and not any(vec == i[0] for i in cola): 
                # Añade el vecino al final (derecha) de la cola, pasándole la ruta que lo alcanzó + el propio vecino.
                cola.append((vec, path + [vec]))
                
    if cam_b: 
        ui_str, _ = generar_comandos_auto(cam_b, puerta_ent)
        res['3'] = {'nombre': "BFS", 'img': trazar_ruta_con_exploracion(cam_b, vis_b, (255, 165, 0)), 'color_ar': (0, 165, 255), 'camino': cam_b, 'pasos': len(cam_b)-1, 't': (time.perf_counter()-ts)*1000, 'n': len(vis_b), 'ui': ui_str}

    # -----------------------------------------------------------------
    # ALGORITMO 4: DFS (Búsqueda en Profundidad) - Laberintico
    # -----------------------------------------------------------------
    ts = time.perf_counter() # Inicia cronómetro.
    
    # DFS usa una Pila o Stack (LIFO: Último en entrar, primero en salir).
    stack, vis_dfs, cam_dfs = [('START', ['START'])], set(), None
    
    while stack:
        # Saca siempre el elemento más NUEVO (El final de la lista).
        # Esto obliga al algoritmo a irse "hasta el fondo" de un pasillo sin mirar los lados hasta chocar con la pared.
        curr, path = stack.pop(); vis_dfs.add(curr)
        
        if curr == 'END': 
            cam_dfs = path; break # Encontró la meta. Rara vez es el camino más corto.
            
        for vec in grafo.get(curr, []):
            # Si el vecino no está visitado:
            if vec not in vis_dfs and not any(vec == i[0] for i in stack):
                # Lo apila al final. Como el siguiente bucle saca del final, esto forzará a seguir por este nuevo vecino de inmediato.
                stack.append((vec, path + [vec]))
                
    if cam_dfs: 
        ui_str, _ = generar_comandos_auto(cam_dfs, puerta_ent)
        res['4'] = {'nombre': "DFS", 'img': trazar_ruta_con_exploracion(cam_dfs, vis_dfs, (128, 0, 128)), 'color_ar': (255, 0, 255), 'camino': cam_dfs, 'pasos': len(cam_dfs)-1, 't': (time.perf_counter()-ts)*1000, 'n': len(vis_dfs), 'ui': ui_str}

    # =================================================================
    # RETORNO FINAL DE RESULTADOS
    # =================================================================
    # Devuelve el diccionario 'res' a main.py. Contiene las imágenes dibujadas, estadísticas de tiempo/nodos y comandos de los 4 algoritmos.
    return res
