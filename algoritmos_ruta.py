import cv2
import numpy as np
import time
import heapq
import itertools
from collections import deque
from PIL import Image, ImageDraw

def detectar_aberturas_360(ruta, cells_x=5, cells_y=5):
    img = Image.open(ruta).convert('RGB')
    px, py, off = img.size[0] / cells_x, img.size[1] / cells_y, 5
    ab = []
    for c in range(cells_x):
        if img.getpixel((int((c+0.5)*px), off))[0] > 128: ab.append(('N', c, int((c+0.5)*px), 0))
        if img.getpixel((int((c+0.5)*px), img.size[1]-1-off))[0] > 128: ab.append(('S', c, int((c+0.5)*px), img.size[1]-1))
    for r in range(cells_y):
        if img.getpixel((off, int((r+0.5)*py)))[0] > 128: ab.append(('O', r, 0, int((r+0.5)*py)))
        if img.getpixel((img.size[0]-1-off, int((r+0.5)*py)))[0] > 128: ab.append(('E', r, img.size[0]-1, int((r+0.5)*py)))
    
    if len(ab) >= 2:
        ab.sort(key=lambda i: (i[2], i[3]))
        return (ab[0][0], ab[0][1]), (ab[1][0], ab[1][1])
    else:
        return ('O', 0), ('E', cells_y - 1)

def generar_comandos_auto(camino, puerta_ent):
    if not camino or len(camino) < 3: return "Sin ruta", []
    lado = puerta_ent[0]
    current_dir = (0, 1) if lado == 'N' else (0, -1) if lado == 'S' else (1, 0) if lado == 'O' else (-1, 0)
    cmds = ["Adelante"] 
    nodos = [n for n in camino if isinstance(n, tuple)]
    for i in range(len(nodos) - 1):
        move = (nodos[i+1][0] - nodos[i][0], nodos[i+1][1] - nodos[i][1])
        if move == current_dir: cmds.append("Adelante")
        elif move == (current_dir[1], -current_dir[0]): cmds.extend(["Izquierda", "Adelante"])
        elif move == (-current_dir[1], current_dir[0]): cmds.extend(["Derecha", "Adelante"])
        else: cmds.extend(["Media_Vuelta", "Adelante"])
        current_dir = move 
    cmds.append("Adelante") 
    comp, adv = [], 0
    for c in cmds:
        if c == "Adelante": adv += 1
        else:
            if adv > 0: comp.append(f"Adelante x{adv}"); adv = 0
            comp.append(c)
    if adv > 0: comp.append(f"Adelante x{adv}")
    return " -> ".join(comp), cmds

def paso4_resolver_laberinto(archivo_entrada, puerta_ent, puerta_sal, cells_x=5, cells_y=5):
    print("\n" + "="*50)
    print(" PASO 4: CALCULANDO ALGORITMOS...")
    print("="*50)
    
    img_base = Image.open(archivo_entrada).convert('RGB')
    ancho_total, alto_total = img_base.size
    px, py = ancho_total / cells_x, alto_total / cells_y

    def obtener_coordenada(columna, fila): 
        return (int((columna + 0.5) * px), int((fila + 0.5) * py))
        
    def hay_pared(x1, y1, x2, y2): 
        for i in range(1, 16):
            pt_x = int(x1 + (x2 - x1) * (i / 16.0))
            pt_y = int(y1 + (y2 - y1) * (i / 16.0))
            if img_base.getpixel((pt_x, pt_y))[0] < 128: 
                return True
        return False

    grafo = {}
    for fila in range(cells_y):
        for col in range(cells_x):
            nodo = (col, fila)
            grafo[nodo] = []
            x1, y1 = obtener_coordenada(col, fila)
            if col < cells_x - 1 and not hay_pared(x1, y1, *obtener_coordenada(col + 1, fila)): grafo[nodo].append((col + 1, fila))
            if fila < cells_y - 1 and not hay_pared(x1, y1, *obtener_coordenada(col, fila + 1)): grafo[nodo].append((col, fila + 1))
            if col > 0 and not hay_pared(x1, y1, *obtener_coordenada(col - 1, fila)): grafo[nodo].append((col - 1, fila))
            if fila > 0 and not hay_pared(x1, y1, *obtener_coordenada(col, fila - 1)): grafo[nodo].append((col, fila - 1))

    def celda_desde_puerta(p):
        if p[0] == 'N': return (p[1], 0)
        if p[0] == 'S': return (p[1], cells_y - 1)
        if p[0] == 'O': return (0, p[1])
        if p[0] == 'E': return (cells_x - 1, p[1])

    def coord_puerta(p):
        if p[0] == 'N': return (int((p[1] + 0.5) * px), 0)
        if p[0] == 'S': return (int((p[1] + 0.5) * px), alto_total - 1)
        if p[0] == 'O': return (0, int((p[1] + 0.5) * py))
        if p[0] == 'E': return (ancho_total - 1, int((p[1] + 0.5) * py))

    nodo_ent, nodo_sal = celda_desde_puerta(puerta_ent), celda_desde_puerta(puerta_sal)
    grafo['START'], grafo['END'] = [nodo_ent], []
    grafo[nodo_ent].append('START')
    grafo[nodo_sal].append('END')

    def trazar_ruta_con_exploracion(camino, visitados, color_rgb):
        img_rgba = img_base.convert("RGBA")
        overlay = Image.new("RGBA", img_rgba.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        radio_vis = int(px * 0.25) 
        color_transparente = (*color_rgb, 100) 
        
        for n in visitados:
            if n in ['START', 'END']: continue
            cx, cy = obtener_coordenada(*n)
            draw_overlay.ellipse([cx - radio_vis, cy - radio_vis, cx + radio_vis, cy + radio_vis], fill=color_transparente)
            
        img_result = Image.alpha_composite(img_rgba, overlay).convert("RGB")
        draw_final = ImageDraw.Draw(img_result)
        
        puntos_linea = []
        for n in camino:
            if n == 'START': puntos_linea.append(coord_puerta(puerta_ent))
            elif n == 'END': puntos_linea.append(coord_puerta(puerta_sal))
            else: puntos_linea.append(obtener_coordenada(*n))
                
        draw_final.line(puntos_linea, fill=color_rgb, width=int(px*0.1), joint="curve")
        return cv2.cvtColor(np.array(img_result), cv2.COLOR_RGB2BGR) 

    res = {}
    
    # A*
    ts = time.perf_counter()
    h = lambda n: abs(nodo_ent[0]-nodo_sal[0])+abs(nodo_ent[1]-nodo_sal[1]) if n=='START' else 0 if n=='END' else abs(n[0]-nodo_sal[0])+abs(n[1]-nodo_sal[1])
    ct, op, cf, g, f = itertools.count(), [], {}, {n: float('inf') for n in grafo}, {n: float('inf') for n in grafo}
    heapq.heappush(op, (0, next(ct), 'START')); g['START'], f['START'] = 0, h('START')
    cam_a, vis_a = None, set()
    while op:
        _, _, curr = heapq.heappop(op); vis_a.add(curr)
        if curr == 'END':
            cam_a = [curr]; 
            while curr in cf: curr = cf[curr]; cam_a.append(curr)
            cam_a.reverse(); break
        for vec in grafo[curr]:
            tg = g[curr] + 1
            if tg < g[vec]: cf[vec], g[vec], f[vec] = curr, tg, tg + h(vec); heapq.heappush(op, (f[vec], next(ct), vec))
    if cam_a: 
        ui_str, _ = generar_comandos_auto(cam_a, puerta_ent)
        res['1'] = {'nombre': "A*", 'img': trazar_ruta_con_exploracion(cam_a, vis_a, (0, 200, 255)), 'color_ar': (255, 200, 0), 'camino': cam_a, 'pasos': len(cam_a)-1, 't': (time.perf_counter()-ts)*1000, 'n': len(vis_a), 'ui': ui_str}

    # Dijkstra
    ts = time.perf_counter()
    ct, op, cf, g = itertools.count(), [], {}, {n: float('inf') for n in grafo}
    heapq.heappush(op, (0, next(ct), 'START')); g['START'] = 0
    cam_d, vis_d = None, set()
    while op:
        _, _, curr = heapq.heappop(op); vis_d.add(curr)
        if curr == 'END':
            cam_d = [curr]; 
            while curr in cf: curr = cf[curr]; cam_d.append(curr)
            cam_d.reverse(); break
        for vec in grafo[curr]:
            tg = g[curr] + 1
            if tg < g[vec]: cf[vec], g[vec] = curr, tg; heapq.heappush(op, (tg, next(ct), vec))
    if cam_d: 
        ui_str, _ = generar_comandos_auto(cam_d, puerta_ent)
        res['2'] = {'nombre': "Dijkstra", 'img': trazar_ruta_con_exploracion(cam_d, vis_d, (50, 205, 50)), 'color_ar': (50, 205, 50), 'camino': cam_d, 'pasos': len(cam_d)-1, 't': (time.perf_counter()-ts)*1000, 'n': len(vis_d), 'ui': ui_str}

    # BFS
    ts = time.perf_counter()
    cola, vis_b, cam_b = deque([('START', ['START'])]), set(), None
    while cola:
        curr, path = cola.popleft(); vis_b.add(curr)
        if curr == 'END': cam_b = path; break
        for vec in grafo.get(curr, []):
            if vec not in vis_b and not any(vec == i[0] for i in cola): cola.append((vec, path + [vec]))
    if cam_b: 
        ui_str, _ = generar_comandos_auto(cam_b, puerta_ent)
        res['3'] = {'nombre': "BFS", 'img': trazar_ruta_con_exploracion(cam_b, vis_b, (255, 165, 0)), 'color_ar': (0, 165, 255), 'camino': cam_b, 'pasos': len(cam_b)-1, 't': (time.perf_counter()-ts)*1000, 'n': len(vis_b), 'ui': ui_str}

    # DFS
    ts = time.perf_counter()
    stack, vis_dfs, cam_dfs = [('START', ['START'])], set(), None
    while stack:
        curr, path = stack.pop(); vis_dfs.add(curr)
        if curr == 'END': cam_dfs = path; break
        for vec in grafo.get(curr, []):
            if vec not in vis_dfs and not any(vec == i[0] for i in stack): stack.append((vec, path + [vec]))
    if cam_dfs: 
        ui_str, _ = generar_comandos_auto(cam_dfs, puerta_ent)
        res['4'] = {'nombre': "DFS", 'img': trazar_ruta_con_exploracion(cam_dfs, vis_dfs, (150, 0, 255)), 'color_ar': (255, 0, 150), 'camino': cam_dfs, 'pasos': len(cam_dfs)-1, 't': (time.perf_counter()-ts)*1000, 'n': len(vis_dfs), 'ui': ui_str}

    return res