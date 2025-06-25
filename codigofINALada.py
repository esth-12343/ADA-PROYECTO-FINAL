import zipfile
import os
from tqdm import tqdm

# Constantes para los nombres de archivo esperados dentro de los ZIP
USER_FILE_NAME = "10_million_user.txt"
LOCATION_FILE_NAME = "10_million_location.txt"

def cargar_ubicaciones(ruta_zip, limit=None):
    """
    Carga las ubicaciones de los usuarios desde un archivo ZIP.

    Args:
        ruta_zip (str): Ruta al archivo ZIP que contiene los datos de ubicación.
        limit (int, optional): Limita el número de ubicaciones a cargar. 
                               Por defecto es None (cargar todos).

    Returns:
        dict: Un diccionario donde las claves son IDs de usuario (int, 1-indexed) 
              y los valores son tuplas (latitud, longitud).
              Retorna un diccionario vacío si hay errores.
    """
    ubicaciones = {}
    if not os.path.exists(ruta_zip):
        print(f"Error: El archivo ZIP de ubicaciones no se encontró en {ruta_zip}")
        return ubicaciones

    try:
        with zipfile.ZipFile(ruta_zip) as z:
            # Asumimos que el archivo de texto es el primero o tiene un nombre específico
            # Intentamos encontrar el archivo por un nombre conocido
            target_file = None
            for name in z.namelist():
                if LOCATION_FILE_NAME in name: # Flexible para subcarpetas
                    target_file = name
                    break
            
            if not target_file:
                if z.namelist():
                    target_file = z.namelist()[0] # Tomar el primero si no se encuentra por nombre
                    print(f"Advertencia: No se encontró '{LOCATION_FILE_NAME}' en {ruta_zip}. Usando '{target_file}' en su lugar.")
                else:
                    print(f"Error: El archivo ZIP {ruta_zip} está vacío.")
                    return ubicaciones

            with z.open(target_file) as f:
                print(f"Cargando ubicaciones desde {ruta_zip}/{target_file}...")
                for i, line_bytes in enumerate(tqdm(f, desc="Cargando ubicaciones", unit="líneas")):
                    if limit is not None and i >= limit:
                        print(f"Límite de {limit} ubicaciones alcanzado.")
                        break
                    try:
                        line = line_bytes.decode().strip()
                        if not line:  # Omitir líneas vacías
                            continue
                        parts = line.split(',')
                        if len(parts) == 2:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            ubicaciones[i + 1] = (lat, lon)  # IDs son 1-indexed
                        else:
                            print(f"Advertencia: Línea {i+1} mal formada en ubicaciones: '{line}'")
                    except ValueError as ve:
                        print(f"Advertencia: Error de valor en línea {i+1} de ubicaciones ('{line}'): {ve}")
                    except Exception as e_line:
                        print(f"Advertencia: Error procesando línea {i+1} de ubicaciones ('{line}'): {e_line}")
        print(f"Se cargaron {len(ubicaciones)} ubicaciones.")
        return ubicaciones
    except zipfile.BadZipFile:
        print(f"Error: El archivo ZIP de ubicaciones está corrupto: {ruta_zip}")
        return {}
    except Exception as e:
        print(f"Error inesperado cargando ubicaciones desde {ruta_zip}: {e}")
        return {}

def cargar_usuarios_grafo(ruta_zip, grafo, limit=None):
    """
    Carga las conexiones de usuarios desde un archivo ZIP y las añade a un grafo.

    Args:
        ruta_zip (str): Ruta al archivo ZIP que contiene los datos de usuarios.
        grafo (GrafoDirigido): Instancia de la clase GrafoDirigido donde se añadirán los nodos y aristas.
        limit (int, optional): Limita el número de usuarios (líneas) a procesar.
                               Por defecto es None (procesar todos).

    Returns:
        None: Modifica el objeto grafo directamente.
    """
    if not os.path.exists(ruta_zip):
        print(f"Error: El archivo ZIP de usuarios no se encontró en {ruta_zip}")
        return

    try:
        with zipfile.ZipFile(ruta_zip) as z:
            target_file = None
            for name in z.namelist():
                if USER_FILE_NAME in name: # Flexible para subcarpetas
                    target_file = name
                    break
            
            if not target_file:
                if z.namelist():
                    target_file = z.namelist()[0]
                    print(f"Advertencia: No se encontró '{USER_FILE_NAME}' en {ruta_zip}. Usando '{target_file}' en su lugar.")
                else:
                    print(f"Error: El archivo ZIP {ruta_zip} está vacío.")
                    return

            with z.open(target_file) as f:
                print(f"Cargando conexiones de usuarios desde {ruta_zip}/{target_file}...")
                for i, line_bytes in enumerate(tqdm(f, desc="Cargando usuarios", unit="líneas")):
                    usuario_id = i + 1  # Los IDs de usuario son 1-indexed
                    
                    if limit is not None and usuario_id > limit: # Si el límite es por N usuarios principales
                        print(f"Límite de {limit} usuarios procesados alcanzado.")
                        break
                    
                    grafo.agregar_nodo(usuario_id) # Asegurar que el nodo exista
                    
                    try:
                        line = line_bytes.decode().strip()
                        if not line: # Puede haber usuarios sin conexiones
                            continue 
                        
                        vecinos_str = line.split(',')
                        for vecino_str in vecinos_str:
                            try:
                                vecino_id = int(vecino_str.strip())
                                if limit is not None and (vecino_id > limit): # Si el vecino excede el límite de nodos a considerar
                                    continue # Opcional: no añadir aristas a nodos fuera del subconjunto limitado
                                grafo.agregar_arista(usuario_id, vecino_id)
                            except ValueError:
                                print(f"Advertencia: ID de vecino no válido '{vecino_str}' para usuario {usuario_id} en línea '{line}'")
                    except Exception as e_line:
                        print(f"Advertencia: Error procesando línea {i+1} de usuarios ('{line}'): {e_line}")
        print(f"Carga de conexiones de usuarios completada. Nodos en grafo: {grafo.obtener_numero_nodos()}, Aristas: {grafo.obtener_numero_aristas()}")

    except zipfile.BadZipFile:
        print(f"Error: El archivo ZIP de usuarios está corrupto: {ruta_zip}")
    except Exception as e:
        print(f"Error inesperado cargando usuarios desde {ruta_zip}: {e}")

if __name__ == '__main__':
    # Ejemplo de uso (requiere crear un dummy GrafoDirigido y archivos zip de prueba)
    
    # Crear clase GrafoDirigido dummy para prueba
    class GrafoDirigidoDummy:
        def __init__(self):
            self.adj = {}
            self.nodes = set()
            self.num_aristas = 0

        def agregar_nodo(self, nodo_id):
            self.nodes.add(nodo_id)
            if nodo_id not in self.adj:
                self.adj[nodo_id] = []

        def agregar_arista(self, u, v):
            self.agregar_nodo(u) # Asegurar que ambos nodos existan
            self.agregar_nodo(v)
            if v not in self.adj[u]: # Evitar duplicados si la lógica del grafo lo requiere
                self.adj[u].append(v)
                self.num_aristas +=1

        def obtener_numero_nodos(self):
            return len(self.nodes)

        def obtener_numero_aristas(self):
            return self.num_aristas

    # Crear archivos ZIP de prueba si no existen
    ruta_zip_usuarios_test = 'dummy_users.zip'
    ruta_zip_ubicaciones_test = 'dummy_locations.zip'

    if not os.path.exists(ruta_zip_usuarios_test):
        with zipfile.ZipFile(ruta_zip_usuarios_test, 'w') as zf:
            zf.writestr(USER_FILE_NAME, "2,3\n1,3\n1,2,4\n3\n") # Usuario 5 no sigue a nadie
        print(f"Creado '{ruta_zip_usuarios_test}' para pruebas.")

    if not os.path.exists(ruta_zip_ubicaciones_test):
        with zipfile.ZipFile(ruta_zip_ubicaciones_test, 'w') as zf:
            zf.writestr(LOCATION_FILE_NAME, "10.0,20.0\n11.0,21.0\n12.0,22.0\n13.0,23.0\n14.0,24.0\n")
        print(f"Creado '{ruta_zip_ubicaciones_test}' para pruebas.")

    print("\n--- Prueba de carga_ubicaciones (limitado a 3) ---")
    ubicaciones_test = cargar_ubicaciones(ruta_zip_ubicaciones_test, limit=3)
    for uid, coords in ubicaciones_test.items():
        print(f"Usuario {uid}: {coords}")
    
    print("\n--- Prueba de carga_ubicaciones (completo) ---")
    ubicaciones_test_full = cargar_ubicaciones(ruta_zip_ubicaciones_test)
    print(f"Total de ubicaciones cargadas: {len(ubicaciones_test_full)}")


    print("\n--- Prueba de carga_usuarios_grafo (limitado a 3 nodos principales) ---")
    grafo_test_limitado = GrafoDirigidoDummy()
    cargar_usuarios_grafo(ruta_zip_usuarios_test, grafo_test_limitado, limit=3)
    # Los nodos 4 y 5 podrían existir si son mencionados por 1,2,3 pero no tendrán aristas salientes procesadas.
    # Las aristas a nodos >3 desde nodos <=3 no se añadirán.
    print(f"Nodos en grafo limitado: {sorted(list(grafo_test_limitado.nodes))}")
    print(f"Adyacencia grafo limitado: {grafo_test_limitado.adj}")
    print(f"Número de aristas grafo limitado: {grafo_test_limitado.obtener_numero_aristas()}")


    print("\n--- Prueba de carga_usuarios_grafo (completo) ---")
    grafo_test_completo = GrafoDirigidoDummy()
    cargar_usuarios_grafo(ruta_zip_usuarios_test, grafo_test_completo)
    print(f"Nodos en grafo completo: {sorted(list(grafo_test_completo.nodes))}")
    print(f"Adyacencia grafo completo: {grafo_test_completo.adj}")
    print(f"Número de aristas grafo completo: {grafo_test_completo.obtener_numero_aristas()}")

    # Limpiar archivos dummy creados
    # os.remove(ruta_zip_usuarios_test)
    # os.remove(ruta_zip_ubicaciones_test)
    # print("\nArchivos dummy eliminados.")
    print("\nArchivos dummy NO eliminados para permitir re-ejecuciones.")


import time
import os
import random

from grafo import GrafoDirigido
from carga_datos import cargar_ubicaciones, cargar_usuarios_grafo
from metricas_red import calcular_estadisticas_basicas, top_n_usuarios_por_grado_salida, top_n_usuarios_por_grado_entrada
from algoritmos_grafos import (
    longitud_promedio_camino_mas_corto_muestreo,
    detectar_comunidades_louvain, # Recordar que es una versión simplificada
    arbol_expansion_minima_prim
)
from visualizacion import visualizar_red_plotly

# --- Configuración ---
# Rutas a los archivos ZIP (ajustar si es necesario, o usar variables de entorno)
# Por ahora, asumimos que están en el mismo directorio que main.py o se proporcionan rutas completas.
# Si se suben al repo, podrían estar en una carpeta 'data/'
RUTA_ZIP_USUARIOS = "10_million_user.txt.zip"
RUTA_ZIP_UBICACIONES = "10_million_location.txt.zip"

# Para desarrollo y pruebas rápidas, se puede limitar el número de nodos/ubicaciones a cargar.
# Poner en None para cargar todos los datos.
LIMITE_NODOS_CARGA = 10000  # Ejemplo: Cargar solo los primeros 10,000 usuarios y sus conexiones
# LIMITE_NODOS_CARGA = None # Para cargar los 10 millones completos

LIMITE_UBICACIONES_CARGA = LIMITE_NODOS_CARGA # Cargar ubicaciones para los mismos nodos
# LIMITE_UBICACIONES_CARGA = None

MAX_NODOS_VISUALIZAR = 500 # Límite para Plotly para mantener la interactividad
NUM_MUESTRAS_CAMINO_PROMEDIO = 100 # Muestras para el cálculo de camino promedio

def main():
    print("Inicio del Análisis de Red Social 'X'")
    print("======================================")

    # --- 1. Construcción del Grafo ---
    print("\n--- Fase 1: Carga y Construcción del Grafo ---")
    grafo_principal = GrafoDirigido()

    # --- 1. Construcción del Grafo ---
    print("\n--- Fase 1: Carga y Construcción del Grafo ---")
    grafo_principal = GrafoDirigido()

    # Definir rutas efectivas, que podrían cambiar a dummy si los archivos no existen
    effective_ruta_zip_usuarios = RUTA_ZIP_USUARIOS
    effective_ruta_zip_ubicaciones = RUTA_ZIP_UBICACIONES
    ubicaciones = {}

    # Comprobar si los archivos reales existen
    usuarios_reales_existen = os.path.exists(RUTA_ZIP_USUARIOS)
    ubicaciones_reales_existen = os.path.exists(RUTA_ZIP_UBICACIONES)

    if not usuarios_reales_existen:
        print(f"ADVERTENCIA: Archivo de usuarios '{RUTA_ZIP_USUARIOS}' no encontrado.")
        effective_ruta_zip_usuarios = 'dummy_main_users.zip'
        if not os.path.exists(effective_ruta_zip_usuarios):
            with zipfile.ZipFile(effective_ruta_zip_usuarios, 'w') as zf_u:
                zf_u.writestr("10_million_user.txt", "2,3\n1,4\n1,2\n2,1\n6\n5\n") # 6 nodos, usuario 7 aislado
            print(f"Creado '{effective_ruta_zip_usuarios}' dummy.")

    if not ubicaciones_reales_existen:
        print(f"ADVERTENCIA: Archivo de ubicaciones '{RUTA_ZIP_UBICACIONES}' no encontrado.")
        effective_ruta_zip_ubicaciones = 'dummy_main_locations.zip'
        if not os.path.exists(effective_ruta_zip_ubicaciones):
            with zipfile.ZipFile(effective_ruta_zip_ubicaciones, 'w') as zf_l:
                zf_l.writestr("10_million_location.txt", "10,20\n11,21\n12,22\n13,23\n14,24\n15,25\n16,26\n") # 7 ubicaciones
            print(f"Creado '{effective_ruta_zip_ubicaciones}' dummy.")

    # Cargar ubicaciones
    tiempo_inicio_ubicaciones = time.time()
    if os.path.exists(effective_ruta_zip_ubicaciones):
        ubicaciones = cargar_ubicaciones(effective_ruta_zip_ubicaciones, limit=LIMITE_UBICACIONES_CARGA)
    else:
        print(f"ADVERTENCIA: No se pudo cargar ni encontrar el archivo de ubicaciones: {effective_ruta_zip_ubicaciones}")
    tiempo_fin_ubicaciones = time.time()
    print(f"Tiempo de carga de ubicaciones: {tiempo_fin_ubicaciones - tiempo_inicio_ubicaciones:.2f} segundos.")

    # Cargar grafo de usuarios
    tiempo_inicio_grafo = time.time()
    if os.path.exists(effective_ruta_zip_usuarios):
        cargar_usuarios_grafo(effective_ruta_zip_usuarios, grafo_principal, limit=LIMITE_NODOS_CARGA)
    else:
        print(f"ERROR CRÍTICO: No se pudo cargar ni encontrar el archivo de usuarios: {effective_ruta_zip_usuarios}")

    tiempo_fin_grafo = time.time()
    print(f"Tiempo de construcción del grafo: {tiempo_fin_grafo - tiempo_inicio_grafo:.2f} segundos.")
    print(f"Grafo construido con {grafo_principal.obtener_numero_nodos()} nodos y {grafo_principal.obtener_numero_aristas()} aristas.")

    if grafo_principal.obtener_numero_nodos() == 0:
        print("El grafo está vacío después de la carga. Terminando el análisis.")
        return

    # --- 2. Análisis Exploratorio de Datos (EDA) y Métricas Básicas ---
    print("\n--- Fase 2: Análisis Exploratorio y Métricas Básicas ---")
    tiempo_inicio_metricas = time.time()
    estadisticas = calcular_estadisticas_basicas(grafo_principal)
    print("Estadísticas Básicas de la Red:")
    for k, v in estadisticas.items():
        print(f"  {k}: {v}")

    top_5_salida = top_n_usuarios_por_grado_salida(grafo_principal, n=5)
    print(f"  Top 5 usuarios por grado de salida: {top_5_salida}")

    # El cálculo de grado de entrada puede ser lento para grafos grandes
    if grafo_principal.obtener_numero_nodos() <= 20000: # Umbral para calcular top grado entrada
        top_5_entrada = top_n_usuarios_por_grado_entrada(grafo_principal, n=5)
        print(f"  Top 5 usuarios por grado de entrada: {top_5_entrada}")
    else:
        print("  Cálculo de Top N por grado de entrada omitido por tamaño del grafo (optimización).")
    tiempo_fin_metricas = time.time()
    print(f"Tiempo de cálculo de métricas básicas: {tiempo_fin_metricas - tiempo_inicio_metricas:.2f} segundos.")

    # --- 3. Propiedades y Métricas Avanzadas de la Red ---
    print("\n--- Fase 3: Propiedades y Métricas Avanzadas ---")

    # Detección de Comunidades (Louvain)
    print("\n  Iniciando Detección de Comunidades (Louvain)...")
    tiempo_inicio_louvain = time.time()
    # Para grafos muy grandes, la implementación manual de Louvain puede ser muy lenta.
    # Considerar ejecutar en una muestra más pequeña si es necesario o si la implementación no está optimizada.
    grafo_para_louvain = grafo_principal # o una submuestra si es muy grande
    if LIMITE_NODOS_CARGA and LIMITE_NODOS_CARGA > 5000: # Umbral arbitrario
        print(f"    Louvain se ejecutará sobre el grafo cargado de {grafo_para_louvain.obtener_numero_nodos()} nodos.")
        print(f"    ADVERTENCIA: La implementación manual de Louvain puede ser lenta para este tamaño.")

    particion_comunidades = detectar_comunidades_louvain(grafo_para_louvain)
    tiempo_fin_louvain = time.time()
    if particion_comunidades:
        num_comunidades = len(set(particion_comunidades.values()))
        print(f"  Detección de Comunidades (Louvain) completada. Encontradas: {num_comunidades} comunidades.")
        # print(f"  Ejemplo de partición: {dict(list(particion_comunidades.items())[:5])}")
    else:
        print("  No se pudieron detectar comunidades.")
    print(f"  Tiempo de detección de comunidades: {tiempo_fin_louvain - tiempo_inicio_louvain:.2f} segundos.")

    # Análisis de Camino Más Corto Promedio
    print("\n  Calculando Longitud Promedio del Camino Más Corto (muestreo)...")
    tiempo_inicio_camino = time.time()
    long_prom_camino = longitud_promedio_camino_mas_corto_muestreo(grafo_principal, num_muestras=NUM_MUESTRAS_CAMINO_PROMEDIO)
    tiempo_fin_camino = time.time()
    print(f"  Longitud promedio del camino más corto (estimada): {long_prom_camino:.2f}")
    print(f"  Tiempo de cálculo de camino promedio: {tiempo_fin_camino - tiempo_inicio_camino:.2f} segundos.")

    # Árbol de Expansión Mínima (Prim)
    # MST es para grafos no dirigidos. Lo aplicaremos tratando el grafo como no dirigido.
    print("\n  Calculando Árbol de Expansión Mínima (Prim adaptado)...")
    tiempo_inicio_mst = time.time()
    mst = arbol_expansion_minima_prim(grafo_principal)
    tiempo_fin_mst = time.time()
    print(f"  Árbol de Expansión Mínima construido con {mst.obtener_numero_nodos()} nodos y {mst.obtener_numero_aristas()} aristas.")
    print(f"  Tiempo de cálculo de MST: {tiempo_fin_mst - tiempo_inicio_mst:.2f} segundos.")


    # --- 4. Visualización ---
    print("\n--- Fase 4: Visualización ---")
    print(f"Se visualizarán hasta {MAX_NODOS_VISUALIZAR} nodos.")

    # Visualización de la Red General (subconjunto)
    print("\n  Generando visualización de la red general (subconjunto)...")
    # Decidir si usar ubicaciones o layout aleatorio
    layout_para_visualizacion = None
    usar_ubicaciones_para_layout = False
    if ubicaciones and len(ubicaciones) >= MAX_NODOS_VISUALIZAR * 0.5: # Si tenemos suficientes ubicaciones
        print("  Intentando usar ubicaciones geográficas para el layout de la red general.")
        usar_ubicaciones_para_layout = True

    visualizar_red_plotly(grafo_principal,
                          titulo="Red Social 'X' (Muestra)",
                          max_nodos_visualizar=MAX_NODOS_VISUALIZAR,
                          ubicaciones=ubicaciones if usar_ubicaciones_para_layout else None)

    # Visualización de Comunidades (subconjunto)
    if particion_comunidades:
        print("\n  Generando visualización de comunidades (subconjunto)...")
        usar_ubicaciones_para_layout_com = False
        if ubicaciones and len(ubicaciones) >= MAX_NODOS_VISUALIZAR * 0.5:
            print("  Intentando usar ubicaciones geográficas para el layout de comunidades.")
            usar_ubicaciones_para_layout_com = True

        visualizar_red_plotly(grafo_principal,
                              titulo="Comunidades en Red Social 'X' (Muestra)",
                              max_nodos_visualizar=MAX_NODOS_VISUALIZAR,
                              particion_comunidades=particion_comunidades,
                              ubicaciones=ubicaciones if usar_ubicaciones_para_layout_com else None)
    else:
        print("\n  No hay partición de comunidades para visualizar.")

    # Visualización del MST (subconjunto, si es muy grande)
    # print("\n  Generando visualización del Árbol de Expansión Mínima (subconjunto)...")
    # visualizar_red_plotly(mst,
    #                       titulo="Árbol de Expansión Mínima (Muestra)",
    #                       max_nodos_visualizar=min(MAX_NODOS_VISUALIZAR, mst.obtener_numero_nodos()),
    #                       ubicaciones=None) # MST raramente se visualiza con geo-ubicaciones originales

    print("\n======================================")
    print("Análisis Completado.")
    print("Los gráficos interactivos de Plotly deberían haberse abierto en tu navegador.")

if __name__ == '__main__':
    # Para el entorno de prueba, necesitamos crear los archivos dummy si los reales no están.
    # Esto se maneja dentro de main() ahora.
    import zipfile # Necesario para la creación de dummies si los archivos no existen
    main()


import plotly.graph_objects as go
import random

# Nota: La clase GrafoDirigido se importa desde grafo.py
# from grafo import GrafoDirigido

def generar_layout_aleatorio(grafo, nodos_a_visualizar):
    """
    Genera posiciones aleatorias para los nodos especificados.
    Es un layout muy básico y no refleja la estructura del grafo.
    """
    pos = {}
    for nodo_id in nodos_a_visualizar:
        pos[nodo_id] = (random.uniform(-1, 1), random.uniform(-1, 1))
    return pos

def visualizar_red_plotly(grafo, titulo="Visualización de Red Social", max_nodos_visualizar=1000, layout_func=None, particion_comunidades=None, ubicaciones=None):
    """
    Crea una visualización interactiva de la red usando Plotly.

    Args:
        grafo (GrafoDirigido): El grafo a visualizar.
        titulo (str): Título del gráfico.
        max_nodos_visualizar (int): Número máximo de nodos a mostrar para evitar sobrecarga.
                                    Los nodos se seleccionan aleatoriamente si se supera.
        layout_func (function, optional): Una función que toma (grafo, nodos_a_visualizar)
                                          y devuelve un diccionario de posiciones {nodo_id: (x,y)}.
                                          Si es None, se usa un layout aleatorio simple.
                                          Para layouts más avanzados (Fruchterman-Reingold, Kamada-Kawai),
                                          se necesitaría una implementación manual o una librería permitida
                                          que no sea networkx para el cálculo del layout en sí.
        particion_comunidades (dict, optional): Un diccionario {nodo_id: comunidad_id} para colorear nodos.
        ubicaciones (dict, optional): Un diccionario {nodo_id: (lat, lon)} para usar como layout si está disponible.
    """
    if not grafo.obtener_numero_nodos():
        print("El grafo está vacío, no se puede visualizar.")
        return

    todos_los_nodos = grafo.obtener_todos_los_nodos()
    if len(todos_los_nodos) > max_nodos_visualizar:
        print(f"Mostrando una submuestra de {max_nodos_visualizar} de {len(todos_los_nodos)} nodos.")
        nodos_a_visualizar = random.sample(todos_los_nodos, max_nodos_visualizar)
    else:
        nodos_a_visualizar = todos_los_nodos

    # Layout de nodos
    if ubicaciones:
        print("Usando ubicaciones geográficas para el layout.")
        pos = {nodo_id: (ubicaciones[nodo_id][1], ubicaciones[nodo_id][0]) # (lon, lat) para Plotly scatter
               for nodo_id in nodos_a_visualizar if nodo_id in ubicaciones}
        # Filtrar nodos_a_visualizar para que solo incluya aquellos con ubicaciones
        nodos_a_visualizar_con_pos = list(pos.keys())
        if not nodos_a_visualizar_con_pos:
            print("Advertencia: Ninguno de los nodos a visualizar tiene ubicaciones. Usando layout aleatorio.")
            if layout_func:
                pos = layout_func(grafo, nodos_a_visualizar)
            else:
                pos = generar_layout_aleatorio(grafo, nodos_a_visualizar)
            nodos_a_visualizar_con_pos = nodos_a_visualizar
        else:
             nodos_a_visualizar = nodos_a_visualizar_con_pos # Actualizar la lista de nodos que realmente se mostrarán
    elif layout_func:
        print("Usando función de layout proporcionada.")
        pos = layout_func(grafo, nodos_a_visualizar)
    else:
        print("Usando layout aleatorio.")
        pos = generar_layout_aleatorio(grafo, nodos_a_visualizar)

    edge_x = []
    edge_y = []
    for nodo_origen in nodos_a_visualizar:
        if nodo_origen in pos: # Asegurar que el nodo origen tenga posición
            x0, y0 = pos[nodo_origen]
            for nodo_destino in grafo.obtener_vecinos(nodo_origen):
                if nodo_destino in nodos_a_visualizar and nodo_destino in pos: # Asegurar que el nodo destino esté en la muestra y tenga posición
                    x1, y1 = pos[nodo_destino]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color_values = []

    for nodo_id in nodos_a_visualizar:
        if nodo_id in pos: # Solo añadir nodos que tienen posición
            x, y = pos[nodo_id]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"ID: {nodo_id}" + (f"<br>Com: {particion_comunidades[nodo_id]}" if particion_comunidades and nodo_id in particion_comunidades else ""))
            if particion_comunidades and nodo_id in particion_comunidades:
                node_color_values.append(particion_comunidades[nodo_id])
            else:
                node_color_values.append(0) # Color por defecto si no hay comunidades o el nodo no está en una


    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True if particion_comunidades else False,
            colorscale='Viridis', # Popular para comunidades
            reversescale=False,
            color=node_color_values if particion_comunidades else 'blue',
            size=5,
            colorbar=dict(
                thickness=15,
                title={'text': 'Comunidad', 'side': 'right'} # Correct way to set title text and side
            ) if particion_comunidades else None,
            line_width=0.5))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text=titulo, font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Visualización interactiva de red. Zoom y pan para explorar.",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()

if __name__ == '__main__':
    from grafo import GrafoDirigido # Importar para pruebas

    print("--- Prueba de Visualización ---")
    g_viz = GrafoDirigido()
    # Crear un grafo de prueba más interesante
    for i in range(1, 21): # 20 nodos
        g_viz.agregar_nodo(i)

    # Pequeñas comunidades
    for i in range(1, 6): # Comunidad 0 (nodos 1-5)
        for j in range(1, 6):
            if i != j and random.random() < 0.6: g_viz.agregar_arista(i,j)

    for i in range(6, 11): # Comunidad 1 (nodos 6-10)
        for j in range(6, 11):
            if i != j and random.random() < 0.6: g_viz.agregar_arista(i,j)

    for i in range(11, 16): # Comunidad 2 (nodos 11-15)
        for j in range(11, 16):
            if i != j and random.random() < 0.6: g_viz.agregar_arista(i,j)

    # Puentes entre comunidades
    g_viz.agregar_arista(3, 7)
    g_viz.agregar_arista(8, 12)
    g_viz.agregar_arista(1,15)
    g_viz.agregar_arista(16,1) # Nodo 16-20 más dispersos
    g_viz.agregar_arista(17,5)
    g_viz.agregar_arista(18,10)
    g_viz.agregar_arista(19,13)
    g_viz.agregar_arista(20,2)


    # Partición de comunidades de ejemplo
    particion_ejemplo = {}
    for i in range(1,6): particion_ejemplo[i] = 0
    for i in range(6,11): particion_ejemplo[i] = 1
    for i in range(11,16): particion_ejemplo[i] = 2
    for i in range(16,21): particion_ejemplo[i] = 3

    # Ubicaciones de ejemplo (simuladas, no geográficamente precisas para la estructura)
    ubicaciones_ejemplo = {i: (random.uniform(0,10), random.uniform(0,10)) for i in range(1,21)}


    print("\nVisualizando red con layout aleatorio y colores de comunidad:")
    visualizar_red_plotly(g_viz,
                          titulo="Red de Prueba - Colores por Comunidad (Layout Aleatorio)",
                          max_nodos_visualizar=20,
                          particion_comunidades=particion_ejemplo)

    print("\nVisualizando red con layout por 'ubicaciones' y colores de comunidad:")
    # Simular algunas ubicaciones que podrían agrupar comunidades visualmente
    ubic_agrupadas = {}
    for i in range(1,6): ubic_agrupadas[i] = (random.uniform(0,2), random.uniform(0,2)) # Com 0
    for i in range(6,11): ubic_agrupadas[i] = (random.uniform(4,6), random.uniform(4,6)) # Com 1
    for i in range(11,16): ubic_agrupadas[i] = (random.uniform(0,2), random.uniform(4,6)) # Com 2
    for i in range(16,21): ubic_agrupadas[i] = (random.uniform(4,6), random.uniform(0,2)) # Com 3

    visualizar_red_plotly(g_viz,
                          titulo="Red de Prueba - Colores por Comunidad (Layout por Ubicaciones Simuladas)",
                          max_nodos_visualizar=20,
                          particion_comunidades=particion_ejemplo,
                          ubicaciones=ubic_agrupadas)

    print("\nVisualizando red sin colores de comunidad (layout aleatorio):")
    visualizar_red_plotly(g_viz,
                          titulo="Red de Prueba - Sin Comunidades (Layout Aleatorio)",
                          max_nodos_visualizar=20)

    print("\nPrueba con un subconjunto de nodos (max_nodos_visualizar=10):")
    visualizar_red_plotly(g_viz,
                          titulo="Red de Prueba - Subconjunto de 10 Nodos (Layout Aleatorio)",
                          max_nodos_visualizar=10,
                          particion_comunidades=particion_ejemplo)

    print("\nPruebas de visualizacion.py completadas. Revisa las pestañas del navegador para ver los gráficos Plotly.")


class GrafoDirigido:
    """
    Clase para representar un grafo dirigido utilizando listas de adyacencia.
    Los IDs de los nodos se asumen como enteros.
    """
    def __init__(self):
        """
        Inicializa un grafo vacío.
        self.adj: Diccionario para almacenar las listas de adyacencia.
                  Clave: nodo_id, Valor: lista de nodos a los que apunta.
        self.nodes: Un conjunto para almacenar todos los nodos únicos presentes en el grafo.
        self.num_aristas: Contador para el número de aristas.
        """
        self.adj = {}
        self.nodes = set()
        self.num_aristas = 0

    def agregar_nodo(self, nodo_id):
        """
        Agrega un nodo al grafo si no existe.

        Args:
            nodo_id: El identificador del nodo.
        """
        self.nodes.add(nodo_id)
        if nodo_id not in self.adj:
            self.adj[nodo_id] = []

    def agregar_arista(self, u, v):
        """
        Agrega una arista dirigida desde el nodo u al nodo v.
        Si los nodos u o v no existen, se agregarán al grafo.

        Args:
            u: El nodo de origen.
            v: El nodo de destino.
        """
        self.agregar_nodo(u)  # Asegura que el nodo origen exista
        self.agregar_nodo(v)  # Asegura que el nodo destino exista

        # Evitar aristas duplicadas y auto-bucles si no se desean (común en redes sociales)
        if v not in self.adj[u]: # Comprobar si la arista ya existe
            self.adj[u].append(v)
            self.num_aristas += 1
        # Si se permiten múltiples aristas entre los mismos nodos o auto-bucles,
        # se podría simplemente hacer self.adj[u].append(v) y self.num_aristas += 1

    def obtener_vecinos(self, nodo_id):
        """
        Obtiene la lista de vecinos a los que apunta un nodo.

        Args:
            nodo_id: El identificador del nodo.

        Returns:
            Una lista de vecinos. Retorna una lista vacía si el nodo no existe o no tiene vecinos salientes.
        """
        return self.adj.get(nodo_id, [])

    def obtener_numero_nodos(self):
        """
        Retorna el número total de nodos en el grafo.
        """
        return len(self.nodes)

    def obtener_numero_aristas(self):
        """
        Retorna el número total de aristas en el grafo.
        """
        return self.num_aristas

    def obtener_todos_los_nodos(self):
        """
        Retorna una lista de todos los nodos en el grafo.
        """
        return list(self.nodes)

    def grado_salida(self, nodo_id):
        """
        Calcula el grado de salida de un nodo (número de aristas que salen de él).

        Args:
            nodo_id: El identificador del nodo.

        Returns:
            El grado de salida del nodo. Retorna 0 si el nodo no existe.
        """
        if nodo_id in self.adj:
            return len(self.adj[nodo_id])
        return 0

    def grado_entrada(self, nodo_id_destino):
        """
        Calcula el grado de entrada de un nodo (número de aristas que apuntan a él).
        Esta operación puede ser costosa (O(N+M)) si no se mantiene una lista de adyacencia inversa.

        Args:
            nodo_id_destino: El identificador del nodo destino.

        Returns:
            El grado de entrada del nodo.
        """
        grado = 0
        if nodo_id_destino in self.nodes: # Solo calcular si el nodo existe
            for nodo_origen in self.adj:
                if nodo_id_destino in self.adj[nodo_origen]:
                    grado += 1
        return grado

    def __str__(self):
        """
        Representación en cadena del grafo.
        """
        res = f"Grafo Dirigido con {self.obtener_numero_nodos()} nodos y {self.obtener_numero_aristas()} aristas.\n"
        # Imprimir solo una pequeña parte para no sobrecargar la salida
        # count = 0
        # for nodo, vecinos in self.adj.items():
        #     if count < 5: # Imprimir solo las primeras 5 entradas de adyacencia
        #         res += f"Nodo {nodo}: {vecinos}\n"
        #         count +=1
        #     else:
        #         res += "...\n"
        #         break
        return res

if __name__ == '__main__':
    # Pruebas de la clase GrafoDirigido
    g = GrafoDirigido()

    # Agregar nodos y aristas
    g.agregar_arista(1, 2)
    g.agregar_arista(1, 3)
    g.agregar_arista(2, 3)
    g.agregar_arista(3, 1) # Ciclo
    g.agregar_arista(3, 4)
    g.agregar_nodo(5) # Nodo aislado

    print(g)
    print(f"Nodos: {g.obtener_todos_los_nodos()}")
    print(f"Vecinos de 1: {g.obtener_vecinos(1)}")
    print(f"Vecinos de 5 (aislado): {g.obtener_vecinos(5)}")
    print(f"Vecinos de 6 (no existe): {g.obtener_vecinos(6)}")

    print(f"Grado de salida de 1: {g.grado_salida(1)}")
    print(f"Grado de salida de 4: {g.grado_salida(4)}") # Debería ser 0
    print(f"Grado de salida de 5: {g.grado_salida(5)}") # Debería ser 0

    print(f"Grado de entrada de 1: {g.grado_entrada(1)}")
    print(f"Grado de entrada de 3: {g.grado_entrada(3)}")
    print(f"Grado de entrada de 4: {g.grado_entrada(4)}")
    print(f"Grado de entrada de 5: {g.grado_entrada(5)}")

    # Probar agregar arista duplicada
    print(f"Número de aristas antes de duplicar: {g.obtener_numero_aristas()}")
    g.agregar_arista(1, 2) # Intentar agregar arista duplicada
    print(f"Número de aristas después de intentar duplicar: {g.obtener_numero_aristas()}")
    print(f"Vecinos de 1 después de intentar duplicar: {g.obtener_vecinos(1)}")

    # Probar agregar nodo existente
    num_nodos_antes = g.obtener_numero_nodos()
    g.agregar_nodo(1)
    print(f"Número de nodos no debería cambiar: {num_nodos_antes} -> {g.obtener_numero_nodos()}")

    # Prueba con un grafo más grande simulado para grado_entrada
    # for i in range(100, 1000):
    #     g.agregar_arista(i,1) # Muchos apuntan a 1
    # print(f"Grado de entrada de 1 (después de añadir más aristas): {g.grado_entrada(1)}")


import collections
import heapq # Para el árbol de expansión mínima (Prim)
import random # Para muestreo en algunas funciones

# Nota: La clase GrafoDirigido se importa desde grafo.py en main.py o cuando se use.
from grafo import GrafoDirigido

def bfs_caminos_mas_cortos_desde_fuente(grafo, fuente):
    """
    Realiza una Búsqueda en Amplitud (BFS) para encontrar los caminos más cortos
    desde un nodo fuente a todos los demás nodos alcanzables en un grafo dirigido.

    Args:
        grafo (GrafoDirigido): El grafo sobre el cual operar.
        fuente: El nodo de inicio para el BFS.

    Returns:
        dict: Un diccionario de distancias donde las claves son los nodos alcanzables
              y los valores son la longitud del camino más corto desde la fuente.
              Si la fuente no está en el grafo, retorna un diccionario vacío.
    """
    if fuente not in grafo.nodes:
        # print(f"Advertencia: Nodo fuente {fuente} no encontrado en el grafo.")
        return {}

    distancias = {nodo: float('inf') for nodo in grafo.obtener_todos_los_nodos()}
    distancias[fuente] = 0

    cola = collections.deque([fuente])

    visitados_bfs = {fuente} # Para evitar ciclos y reprocesamiento en BFS

    while cola:
        nodo_actual = cola.popleft()

        for vecino in grafo.obtener_vecinos(nodo_actual):
            if vecino not in visitados_bfs: # Si no ha sido visitado en esta BFS
                visitados_bfs.add(vecino)
                distancias[vecino] = distancias[nodo_actual] + 1
                cola.append(vecino)
            # Si se quisiera encontrar todos los caminos (no solo el más corto) o manejar pesos,
            # la lógica aquí cambiaría. Para BFS estándar, esto es suficiente.
            # En un grafo no ponderado, la primera vez que alcanzamos un nodo es por un camino más corto.

    # Filtrar solo los nodos alcanzables (distancia finita)
    alcanzables = {nodo: dist for nodo, dist in distancias.items() if dist != float('inf')}
    return alcanzables

def longitud_promedio_camino_mas_corto_muestreo(grafo, num_muestras=1000):
    """
    Calcula la longitud promedio del camino más corto en el grafo utilizando muestreo.
    Realiza BFS desde un número de nodos fuente seleccionados aleatoriamente.

    Args:
        grafo (GrafoDirigido): El grafo.
        num_muestras (int): Número de nodos fuente a muestrear.

    Returns:
        float: La longitud promedio del camino más corto, o 0.0 si no se pueden calcular caminos.
    """
    if not grafo.obtener_numero_nodos() or not grafo.obtener_numero_aristas():
        return 0.0

    todos_los_nodos = grafo.obtener_todos_los_nodos()
    if not todos_los_nodos:
        return 0.0

    # Asegurar que num_muestras no sea mayor que el número de nodos
    num_muestras = min(num_muestras, len(todos_los_nodos))
    if num_muestras == 0:
        return 0.0

    nodos_muestra = random.sample(todos_los_nodos, num_muestras)

    longitudes_totales = 0
    pares_alcanzables_contados = 0

    print(f"Calculando longitud promedio de caminos más cortos con {num_muestras} muestras...")
    for i, fuente in enumerate(nodos_muestra):
        # print(f"  BFS desde nodo muestra {i+1}/{num_muestras}: {fuente}")
        distancias_desde_fuente = bfs_caminos_mas_cortos_desde_fuente(grafo, fuente)

        for nodo_destino, dist in distancias_desde_fuente.items():
            if nodo_destino != fuente: # No contar el camino de un nodo a sí mismo (distancia 0)
                longitudes_totales += dist
                pares_alcanzables_contados += 1

    if pares_alcanzables_contados == 0:
        print("Advertencia: No se encontraron pares alcanzables en el muestreo.")
        return 0.0

    promedio = longitudes_totales / pares_alcanzables_contados
    return promedio

# --- Implementación del Algoritmo de Louvain (simplificado) ---
# Louvain es complejo. Una implementación completa y optimizada es un gran esfuerzo.
# Aquí se presenta una versión conceptual simplificada para cumplir el requisito "manual".
# Esta versión puede no ser tan eficiente o precisa como las de librerías optimizadas.

def modularidad(grafo, particion, m_total_aristas_ponderadas=None):
    """
    Calcula la modularidad de una partición dada del grafo.
    Para grafos dirigidos, la definición de modularidad puede variar.
    Esta es una adaptación común.
    Q = (1/2m) * sum_{c in C} sum_{i,j in c} (A_ij - (k_i^out * k_j^in) / 2m)
    Donde A_ij es 1 si hay arista i->j, 0 si no.
    k_i^out es el grado de salida de i, k_j^in es el grado de entrada de j.
    2m es el doble del número total de aristas (o suma de pesos si es ponderado).

    Simplificación para grafos no ponderados:
    Q = sum_{c in C} [ (e_c / M) - ( (d_c_out * d_c_in) / (2M^2) ) ] (aproximación)
    donde e_c es el número de aristas dentro de la comunidad c,
    M es el número total de aristas en el grafo.
    d_c_out es la suma de grados de salida de nodos en c.
    d_c_in es la suma de grados de entrada de nodos en c.

    Una definición más estándar para grafos no dirigidos es:
    Q = (1/2m) * sum_{i,j} [A_ij - (k_i * k_j) / 2m] * delta(c_i, c_j)
    donde delta(c_i, c_j) es 1 si i y j están en la misma comunidad, 0 si no.
    k_i es el grado total del nodo i.

    Usaremos una versión que se enfoca en aristas internas vs esperadas.
    """
    if m_total_aristas_ponderadas is None:
        m_total_aristas_ponderadas = grafo.obtener_numero_aristas()

    if m_total_aristas_ponderadas == 0:
        return 0.0

    q = 0.0
    comunidades = {} # comunidad_id -> [nodos]
    for nodo, cid in particion.items():
        if cid not in comunidades:
            comunidades[cid] = []
        comunidades[cid].append(nodo)

    for cid, nodos_en_comunidad in comunidades.items():
        aristas_internas_comunidad = 0
        suma_grados_salida_comunidad = 0
        # suma_grados_entrada_comunidad = 0 # No se usa directamente en la fórmula simplificada de Newman para no dirigidos

        for nodo_i in nodos_en_comunidad:
            suma_grados_salida_comunidad += grafo.grado_salida(nodo_i)
            # suma_grados_entrada_comunidad += grafo.grado_entrada(nodo_i) # Para versiones dirigidas más complejas

            for vecino in grafo.obtener_vecinos(nodo_i):
                if particion.get(vecino) == cid: # Si el vecino está en la misma comunidad
                    aristas_internas_comunidad += 1

        # Para grafos dirigidos, la arista interna ya está contada una vez.
        # Para no dirigidos, se dividiría por 2 si se contaran ambas direcciones.
        # Aquí, como es dirigido, aristas_internas_comunidad es L_c.

        # Fórmula de Newman para modularidad (adaptada, considerando aristas dirigidas):
        # Q = sum_c [ (L_c / M) - ( (sum_k_out_c * sum_k_in_c) / M^2 ) ] (si se usara M^2 para normalizar grados)
        # O una más simple para no dirigidos: Q = sum_c [ (L_c / M) - ( (sum_k_c / 2M)^2 ) ]
        # donde sum_k_c es la suma de grados (salida+entrada) de nodos en c.
        # Por simplicidad y el "sin librerías", usaremos la idea de aristas internas vs. grado total de la comunidad.
        # Esta es una aproximación y podría no ser la definición canónica para grafos dirigidos.

        # Usando la fórmula Q = (1/M) * sum_c ( L_c - ( (sum_k_out_c * sum_k_in_c) / M ) )
        # donde L_c son las aristas internas.
        # Para una versión más simple: Q = sum_c [ (L_c / M) - ( (d_c / 2M)^2 ) ]
        # d_c es la suma de grados (salida) de los nodos en la comunidad.
        # Si el grafo fuera no dirigido, d_c sería la suma de grados y M el doble de aristas.
        # Como es dirigido, M es el número de aristas.

        termino_aristas_internas = aristas_internas_comunidad / m_total_aristas_ponderadas
        # El término de grados esperados para un grafo dirigido es más complejo.
        # Una aproximación: (suma_grados_salida_comunidad / (2*m_total_aristas_ponderadas)) * (suma_grados_entrada_comunidad_calculada / (2*m_total_aristas_ponderadas))
        # Para simplificar, usaremos la suma de grados de salida como proxy para el "tamaño" de la comunidad en términos de conexiones.
        termino_grados_esperados = (suma_grados_salida_comunidad / (2 * m_total_aristas_ponderadas)) ** 2

        q += (termino_aristas_internas - termino_grados_esperados)

    return q


def fase1_louvain(grafo, particion_actual):
    """
    Fase 1 del algoritmo de Louvain: Mover nodos entre comunidades para maximizar modularidad.
    Esta es una versión muy simplificada.
    """
    nodos = list(grafo.obtener_todos_los_nodos())
    random.shuffle(nodos) # Procesar nodos en orden aleatorio

    mejora_total = True
    iteraciones_fase1 = 0
    max_iter_fase1 = 10 # Límite para evitar bucles infinitos en casos difíciles

    while mejora_total and iteraciones_fase1 < max_iter_fase1:
        mejora_total = False
        iteraciones_fase1 += 1
        # print(f"    Iteración Fase 1: {iteraciones_fase1}")

        for nodo_i in nodos:
            comunidad_original_nodo_i = particion_actual[nodo_i]
            mejor_ganancia_modularidad = 0.0
            mejor_comunidad_para_nodo_i = comunidad_original_nodo_i

            # Considerar mover nodo_i a la comunidad de cada uno de sus vecinos
            # o mantenerlo en su comunidad actual.
            comunidades_a_probar = {comunidad_original_nodo_i}
            for vecino in grafo.obtener_vecinos(nodo_i): # Vecinos salientes
                comunidades_a_probar.add(particion_actual[vecino])

            # También considerar vecinos entrantes para una mejor exploración (más costoso)
            # for otro_nodo in grafo.obtener_todos_los_nodos():
            #     if nodo_i in grafo.obtener_vecinos(otro_nodo): # Si otro_nodo apunta a nodo_i
            #         comunidades_a_probar.add(particion_actual[otro_nodo])


            for cid_prueba in comunidades_a_probar:
                if cid_prueba == comunidad_original_nodo_i: # Ganancia de no moverse es 0
                    continue

                # Calcular ganancia de modularidad si nodo_i se mueve a cid_prueba
                # Esto implica calcular Q_nueva - Q_vieja.
                # Una forma eficiente es calcular el cambio local en la modularidad (delta_Q).
                # Delta_Q = [ (sum_in_new + 2*k_i_in_new) / 2m - ( (sum_tot_new + k_i_new) / 2m )^2 ] -
                #           [ (sum_in_old) / 2m - ( (sum_tot_old) / 2m )^2 - ( (k_i_old) / 2m )^2 ]
                # Esta fórmula es para grafos no dirigidos y es compleja de implementar correctamente sin librerías.

                # Simplificación: calcular modularidad completa antes y después del movimiento (ineficiente pero manual)
                q_antes = modularidad(grafo, particion_actual)
                particion_temporal = particion_actual.copy()
                particion_temporal[nodo_i] = cid_prueba
                q_despues = modularidad(grafo, particion_temporal)

                ganancia = q_despues - q_antes

                if ganancia > mejor_ganancia_modularidad:
                    mejor_ganancia_modularidad = ganancia
                    mejor_comunidad_para_nodo_i = cid_prueba

            if mejor_comunidad_para_nodo_i != comunidad_original_nodo_i and mejor_ganancia_modularidad > 0:
                particion_actual[nodo_i] = mejor_comunidad_para_nodo_i
                mejora_total = True # Hubo al menos una mejora en esta pasada

    return particion_actual, mejora_total

def fase2_louvain(grafo, particion_fase1):
    """
    Fase 2 del algoritmo de Louvain: Construir un nuevo grafo donde las comunidades son nodos.
    """
    nuevo_grafo = GrafoDirigido() # Usar nuestra clase
    mapa_comunidad_a_nuevo_nodo = {cid: i for i, cid in enumerate(set(particion_fase1.values()))}

    nueva_particion_para_nuevo_grafo = {} # La partición inicial para el nuevo grafo es cada nodo en su propia comunidad

    for cid, nuevo_nodo_id in mapa_comunidad_a_nuevo_nodo.items():
        nuevo_grafo.agregar_nodo(nuevo_nodo_id)
        nueva_particion_para_nuevo_grafo[nuevo_nodo_id] = nuevo_nodo_id # Cada nuevo nodo es su propia comunidad


    # Agregar aristas entre los nuevos nodos (comunidades)
    # El peso de una arista entre comunidad C1 y C2 es la suma de aristas entre nodos de C1 y C2.
    aristas_entre_comunidades = {} # (cid1, cid2) -> peso

    for u, vecinos_u in grafo.adj.items():
        cid_u = particion_fase1[u]
        nuevo_nodo_u = mapa_comunidad_a_nuevo_nodo[cid_u]

        for v in vecinos_u:
            cid_v = particion_fase1[v]
            nuevo_nodo_v = mapa_comunidad_a_nuevo_nodo[cid_v]

            # No agregar auto-bucles en el grafo de comunidades si no es significativo
            # if nuevo_nodo_u == nuevo_nodo_v:
            #     continue # O manejar pesos de aristas internas si es necesario

            # Para grafos no ponderados, el "peso" es el número de aristas
            # Para grafos dirigidos, la arista u->v contribuye al peso C(u)->C(v)
            par_comunidades = (nuevo_nodo_u, nuevo_nodo_v)
            aristas_entre_comunidades[par_comunidades] = aristas_entre_comunidades.get(par_comunidades, 0) + 1

    for (n_u, n_v), peso in aristas_entre_comunidades.items():
        for _ in range(peso): # Simular peso añadiendo múltiples aristas (si la clase Grafo no maneja pesos)
             nuevo_grafo.agregar_arista(n_u, n_v) # Asumimos que nuestra clase GrafoDirigido no tiene pesos
                                                 # Si los tuviera, sería nuevo_grafo.agregar_arista(n_u, n_v, peso)

    return nuevo_grafo, nueva_particion_para_nuevo_grafo, mapa_comunidad_a_nuevo_nodo


def detectar_comunidades_louvain(grafo, max_pasadas=10):
    """
    Detecta comunidades en el grafo usando el algoritmo de Louvain (versión simplificada).
    """
    if not grafo.obtener_numero_nodos():
        return {}

    # Inicialización: cada nodo es su propia comunidad
    particion_actual = {nodo: nodo for nodo in grafo.obtener_todos_los_nodos()}

    grafo_actual = grafo
    particion_para_grafo_actual = particion_actual.copy()

    mejoras_globales = True
    pasada_actual = 0

    mapeos_de_pasadas = [] # Para reconstruir la partición final

    print("Iniciando detección de comunidades Louvain (simplificado)...")
    while mejoras_globales and pasada_actual < max_pasadas:
        pasada_actual += 1
        print(f"  Pasada Louvain {pasada_actual}:")

        # Fase 1: Optimización local de la modularidad
        # print(f"    Iniciando Fase 1 en grafo con {grafo_actual.obtener_numero_nodos()} nodos...")
        particion_despues_fase1, mejora_fase1 = fase1_louvain(grafo_actual, particion_para_grafo_actual)

        if not mejora_fase1:
            # print("    No hubo mejora en la modularidad en la Fase 1. Deteniendo.")
            mejoras_globales = False
            break # Salir del bucle principal si no hay mejora

        # Actualizar la partición general con los resultados de la fase 1
        # Si es la primera pasada, particion_despues_fase1 es la partición_actual
        # Si no, necesitamos mapear las comunidades del grafo agregado a las originales
        if pasada_actual == 1:
            particion_actual = particion_despues_fase1.copy()
        else:
            # Reconstruir: mapear las comunidades del grafo_actual (que son comunidades de la pasada anterior)
            # a las comunidades encontradas en particion_despues_fase1
            particion_actual_temp = {}
            mapa_anterior = mapeos_de_pasadas[-1] # El mapeo de la pasada anterior
            for nodo_original, comunidad_nivel_anterior in particion_actual.items():
                # comunidad_nivel_anterior es un nodo en el grafo_actual (antes de la fase 2 de esta pasada)
                # particion_despues_fase1 tiene las nuevas comunidades para estos "nodos"
                nueva_comunidad_agregada = particion_despues_fase1[mapa_anterior[comunidad_nivel_anterior]]
                particion_actual_temp[nodo_original] = nueva_comunidad_agregada
            particion_actual = particion_actual_temp

        # Fase 2: Agregación de la red
        # print(f"    Iniciando Fase 2...")
        grafo_agregado, particion_para_grafo_agregado, mapa_actual = fase2_louvain(grafo_actual, particion_despues_fase1)

        mapeos_de_pasadas.append(mapa_actual) # Guardar el mapeo de esta pasada

        grafo_actual = grafo_agregado
        particion_para_grafo_actual = particion_para_grafo_agregado

        if grafo_actual.obtener_numero_nodos() == len(set(particion_para_grafo_actual.values())):
            # Si cada nodo en el grafo agregado es su propia comunidad y no hubo cambios en fase 1,
            # o si el número de nodos agregados es muy pequeño.
            # print("    Convergencia: El grafo agregado no puede reducirse más o no hubo mejora.")
            mejoras_globales = False # Podría ser redundante si mejora_fase1 ya fue False
            break

    print(f"Louvain completado después de {pasada_actual} pasadas.")

    # Reconstrucción final de la partición para el grafo original
    # Si mapeos_de_pasadas está vacío (solo una pasada, sin agregación efectiva o solo fase 1)
    if not mapeos_de_pasadas:
         return particion_actual # O particion_despues_fase1 si fue la única fase hecha

    # La particion_actual ya debería estar mapeada a los IDs de comunidad del último nivel de agregación.
    # Necesitamos asegurarnos de que los IDs de comunidad sean únicos y secuenciales si es necesario.
    # La particion_actual ya contiene los IDs de comunidad finales para los nodos originales.

    # Renumerar comunidades para que sean 0, 1, 2...
    comunidades_finales_unicas = sorted(list(set(particion_actual.values())))
    mapa_renumeracion = {cid_viejo: i for i, cid_viejo in enumerate(comunidades_finales_unicas)}

    particion_renumerada = {nodo: mapa_renumeracion[cid] for nodo, cid in particion_actual.items()}

    return particion_renumerada


# --- Árbol de Expansión Mínima (Prim) ---
# MST es para grafos no dirigidos y ponderados.
# Para un grafo dirigido, el concepto análogo es un "árbol de arborescencia mínima" o "spanning arborescence".
# El algoritmo de Edmonds es común para esto, pero es más complejo.
# Si se pide un MST "clásico", usualmente se convierte el grafo a no dirigido
# y se asignan pesos (si no los hay, se asume peso 1 para todas las aristas).

def arbol_expansion_minima_prim(grafo):
    """
    Encuentra un Árbol de Expansión Mínima (MST) usando el algoritmo de Prim.
    Asume que el grafo es no dirigido o se tratará como tal, y las aristas tienen peso 1.
    Si el grafo no es conexo, encontrará un MST para el componente conexo del nodo de inicio.

    Args:
        grafo (GrafoDirigido): El grafo. Se tratarán las aristas como no dirigidas con peso 1.

    Returns:
        GrafoDirigido: Un nuevo grafo que representa el MST.
                       Retorna un grafo vacío si el grafo original está vacío.
    """
    if not grafo.obtener_numero_nodos():
        return GrafoDirigido()

    mst = GrafoDirigido()
    nodos_en_mst = set()

    # Elegir un nodo de inicio arbitrario
    todos_los_nodos = grafo.obtener_todos_los_nodos()
    if not todos_los_nodos:
        return GrafoDirigido()
    nodo_inicial = todos_los_nodos[0]

    # (peso, nodo_origen, nodo_destino)
    # Para Prim, el nodo_origen es el nodo ya en el MST, nodo_destino es el que se considera agregar.
    # Como los pesos son 1, podemos simplificar.
    # heapq almacenará tuplas (peso, nodo_actual, nodo_adyacente_fuera_del_mst)
    # o más bien (peso, nodo_adyacente_fuera_del_mst, nodo_origen_en_mst) para reconstruir la arista

    # Min-heap de aristas candidatas: (peso, nodo_destino, nodo_origen_en_mst)
    aristas_candidatas = []

    nodos_en_mst.add(nodo_inicial)
    mst.agregar_nodo(nodo_inicial)

    # Añadir todas las aristas salientes y entrantes (tratando como no dirigido) desde el nodo inicial
    # Aristas salientes
    for vecino in grafo.obtener_vecinos(nodo_inicial):
        heapq.heappush(aristas_candidatas, (1, vecino, nodo_inicial)) # (peso=1, destino, origen)
    # Aristas entrantes
    for nodo_u in todos_los_nodos:
        if nodo_u != nodo_inicial and nodo_inicial in grafo.obtener_vecinos(nodo_u):
            heapq.heappush(aristas_candidatas, (1, nodo_inicial, nodo_u)) # Esto no es correcto para Prim estándar
                                                                        # Debería ser (1, nodo_u, nodo_inicial)
                                                                        # si nodo_u es el que se agrega.
            # Corrección: si nodo_u es el que se considera agregar y nodo_inicial está en MST
            heapq.heappush(aristas_candidatas, (1, nodo_u, nodo_inicial))


    print("Construyendo MST con Prim (adaptado)...")
    while aristas_candidatas and len(nodos_en_mst) < grafo.obtener_numero_nodos():
        peso, u, v = heapq.heappop(aristas_candidatas) # v está en MST, u es candidato

        # Si u ya está en el MST, esta arista formaría un ciclo (o es una arista a un nodo ya procesado)
        if u not in nodos_en_mst:
            nodos_en_mst.add(u)
            nodos_en_mst.add(u)
            mst.agregar_nodo(u)
            # Para Prim, la arista es entre v (en MST) y u (nuevo).
            # Si el MST debe ser no dirigido, la clase GrafoDirigido necesitaría
            # un método para agregar aristas no dirigidas o se usaría una clase GrafoNoDirigido.
            # Por ahora, añadimos v -> u, asumiendo que la estructura del MST es la conectividad.
            mst.agregar_arista(v, u)

            # Añadir nuevas aristas candidatas desde el nodo recién agregado 'u'
            # Considerando el grafo como no dirigido para Prim:
            # 1. Aristas salientes desde u: (u, vecino_de_u)
            for vecino_de_u in grafo.obtener_vecinos(u): # u -> vecino_de_u
                if vecino_de_u not in nodos_en_mst:
                    heapq.heappush(aristas_candidatas, (1, vecino_de_u, u)) # (peso, nodo_nuevo, nodo_en_mst)

            # 2. Aristas entrantes a u: (otro_nodo, u)
            # Necesitamos iterar sobre todos los nodos para encontrar los que apuntan a 'u'
            # Esto es costoso. Una lista de adyacencia inversa sería útil aquí.
            # Si grafo.adj_inversa existe:
            # for nodo_origen_de_u in grafo.adj_inversa.get(u, []):
            #    if nodo_origen_de_u not in nodos_en_mst:
            #        heapq.heappush(aristas_candidatas, (1, nodo_origen_de_u, u)) # Incorrecto, u es el nuevo
            #                                                                    # Debería ser (1, u, nodo_origen_de_u) si nodo_origen_de_u está en MST
            #                                                                    # O (1, nodo_origen_de_u, u) si nodo_origen_de_u es el candidato

            # Alternativa para simular no dirigido: iterar todos los nodos y sus listas de adyacencia
            for nodo_potencial_en_grafo in todos_los_nodos:
                if nodo_potencial_en_grafo not in nodos_en_mst: # Si el nodo_potencial_en_grafo aún no está en el MST
                    # Chequear arista (u, nodo_potencial_en_grafo) - ya cubierto arriba si es u->vecino
                    # Chequear arista (nodo_potencial_en_grafo, u)
                    if u in grafo.obtener_vecinos(nodo_potencial_en_grafo): # Si nodo_potencial_en_grafo -> u
                        # u está en MST, nodo_potencial_en_grafo es el candidato
                        heapq.heappush(aristas_candidatas, (1, nodo_potencial_en_grafo, u))

        # Si el grafo no es conexo, Prim solo encuentra el MST del componente del nodo_inicial.
        # Para un bosque de expansión mínima, se necesitaría reiniciar Prim en nodos no visitados.
        # La lógica actual se detendrá cuando el primer componente esté completo.
        if not aristas_candidatas and len(nodos_en_mst) < grafo.obtener_numero_nodos():
            # print(f"  MST: Componente actual con {len(nodos_en_mst)} nodos. Buscando nuevo componente...")
            for nodo_restante in todos_los_nodos:
                if nodo_restante not in nodos_en_mst:
                    # print(f"    Iniciando nuevo MST desde {nodo_restante}")
                    nodos_en_mst.add(nodo_restante)
                    mst.agregar_nodo(nodo_restante)
                    # Añadir aristas desde este nuevo nodo inicial para el nuevo componente
                    for vecino in grafo.obtener_vecinos(nodo_restante):
                        if vecino not in nodos_en_mst: # Aunque no debería estar si es un nuevo componente
                            heapq.heappush(aristas_candidatas, (1, vecino, nodo_restante))
                    for n_orig in todos_los_nodos: # Para aristas entrantes
                        if n_orig not in nodos_en_mst:
                             if nodo_restante in grafo.obtener_vecinos(n_orig):
                                heapq.heappush(aristas_candidatas, (1, n_orig, nodo_restante)) # n_orig candidato, nodo_restante en MST
                    break # Salir del bucle de nodo_restante y continuar el while principal de Prim
            if not aristas_candidatas: # Si después de buscar no hay más inicios posibles
                 break

    num_nodos_final_mst = mst.obtener_numero_nodos()
    num_aristas_final_mst = mst.obtener_numero_aristas()
    print(f"MST (o bosque) construido con {num_nodos_final_mst} nodos y {num_aristas_final_mst} aristas.")

    # Verificación: Un bosque con C componentes y N nodos en total debería tener N-C aristas.
    # Esto es más difícil de verificar aquí sin conocer C explícitamente.
    # Si el grafo original es conexo, el MST debe tener N-1 aristas.
    if num_nodos_final_mst > 0 and num_aristas_final_mst < num_nodos_final_mst -1 :
        pass # Podría ser un bosque
        # print(f"Advertencia: El MST parece tener menos de N-1 aristas ({num_aristas_final_mst} vs {num_nodos_final_mst-1}). Puede ser un bosque si el grafo no era conexo.")
    elif num_nodos_final_mst > 0 and num_aristas_final_mst > num_nodos_final_mst -1:
         print(f"Advertencia: El MST parece tener más de N-1 aristas ({num_aristas_final_mst} vs {num_nodos_final_mst-1}). Esto no debería ocurrir en un MST simple.")


    return mst


if __name__ == '__main__':
    # --- Pruebas para los algoritmos ---
    from grafo import GrafoDirigido # Importar aquí para pruebas locales

    # 1. Prueba de BFS y Longitud Promedio de Camino
    print("\n--- Prueba BFS y Longitud Promedio ---")
    g_bfs = GrafoDirigido()
    g_bfs.agregar_arista(1, 2)
    g_bfs.agregar_arista(1, 3)
    g_bfs.agregar_arista(2, 4)
    g_bfs.agregar_arista(3, 4)
    g_bfs.agregar_arista(4, 5)
    g_bfs.agregar_nodo(6) # Nodo aislado

    dist_desde_1 = bfs_caminos_mas_cortos_desde_fuente(g_bfs, 1)
    print(f"Distancias desde nodo 1: {dist_desde_1}")
    # Esperado: {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}

    dist_desde_6 = bfs_caminos_mas_cortos_desde_fuente(g_bfs, 6)
    print(f"Distancias desde nodo 6 (aislado): {dist_desde_6}")
    # Esperado: {6: 0}

    dist_desde_inexistente = bfs_caminos_mas_cortos_desde_fuente(g_bfs, 7)
    print(f"Distancias desde nodo 7 (no existe): {dist_desde_inexistente}")
    # Esperado: {}

    promedio_camino = longitud_promedio_camino_mas_corto_muestreo(g_bfs, num_muestras=g_bfs.obtener_numero_nodos())
    print(f"Longitud promedio del camino más corto (muestra completa): {promedio_camino:.2f}")
    # Caminos:
    # 1->2 (1), 1->3 (1), 1->4 (2), 1->5 (3)
    # 2->4 (1), 2->5 (2)
    # 3->4 (1), 3->5 (2)
    # 4->5 (1)
    # Suma = 1+1+2+3 + 1+2 + 1+2 + 1 = 14. Pares = 9. Promedio = 14/9 = 1.55

    # 2. Prueba de Louvain (muy simplificada)
    # print("\n--- Prueba Louvain (simplificada) ---")
    # g_louvain = GrafoDirigido()
    # g_louvain.agregar_arista(1,2); g_louvain.agregar_arista(2,1) # Com1
    # g_louvain.agregar_arista(1,3); g_louvain.agregar_arista(3,1) # Com1
    # g_louvain.agregar_arista(2,3); g_louvain.agregar_arista(3,2) # Com1
    # g_louvain.agregar_arista(4,5); g_louvain.agregar_arista(5,4) # Com2
    # g_louvain.agregar_arista(4,6); g_louvain.agregar_arista(6,4) # Com2
    # g_louvain.agregar_arista(5,6); g_louvain.agregar_arista(6,5) # Com2
    # g_louvain.agregar_arista(3,4) # Puente
    # g_louvain.agregar_nodo(7) # Aislado

    # particion_inicial_louvain = {n:n for n in g_louvain.obtener_todos_los_nodos()}
    # q_inicial = modularidad(g_louvain, particion_inicial_louvain)
    # print(f"Modularidad inicial (cada nodo es una comunidad): {q_inicial:.4f}")

    # comunidades_louvain = detectar_comunidades_louvain(g_louvain, max_pasadas=5)
    # print(f"Partición Louvain final: {comunidades_louvain}")
    # q_final = modularidad(g_louvain, comunidades_louvain)
    # print(f"Modularidad final: {q_final:.4f}")
    # # Esperado (idealmente): {1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:2} o similar con IDs diferentes

    # 3. Prueba de MST (Prim)
    print("\n--- Prueba MST (Prim adaptado) ---")
    g_mst = GrafoDirigido()
    # Componente 1
    g_mst.agregar_arista(1,2); g_mst.agregar_arista(2,1)
    g_mst.agregar_arista(1,3); g_mst.agregar_arista(3,1)
    g_mst.agregar_arista(2,3); g_mst.agregar_arista(3,2)
    g_mst.agregar_arista(2,4); g_mst.agregar_arista(4,2)
    # Componente 2 (no conectado al primero directamente por aristas bidireccionales para Prim simple)
    g_mst.agregar_arista(5,6); g_mst.agregar_arista(6,5)
    g_mst.agregar_arista(7,7) # Auto-bucle, no debería estar en MST

    mst_result = arbol_expansion_minima_prim(g_mst)
    print("MST resultante:")
    for nodo in mst_result.obtener_todos_los_nodos():
        print(f"  Nodo {nodo} en MST, vecinos_mst: {mst_result.obtener_vecinos(nodo)}")
    # Esperado para el componente de 1: (1,2), (1,3) o (2,1),(2,3) etc. 3 aristas para 4 nodos.
    # Y para el componente de 5: (5,6). 1 arista para 2 nodos.
    # Total aristas en el bosque = (N1-1) + (N2-1) = 3 + 1 = 4 (si Prim se adapta para bosques)
    # Esta implementación simple de Prim encontrará el MST del componente del nodo inicial.
    # Si el grafo no es conexo, solo se procesa un componente.
    # La salida actual de aristas en MST podría ser (v->u) donde v estaba en MST y u se añadió.
    # MST resultante:
    #   Nodo 1 en MST, vecinos_mst: []
    #   Nodo 2 en MST, vecinos_mst: [1]
    #   Nodo 3 en MST, vecinos_mst: [1]
    #   Nodo 4 en MST, vecinos_mst: [2]
    #   (El componente 5,6,7 no se alcanzaría con el Prim simple desde nodo 1)
    # La adaptación para manejar múltiples componentes (bosque) es más compleja para Prim.
    # Kruskal es naturalmente mejor para encontrar bosques.
    # La implementación actual de Prim encontrará un MST para el componente conexo del nodo inicial.
    # Re-ejecutando Prim para nodos no visitados es una forma de obtener un bosque.
    # La lógica actual de Prim solo construye para el primer componente.
    print(f"MST Nodos: {mst_result.obtener_numero_nodos()}, Aristas: {mst_result.obtener_numero_aristas()}")

    print("\nPruebas de algoritmos_grafos completadas.")


# Instalar las librerías necesarias
!pip install dash dash-core-components dash-html-components plotly networkx numpy pandas
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import networkx as nx
import random
import os
import time
from collections import defaultdict

# Configuración
MAX_NODES = 5000  # Límite de nodos para carga
COORDS_FILE = '10_million_location.txt'
NEIGHBORS_FILE = '10_million_user.txt'

# Crear archivos dummy si no existen
def create_dummy_files():
    if not os.path.exists(COORDS_FILE):
        print(f"Creando archivo dummy de ubicaciones: {COORDS_FILE}")
        with open(COORDS_FILE, 'w') as f:
            for i in range(1, MAX_NODES + 1):
                f.write(f"{random.uniform(0, 100)},{random.uniform(0, 100)}\n")

    if not os.path.exists(NEIGHBORS_FILE):
        print(f"Creando archivo dummy de conexiones: {NEIGHBORS_FILE}")
        with open(NEIGHBORS_FILE, 'w') as f:
            for i in range(1, MAX_NODES + 1):
                neighbors = random.sample(range(1, MAX_NODES + 1), random.randint(1, 5))
                f.write(','.join(map(str, neighbors)) + '\n')

# Cargar coordenadas con límite
def load_coordinates(file_path, limit=MAX_NODES):
    coords = {}
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            if idx >= limit:
                break
            try:
                x, y = map(float, line.strip().split(','))
                coords[idx + 1] = (x, y)
            except:
                continue
    return coords

# Cargar vecinos con límite
def load_neighbors(file_path, limit=MAX_NODES):
    neighbors = {}
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            if idx >= limit:
                break
            try:
                parts = list(map(int, line.strip().replace(',', ' ').split()))
                neighbors[idx + 1] = parts
            except:
                continue
    return neighbors

# Construir grafo
def build_graph(coords, neighbors):
    G = nx.DiGraph()

    # Añadir nodos con coordenadas
    for node, pos in coords.items():
        G.add_node(node, pos=pos)

    # Añadir aristas
    for node, nbrs in neighbors.items():
        if node in G:
            for nbr in nbrs:
                if nbr in G:
                    G.add_edge(node, nbr, weight=1)
    return G

# Calcular métricas para una componente
def calculate_metrics(subgraph):
    metrics = {}

    # Métricas básicas
    metrics['num_nodos'] = subgraph.number_of_nodes()
    metrics['num_aristas'] = subgraph.number_of_edges()
    metrics['densidad'] = nx.density(subgraph)

    # Grado medio
    degrees = dict(subgraph.degree())
    metrics['grado_medio'] = sum(degrees.values()) / metrics['num_nodos']

    # Camino más corto promedio
    try:
        metrics['camino_promedio'] = nx.average_shortest_path_length(subgraph)
    except:
        metrics['camino_promedio'] = "No calculable (grafo no conectado)"

    # Diámetro
    try:
        metrics['diametro'] = nx.diameter(subgraph)
    except:
        metrics['diametro'] = "No calculable"

    # Coeficiente de clustering
    try:
        metrics['clustering'] = nx.average_clustering(subgraph)
    except:
        metrics['clustering'] = "No calculable"

    # Árbol de expansión mínima
    try:
        undirected = subgraph.to_undirected()
        mst = nx.minimum_spanning_tree(undirected)
        metrics['mst_peso'] = sum(data['weight'] for _, _, data in mst.edges(data=True))
    except:
        metrics['mst_peso'] = "No calculable"

    return metrics

# Crear aplicación Dash
app = dash.Dash(__name__)
app.title = "Análisis de Red Social X"

# Layout de la aplicación
app.layout = html.Div([
    html.Div([
        html.H1("Análisis de Red Social 'X'", style={'textAlign': 'center'}),
        html.P("Visualización interactiva de la red social y sus componentes conectadas",
               style={'textAlign': 'center', 'marginBottom': '30px'})
    ]),

    html.Div([
        html.Div([
            html.H3("Selección de Componente"),
            dcc.Dropdown(
                id='component-selector',
                placeholder="Seleccione una componente conectada",
                style={'width': '100%', 'marginBottom': '20px'}
            ),
            html.Div(id='component-info', style={'marginBottom': '20px'})
        ], style={'width': '30%', 'padding': '20px', 'borderRight': '1px solid #ddd'}),

        html.Div([
            dcc.Graph(
                id='graph',
                config={'scrollZoom': True},
                style={'height': '70vh'}
            )
        ], style={'width': '70%', 'padding': '20px'})
    ], style={'display': 'flex'}),

    html.Div([
        html.H3("Métricas de la Red"),
        html.Div(id='analysis-output')
    ], style={'marginTop': '30px', 'padding': '20px', 'borderTop': '1px solid #ddd'}),

    dcc.Store(id='graph-store'),  # Para almacenar el grafo completo
    dcc.Store(id='components-store')  # Para almacenar las componentes
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': '0 auto'})

# Callback para cargar datos y construir el grafo
@app.callback(
    [Output('graph-store', 'data'),
     Output('components-store', 'data'),
     Output('component-selector', 'options')],
    [Input('graph-store', 'data')]  # Disparador inicial
)
def load_graph_data(_):
    start_time = time.time()

    # Crear archivos dummy si no existen
    create_dummy_files()

    # Cargar datos
    coords = load_coordinates(COORDS_FILE)
    neighbors = load_neighbors(NEIGHBORS_FILE)

    # Construir grafo
    G = build_graph(coords, neighbors)

    # Calcular componentes débilmente conectadas
    weakly_connected = list(nx.weakly_connected_components(G))

    # Ordenar componentes por tamaño
    weakly_connected.sort(key=len, reverse=True)

    # Crear opciones para el dropdown
    options = [{'label': f'Componente {i+1} ({len(comp)} nodos)', 'value': i}
               for i, comp in enumerate(weakly_connected)]

    # Almacenar datos
    graph_data = {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges(data=True))
    }

    components_data = [
        {'id': i, 'nodes': list(comp)}
        for i, comp in enumerate(weakly_connected)
    ]

    print(f"Tiempo de carga: {time.time() - start_time:.2f} segundos")
    return graph_data, components_data, options

# Callback para actualizar la visualización según la selección
@app.callback(
    [Output('graph', 'figure'),
     Output('analysis-output', 'children'),
     Output('component-info', 'children')],
    [Input('component-selector', 'value'),
     Input('graph-store', 'data'),
     Input('components-store', 'data')]
)
def update_graph(selected_component, graph_data, components_data):
    if selected_component is None or not graph_data or not components_data:
        return go.Figure(), "Seleccione una componente para análisis.", ""

    # Reconstruir el grafo completo
    G = nx.DiGraph()
    for node, data in graph_data['nodes']:
        G.add_node(node, **data)
    for u, v, data in graph_data['edges']:
        G.add_edge(u, v, **data)

    # Obtener la componente seleccionada
    comp_nodes = components_data[selected_component]['nodes']
    subgraph = G.subgraph(comp_nodes)

    # Extraer posiciones
    positions = nx.get_node_attributes(subgraph, 'pos')

    # Calcular layout si no hay posiciones
    if not positions:
        positions = nx.spring_layout(subgraph, seed=42)
        for node in subgraph.nodes():
            subgraph.nodes[node]['pos'] = positions[node]

    # Crear trazos de aristas
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scattergl(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Crear trazos de nodos
    node_x = []
    node_y = []
    node_text = []
    for node in subgraph.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Nodo {node}')

    node_trace = go.Scattergl(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=7,
            colorbar=dict(
                thickness=15,
                title='Grado',
                xanchor='left',
                titleside='right'
            ),
            line_width=1)
    )

    # Calcular grado para colorear
    degrees = dict(subgraph.degree())
    node_trace.marker.color = [degrees[node] for node in subgraph.nodes()]

    # Crear figura
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Componente {selected_component + 1}',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        height=600
                    ))

    # Calcular métricas
    metrics = calculate_metrics(subgraph)

    # Crear tabla de métricas
    metrics_table = html.Table(
        [html.Tr([html.Th("Métrica"), html.Th("Valor")])] +
        [html.Tr([html.Td(k), html.Td(str(v))]) for k, v in metrics.items()],
        style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'}
    )

    # Información de la componente
    comp_info = html.Div([
        html.H4(f"Componente {selected_component + 1}"),
        html.P(f"Nodos: {metrics['num_nodos']}, Aristas: {metrics['num_aristas']}"),
        html.P(f"Densidad: {metrics['densidad']:.4f}, Grado medio: {metrics['grado_medio']:.2f}")
    ])

    return fig, metrics_table, comp_info

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)


# Análisis de Red Social "X" - Versión Corregida y Optimizada
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import networkx as nx
import random
import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# Configuración
MAX_NODES = 1000  # Reducido para mejor rendimiento
COORDS_FILE = '10_million_location.txt'
NEIGHBORS_FILE = '10_million_user.txt'
MAX_VISUALIZATION_NODES = 300  # Límite para visualización

# Crear archivos dummy si sno existen
def create_dummy_files():
    if not os.path.exists(COORDS_FILE):
        print(f"Creando archivo dummy de ubicaciones: {COORDS_FILE}")
        with open(COORDS_FILE, 'w') as f:
            for i in range(1, MAX_NODES + 1):
                # Crear clusters geográficos para datos más realistas
                if i < MAX_NODES * 0.3:
                    f.write(f"{random.gauss(10, 2)},{random.gauss(15, 2)}\n")  # Región 1
                elif i < MAX_NODES * 0.6:
                    f.write(f"{random.gauss(30, 3)},{random.gauss(10, 3)}\n")  # Región 2
                else:
                    f.write(f"{random.gauss(20, 4)},{random.gauss(30, 4)}\n")  # Región 3

    if not os.path.exists(NEIGHBORS_FILE):
        print(f"Creando archivo dummy de conexiones: {NEIGHBORS_FILE}")

        # Crear una estructura de red más realista (mundo pequeño)
        base_graph = nx.watts_strogatz_graph(MAX_NODES, k=6, p=0.1, seed=42)
        connections = {}

        for i in range(1, MAX_NODES + 1):
            # Obtener vecinos de la estructura base
            neighbors = list(base_graph.neighbors(i-1))
            # Convertir índices a 1-based y añadir algunas conexiones aleatorias
            connections[i] = [n+1 for n in neighbors] + random.sample(
                range(1, MAX_NODES+1),
                random.randint(0, 2)
            )
            # Eliminar duplicados y auto-conexiones
            connections[i] = list(set([n for n in connections[i] if n != i]))

        with open(NEIGHBORS_FILE, 'w') as f:
            for i in range(1, MAX_NODES + 1):
                f.write(','.join(map(str, connections.get(i, []))) + '\n')

# Cargar coordenadas con límite
def load_coordinates(file_path, limit=MAX_NODES):
    coords = {}
    try:
        with open(file_path, 'r') as file:
            for idx, line in enumerate(file):
                if idx >= limit:
                    break
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        x = float(parts[0].strip())
                        y = float(parts[1].strip())
                        coords[idx + 1] = (x, y)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error cargando coordenadas: {e}")
    return coords

# Cargar vecinos con límite (versión optimizada)
def load_neighbors(file_path, limit=MAX_NODES):
    neighbors = {}
    try:
        with open(file_path, 'r') as file:
            for idx, line in enumerate(file):
                if idx >= limit:
                    break
                parts = line.strip().replace(',', ' ').split()
                if parts:
                    try:
                        neighbors[idx + 1] = list(map(int, parts))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error cargando vecinos: {e}")
    return neighbors

# Construir grafo (versión optimizada)
def build_graph(coords, neighbors):
    G = nx.DiGraph()

    # Añadir nodos con coordenadas
    for node in coords:
        G.add_node(node, pos=coords[node])

    # Añadir aristas en batch
    edges_to_add = []
    for node, nbrs in neighbors.items():
        if node in G.nodes:
            for nbr in nbrs:
                if nbr in G.nodes:
                    edges_to_add.append((node, nbr, {'weight': 1}))

    G.add_edges_from(edges_to_add)
    return G

# Calcular métricas para una componente (versión optimizada)
def calculate_metrics(subgraph):
    if subgraph.number_of_nodes() == 0:
        return {}

    metrics = {
        'Número de nodos': subgraph.number_of_nodes(),
        'Número de aristas': subgraph.number_of_edges(),
        'Densidad': f"{nx.density(subgraph):.4f}",
    }

    # Grado medio
    degrees = dict(subgraph.degree())
    metrics['Grado medio'] = f"{sum(degrees.values()) / metrics['Número de nodos']:.2f}"

    # Camino más corto promedio (muestreo para grandes grafos)
    if metrics['Número de nodos'] > 300:
        sample_size = min(50, metrics['Número de nodos'])
        sample_nodes = random.sample(list(subgraph.nodes()), sample_size)
        total_paths = 0
        path_count = 0

        for node in sample_nodes:
            try:
                paths = nx.single_source_shortest_path_length(subgraph, node)
                for target, length in paths.items():
                    if target != node:
                        total_paths += length
                        path_count += 1
            except:
                continue

        if path_count > 0:
            metrics['Camino promedio'] = f"{total_paths / path_count:.2f}"
        else:
            metrics['Camino promedio'] = "No calculable"
    else:
        try:
            metrics['Camino promedio'] = f"{nx.average_shortest_path_length(subgraph):.2f}"
        except:
            metrics['Camino promedio'] = "No calculable"

    # Coeficiente de clustering
    try:
        metrics['Coef. clustering'] = f"{nx.average_clustering(subgraph):.4f}"
    except:
        metrics['Coef. clustering'] = "No calculable"

    # Diámetro estimado
    try:
        if metrics['Número de nodos'] > 300:
            metrics['Diámetro'] = f"≈{nx.approximation.diameter(subgraph)}"
        else:
            metrics['Diámetro'] = nx.diameter(subgraph.to_undirected())
    except:
        metrics['Diámetro'] = "No calculable"

    return metrics

# Crear aplicación Dash
app = dash.Dash(__name__)
app.title = "Análisis de Red Social X"

# Layout de la aplicación mejorado
app.layout = html.Div([
    html.Div([
        html.H1("Análisis de Red Social 'X'", style={
            'textAlign': 'center',
            'color': '#2c3e50',
            'marginBottom': '10px'
        }),
        html.P("Visualización interactiva de la red social y sus componentes conectadas", style={
            'textAlign': 'center',
            'color': '#7f8c8d',
            'marginBottom': '30px'
        }),
        html.Div(id='status', style={
            'textAlign': 'center',
            'padding': '10px',
            'backgroundColor': '#f8f9fa',
            'marginBottom': '20px'
        })
    ]),

    html.Div([
        html.Div([
            html.H3("Selección de Componente", style={'color': '#2c3e50'}),
            dcc.Dropdown(
                id='component-selector',
                placeholder="Seleccione una componente conectada",
                style={'width': '100%', 'marginBottom': '20px'}
            ),
            html.Div(id='component-info', style={
                'marginBottom': '20px',
                'padding': '15px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px'
            }),
            html.Button('Cargar Datos', id='load-button', n_clicks=0, style={
                'backgroundColor': '#3498db',
                'color': 'white',
                'border': 'none',
                'padding': '10px 15px',
                'borderRadius': '5px',
                'width': '100%'
            })
        ], style={'width': '30%', 'padding': '20px', 'borderRight': '1px solid #ddd'}),

        html.Div([
            dcc.Graph(
                id='graph',
                config={'scrollZoom': True, 'displayModeBar': True},
                style={'height': '70vh', 'border': '1px solid #ddd', 'borderRadius': '5px'}
            )
        ], style={'width': '70%', 'padding': '20px'})
    ], style={'display': 'flex', 'marginBottom': '30px'}),

    html.Div([
        html.H3("Métricas de la Red", style={'color': '#2c3e50'}),
        html.Div(id='analysis-output', style={
            'marginTop': '15px',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px'
        })
    ], style={'padding': '20px', 'borderTop': '1px solid #ddd'}),

    dcc.Store(id='graph-store'),
    dcc.Store(id='components-store'),
    dcc.Store(id='metrics-store')
], style={
    'fontFamily': 'Arial, sans-serif',
    'maxWidth': '1400px',
    'margin': '0 auto',
    'backgroundColor': 'white',
    'padding': '20px'
})

# Callback para cargar datos y construir el grafo
@app.callback(
    [Output('graph-store', 'data'),
     Output('components-store', 'data'),
     Output('component-selector', 'options'),
     Output('status', 'children')],
    [Input('load-button', 'n_clicks')],
    prevent_initial_call=True
)
def load_graph_data(n_clicks):
    if n_clicks == 0:
        return dash.no_update, dash.no_update, dash.no_update, "Presione 'Cargar Datos' para iniciar"

    start_time = time.time()

    # Crear archivos dummy si no existen
    create_dummy_files()

    # Cargar datos
    coords = load_coordinates(COORDS_FILE)
    neighbors = load_neighbors(NEIGHBORS_FILE)

    # Construir grafo
    G = build_graph(coords, neighbors)

    # Calcular componentes débilmente conectadas
    weakly_connected = list(nx.weakly_connected_components(G))

    # Ordenar componentes por tamaño y limitar a las 5 más grandes
    weakly_connected.sort(key=len, reverse=True)
    weakly_connected = weakly_connected[:5]

    # Crear opciones para el dropdown
    options = [{'label': f'Componente {i+1} ({len(comp)} nodos)', 'value': i}
               for i, comp in enumerate(weakly_connected)]

    # Almacenar datos
    graph_data = {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges(data=True))
    }

    components_data = [
        {'id': i, 'nodes': list(comp)}
        for i, comp in enumerate(weakly_connected)
    ]

    load_time = time.time() - start_time
    status = f"Datos cargados en {load_time:.2f} segundos | {G.number_of_nodes()} nodos | {G.number_of_edges()} aristas"

    return graph_data, components_data, options, status

# Callback para actualizar la visualización según la selección
@app.callback(
    [Output('graph', 'figure'),
     Output('analysis-output', 'children'),
     Output('component-info', 'children')],
    [Input('component-selector', 'value'),
     Input('graph-store', 'data'),
     Input('components-store', 'data')]
)
def update_graph(selected_component, graph_data, components_data):
    if selected_component is None or not graph_data or not components_data:
        return go.Figure(), "Seleccione una componente para análisis.", ""

    # Reconstruir el grafo completo
    G = nx.DiGraph()
    for node, data in graph_data['nodes']:
        G.add_node(node, **data)
    for u, v, data in graph_data['edges']:
        G.add_edge(u, v, **data)

    # Obtener la componente seleccionada
    comp_nodes = components_data[selected_component]['nodes']
    subgraph = G.subgraph(comp_nodes)

    # Limitar nodos para visualización
    if len(subgraph.nodes()) > MAX_VISUALIZATION_NODES:
        sample_nodes = random.sample(list(subgraph.nodes()), MAX_VISUALIZATION_NODES)
        subgraph = subgraph.subgraph(sample_nodes)

    # Extraer posiciones
    positions = nx.get_node_attributes(subgraph, 'pos')

    # Calcular layout si no hay posiciones o no son adecuadas
    if not positions or len(positions) != len(subgraph.nodes()):
        positions = nx.spring_layout(subgraph, seed=42)
        for node in subgraph.nodes():
            subgraph.nodes[node]['pos'] = positions[node]

    # Crear trazos de aristas
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        if edge[0] in positions and edge[1] in positions:
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Crear trazos de nodos
    node_x = []
    node_y = []
    node_text = []
    for node in subgraph.nodes():
        if node in positions:
            x, y = positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'Nodo {node}')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=8,
            colorbar=dict(
                thickness=15,
                title='Grado',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=1, color='DarkSlateGrey'))
    )

    # Calcular grado para colorear
    degrees = dict(subgraph.degree())
    node_trace.marker.color = [degrees.get(node, 0) for node in subgraph.nodes()]

    # Crear figura
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Componente {selected_component + 1} - {len(subgraph.nodes())} nodos',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    ))

    # Calcular métricas
    metrics = calculate_metrics(subgraph)

    # Crear tabla de métricas
    metrics_table = []
    for k, v in metrics.items():
        metrics_table.append(html.Tr([
            html.Td(k, style={'padding': '8px', 'borderBottom': '1px solid #ddd', 'fontWeight': 'bold'}),
            html.Td(v, style={'padding': '8px', 'borderBottom': '1px solid #ddd'})
        ]))

    metrics_div = html.Table(
        html.Tr([html.Th("Métrica", style={'textAlign': 'left'}), html.Th("Valor", style={'textAlign': 'left'})]) + metrics_table,
        style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'}
    )

    # Información de la componente
    comp_info = html.Div([
        html.H4(f"Componente {selected_component + 1}"),
        html.P(f"🔵 Nodos: {metrics.get('Número de nodos', 'N/A')}"),
        html.P(f"🔶 Aristas: {metrics.get('Número de aristas', 'N/A')}"),
        html.P(f"📏 Densidad: {metrics.get('Densidad', 'N/A')}"),
        html.P(f"📊 Grado medio: {metrics.get('Grado medio', 'N/A')}")
    ])

    return fig, metrics_div, comp_info
# Ejecutar la aplicación con el método correcto
if __name__ == '__main__':
    app.run(debug=True, port=8050)  # Puedes usar 8859 si prefieres

# Análisis de Red Social "X" - Versión Corregida y Optimizada
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import networkx as nx
import random
import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# Configuración
MAX_NODES = 1000  # Reducido para mejor rendimiento
COORDS_FILE = '10_million_location.txt'
NEIGHBORS_FILE = '10_million_user.txt'
MAX_VISUALIZATION_NODES = 300  # Límite para visualización

# Crear archivos dummy si sno existen
def create_dummy_files():
    if not os.path.exists(COORDS_FILE):
        print(f"Creando archivo dummy de ubicaciones: {COORDS_FILE}")
        with open(COORDS_FILE, 'w') as f:
            for i in range(1, MAX_NODES + 1):
                # Crear clusters geográficos para datos más realistas
                if i < MAX_NODES * 0.3:
                    f.write(f"{random.gauss(10, 2)},{random.gauss(15, 2)}\n")  # Región 1
                elif i < MAX_NODES * 0.6:
                    f.write(f"{random.gauss(30, 3)},{random.gauss(10, 3)}\n")  # Región 2
                else:
                    f.write(f"{random.gauss(20, 4)},{random.gauss(30, 4)}\n")  # Región 3

    if not os.path.exists(NEIGHBORS_FILE):
        print(f"Creando archivo dummy de conexiones: {NEIGHBORS_FILE}")

        # Crear una estructura de red más realista (mundo pequeño)
        base_graph = nx.watts_strogatz_graph(MAX_NODES, k=6, p=0.1, seed=42)
        connections = {}

        for i in range(1, MAX_NODES + 1):
            # Obtener vecinos de la estructura base
            neighbors = list(base_graph.neighbors(i-1))
            # Convertir índices a 1-based y añadir algunas conexiones aleatorias
            connections[i] = [n+1 for n in neighbors] + random.sample(
                range(1, MAX_NODES+1),
                random.randint(0, 2)
            )
            # Eliminar duplicados y auto-conexiones
            connections[i] = list(set([n for n in connections[i] if n != i]))

        with open(NEIGHBORS_FILE, 'w') as f:
            for i in range(1, MAX_NODES + 1):
                f.write(','.join(map(str, connections.get(i, []))) + '\n')

# Cargar coordenadas con límite
def load_coordinates(file_path, limit=MAX_NODES):
    coords = {}
    try:
        with open(file_path, 'r') as file:
            for idx, line in enumerate(file):
                if idx >= limit:
                    break
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        x = float(parts[0].strip())
                        y = float(parts[1].strip())
                        coords[idx + 1] = (x, y)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error cargando coordenadas: {e}")
    return coords

# Cargar vecinos con límite (versión optimizada)
def load_neighbors(file_path, limit=MAX_NODES):
    neighbors = {}
    try:
        with open(file_path, 'r') as file:
            for idx, line in enumerate(file):
                if idx >= limit:
                    break
                parts = line.strip().replace(',', ' ').split()
                if parts:
                    try:
                        neighbors[idx + 1] = list(map(int, parts))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error cargando vecinos: {e}")
    return neighbors

# Construir grafo (versión optimizada)
def build_graph(coords, neighbors):
    G = nx.DiGraph()

    # Añadir nodos con coordenadas
    for node in coords:
        G.add_node(node, pos=coords[node])

    # Añadir aristas en batch
    edges_to_add = []
    for node, nbrs in neighbors.items():
        if node in G.nodes:
            for nbr in nbrs:
                if nbr in G.nodes:
                    edges_to_add.append((node, nbr, {'weight': 1}))

    G.add_edges_from(edges_to_add)
    return G

# Calcular métricas para una componente (versión optimizada)
def calculate_metrics(subgraph):
    if subgraph.number_of_nodes() == 0:
        return {}

    metrics = {
        'Número de nodos': subgraph.number_of_nodes(),
        'Número de aristas': subgraph.number_of_edges(),
        'Densidad': f"{nx.density(subgraph):.4f}",
    }

    # Grado medio
    degrees = dict(subgraph.degree())
    metrics['Grado medio'] = f"{sum(degrees.values()) / metrics['Número de nodos']:.2f}"

    # Camino más corto promedio (muestreo para grandes grafos)
    if metrics['Número de nodos'] > 300:
        sample_size = min(50, metrics['Número de nodos'])
        sample_nodes = random.sample(list(subgraph.nodes()), sample_size)
        total_paths = 0
        path_count = 0

        for node in sample_nodes:
            try:
                paths = nx.single_source_shortest_path_length(subgraph, node)
                for target, length in paths.items():
                    if target != node:
                        total_paths += length
                        path_count += 1
            except:
                continue

        if path_count > 0:
            metrics['Camino promedio'] = f"{total_paths / path_count:.2f}"
        else:
            metrics['Camino promedio'] = "No calculable"
    else:
        try:
            metrics['Camino promedio'] = f"{nx.average_shortest_path_length(subgraph):.2f}"
        except:
            metrics['Camino promedio'] = "No calculable"

    # Coeficiente de clustering
    try:
        metrics['Coef. clustering'] = f"{nx.average_clustering(subgraph):.4f}"
    except:
        metrics['Coef. clustering'] = "No calculable"

    # Diámetro estimado
    try:
        if metrics['Número de nodos'] > 300:
            metrics['Diámetro'] = f"≈{nx.approximation.diameter(subgraph)}"
        else:
            metrics['Diámetro'] = nx.diameter(subgraph.to_undirected())
    except:
        metrics['Diámetro'] = "No calculable"

    return metrics

# Crear aplicación Dash
app = dash.Dash(__name__)
app.title = "Análisis de Red Social X"

# Layout de la aplicación mejorado
app.layout = html.Div([
    html.Div([
        html.H1("Análisis de Red Social 'X'", style={
            'textAlign': 'center',
            'color': '#2c3e50',
            'marginBottom': '10px'
        }),
        html.P("Visualización interactiva de la red social y sus componentes conectadas", style={
            'textAlign': 'center',
            'color': '#7f8c8d',
            'marginBottom': '30px'
        }),
        html.Div(id='status', style={
            'textAlign': 'center',
            'padding': '10px',
            'backgroundColor': '#f8f9fa',
            'marginBottom': '20px'
        })
    ]),

    html.Div([
        html.Div([
            html.H3("Selección de Componente", style={'color': '#2c3e50'}),
            dcc.Dropdown(
                id='component-selector',
                placeholder="Seleccione una componente conectada",
                style={'width': '100%', 'marginBottom': '20px'}
            ),
            html.Div(id='component-info', style={
                'marginBottom': '20px',
                'padding': '15px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px'
            }),
            html.Button('Cargar Datos', id='load-button', n_clicks=0, style={
                'backgroundColor': '#3498db',
                'color': 'white',
                'border': 'none',
                'padding': '10px 15px',
                'borderRadius': '5px',
                'width': '100%'
            })
        ], style={'width': '30%', 'padding': '20px', 'borderRight': '1px solid #ddd'}),

        html.Div([
            dcc.Graph(
                id='graph',
                config={'scrollZoom': True, 'displayModeBar': True},
                style={'height': '70vh', 'border': '1px solid #ddd', 'borderRadius': '5px'}
            )
        ], style={'width': '70%', 'padding': '20px'})
    ], style={'display': 'flex', 'marginBottom': '30px'}),

    html.Div([
        html.H3("Métricas de la Red", style={'color': '#2c3e50'}),
        html.Div(id='analysis-output', style={
            'marginTop': '15px',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px'
        })
    ], style={'padding': '20px', 'borderTop': '1px solid #ddd'}),

    dcc.Store(id='graph-store'),
    dcc.Store(id='components-store'),
    dcc.Store(id='metrics-store')
], style={
    'fontFamily': 'Arial, sans-serif',
    'maxWidth': '1400px',
    'margin': '0 auto',
    'backgroundColor': 'white',
    'padding': '20px'
})

# Callback para cargar datos y construir el grafo
@app.callback(
    [Output('graph-store', 'data'),
     Output('components-store', 'data'),
     Output('component-selector', 'options'),
     Output('status', 'children')],
    [Input('load-button', 'n_clicks')],
    prevent_initial_call=True
)
def load_graph_data(n_clicks):
    if n_clicks == 0:
        return dash.no_update, dash.no_update, dash.no_update, "Presione 'Cargar Datos' para iniciar"

    start_time = time.time()

    # Crear archivos dummy si no existen
    create_dummy_files()

    # Cargar datos
    coords = load_coordinates(COORDS_FILE)
    neighbors = load_neighbors(NEIGHBORS_FILE)

    # Construir grafo
    G = build_graph(coords, neighbors)

    # Calcular componentes débilmente conectadas
    weakly_connected = list(nx.weakly_connected_components(G))

    # Ordenar componentes por tamaño y limitar a las 5 más grandes
    weakly_connected.sort(key=len, reverse=True)
    weakly_connected = weakly_connected[:5]

    # Crear opciones para el dropdown
    options = [{'label': f'Componente {i+1} ({len(comp)} nodos)', 'value': i}
               for i, comp in enumerate(weakly_connected)]

    # Almacenar datos
    graph_data = {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges(data=True))
    }

    components_data = [
        {'id': i, 'nodes': list(comp)}
        for i, comp in enumerate(weakly_connected)
    ]

    load_time = time.time() - start_time
    status = f"Datos cargados en {load_time:.2f} segundos | {G.number_of_nodes()} nodos | {G.number_of_edges()} aristas"

    return graph_data, components_data, options, status

# Callback para actualizar la visualización según la selección
@app.callback(
    [Output('graph', 'figure'),
     Output('analysis-output', 'children'),
     Output('component-info', 'children')],
    [Input('component-selector', 'value'),
     Input('graph-store', 'data'),
     Input('components-store', 'data')]
)
def update_graph(selected_component, graph_data, components_data):
    if selected_component is None or not graph_data or not components_data:
        return go.Figure(), "Seleccione una componente para análisis.", ""

    # Reconstruir el grafo completo
    G = nx.DiGraph()
    for node, data in graph_data['nodes']:
        G.add_node(node, **data)
    for u, v, data in graph_data['edges']:
        G.add_edge(u, v, **data)

    # Obtener la componente seleccionada
    comp_nodes = components_data[selected_component]['nodes']
    subgraph = G.subgraph(comp_nodes)

    # Limitar nodos para visualización
    if len(subgraph.nodes()) > MAX_VISUALIZATION_NODES:
        sample_nodes = random.sample(list(subgraph.nodes()), MAX_VISUALIZATION_NODES)
        subgraph = subgraph.subgraph(sample_nodes)

    # Extraer posiciones
    positions = nx.get_node_attributes(subgraph, 'pos')

    # Calcular layout si no hay posiciones o no son adecuadas
    if not positions or len(positions) != len(subgraph.nodes()):
        positions = nx.spring_layout(subgraph, seed=42)
        for node in subgraph.nodes():
            subgraph.nodes[node]['pos'] = positions[node]

    # Crear trazos de aristas
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        if edge[0] in positions and edge[1] in positions:
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Crear trazos de nodos
    node_x = []
    node_y = []
    node_text = []
    for node in subgraph.nodes():
        if node in positions:
            x, y = positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'Nodo {node}')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=8,
            colorbar=dict(
                thickness=15,
                title='Grado',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=1, color='DarkSlateGrey'))
    )

    # Calcular grado para colorear
    degrees = dict(subgraph.degree())
    node_trace.marker.color = [degrees.get(node, 0) for node in subgraph.nodes()]

    # Crear figura
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Componente {selected_component + 1} - {len(subgraph.nodes())} nodos',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    ))

    # Calcular métricas
    metrics = calculate_metrics(subgraph)

    # Crear tabla de métricas
    metrics_table = []
    for k, v in metrics.items():
        metrics_table.append(html.Tr([
            html.Td(k, style={'padding': '8px', 'borderBottom': '1px solid #ddd', 'fontWeight': 'bold'}),
            html.Td(v, style={'padding': '8px', 'borderBottom': '1px solid #ddd'})
        ]))

    metrics_div = html.Table(
    [html.Tr([html.Th("Métrica"), html.Th("Valor")])] + metrics_table,
    style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'}
)

    # Información de la componente
    comp_info = html.Div([
        html.H4(f"Componente {selected_component + 1}"),
        html.P(f" Nodos: {metrics.get('Número de nodos', 'N/A')}"),
        html.P(f" Aristas: {metrics.get('Número de aristas', 'N/A')}"),
        html.P(f" Densidad: {metrics.get('Densidad', 'N/A')}"),
        html.P(f" Grado medio: {metrics.get('Grado medio', 'N/A')}")
    ])

    return fig, metrics_div, comp_info
# Ejecutar la aplicación con el método correcto
if __name__ == '__main__':
    app.run(debug=True, port=8050)  # Puedes usar 8859 si prefieres

# from grafo import GrafoDirigido # Se importará en main.py o donde se use

def calcular_estadisticas_basicas(grafo):
    """
    Calcula estadísticas básicas de la red.

    Args:
        grafo (GrafoDirigido): El grafo del cual calcular estadísticas.

    Returns:
        dict: Un diccionario con las estadísticas.
    """
    num_nodos = grafo.obtener_numero_nodos()
    num_aristas = grafo.obtener_numero_aristas()

    grado_salida_promedio = 0
    grado_entrada_promedio = 0
    max_grado_salida = 0
    min_grado_salida = float('inf') if num_nodos > 0 else 0
    max_grado_entrada = 0
    min_grado_entrada = float('inf') if num_nodos > 0 else 0

    grados_salida = []
    grados_entrada = [] # Calcular esto puede ser costoso si se hace nodo por nodo

    if num_nodos > 0:
        suma_grados_salida = 0
        suma_grados_entrada = 0 # Se calcula de forma más eficiente que nodo por nodo

        for nodo in grafo.obtener_todos_los_nodos():
            gs = grafo.grado_salida(nodo)
            grados_salida.append(gs)
            suma_grados_salida += gs
            if gs > max_grado_salida:
                max_grado_salida = gs
            if gs < min_grado_salida:
                min_grado_salida = gs

            # Para grado de entrada, es más eficiente sumar las longitudes de las listas de adyacencia
            # ya que cada arista u->v contribuye con 1 al grado de entrada de v.
            # Sin embargo, para min/max grado de entrada, necesitaríamos calcularlo por nodo o tener adj_inversa
            # ge = grafo.grado_entrada(nodo) # Esto sería ineficiente si se llama para todos los nodos
            # grados_entrada.append(ge)
            # suma_grados_entrada += ge
            # if ge > max_grado_entrada: max_grado_entrada = ge
            # if ge < min_grado_entrada: min_grado_entrada = ge

        grado_salida_promedio = suma_grados_salida / num_nodos

        # El grado de entrada promedio es igual al grado de salida promedio en cualquier grafo dirigido
        # Suma de grados de entrada = Suma de grados de salida = Número de aristas
        grado_entrada_promedio = num_aristas / num_nodos

        # Para min/max grado de entrada, necesitamos una forma más directa o calcularlos
        # Si no tenemos una lista de adyacencia inversa, calcularlo para cada nodo es O(N*(N+M)) o O(N*M) si es disperso
        # Por ahora, dejaremos min/max grado entrada como placeholders o los calcularemos si es necesario
        # de forma explícita aunque sea costoso para grafos muy grandes.
        # Alternativa para grafos no demasiado grandes:
        if num_nodos < 100000: # Umbral arbitrario para calcular grados de entrada individuales
             for nodo_id in grafo.obtener_todos_los_nodos():
                ge = grafo.grado_entrada(nodo_id)
                grados_entrada.append(ge)
                if ge > max_grado_entrada: max_grado_entrada = ge
                if ge < min_grado_entrada: min_grado_entrada = ge
        else:
            print("Advertencia: Min/Max grado de entrada no se calculan para grafos muy grandes por eficiencia.")
            max_grado_entrada = "No calculado"
            min_grado_entrada = "No calculado"


    if num_nodos == 0: # Evitar división por cero y manejar grafo vacío
        min_grado_salida = 0
        min_grado_entrada = 0

    estadisticas = {
        "Número de Nodos": num_nodos,
        "Número de Aristas": num_aristas,
        "Grado de Salida Promedio": f"{grado_salida_promedio:.2f}" if num_nodos > 0 else 0,
        "Grado de Entrada Promedio": f"{grado_entrada_promedio:.2f}" if num_nodos > 0 else 0,
        "Máximo Grado de Salida": max_grado_salida,
        "Mínimo Grado de Salida": min_grado_salida if num_nodos > 0 else 0,
        "Máximo Grado de Entrada": max_grado_entrada if grados_entrada else ("No calculado" if num_nodos > 0 else 0) ,
        "Mínimo Grado de Entrada": min_grado_entrada if grados_entrada and min_grado_entrada != float('inf') else ("No calculado" if num_nodos > 0 else 0),
        # Podríamos añadir densidad aquí: num_aristas / (num_nodos * (num_nodos - 1)) para dirigidos
    }
    return estadisticas

def top_n_usuarios_por_grado_salida(grafo, n=10):
    """
    Encuentra los N usuarios con el mayor grado de salida.
    """
    if not grafo.obtener_numero_nodos():
        return []

    grados = []
    for nodo in grafo.obtener_todos_los_nodos():
        grados.append((nodo, grafo.grado_salida(nodo)))

    # Ordenar por grado de salida en orden descendente
    grados.sort(key=lambda x: x[1], reverse=True)

    return grados[:n]

def top_n_usuarios_por_grado_entrada(grafo, n=10):
    """
    Encuentra los N usuarios con el mayor grado de entrada.
    Esto puede ser costoso si grado_entrada no está optimizado.
    """
    if not grafo.obtener_numero_nodos():
        return []

    grados_entrada_calculados = []
    # Calcular grado de entrada para todos si es necesario (puede ser lento)
    # Alternativa: si el grafo es pequeño o si esta métrica es muy importante.
    print("Calculando grados de entrada para top N (puede ser lento)...")
    for nodo in grafo.obtener_todos_los_nodos():
        grados_entrada_calculados.append((nodo, grafo.grado_entrada(nodo)))

    grados_entrada_calculados.sort(key=lambda x: x[1], reverse=True)

    return grados_entrada_calculados[:n]


if __name__ == '__main__':
    from grafo import GrafoDirigido # Importar para pruebas

    g_metricas = GrafoDirigido()
    g_metricas.agregar_arista(1, 2)
    g_metricas.agregar_arista(1, 3)
    g_metricas.agregar_arista(2, 3)
    g_metricas.agregar_arista(3, 1)
    g_metricas.agregar_arista(3, 4)
    g_metricas.agregar_arista(4, 1) # 1 tiene alto grado de entrada
    g_metricas.agregar_nodo(5)     # Nodo aislado

    print("\n--- Prueba de Estadísticas Básicas ---")
    stats = calcular_estadisticas_basicas(g_metricas)
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n--- Prueba Top N por Grado de Salida ---")
    top_salida = top_n_usuarios_por_grado_salida(g_metricas, n=3)
    print(f"Top 3 por grado de salida: {top_salida}")
    # Esperado (ejemplo): [(1,2), (3,2), (2,1)] o similar, depende de desempates

    print("\n--- Prueba Top N por Grado de Entrada ---")
    top_entrada = top_n_usuarios_por_grado_entrada(g_metricas, n=3)
    print(f"Top 3 por grado de entrada: {top_entrada}")
    # Esperado (ejemplo): [(1,2), (3,2), (2,1)] o similar. Nodo 1 tiene entrada de 3 y 4. Nodo 3 tiene de 1 y 2.
    # 1 <- 3, 4 (ge=2)
    # 2 <- 1 (ge=1)
    # 3 <- 1, 2 (ge=2)
    # 4 <- 3 (ge=1)
    # 5 <- (ge=0)
    # Esperado: [(1,2), (3,2), (2,1)] o [(3,2), (1,2), (2,1)] etc.

    g_vacio = GrafoDirigido()
    print("\n--- Prueba de Estadísticas Básicas (Grafo Vacío) ---")
    stats_vacio = calcular_estadisticas_basicas(g_vacio)
    for k, v in stats_vacio.items():
        print(f"{k}: {v}")

    print("\nPruebas de metricas_red.py completadas.")
