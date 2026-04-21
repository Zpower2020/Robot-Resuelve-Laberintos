# Importaciones necesarias para ROS 2 y métricas de sistema
import rclpy                                # Librería principal de ROS 2 para Python.
from rclpy.node import Node                 # Clase base para crear un "Nodo" (programa independiente) en ROS 2.
from std_msgs.msg import String             # Tipo de mensaje estándar de ROS 2 (texto simple) para mandar y recibir datos.
import time                                 # Librería para medir tiempos (timestamps) y calcular la latencia de la red.
import psutil                               # Librería del sistema operativo para leer el consumo de RAM y CPU de la laptop.

# =====================================================================
# CLASE PRINCIPAL: Nodo de Comunicación del Robot
# =====================================================================
class MazeRobotNode(Node):
    def __init__(self):
        # Inicia el nodo con el nombre oficial 'maze_vision_controller' dentro de la red ROS 2.
        super().__init__('maze_vision_controller')
        
        # EL "ALTAVOZ" (Publicador):
        # Crea un canal (tópico) llamado '/cmd_maze_robot'. Aquí enviará mensajes tipo String al ESP32. 
        # El '10' es el tamaño del buffer (guarda hasta 10 mensajes en cola si hay lag).
        self.publisher_ = self.create_publisher(String, '/cmd_maze_robot', 10)
        
        # EL "MICRÓFONO" (Suscriptor):
        # Escucha un canal llamado '/estado_robot' donde el ESP32 envía sus confirmaciones.
        # Cuando llega un mensaje, ejecuta automáticamente la función 'self.listener_callback'.
        self.subscription = self.create_subscription(String, '/estado_robot', self.listener_callback, 10)
        
        # Variables de estado interno para el control del Lazo Cerrado (Closed-Loop)
        self.estado_auto = ""               # Guarda el último texto literal que respondió el ESP32.
        self.ultimo_comando = ""            # Guarda la última orden que Ubuntu decidió enviar.
        self.ultimo_tiempo_envio = 0        # Guarda el momento exacto (timestamp) en que se disparó el último comando.
        
        # DICCIONARIO DE VALIDACIÓN (El "Traductor"):
        # Relaciona la orden corta que envía Ubuntu con la respuesta larga que devuelve el código de Arduino del ESP32.
        # Esto es vital para saber si el auto realmente nos hizo caso.
        self.mapa_estados = {
            "ADELANTE": "ACCION: Yendo hacia adelante",
            "CORREGIR_DERECHA": "ACCION: Corrigiendo a la derecha",
            "CORREGIR_IZQUIERDA": "ACCION: Corrigiendo a la izquierda",
            "DERECHA": "ACCION: Girando 90 grados a la derecha",
            "IZQUIERDA": "ACCION: Girando 90 grados a la izquierda",
            "STOP": "ACCION: Robot detenido",
            "RETROCEDER": "ACCION: Yendo hacia atras"
        }
        
        # --- VARIABLES PARA RECOLECCIÓN DE DATOS EXPERIMENTALES (TESIS/DIPLOMADO) ---
        self.latencias = []                 # Lista para guardar los tiempos de respuesta (Ping).
        self.jitters = []                   # Lista para guardar la variación de la latencia (inestabilidad de la red).
        self.cpus = []                      # Lista para guardar qué porcentaje del procesador de la laptop se usó.
        self.memorias = []                  # Lista para guardar el consumo de memoria RAM.
        self.energias = []                  # Lista para guardar una estimación matemática de la eficiencia.
        self.recoleccion_activa = False     # Bandera (Switch) que dice si debe o no debe guardar datos ahora mismo.
        self.ultima_latencia = 0.0          # Memoria temporal para calcular el jitter restando la latencia actual menos la anterior.
        self.proceso = psutil.Process()     # Conecta Python con el sistema operativo de la laptop para poder leer sus métricas.
        
        # Imprime un mensaje verde en la terminal de Ubuntu para confirmar que todo arrancó bien.
        self.get_logger().info('🤖 CEREBRO VISUAL INICIADO. Verificación de Lazo Cerrado activada.')

    # =====================================================================
    # FUNCIONES DE CONTROL DE MÉTRICAS (Para generar el archivo TXT)
    # =====================================================================
    def iniciar_recoleccion(self):
        # Esta función es llamada por 'navegacion_ar.py' cuando el usuario presiona la tecla 'i' (Iniciar).
        self.latencias.clear()              # Vacía la basura de pruebas anteriores.
        self.jitters.clear()
        self.cpus.clear()
        self.memorias.clear()
        self.energias.clear()
        self.ultima_latencia = 0.0
        self.recoleccion_activa = True      # "Enciende la grabadora" de datos.

    def detener_recoleccion(self):
        # Esta función es llamada cuando el auto cruza la meta para dejar de recolectar datos inútiles.
        self.recoleccion_activa = False     # "Apaga la grabadora".

    def registrar_metricas(self, latencia):
        # Esta función se activa automáticamente cada vez que se calcula un ping válido (ida y vuelta del mensaje).
        if not self.recoleccion_activa: return # Si la carrera no ha empezado o ya terminó, ignora el dato.
        
        # Calcula el Jitter: La diferencia de tiempo absoluto entre la latencia de este instante y la del instante pasado.
        jitter = abs(latencia - self.ultima_latencia) if self.latencias else 0.0
        self.ultima_latencia = latencia     # Actualiza la variable temporal.
        
        # Lee los sensores físicos de la computadora (Laptop)
        cpu = self.proceso.cpu_percent()    # Qué porcentaje del CPU está usando el programa de Python en este milisegundo.
        mem = self.proceso.memory_info().rss / (1024 * 1024) # Lee la memoria en Bytes y la divide para convertirla a Megabytes (MB).
        
        # Calcula una métrica de "Eficiencia": Empieza en 100% y le resta la mitad del uso del CPU. (Es una fórmula teórica estimada).
        energia = max(0.0, 100.0 - (cpu * 0.5)) 
        
        # Añade todas las métricas calculadas a las listas largas para imprimirlas al final en el TXT.
        self.latencias.append(latencia)
        self.jitters.append(jitter)
        self.cpus.append(cpu)
        self.memorias.append(mem)
        self.energias.append(energia)

    # =====================================================================
    # FUNCIÓN ESCUCHA (Callback del Suscriptor)
    # =====================================================================
    def listener_callback(self, msg):
        # Esta función salta instantáneamente cada vez que el ESP32 del auto nos manda un texto.
        tiempo_recepcion = time.time()      # Anota la hora exacta (milisegundo) en la que llegó la respuesta por WiFi.
        self.estado_auto = msg.data         # Extrae el texto plano del paquete ROS 2 y lo guarda en la memoria.
        
        # Traduce la orden que mandamos ("ADELANTE") a lo que el auto debería estar gritando ("ACCION: Yendo hacia adelante").
        estado_esperado = self.mapa_estados.get(self.ultimo_comando, "")
        
        # CÁLCULO DE LATENCIA PERFECTA (LAZO CERRADO):
        # Si el auto grita exactamente lo que esperábamos escuchar, y si nosotros efectivamente habíamos enviado algo...
        if self.estado_auto == estado_esperado and self.ultimo_tiempo_envio > 0:
            # Calcula la Latencia de Red Total: (Hora de recepción) - (Hora de envío). Es el tiempo del viaje ida+vuelta.
            latencia = tiempo_recepcion - self.ultimo_tiempo_envio
            
            # Manda este dato a la función de arriba para guardarlo en las estadísticas.
            self.registrar_metricas(latencia)
            
            # Resetea el reloj a 0. Esto evita que, si el auto sigue gritando "Estoy yendo hacia adelante",
            # el código cuente esas repeticiones infinitas como nuevas latencias. Solo mide el primer acierto.
            self.ultimo_tiempo_envio = 0 

    # =====================================================================
    # FUNCIÓN HABLANTE (El transmisor de órdenes)
    # =====================================================================
    def publicar_comando(self, comando):
        # Esta función es invocada 30 veces por segundo por 'navegacion_ar.py', pasándole la palabra (ej. "ADELANTE").
        estado_esperado = self.mapa_estados.get(comando, "") # Busca qué debería responder el auto a esta orden.
        tiempo_actual = time.time() # Anota la hora actual.

        # -----------------------------------------------------------
        # EL SISTEMA DE SEGURIDAD DE RED (REINTENTO OBLIGATORIO):
        # -----------------------------------------------------------
        # Verifica 3 cosas simultáneamente:
        # 1. (self.estado_auto != estado_esperado): ¿El auto está haciendo algo distinto a lo que le mandé?
        # 2. (self.ultimo_tiempo_envio > 0): ¿Efectivamente le mandé una orden y estoy esperando confirmación?
        # 3. (tiempo_actual - self.ultimo_tiempo_envio > 0.3): ¿Ya pasaron 0.3 segundos (300ms) desde que le mandé la orden y el auto sigue ignorándome?
        necesita_reenvio = (self.estado_auto != estado_esperado) and (self.ultimo_tiempo_envio > 0) and (tiempo_actual - self.ultimo_tiempo_envio > 0.3)

        # -----------------------------------------------------------
        # EL FILTRO ANTI-SPAM:
        # -----------------------------------------------------------
        # Si la orden es NUEVA (distinta a la anterior) O si el sistema de seguridad dictamina que hay que reenviar porque se perdió el paquete por WiFi...
        if comando != self.ultimo_comando or necesita_reenvio:
            msg = String()          # Crea el empaque vacío del mensaje estándar de ROS 2.
            msg.data = comando      # Mete la palabra dentro del empaque.
            self.publisher_.publish(msg) # ¡DISPARA EL MENSAJE HACIA EL ROBOT POR WIFI!
            
            # (Solamente para imprimir en la consola de Ubuntu y ver qué pasa visualmente)
            if comando != self.ultimo_comando:
                # Si es una orden nueva, imprime en blanco normal.
                self.get_logger().info(f'📡 Transmitiendo orden: {comando}')
            else:
                # Si es un reenvío por falla de red, imprime una ADVERTENCIA en amarillo.
                self.get_logger().warn(f'🔄 Reintentando orden: {comando} (El ESP32 reporta: {self.estado_auto})')
            
            # Actualiza las memorias vitales para el ciclo:
            self.ultimo_comando = comando            # Anota cuál fue la última orden que mandó.
            self.ultimo_tiempo_envio = tiempo_actual # Enciende el cronómetro de la latencia.

    # =====================================================================
    # FUNCIÓN DE VERIFICACIÓN (Check de Sincronía)
    # =====================================================================
    def comando_confirmado(self):
        # Esta función es consultada por la Lógica de Atascos en 'navegacion_ar.py'.
        # Si no hemos mandado nada aún, retorna Falso (no estamos sincronizados).
        if not self.ultimo_comando:
            return False
            
        # Pregunta al traductor: ¿Qué respuesta espero para la última orden que mandé?
        estado_esperado = self.mapa_estados.get(self.ultimo_comando, "")
        
        # Devuelve VERDADERO si el auto está gritando exactamente la respuesta que esperábamos,
        # lo que significa que el WiFi funciona perfecto y ambos (Ubuntu y ESP32) están en perfecta sincronía.
        return self.estado_auto == estado_esperado
