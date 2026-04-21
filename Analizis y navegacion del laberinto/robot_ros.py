import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import psutil  # NUEVO: Libreria para medir CPU y RAM

class MazeRobotNode(Node):
    def __init__(self):
        super().__init__('maze_vision_controller')
        self.publisher_ = self.create_publisher(String, '/cmd_maze_robot', 10)
        self.subscription = self.create_subscription(String, '/estado_robot', self.listener_callback, 10)
        self.estado_auto = "" 
        self.ultimo_comando = ""
        self.ultimo_tiempo_envio = 0
        
        # Diccionario de validación: Comando enviado -> Respuesta esperada
        self.mapa_estados = {
            "ADELANTE": "ACCION: Yendo hacia adelante",
            "CORREGIR_DERECHA": "ACCION: Corrigiendo a la derecha",
            "CORREGIR_IZQUIERDA": "ACCION: Corrigiendo a la izquierda",
            "DERECHA": "ACCION: Girando 90 grados a la derecha",
            "IZQUIERDA": "ACCION: Girando 90 grados a la izquierda",
            "STOP": "ACCION: Robot detenido",
            "RETROCEDER": "ACCION: Yendo hacia atras"
        }
        
        # --- NUEVAS VARIABLES PARA MÉTRICAS DE RED Y HARDWARE ---
        self.latencias = []
        self.jitters = []
        self.cpus = []
        self.memorias = []
        self.energias = []
        self.recoleccion_activa = False
        self.ultima_latencia = 0.0
        self.proceso = psutil.Process()
        
        self.get_logger().info('🤖 CEREBRO VISUAL INICIADO. Verificación de Lazo Cerrado activada.')

    # --- FUNCIONES DE CONTROL DE MÉTRICAS ---
    def iniciar_recoleccion(self):
        self.latencias.clear()
        self.jitters.clear()
        self.cpus.clear()
        self.memorias.clear()
        self.energias.clear()
        self.ultima_latencia = 0.0
        self.recoleccion_activa = True

    def detener_recoleccion(self):
        self.recoleccion_activa = False

    def registrar_metricas(self, latencia):
        if not self.recoleccion_activa: return
        jitter = abs(latencia - self.ultima_latencia) if self.latencias else 0.0
        self.ultima_latencia = latencia
        
        cpu = self.proceso.cpu_percent()
        mem = self.proceso.memory_info().rss / (1024 * 1024) # En MB
        energia = max(0.0, 100.0 - (cpu * 0.5)) # Eficiencia estimada en base a la carga de procesamiento
        
        self.latencias.append(latencia)
        self.jitters.append(jitter)
        self.cpus.append(cpu)
        self.memorias.append(mem)
        self.energias.append(energia)

    def listener_callback(self, msg):
        tiempo_recepcion = time.time()
        self.estado_auto = msg.data
        
        # Calcular latencia solo cuando el ESP32 confirma la instrucción exacta
        estado_esperado = self.mapa_estados.get(self.ultimo_comando, "")
        if self.estado_auto == estado_esperado and self.ultimo_tiempo_envio > 0:
            latencia = tiempo_recepcion - self.ultimo_tiempo_envio
            self.registrar_metricas(latencia)
            self.ultimo_tiempo_envio = 0 # Reset para no contar la misma latencia dos veces

    def publicar_comando(self, comando):
        estado_esperado = self.mapa_estados.get(comando, "")
        tiempo_actual = time.time()

        # LÓGICA DE REINTENTO: 
        # Si el robot no reporta el estado correcto y ha pasado medio segundo, forzamos un reenvío
        necesita_reenvio = (self.estado_auto != estado_esperado) and (self.ultimo_tiempo_envio > 0) and (tiempo_actual - self.ultimo_tiempo_envio > 0.3)

        if comando != self.ultimo_comando or necesita_reenvio:
            msg = String()
            msg.data = comando
            self.publisher_.publish(msg)
            
            if comando != self.ultimo_comando:
                self.get_logger().info(f'📡 Transmitiendo orden: {comando}')
            else:
                self.get_logger().warn(f'🔄 Reintentando orden: {comando} (El ESP32 reporta: {self.estado_auto})')
            
            self.ultimo_comando = comando
            self.ultimo_tiempo_envio = tiempo_actual # Guardamos tiempo para calcular la latencia

    # --- NUEVA FUNCIÓN: Verifica si el robot y Ubuntu están sincronizados ---
    def comando_confirmado(self):
        if not self.ultimo_comando:
            return False
        estado_esperado = self.mapa_estados.get(self.ultimo_comando, "")
        return self.estado_auto == estado_esperado
