#include <Arduino.h>
#include <micro_ros_arduino.h>

#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/string.h>

// --- Pines de los Motores ---
int motor1Pin1 = 32; 
int motor1Pin2 = 33; 
int enable1Pin = 18; // PWM Izquierdo

int motor2Pin1 = 26; 
int motor2Pin2 = 25; 
int enable2Pin = 27; // PWM Derecho

// --- Configuración PWM (ESP32 v3.x) ---
const int freq = 20000;
const int resolution = 8; 

// --- VARIABLES DE VELOCIDAD (PWM de 0 a 255) ---
int pwmBase = 41;      // Velocidad normal para ir recto
int pwmGiro = 66;
int pwmRetroceder = 45;

// --- VARIABLES DE ROS 2 ---
rcl_subscription_t subscriber;
rcl_publisher_t publisher; // NUEVO: Publicador
std_msgs__msg__String msg_sub; // Mensaje entrante
std_msgs__msg__String msg_pub; // NUEVO: Mensaje saliente

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;

// Macros de chequeo de errores para micro-ROS
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

void error_loop(){
  while(1){
    delay(100);
  }
}

// ==========================================
// NUEVO: FUNCIÓN PARA PUBLICAR MENSAJES
// ==========================================
void publicarMensaje(const char* texto) {
  strcpy(msg_pub.data.data, texto);
  msg_pub.data.size = strlen(msg_pub.data.data);
  
  // Publicamos el mensaje
  rcl_publish(&publisher, &msg_pub, NULL);
  
  // También lo mostramos por el puerto serial para debugging
  Serial.print("📢 Enviando a ROS 2: ");
  Serial.println(texto);
}

// ==========================================
// FUNCIONES DE MOVIMIENTO
// ==========================================
void avanzar() {
  digitalWrite(motor1Pin1, HIGH); digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH); digitalWrite(motor2Pin2, LOW);
  ledcWrite(enable1Pin, pwmBase);
  ledcWrite(enable2Pin, pwmBase);
}

void retroceder() {
  digitalWrite(motor1Pin1, LOW); digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW); digitalWrite(motor2Pin2, HIGH);
  ledcWrite(enable1Pin, pwmRetroceder);
  ledcWrite(enable2Pin, pwmRetroceder);
}

void corregirDerecha() {
  digitalWrite(motor1Pin1, HIGH); digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH); digitalWrite(motor2Pin2, LOW); 
  ledcWrite(enable1Pin, pwmGiro); 
  ledcWrite(enable2Pin, 0);       
}

void corregirIzquierda() {
  digitalWrite(motor1Pin1, HIGH); digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH); digitalWrite(motor2Pin2, LOW); 
  ledcWrite(enable1Pin, 0);       
  ledcWrite(enable2Pin, pwmGiro); 
}

void girarDerecha() {
  digitalWrite(motor1Pin1, LOW); digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, HIGH);  digitalWrite(motor2Pin2, LOW); 
  ledcWrite(enable1Pin, pwmGiro); 
  ledcWrite(enable2Pin, pwmGiro);       
}

void girarIzquierda() {
  digitalWrite(motor1Pin1, HIGH);  digitalWrite(motor1Pin2, LOW); 
  digitalWrite(motor2Pin1, LOW); digitalWrite(motor2Pin2, HIGH);
  ledcWrite(enable1Pin, pwmGiro);       
  ledcWrite(enable2Pin, pwmGiro); 
}

void detener() {
  digitalWrite(motor1Pin1, LOW); digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW); digitalWrite(motor2Pin2, LOW);
  ledcWrite(enable1Pin, 0);
  ledcWrite(enable2Pin, 0);
}

// ==========================================
// CALLBACK: RECIBIR ÓRDENES DE ROS 2
// ==========================================
void subscription_callback(const void * msgin) {
  const std_msgs__msg__String * msg = (const std_msgs__msg__String *)msgin;
  String comando = String(msg->data.data);

  Serial.print("📡 Orden recibida: ");
  Serial.println(comando);

  if (comando == "ADELANTE") {
    publicarMensaje("ACCION: Yendo hacia adelante");
    avanzar();
  } 
  else if (comando == "RETROCEDER") {
    publicarMensaje("ACCION: Yendo hacia atras");
    retroceder();
  } 
  else if (comando == "CORREGIR_DERECHA") {
    publicarMensaje("ACCION: Corrigiendo a la derecha");
    corregirDerecha();
  } 
  else if (comando == "CORREGIR_IZQUIERDA") {
    publicarMensaje("ACCION: Corrigiendo a la izquierda");
    corregirIzquierda();
  } 
  else if (comando == "DERECHA") {
    publicarMensaje("ACCION: Girando 90 grados a la derecha");
    girarDerecha();
  } 
  else if (comando == "IZQUIERDA") {
    publicarMensaje("ACCION: Girando 90 grados a la izquierda");
    girarIzquierda();
  } 
  else if (comando == "STOP") {
    publicarMensaje("ACCION: Robot detenido");
    detener();
  } 
  else {
    publicarMensaje("ALERTA: Comando desconocido, freno de emergencia");
    detener(); 
  }
}

// ==========================================
// CONFIGURACIÓN PRINCIPAL
// ==========================================
void setup() {
  Serial.begin(115200);

  // 1. CONFIGURACIÓN WIFI Y MICRO-ROS AGENT
  //set_microros_wifi_transports("ZEGACES", "Elmasfacil1998@", "192.168.0.24", 8888);
  set_microros_wifi_transports("MATEO", "261298mat", "10.172.204.201", 8888);
  //set_microros_wifi_transports("ROS2", "261298mat", "192.168.1.163", 8888);
  //set_microros_wifi_transports("ROS2_5G", "261298mat", "192.168.1.163", 8888);

  // 2. CONFIGURACIÓN DE PINES (Hardware)
  pinMode(enable1Pin, OUTPUT);
  pinMode(enable2Pin, OUTPUT);
  digitalWrite(enable1Pin, LOW); 
  digitalWrite(enable2Pin, LOW);
  pinMode(motor1Pin1, OUTPUT); pinMode(motor1Pin2, OUTPUT);
  pinMode(motor2Pin1, OUTPUT); pinMode(motor2Pin2, OUTPUT);
  ledcAttach(enable1Pin, freq, resolution);
  ledcAttach(enable2Pin, freq, resolution);
  ledcWrite(enable1Pin, 0);
  ledcWrite(enable2Pin, 0);

  // Freno inicial por seguridad
  detener();
  delay(2000); 

  // 3. INICIALIZACIÓN DEL NODO ROS 2
  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "esp32_motor_controller", "", &support));

  // 4. SUSCRIPCIÓN (Recibir comandos)
  RCCHECK(rclc_subscription_init_default(
    &subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "/cmd_maze_robot"));

  msg_sub.data.data = (char * ) malloc(50 * sizeof(char));
  msg_sub.data.capacity = 50;

  // 5. NUEVO: PUBLICADOR (Enviar estados)
  RCCHECK(rclc_publisher_init_default(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "/estado_robot"));

  msg_pub.data.data = (char * ) malloc(50 * sizeof(char));
  msg_pub.data.capacity = 50;

  // 6. INICIALIZACIÓN DEL EJECUTOR
  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &msg_sub, &subscription_callback, ON_NEW_DATA));
  
  // 7. MENSAJE DE ARRANQUE (Detecta reinicios por caídas de voltaje)
  // Como micro-ROS ya está conectado, mandamos el aviso.
  publicarMensaje("¡SISTEMA INICIADO / ESP32 REINICIADA!");

  Serial.println("✅ Nodo ESP32 iniciado. Esperando órdenes...");
}

// ==========================================
// BUCLE INFINITO
// ==========================================
void loop() {
  delay(10);
  // Mantener a ROS 2 escuchando nuevos mensajes constantemente
  RCCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100)));
}
