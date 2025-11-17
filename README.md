# BigData-CNN-Proyecto
Proyecto final del módulo CNN de la materia BigData

**Creado por Juan Manuel Garzón Gil cod 20251695004**

# Inventario Automático del Salón de Cómputo
Sistema de detección automática de objetos usando YOLOv11 para inventariar elementos del salón de cómputo (CPU, mesa, mouse, pantalla, silla, teclado).

## Arquitectura del Proyecto

El proyecto usa:
- **Backend**: Servidor Python con Flask que ejecuta el modelo YOLOv11
- **Frontend**: Interfaz web HTML/JavaScript que se comunica con el servidor
- **Modelo**: YOLOv11 nano entrenado con transfer learning en dataset personalizado de Roboflow

## Requisitos Previos

- Python 3.11+
- pip (gestor de paquetes de Python)
- Navegador web moderno

## Instalación

### 1. Clonar o descargar el repositorio

```bash
git clone <URL-del-repositorio>
cd proyecto_inventario
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Si tiene problemas con NumPy, ejecute:

```bash
pip install "numpy<2"
```

### 3. Descargar el modelo entrenado

**IMPORTANTE**: El archivo `best.pt` NO está incluido en el repositorio por su tamaño (~5MB).

Debes obtenerlo de una de estas formas:

#### Opción A: Descarga rápida desde Google Drive (RECOMENDADO)

Si tienes dificultades o quieres todo listo de una vez, descarga el archivo completo desde Google Drive:

**[DESCARGAR PROYECTO COMPLETO CON MODELO](https://drive.google.com/file/d/1zjUz9k0nU72QeMyFWKnH0dsKPCSQlGSy/view?usp=drive_link)**

**[Visualizar el codigo del entrenamiento en colab](https://colab.research.google.com/drive/1I5JoN4pPWJ7nf4qonWL7sEuy4gLmoKZb?usp=sharing)**

Este archivo `.rar` contiene:
- Proyecto completo estructurado
- Modelo `best.pt` ya incluido
- Todos los archivos necesarios para ejecutar
- Solo descomprime y sigue el paso 4

#### Opción B: Desde Colab (si prefieres entrenar o regenerar)

1. Accede al notebook: `proyecto1.ipynb` (proporcionado en el repositorio)
2. Ejecuta en Google Colab
3. En la última celda, descarga el archivo `best.pt`
4. Coloca el archivo en la raíz del proyecto:

```
proyecto_inventario/
├── best.pt          <-- AQUI
├── servidor.py
├── index.html
├── app.js
├── styles.css
├── requirements.txt
└── model/
    └── labels.txt
```

#### Opción C: Solicitar directamente

Si no tienes acceso a Colab ni Google Drive, solicita el archivo `best.pt` al desarrollador del proyecto.

### 4. Verificar estructura de carpetas

Asegúrate de que tu proyecto esté así:

```
proyecto_inventario/
├── best.pt
├── servidor.py
├── index.html
├── app.js
├── styles.css
├── requirements.txt
├── README.md
├── .gitignore
└── model/
    └── labels.txt
```

## Uso

### Paso 1: Iniciar el servidor

Abre PowerShell/Terminal en la carpeta del proyecto y ejecuta:

```bash
python servidor.py
```

Deberías ver:

```
Intentando cargar modelo...
Modelo YOLO cargado.
Servidor corriendo en http://127.0.0.1:5000
```

**Deja este terminal abierto mientras uses la aplicación.**

### Paso 2: Abrir la interfaz web

1. Abre tu navegador
2. Ve a: `http://127.0.0.1:3000/index.html` (o el puerto local que uses)
3. Deberías ver la interfaz del inventario

### Paso 3: Usar la aplicación

1. Haz clic en "Seleccionar archivo" y elige una imagen del salón de cómputo
2. Haz clic en "Detectar objetos"
3. El servidor procesará la imagen y mostrará:
   - Imagen original (izquierda)
   - Resultado con bounding boxes (derecha)
   - Tabla con conteo de objetos detectados

## Cómo Funciona el Modelo

### Entrenamiento

El modelo fue entrenado con:
- **Base**: YOLOv11 nano (preentrenado en COCO)
- **Dataset**: 290 imágenes del salón de cómputo
  - 203 imágenes de entrenamiento
  - 58 imágenes de validación
  - 29 imágenes de prueba
- **Clases**: 6 objetos
  - cpu
  - mesa
  - mouse
  - pantalla
  - silla
  - teclado
- **Epochs**: 50
- **Optimizer**: Adam
- **Augmentación**: Rotación, escalado, perspectiva

### Rendimiento Final

Métricas de validación después del entrenamiento:

- **Precisión (mAP50)**: 0.938 (93.8%)
- **Recall**: 0.877 (87.7%)
- **Confianza umbral**: 0.45 (detecciones con >45% de confianza)

Por clase:
- CPU: 0.956 mAP50
- Silla: 0.995 mAP50
- Mouse: 0.973 mAP50
- Pantalla: 0.892 mAP50
- Mesa: 0.855 mAP50
- Teclado: 0.933 mAP50

### Flujo de Detección

1. Usuario carga imagen en navegador
2. Navegador envía imagen base64 al servidor (puerto 5000)
3. Servidor recibe la imagen y la procesa con YOLOv11
4. Modelo YOLOv11 predice bounding boxes y clases
5. Servidor devuelve detecciones en formato JSON
6. Navegador dibuja bounding boxes sobre la imagen
7. Navegador muestra tabla con conteo de objetos

## Archivos del Proyecto

- **servidor.py**: Backend Flask que carga el modelo y procesa imágenes
- **index.html**: Interfaz web (estructura HTML)
- **app.js**: Lógica de frontend (comunicación con servidor, dibujo)
- **styles.css**: Estilos visuales
- **model/labels.txt**: Nombres de las 6 clases a detectar
- **requirements.txt**: Dependencias de Python
- **proyecto1.ipynb**: Notebook de Colab con todo el entrenamiento

## Troubleshooting

### Error: "No such file or directory: 'best.pt'"

Solución: El archivo `best.pt` no está en la raíz del proyecto. Descárgalo de Colab siguiendo las instrucciones de la sección "Descargar el modelo".

### Error: "ModuleNotFoundError: No module named 'flask'"

Solución: Instala las dependencias:

```bash
pip install -r requirements.txt
```

### Error: "Cannot read properties of undefined"

Solución: Asegúrate de que el servidor está corriendo (`python servidor.py`) en otra terminal.

### La aplicación es lenta

Normal. La primera detección tarda más. El modelo necesita cargar en memoria. Las detecciones posteriores son más rápidas.

## Tecnologías Utilizadas

- **YOLOv11**: Framework de detección de objetos
- **Flask**: Servidor web Python
- **OpenCV**: Procesamiento de imágenes
- **Roboflow**: Plataforma para datasets
- **TensorFlow/PyTorch**: Backend del modelo YOLO

## Notas Importantes

1. El modelo solo detecta los 6 objetos para los que fue entrenado
2. La precisión depende de la calidad de la imagen y el ángulo
3. Funciona mejor con imágenes del salón de cómputo (no generaliza a otros espacios)
4. Requiere una conexión HTTP local (no funciona en HTTPS sin configuración especial)

## Autor

Desarrollado como proyecto de Big Data y Machine Learning.

## Licencia

MIT (o la que especifiques)

## Contacto

Para preguntas o problemas, contacta al desarrollador.
